import collections
import time
from ortools.sat.python import cp_model

NO_EQUIPMENT = "NO_EQUIPMENT"


# -------------------------------
# 중간 해 출력 콜백
# -------------------------------
class ProgressSolutionPrinter(cp_model.CpSolverSolutionCallback):
    def __init__(self, makespan_var, interval_sec=5):
        super().__init__()
        self._makespan = makespan_var
        self._solution_count = 0
        self._last_print_time = time.time()
        self._interval = interval_sec

    def on_solution_callback(self):
        self._solution_count += 1
        print(
            f"[Solution #{self._solution_count}] "
            f"time={self.WallTime():.1f}s "
            f"makespan={self.Value(self._makespan)}"
        )


# -------------------------------
# 데이터 전처리
# -------------------------------
def flatten_data(data):
    rows = []
    for product in data["scenarioProductList"]:
        product_id = product["id"]
        qty = product.get("qty", 1)

        for i in range(qty):
            pid = f"{product_id}__{i+1}"
            for task in product["scenarioTasks"]:
                rows.append({
                    "product_instance_id": pid,
                    "product_id": product_id,
                    "task_id": task["id"],
                    "seq": task["seq"],
                    "duration": task["duration"],
                    "required_workers": task.get("requiredWorkers", 1),
                    "tool_category_id": task.get("toolCategory", {}).get("id"),
                })
    rows.sort(key=lambda r: (r["product_instance_id"], r["seq"]))
    return rows


def group_by_product(rows):
    d = collections.defaultdict(list)
    for r in rows:
        d[r["product_instance_id"]].append(r)
    return d


def build_tools_by_category(tools):
    d = collections.defaultdict(list)
    for t in tools:
        d[t["category"]["id"]].append(t["id"])
    return dict(d)


def build_tool_to_category_map(tools):
    return {t["id"]: t["category"]["id"] for t in tools}


# -------------------------------
# 메인 Solver
# -------------------------------
def solve_scenario(data):
    rows = flatten_data(data)
    products = group_by_product(rows)
    tools_by_category = build_tools_by_category(data["tools"])
    tool_to_category = build_tool_to_category_map(data["tools"])

    horizon = sum(r["duration"] for r in rows)

    model = cp_model.CpModel()

    var_map = {}
    all_end_vars = []

    worker_intervals = []
    worker_demands = []

    tool_intervals = collections.defaultdict(list)

    # 🔥 핵심: 같은 task_id 전역 직렬화
    task_intervals_by_task_id = collections.defaultdict(list)

    for pid, tasks in products.items():
        prev_end = None

        for task in tasks:
            start = model.NewIntVar(0, horizon, f"s_{pid}_{task['seq']}")
            end = model.NewIntVar(0, horizon, f"e_{pid}_{task['seq']}")
            model.Add(end == start + task["duration"])

            if prev_end is not None:
                model.Add(start >= prev_end)
            prev_end = end

            interval = model.NewIntervalVar(
                start, task["duration"], end,
                f"iv_{pid}_{task['task_id']}"
            )

            # Worker
            worker_intervals.append(interval)
            worker_demands.append(task["required_workers"])

            # Tool
            cat = task["tool_category_id"]
            if cat and cat != NO_EQUIPMENT:
                tool_intervals[cat].append(interval)

            # 🔥 같은 task_id 묶기
            task_intervals_by_task_id[task["task_id"]].append(interval)

            var_map[(pid, task["task_id"])] = {
                "start": start,
                "end": end,
                "duration": task["duration"],
                "tool_category_id": cat,
                "product_id": task["product_id"],
                "required_workers": task["required_workers"],
            }

            all_end_vars.append(end)

    # -------------------------------
    # 제약 조건
    # -------------------------------
    model.AddCumulative(
        worker_intervals,
        worker_demands,
        data["scenario"]["maxWorkerCount"]//2
    )

    for cat, intervals in tool_intervals.items():
        model.AddCumulative(
            intervals,
            [1] * len(intervals),
            len(tools_by_category[cat])
        )

    # 🔥🔥🔥 절대 겹침 방지 (같은 task_id)
    for task_id, intervals in task_intervals_by_task_id.items():
        model.AddNoOverlap(intervals)

    makespan = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(makespan, all_end_vars)
    model.Minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 43200
    solver.parameters.log_search_progress = True

    status = solver.Solve(
        model,
        ProgressSolutionPrinter(makespan)
    )

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {
            "status": solver.StatusName(status),
            "makespan": 0,
            "schedules": [],
            "analysis": None,
        }

    # -------------------------------
    # 결과 생성
    # -------------------------------
    timeline = []
    for (pid, tid), v in var_map.items():
        timeline.append({
            "product_id": v["product_id"],
            "task_id": tid,
            "tool_category_id": v["tool_category_id"],
            "tool_id": None,
            "start": solver.Value(v["start"]),
            "end": solver.Value(v["end"]),
            "duration": v["duration"],
            "required_workers": v["required_workers"],
        })

    # -------------------------------
    # Tool 배정 (null 방지)
    # -------------------------------
    tasks_by_cat = collections.defaultdict(list)
    for t in timeline:
        if t["tool_category_id"] and t["tool_category_id"] != NO_EQUIPMENT:
            tasks_by_cat[t["tool_category_id"]].append(t)

    for cat, tasks in tasks_by_cat.items():
        tools = tools_by_category.get(cat, [])
        avail = {tid: 0 for tid in tools}
        tasks.sort(key=lambda x: x["start"])

        for task in tasks:
            for tid in tools:
                if avail[tid] <= task["start"]:
                    task["tool_id"] = tid
                    avail[tid] = task["end"]
                    break

            if task["tool_id"] is None and tools:
                task["tool_id"] = tools[0]  # 안전장치

    # -------------------------------
    # 🔥 분석 계산 (완전 복구)
    # -------------------------------
    total_time = solver.Value(makespan)

    tool_usage = collections.defaultdict(int)
    for t in timeline:
        if t["tool_id"]:
            tool_usage[t["tool_id"]] += t["duration"]

    if tool_usage:
        tool_id, usage = max(tool_usage.items(), key=lambda x: x[1])
        bottleneck_tool = {
            "tool": tool_id,
            "toolCategoryId": tool_to_category.get(tool_id),
            "totalUsageTime": usage,
        }
    else:
        bottleneck_tool = {
            "tool": None,
            "toolCategoryId": None,
            "totalUsageTime": 0,
        }

    total_worker_time = sum(
        t["duration"] * t["required_workers"] for t in timeline
    )
    worker_util = total_worker_time / (
        total_time * data["scenario"]["maxWorkerCount"]
    )

    idle_times = []
    jobs = collections.defaultdict(list)
    for t in timeline:
        jobs[t["product_id"]].append(t)

    for tasks in jobs.values():
        tasks.sort(key=lambda x: x["start"])
        for i in range(len(tasks) - 1):
            idle = tasks[i + 1]["start"] - tasks[i]["end"]
            if idle > 0:
                idle_times.append(idle)

    avg_idle = sum(idle_times) / len(idle_times) if idle_times else 0

    events = []
    for t in timeline:
        events.append((t["start"], t["required_workers"]))
        events.append((t["end"], -t["required_workers"]))
    events.sort()

    cur = peak = 0
    for _, w in events:
        cur += w
        peak = max(peak, cur)

    total_equipment_time = sum(tool_usage.values())
    total_equipment_capacity = (
        sum(len(v) for v in tools_by_category.values()) * total_time
    )
    equipment_util = (
        total_equipment_time / total_equipment_capacity
        if total_equipment_capacity
        else 0
    )

    bottleneck_process = max(timeline, key=lambda x: x["duration"])

    return {
        "status": solver.StatusName(status),
        "makespan": total_time,
        "schedules": timeline,
        "analysis": {
            "bottleneckTool": bottleneck_tool,
            "workerUtilization": round(worker_util, 4),
            "averageIdleTimeBetweenTasks": round(avg_idle, 2),
            "peakConcurrentWorkers": peak,
            "equipmentUtilization": round(equipment_util, 4),
            "bottleneckProcess": {
                "taskId": bottleneck_process["task_id"],
                "productId": bottleneck_process["product_id"],
                "duration": bottleneck_process["duration"],
            },
        },
    }
