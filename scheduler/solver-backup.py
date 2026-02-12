import collections
import time
from ortools.sat.python import cp_model

NO_EQUIPMENT = "NO_EQUIPMENT"


# -------------------------------
# 중간 해 출력 콜백
# -------------------------------
class ProgressSolutionPrinter(cp_model.CpSolverSolutionCallback):
    def __init__(self, makespan_var):
        super().__init__()
        self._makespan = makespan_var
        self._solution_count = 0

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

    # 🔥 추가: 같은 task_id 직렬화용
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

            # -------------------------------
            # Worker cumulative
            # -------------------------------
            worker_intervals.append(interval)
            worker_demands.append(task["required_workers"])

            # -------------------------------
            # Tool cumulative
            # -------------------------------
            cat = task["tool_category_id"]
            if cat and cat != NO_EQUIPMENT:
                tool_intervals[cat].append(interval)

            # -------------------------------
            # 🔥 같은 작업(task_id) 직렬화
            # -------------------------------
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
        data["scenario"]["maxWorkerCount"]
    )

    for cat, intervals in tool_intervals.items():
        model.AddCumulative(
            intervals,
            [1] * len(intervals),
            len(tools_by_category[cat])
        )

    # 🔥 핵심 제약: 같은 작업은 동시에 하나만
    for task_id, intervals in task_intervals_by_task_id.items():
        model.AddNoOverlap(intervals)

    makespan = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(makespan, all_end_vars)
    model.Minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 100
    solver.parameters.log_search_progress = True

    status = solver.Solve(
        model,
        ProgressSolutionPrinter(makespan)
    )

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {"status": solver.StatusName(status), "makespan": 0, "schedules": []}

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

    return {
        "status": solver.StatusName(status),
        "makespan": solver.Value(makespan),
        "schedules": timeline,
    }
