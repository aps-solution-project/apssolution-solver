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
            pid = f"{product_id}__{i + 1}"
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


def is_day_shift(end_time_minutes):
    """
    종료 시간이 주간 근무 시간(06:00-18:00)에 속하는지 확인
    end_time_minutes: 하루 내 분 단위 시간 (0-1439)
    """
    # 06:00 = 360분, 18:00 = 1080분
    time_of_day = end_time_minutes % 1440  # 24시간 = 1440분
    return 360 <= time_of_day < 1080


# -------------------------------
# 메인 Solver
# -------------------------------
def solve_scenario(data):
    """
    data: Java SolveScenarioRequest 객체 (JSON 형태)
    {
        "scenario": {...},
        "scenarioProductList": [...],
        "tools": [...],
        "dayWorkers": [{"id": "...", "name": "..."}, ...],
        "nightWorkers": [{"id": "...", "name": "..."}, ...]
    }
    """
    rows = flatten_data(data)
    products = group_by_product(rows)
    tools_by_category = build_tools_by_category(data["tools"])
    tool_to_category = build_tool_to_category_map(data["tools"])

    # 주간/야간 작업자 분리
    day_workers = data.get("dayWorkers", [])
    night_workers = data.get("nightWorkers", [])

    day_worker_ids = [acc["id"] for acc in day_workers]
    night_worker_ids = [acc["id"] for acc in night_workers]

    num_day_workers = len(day_worker_ids)
    num_night_workers = len(night_worker_ids)

    if num_day_workers == 0 and num_night_workers == 0:
        raise ValueError("최소 1명 이상의 작업자가 필요합니다.")

    horizon = sum(r["duration"] for r in rows)

    model = cp_model.CpModel()

    var_map = {}
    all_end_vars = []

    worker_intervals = []
    worker_demands = []

    tool_intervals = collections.defaultdict(list)

    # 🔥 핵심: 같은 task_id 전역 직렬화
    task_intervals_by_task_id = collections.defaultdict(list)

    # 🆕 각 account별로 배정된 interval을 추적 (중복 방지용)
    day_account_intervals = collections.defaultdict(list)
    night_account_intervals = collections.defaultdict(list)

    # 🆕 각 작업에 배정될 작업자들을 저장할 변수
    task_worker_assignments = {}

    task_counter = 0  # 각 작업에 고유 ID 부여

    for pid, tasks in products.items():
        prev_end = None

        for task in tasks:
            task_counter += 1
            unique_task_key = f"{pid}_{task['task_id']}_{task_counter}"

            start = model.NewIntVar(0, horizon, f"s_{unique_task_key}")
            end = model.NewIntVar(0, horizon, f"e_{unique_task_key}")
            model.Add(end == start + task["duration"])

            if prev_end is not None:
                model.Add(start >= prev_end)
            prev_end = end

            interval = model.NewIntervalVar(
                start, task["duration"], end,
                f"iv_{unique_task_key}"
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

            # 🆕 작업자 배정 변수 생성 (0명 또는 1명만)
            required_workers = task["required_workers"]
            assigned_worker_info = None  # 배정된 작업자 정보

            if required_workers == 1:
                # 🔥🔥 시간대 판별을 위한 Boolean 변수
                is_day_shift_var = model.NewBoolVar(f"is_day_{unique_task_key}")

                # end 시간의 하루 내 위치 계산 (0-1439)
                time_of_day = model.NewIntVar(0, 1439, f"time_of_day_{unique_task_key}")
                model.AddModuloEquality(time_of_day, end, 1440)

                # 06:00(360분) <= time_of_day < 18:00(1080분)이면 주간
                # 주간 조건: time_of_day >= 360 AND time_of_day < 1080
                is_gte_360 = model.NewBoolVar(f"gte_360_{unique_task_key}")
                is_lt_1080 = model.NewBoolVar(f"lt_1080_{unique_task_key}")

                model.Add(time_of_day >= 360).OnlyEnforceIf(is_gte_360)
                model.Add(time_of_day < 360).OnlyEnforceIf(is_gte_360.Not())
                model.Add(time_of_day < 1080).OnlyEnforceIf(is_lt_1080)
                model.Add(time_of_day >= 1080).OnlyEnforceIf(is_lt_1080.Not())

                # 둘 다 True면 주간
                model.AddBoolAnd([is_gte_360, is_lt_1080]).OnlyEnforceIf(is_day_shift_var)
                model.AddBoolOr([is_gte_360.Not(), is_lt_1080.Not()]).OnlyEnforceIf(is_day_shift_var.Not())

                # 주간 작업자 선택 (dayWorkers 중에서)
                day_worker_var = None
                if num_day_workers > 0:
                    day_worker_var = model.NewIntVar(
                        0, num_day_workers - 1,
                        f"day_worker_{unique_task_key}"
                    )

                # 야간 작업자 선택 (nightWorkers 중에서)
                night_worker_var = None
                if num_night_workers > 0:
                    night_worker_var = model.NewIntVar(
                        0, num_night_workers - 1,
                        f"night_worker_{unique_task_key}"
                    )

                # 각 주간 작업자에 대해 Optional Interval 생성
                if num_day_workers > 0:
                    for acc_idx, acc_id in enumerate(day_worker_ids):
                        # 조건: 주간이고(is_day_shift_var) AND 이 작업자가 선택됨(day_worker_var == acc_idx)
                        is_this_day_worker = model.NewBoolVar(
                            f"is_day_worker_{unique_task_key}_{acc_id}"
                        )

                        # is_this_day_worker = (is_day_shift_var AND day_worker_var == acc_idx)
                        model.Add(day_worker_var == acc_idx).OnlyEnforceIf(is_this_day_worker)
                        model.Add(day_worker_var != acc_idx).OnlyEnforceIf(is_this_day_worker.Not())

                        # 최종 활성화 조건: 주간이면서 이 작업자가 선택됨
                        is_active = model.NewBoolVar(f"active_day_{unique_task_key}_{acc_id}")
                        model.AddBoolAnd([is_day_shift_var, is_this_day_worker]).OnlyEnforceIf(is_active)
                        model.AddBoolOr([is_day_shift_var.Not(), is_this_day_worker.Not()]).OnlyEnforceIf(
                            is_active.Not())

                        opt_interval = model.NewOptionalIntervalVar(
                            start, task["duration"], end,
                            is_active,
                            f"opt_day_iv_{unique_task_key}_{acc_id}"
                        )
                        day_account_intervals[acc_id].append(opt_interval)

                # 각 야간 작업자에 대해 Optional Interval 생성
                if num_night_workers > 0:
                    for acc_idx, acc_id in enumerate(night_worker_ids):
                        # 조건: 야간이고(is_day_shift_var.Not()) AND 이 작업자가 선택됨(night_worker_var == acc_idx)
                        is_this_night_worker = model.NewBoolVar(
                            f"is_night_worker_{unique_task_key}_{acc_id}"
                        )

                        # is_this_night_worker = (night_worker_var == acc_idx)
                        model.Add(night_worker_var == acc_idx).OnlyEnforceIf(is_this_night_worker)
                        model.Add(night_worker_var != acc_idx).OnlyEnforceIf(is_this_night_worker.Not())

                        # 최종 활성화 조건: 야간이면서 이 작업자가 선택됨
                        is_active = model.NewBoolVar(f"active_night_{unique_task_key}_{acc_id}")
                        model.AddBoolAnd([is_day_shift_var.Not(), is_this_night_worker]).OnlyEnforceIf(is_active)
                        model.AddBoolOr([is_day_shift_var, is_this_night_worker.Not()]).OnlyEnforceIf(is_active.Not())

                        opt_interval = model.NewOptionalIntervalVar(
                            start, task["duration"], end,
                            is_active,
                            f"opt_night_iv_{unique_task_key}_{acc_id}"
                        )
                        night_account_intervals[acc_id].append(opt_interval)

                assigned_worker_info = {
                    "is_day_shift": is_day_shift_var,
                    "day_worker": day_worker_var,
                    "night_worker": night_worker_var,
                }

            task_worker_assignments[unique_task_key] = {
                "start": start,
                "end": end,
                "duration": task["duration"],
                "tool_category_id": cat,
                "product_id": task["product_id"],
                "task_id": task["task_id"],
                "required_workers": task["required_workers"],
                "assigned_worker": assigned_worker_info,
            }

            var_map[(pid, task["task_id"], task_counter)] = task_worker_assignments[unique_task_key]

            all_end_vars.append(end)

    # -------------------------------
    # 제약 조건
    # -------------------------------
    # requiredWorkers > 0인 작업만 동시 작업 인원 제약 적용
    worker_intervals_filtered = []
    worker_demands_filtered = []

    for interval, demand in zip(worker_intervals, worker_demands):
        if demand > 0:
            worker_intervals_filtered.append(interval)
            worker_demands_filtered.append(demand)

    # 주간/야간 최대 인원은 각각의 작업자 수로 설정
    # 전체적인 제약은 시간대별로 나누어 적용
    max_workers = max(num_day_workers, num_night_workers) if (num_day_workers > 0 or num_night_workers > 0) else 1

    if worker_intervals_filtered:
        model.AddCumulative(
            worker_intervals_filtered,
            worker_demands_filtered,
            max_workers
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

    # 🆕 각 account별로 시간 겹침 방지 (동일 작업자 중복 배정 불가)
    for acc_id, intervals in day_account_intervals.items():
        if len(intervals) > 0:
            model.AddNoOverlap(intervals)

    for acc_id, intervals in night_account_intervals.items():
        if len(intervals) > 0:
            model.AddNoOverlap(intervals)

    # 🆕🆕🆕 시간대별 동시 작업 가능 인원 제한
    # 주간: dayWorkers.size, 야간: nightWorkers.size
    day_intervals = []
    day_demands = []
    night_intervals = []
    night_demands = []

    for unique_task_key, task_info in task_worker_assignments.items():
        if task_info["required_workers"] > 0 and task_info["assigned_worker"] is not None:
            is_day = task_info["assigned_worker"]["is_day_shift"]

            # 주간 작업 interval
            if num_day_workers > 0:
                day_interval = model.NewOptionalIntervalVar(
                    task_info["start"],
                    task_info["duration"],
                    task_info["end"],
                    is_day,
                    f"day_count_{unique_task_key}"
                )
                day_intervals.append(day_interval)
                day_demands.append(1)

            # 야간 작업 interval
            if num_night_workers > 0:
                night_interval = model.NewOptionalIntervalVar(
                    task_info["start"],
                    task_info["duration"],
                    task_info["end"],
                    is_day.Not(),
                    f"night_count_{unique_task_key}"
                )
                night_intervals.append(night_interval)
                night_demands.append(1)

    if day_intervals and num_day_workers > 0:
        model.AddCumulative(day_intervals, day_demands, num_day_workers)

    if night_intervals and num_night_workers > 0:
        model.AddCumulative(night_intervals, night_demands, num_night_workers)

    makespan = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(makespan, all_end_vars)
    model.Minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 300
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
    for (pid, tid, counter), v in var_map.items():
        # 배정된 작업자 ID 추출 (0명 또는 1명)
        account_id = None
        if v["assigned_worker"] is not None:
            is_day = solver.Value(v["assigned_worker"]["is_day_shift"])

            if is_day and v["assigned_worker"]["day_worker"] is not None:
                worker_idx = solver.Value(v["assigned_worker"]["day_worker"])
                account_id = day_worker_ids[worker_idx]
            elif not is_day and v["assigned_worker"]["night_worker"] is not None:
                worker_idx = solver.Value(v["assigned_worker"]["night_worker"])
                account_id = night_worker_ids[worker_idx]

        timeline.append({
            "product_id": v["product_id"],
            "task_id": tid,
            "tool_category_id": v["tool_category_id"],
            "tool_id": None,
            "start": solver.Value(v["start"]),
            "end": solver.Value(v["end"]),
            "duration": v["duration"],
            "required_workers": v["required_workers"],
            "accountId": account_id,  # 🆕 작업자 ID (null 또는 단일 값)
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
    # 🔥 분석 계산
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
    max_workers_total = num_day_workers + num_night_workers
    worker_util = total_worker_time / (
            total_time * max_workers_total
    ) if max_workers_total > 0 else 0

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