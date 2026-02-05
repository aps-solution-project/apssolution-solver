import json
import collections
from ortools.sat.python import cp_model
import time
from ortools.sat.python import cp_model

NO_EQUIPMENT = "NO_EQUIPMENT"

# -------------------------------
# 중간 해 출력 콜백 (해 하나 나올 때마다 무조건 출력)
# -------------------------------
import time
from ortools.sat.python import cp_model


class ProgressSolutionPrinter(cp_model.CpSolverSolutionCallback):
    def __init__(self, makespan_var, interval_sec=5):
        super().__init__()
        self._makespan = makespan_var
        self._solution_count = 0
        self._last_print_time = time.time()
        self._interval = interval_sec
        self._best_makespan = None

    def on_solution_callback(self):
        self._solution_count += 1
        self._best_makespan = self.Value(self._makespan)

        print(
            f"[Solution #{self._solution_count}] "
            f"time={self.WallTime():.1f}s "
            f"makespan={self._best_makespan}"
        )

        self._maybe_print_progress()

    def _maybe_print_progress(self):
        now = time.time()
        if now - self._last_print_time >= self._interval:
            bound = self.BestObjectiveBound()
            print(
                f"[Progress] time={self.WallTime():.1f}s "
                f"best={self._best_makespan} "
                f"bound={bound}"
            )
            self._last_print_time = now



# -------------------------------
# 1. 데이터 평탄화
# -------------------------------
def flatten_data(data: dict):
    rows = []

    for product in data["scenarioProductList"]:
        product_id = product["id"]
        quantity = product.get("qty", 1)

        for i in range(quantity):
            product_instance_id = f"{product_id}__{i+1}"

            for task in product["scenarioTasks"]:
                rows.append({
                    "product_instance_id": product_instance_id,
                    "product_id": product_id,
                    "task_id": task["id"],
                    "seq": task["seq"],
                    "duration": task["duration"],
                    "required_workers": task.get("requiredWorkers", 1),
                    "tool_category_id": task.get("toolCategory", {}).get("id"),
                })

    rows.sort(key=lambda r: (r["product_instance_id"], r["seq"]))
    return rows


# -------------------------------
# 2. Product 그룹화
# -------------------------------
def group_by_product(rows):
    products = collections.defaultdict(list)
    for r in rows:
        products[r["product_instance_id"]].append(r)
    return products


# -------------------------------
# 3. Tool 카테고리별 보유 장비 정리
# -------------------------------
def build_tools_by_category(tools: list):
    result = collections.defaultdict(list)
    for tool in tools:
        result[tool["category"]["id"]].append(tool["id"])
    return dict(result)


# -------------------------------
# 4. 스케줄링 모델 생성 및 풀이
# -------------------------------
def solve_scenario(data: dict):
    rows = flatten_data(data)
    products = group_by_product(rows)
    tools_by_category = build_tools_by_category(data["tools"])

    horizon = sum(r["duration"] for r in rows)

    model = cp_model.CpModel()

    var_map = {}
    all_end_vars = []

    worker_intervals = []
    worker_demands = []
    tool_intervals = collections.defaultdict(list)

    # ---------------------------
    # 작업별 변수 생성
    # ---------------------------
    for product_instance_id, tasks in products.items():
        prev_end = None

        for task in tasks:
            start = model.NewIntVar(0, horizon, f"s_{product_instance_id}_{task['seq']}")
            end = model.NewIntVar(0, horizon, f"e_{product_instance_id}_{task['seq']}")

            model.Add(end == start + task["duration"])

            if prev_end is not None:
                model.Add(start >= prev_end)
            prev_end = end

            worker_iv = model.NewIntervalVar(
                start, task["duration"], end,
                f"iv_worker_{product_instance_id}_{task['seq']}"
            )
            worker_intervals.append(worker_iv)
            worker_demands.append(task["required_workers"])

            tool_cat = task["tool_category_id"]
            if tool_cat and tool_cat != NO_EQUIPMENT:
                tool_iv = model.NewIntervalVar(
                    start, task["duration"], end,
                    f"iv_tool_{tool_cat}_{product_instance_id}_{task['seq']}"
                )
                tool_intervals[tool_cat].append(tool_iv)

            var_map[(product_instance_id, task["task_id"])] = {
                "start": start,
                "end": end,
                "duration": task["duration"],
                "tool_category_id": task["tool_category_id"],
                "product_id": task["product_id"],
                "required_workers": task["required_workers"],
            }

            all_end_vars.append(end)

    # ---------------------------
    # 인력 Cumulative
    # ---------------------------
    model.AddCumulative(
        worker_intervals,
        worker_demands,
        data["scenario"]["maxWorkerCount"]
    )

    # ---------------------------
    # 설비 Cumulative
    # ---------------------------
    for tool_category_id, intervals in tool_intervals.items():
        tools = tools_by_category.get(tool_category_id, [])
        if not tools:
            raise ValueError(f"Tool category not available: {tool_category_id}")

        model.AddCumulative(intervals, [1] * len(intervals), len(tools))

    # ---------------------------
    # Makespan 최소화
    # ---------------------------
    makespan = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(makespan, all_end_vars)
    model.Minimize(makespan)

    # ---------------------------
    # Solver 실행 + 콜백
    # ---------------------------
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 43000
    solver.parameters.log_search_progress = False

    solution_printer = ProgressSolutionPrinter(makespan, interval_sec=1)

    status = solver.Solve(model, solution_printer)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {"status": solver.StatusName(status), "makespan": 0, "schedules": []}

    # ---------------------------
    # 결과 타임라인 생성
    # ---------------------------
    timeline = []

    for (product_instance_id, task_id), v in var_map.items():
        timeline.append({
            "product_instance_id": product_instance_id,
            "product_id": v["product_id"],
            "task_id": task_id,
            "tool_category_id": v["tool_category_id"],
            "tool_id": None,
            "start": solver.Value(v["start"]),
            "end": solver.Value(v["end"]),
            "duration": v["duration"],
            "required_workers": v["required_workers"],
        })

    # ---------------------------
    # Tool ID 후처리 배정
    # ---------------------------
    tasks_by_category = collections.defaultdict(list)
    for t in timeline:
        if t["tool_category_id"] and t["tool_category_id"] != NO_EQUIPMENT:
            tasks_by_category[t["tool_category_id"]].append(t)

    for tool_category_id, tasks in tasks_by_category.items():
        tool_ids = tools_by_category[tool_category_id]
        tool_available_at = {tool_id: 0 for tool_id in tool_ids}

        tasks.sort(key=lambda x: x["start"])

        for task in tasks:
            for tool_id, available_at in tool_available_at.items():
                if available_at <= task["start"]:
                    task["tool_id"] = tool_id
                    tool_available_at[tool_id] = task["end"]
                    break

            if task["tool_id"] is None:
                raise RuntimeError(f"Tool assignment failed: {tool_category_id}")

    return {
        "status": solver.StatusName(status),
        "makespan": solver.Value(makespan),
        "schedules": timeline,
    }


# -------------------------------
# 실행 테스트용
# -------------------------------
if __name__ == "__main__":
    data = {}

    result = solve_scenario(data)

    print("\n=== SOLVER RESULT ===")
    print("Status   :", result["status"])
    print("Makespan :", result["makespan"])
