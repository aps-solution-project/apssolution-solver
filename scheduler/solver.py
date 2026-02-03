import json
import collections
from ortools.sat.python import cp_model

NO_EQUIPMENT = "NO_EQUIPMENT"


# -------------------------------
# 1. 데이터 평탄화
# -------------------------------
def flatten_data(data: dict):
    rows = []

    for product in data["scenarioProductList"]:
        product_id = product["id"]
        quantity = product.get("qty", 1)

        for i in range(quantity):
            job_id = f"{product_id}__{i+1}"

            for task in product["scenarioTasks"]:
                rows.append({
                    "job_id": job_id,
                    "product_id": product_id,
                    "task_id": task["id"],
                    "seq": task["seq"],
                    "duration": task["duration"],
                    "required_workers": task.get("requiredWorkers", 1),  # ⭐ 추가
                    "tool_category_id": task.get("toolCategory", {}).get("id"),
                })

    rows.sort(key=lambda r: (r["job_id"], r["seq"]))
    return rows


# -------------------------------
# 2. Job 그룹화
# -------------------------------
def group_by_job(rows):
    jobs = collections.defaultdict(list)
    for r in rows:
        jobs[r["job_id"]].append(r)
    return jobs


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
    jobs = group_by_job(rows)
    tools_by_category = build_tools_by_category(data["tools"])

    horizon = sum(r["duration"] for r in rows)

    model = cp_model.CpModel()

    var_map = {}
    all_end_vars = []

    worker_intervals = []
    worker_demands = []  # ⭐ 작업별 인원 수요

    tool_intervals = collections.defaultdict(list)

    # ---------------------------
    # 작업별 변수 생성
    # ---------------------------
    for job_id, tasks in jobs.items():
        prev_end = None

        for task in tasks:
            start = model.new_int_var(0, horizon, f"s_{job_id}_{task['seq']}")
            end = model.new_int_var(0, horizon, f"e_{job_id}_{task['seq']}")

            model.add(end == start + task["duration"])

            if prev_end is not None:
                model.add(start >= prev_end)
            prev_end = end

            # 👷 작업 인력 interval
            worker_iv = model.new_interval_var(
                start, task["duration"], end,
                f"iv_worker_{job_id}_{task['seq']}"
            )
            worker_intervals.append(worker_iv)
            worker_demands.append(task["required_workers"])  # ⭐ 핵심

            # 🛠 설비 interval
            tool_cat = task["tool_category_id"]
            if tool_cat and tool_cat != NO_EQUIPMENT:
                tool_iv = model.new_interval_var(
                    start, task["duration"], end,
                    f"iv_tool_{tool_cat}_{job_id}_{task['seq']}"
                )
                tool_intervals[tool_cat].append(tool_iv)

            var_map[(job_id, task["task_id"])] = {
                "start": start,
                "end": end,
                "duration": task["duration"],
                "tool_category_id": task["tool_category_id"],
                "product_id": task["product_id"],
                "required_workers": task["required_workers"],
            }

            all_end_vars.append(end)

    # ---------------------------
    # 👷 인력 Cumulative 제약
    # ---------------------------
    model.add_cumulative(
        worker_intervals,
        worker_demands,  # 작업별 인원 사용량
        data["scenario"]["maxWorkerCount"]  # 전체 인원 한도
    )

    # ---------------------------
    # 🛠 설비 Cumulative 제약
    # ---------------------------
    for tool_category_id, intervals in tool_intervals.items():
        tools = tools_by_category.get(tool_category_id, [])
        if not tools:
            raise ValueError(f"Tool category not available: {tool_category_id}")

        model.add_cumulative(intervals, [1] * len(intervals), len(tools))

    # ---------------------------
    # Makespan 최소화
    # ---------------------------
    makespan = model.new_int_var(0, horizon, "makespan")
    model.add_max_equality(makespan, all_end_vars)
    model.minimize(makespan)

    solver = cp_model.CpSolver()
    status = solver.solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {"status": solver.status_name(status), "makespan": 0, "schedules": []}

    # ---------------------------
    # 결과 타임라인 생성
    # ---------------------------
    timeline = []

    for (job_id, task_id), v in var_map.items():
        timeline.append({
            "job_id": job_id,
            "product_id": v["product_id"],
            "task_id": task_id,
            "tool_category_id": v["tool_category_id"],
            "tool_id": None,
            "start": solver.value(v["start"]),
            "end": solver.value(v["end"]),
            "duration": v["duration"],
            "required_workers": v["required_workers"],  # ⭐ 표시
        })

    # ---------------------------
    # 실제 Tool ID 배정 (후처리)
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
        "status": solver.status_name(status),
        "makespan": solver.value(makespan),
        "schedules": timeline,
    }


# -------------------------------
# 실행용 메인
# -------------------------------
if __name__ == "__main__":
    with open("input.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    result = solve_scenario(data)

    print("\n=== SOLVER RESULT ===")
    print("Status   :", result["status"])
    print("Makespan :", result["makespan"])
    print("\nSchedules:")
    for s in sorted(result["schedules"], key=lambda x: (x["start"], x["job_id"])):
        print(s)
