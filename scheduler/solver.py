import collections
from ortools.sat.python import cp_model


def flatten_data(data: dict):
    rows = []

    for product in data["scenarioProductList"]:
        job_id = product["id"]

        for task in product["scenarioTasks"]:
            rows.append({
                "job_id": job_id,
                "task_id": task["id"],
                "seq": task["seq"],
                "duration": task["duration"],
                "tool_category_id": task.get("toolCategory", {}).get("id"),
            })

    rows.sort(key=lambda r: (r["job_id"], r["seq"]))
    return rows


def group_by_job(rows):
    jobs = collections.defaultdict(list)
    for r in rows:
        jobs[r["job_id"]].append(r)
    return jobs


def build_tools_by_category(tools: list):
    """
    toolCategoryId -> [toolId, toolId, ...]
    """
    result = collections.defaultdict(list)

    for tool in tools:
        category_id = tool["category"]["id"]
        result[category_id].append(tool["id"])

    return dict(result)


def solve_scenario(data: dict):
    rows = flatten_data(data)
    jobs = group_by_job(rows)

    tools_by_category = build_tools_by_category(data["tools"])

    horizon = sum(r["duration"] for r in rows)

    model = cp_model.CpModel()

    var_map = {}
    all_end_vars = []

    worker_intervals = []
    tool_intervals = collections.defaultdict(list)

    for job_id, tasks in jobs.items():
        prev_end = None

        for task in tasks:
            start = model.new_int_var(0, horizon, f"s_{job_id}_{task['seq']}")
            end = model.new_int_var(0, horizon, f"e_{job_id}_{task['seq']}")

            model.add(end == start + task["duration"])

            if prev_end is not None:
                model.add(start >= prev_end)
            prev_end = end

            # worker 1명 소모
            worker_iv = model.new_interval_var(
                start, task["duration"], end,
                f"iv_worker_{job_id}_{task['seq']}"
            )
            worker_intervals.append(worker_iv)

            # toolCategory
            if task["tool_category_id"]:
                tool_iv = model.new_interval_var(
                    start, task["duration"], end,
                    f"iv_tool_{task['tool_category_id']}_{job_id}_{task['seq']}"
                )
                tool_intervals[task["tool_category_id"]].append(tool_iv)

            var_map[(job_id, task["task_id"])] = {
                "start": start,
                "end": end,
                "duration": task["duration"],
                "tool_category_id": task["tool_category_id"],
            }

            all_end_vars.append(end)

    # 1) 작업자 수 제한
    model.add_cumulative(
        worker_intervals,
        [1] * len(worker_intervals),
        data["scenario"]["maxWorkerCount"]
    )

    # 2) toolCategory capacity 제한
    for tool_category_id, intervals in tool_intervals.items():
        tools = tools_by_category.get(tool_category_id, [])

        if not tools:
            raise ValueError(f"Tool category not available: {tool_category_id}")

        model.add_cumulative(
            intervals,
            [1] * len(intervals),
            len(tools)
        )

    makespan = model.new_int_var(0, horizon, "makespan")
    model.add_max_equality(makespan, all_end_vars)
    model.minimize(makespan)

    solver = cp_model.CpSolver()
    status = solver.solve(model)

    timeline = []

    for (job_id, task_id), v in var_map.items():
        timeline.append({
            "productId": job_id,
            "taskId": task_id,
            "toolCategoryId": v["tool_category_id"],
            "toolId": None,
            "start": solver.value(v["start"]),
            "end": solver.value(v["end"]),
            "duration": v["duration"],
        })

    # toolCategory별 작업 정렬
    tasks_by_category = collections.defaultdict(list)
    for t in timeline:
        if t["toolCategoryId"]:
            tasks_by_category[t["toolCategoryId"]].append(t)

    for tool_category_id, tasks in tasks_by_category.items():
        tool_ids = tools_by_category[tool_category_id]

        # tool_id -> 해당 tool의 마지막 종료 시간
        tool_available_at = {tool_id: 0 for tool_id in tool_ids}

        # 시작 시간 기준 정렬
        tasks.sort(key=lambda x: x["start"])

        for task in tasks:
            for tool_id, available_at in tool_available_at.items():
                if available_at <= task["start"]:
                    task["toolId"] = tool_id
                    tool_available_at[tool_id] = task["end"]
                    break

            if task["toolId"] is None:
                raise RuntimeError(
                    f"Tool assignment failed: {tool_category_id} at {task}"
                )

    status_name = solver.status_name(status)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {
            "status": status_name,
            "makespan": solver.value(makespan),
            "timeline": timeline,
        }
    else:
        return {
            "status": status_name,
            "makespan": 0,
            "timeline": [],
        }


if __name__ == "__main__":
    data = "test"
    solve_scenario(data)
