import collections
from ortools.sat.python import cp_model


# step1, 데이터 평탄화
def flatten_data(data):
    rows = []
    for sj in data["scenarioJobs"]:
        scenario_job_id = sj["id"]
        job = sj["job"]
        job_id = job["id"]
        for task in job["tasks"]:
            task_id = task["id"]
            tool_id = task["tool"]["id"] if task["tool"] else None
            seq = task["seq"]
            duration = task["duration"]

            rows.append({"id": scenario_job_id, "job_id": job_id, "task_id": task_id, "tool_id": tool_id, "seq": seq,
                         "duration": duration})

    # 기본 [].sort() 하면 괜찮은데, 객체거나 딕셔너리면 안에 리턴값 설정해야함
    # 아래는 row라는 이름으로 받아서 job_id로 소팅하고, 그 상태에서 seq순서로 소팅하겠다는 뜻 => (row["job_id"], row["seq"]) 라는 튜플!
    rows.sort(key=lambda row: (row["job_id"], row["seq"]))

    return rows


# step2-1, 작업으로 테스크 그룹화
def group_by_job(rows):
    jobs = collections.defaultdict(list)
    for row in rows:
        jobs[row["job_id"]].append(row)

    return jobs


# step2-2, tool로 그룹화
def group_by_tool(rows):
    tools = collections.defaultdict(list)
    for row in rows:
        tools[row["tool_id"]].append(row)

    return tools


def solver_job_scheduler(data: dict):
    """
    or-tools로 스케쥴링 실행
    :return:
    """
    rows = flatten_data(data)
    jobs = group_by_job(rows)
    tools = group_by_tool(rows)
    job_ids = list(jobs.keys())
    tool_ids = list(tools.keys())

    # 뇌 빼고 계산한 시간 최대치(하나 끝내고 다음거 하고... 느낌)
    horizon = sum(r["duration"] for r in rows)

    # step3, 상황 및 제약조건 설정
    model = cp_model.CpModel()
    var_map = {}
    all_end = []
    tool_intervals = collections.defaultdict(list)

    for job_id, tasks in jobs.items():
        prev_end = None

        for idx, task in enumerate(tasks):
            start = model.new_int_var(0, horizon, f"s_{job_id}_{task["seq"]}")
            end = model.new_int_var(0, horizon, f"s_{job_id}_{task["seq"]}")
            # 한 task의 end값은 start + duration으로 해달라
            model.add(task["duration"] == end - start)

            if idx == 0:
                prev_end = end
            else:
                # 한 task의 start 값은 이전 end값보다 크게 해달라
                model.add(start >= prev_end)
                prev_end = end

            if task["tool_id"]:
                iv = model.new_interval_var(start, task["duration"], end, f"iv_{job_id}_{task["seq"]}")
                tool_intervals[task["tool_id"]].append(iv)

            var_map[(job_id, task["task_id"])] = (start, end, task["duration"], task["tool_id"])
            all_end.append(end)

    # step4, 같은 tool은 중복 불가 제약 추가
    for tool_id, intervals in tool_intervals.items():
        print(tool_id)
        model.add_no_overlap(intervals)

    # step5, 목적함수 추가
    makespan = model.new_int_var(0, horizon, "makespan")
    model.add_max_equality(makespan, all_end)
    model.minimize(makespan)

    solver = cp_model.CpSolver()
    status = solver.solve(model)

    print(solver.status_name(status))
    print(horizon)
    print(solver.Value(makespan))

    timeline = []
    for (job_id, task_id), (s_var, e_var, duration, tool_id) in var_map.items():
        print(job_id, task_id, solver.value(s_var), solver.value(e_var), duration, tool_id)
        timeline.append({"job_id": job_id, "task_id": task_id, "tool_id": tool_id, "start": solver.value(s_var), "end": solver.value(e_var),
                         "duration": duration})

    resp = {
        "status": solver.status_name(status),
        "elapsed": solver.value(makespan),
        "timeline": timeline
    }

    return resp

if __name__ == "__main__":
    #테스트용
    data = {'id': 'BAFEF806ABAC', 'description': '시나리오 생성 테스트!', 'done': False, 'createdAt': [2026, 1, 13, 16, 10, 37],
            'scenarioJobs': [{'id': 19,
                              'job': {'id': 'JB_BREAD_BASIC', 'name': '기본 식빵 제조', 'description': '일반 식빵 생산을 위한 표준 공정',
                                      'active': True, 'tasks': [{'id': 'TSK_BB_BAKE',
                                                                 'tool': {'id': 'OVEN_DECK', 'name': '데크 오븐',
                                                                          'description': '빵을 굽는 기본 오븐'},
                                                                 'seq': 5,
                                                                 'name': '굽기', 'description': '데크 오븐에서 굽기!!',
                                                                 'duration': 3}, {'id': 'TSK_BB_COOL',
                                                                                  'tool': {'id': 'COOKING_RACK',
                                                                                           'name': '식힘 랙',
                                                                                           'description': '구운 빵 냉각용 작업 공간'},
                                                                                  'seq': 6, 'name': '식힘',
                                                                                  'description': '식힘 랙에서 냉각',
                                                                                  'duration': 2}, {'id': 'TSK_BB_MIX',
                                                                                                   'tool': {
                                                                                                       'id': 'MIXER_SPIRAL',
                                                                                                       'name': '스파이럴 반죽기',
                                                                                                       'description': '대량 빵 반죽용 반죽기'},
                                                                                                   'seq': 1,
                                                                                                   'name': '반죽',
                                                                                                   'description': '스파이럴 반죽기로 반죽 혼합',
                                                                                                   'duration': 2},
                                                                {'id': 'TSK_BB_PACK',
                                                                 'tool': {'id': 'PACK_TABLE', 'name': '포장 작업대',
                                                                          'description': '완제품 수작업 포장 설비'}, 'seq': 7,
                                                                 'name': '포장', 'description': '완제품 포장', 'duration': 1},
                                                                {'id': 'TSK_BB_PROOF1',
                                                                 'tool': {'id': 'PROOFER', 'name': '발효기',
                                                                          'description': '반죽 발효용 온습도 제어 설비'}, 'seq': 2,
                                                                 'name': '1차 발효', 'description': '발효기에서 1차 발효',
                                                                 'duration': 6}, {'id': 'TSK_BB_PROOF2',
                                                                                  'tool': {'id': 'PROOFER',
                                                                                           'name': '발효기',
                                                                                           'description': '반죽 발효용 온습도 제어 설비'},
                                                                                  'seq': 4, 'name': '2차 발효',
                                                                                  'description': '발효기에서 2차 발효',
                                                                                  'duration': 5}, {'id': 'TSK_BB_SHAPE',
                                                                                                   'tool': {
                                                                                                       'id': 'PACK_TABLE',
                                                                                                       'name': '포장 작업대',
                                                                                                       'description': '완제품 수작업 포장 설비'},
                                                                                                   'seq': 3,
                                                                                                   'name': '분할/성형',
                                                                                                   'description': '반죽 분할 및 성형/팬닝',
                                                                                                   'duration': 2}]}},
                             {'id': 20,
                              'job': {'id': 'JB_BREAD_WHOLE', 'name': '통밀 식빵 제조', 'description': '통밀 반죽을 사용하는 식빵 생산 공정',
                                      'active': True, 'tasks': [{'id': 'TSK_BW_BAKE',
                                                                 'tool': {'id': 'OVEN_DECK', 'name': '데크 오븐',
                                                                          'description': '빵을 굽는 기본 오븐'}, 'seq': 5,
                                                                 'name': '굽기', 'description': '데크 오븐에서 굽기',
                                                                 'duration': 3}, {'id': 'TSK_BW_COOL',
                                                                                  'tool': {'id': 'COOKING_RACK',
                                                                                           'name': '식힘 랙',
                                                                                           'description': '구운 빵 냉각용 작업 공간'},
                                                                                  'seq': 6, 'name': '식힘',
                                                                                  'description': '식힘 랙에서 냉각',
                                                                                  'duration': 2}, {'id': 'TSK_BW_MIX',
                                                                                                   'tool': {
                                                                                                       'id': 'MIXER_SPIRAL',
                                                                                                       'name': '스파이럴 반죽기',
                                                                                                       'description': '대량 빵 반죽용 반죽기'},
                                                                                                   'seq': 1,
                                                                                                   'name': '반죽',
                                                                                                   'description': '스파이럴 반죽기로 통밀 반죽 혼합',
                                                                                                   'duration': 3},
                                                                {'id': 'TSK_BW_PACK',
                                                                 'tool': {'id': 'PACK_TABLE', 'name': '포장 작업대',
                                                                          'description': '완제품 수작업 포장 설비'}, 'seq': 7,
                                                                 'name': '포장', 'description': '완제품 포장', 'duration': 1},
                                                                {'id': 'TSK_BW_PROOF1',
                                                                 'tool': {'id': 'PROOFER', 'name': '발효기',
                                                                          'description': '반죽 발효용 온습도 제어 설비'}, 'seq': 2,
                                                                 'name': '1차 발효',
                                                                 'description': '발효기에서 1차 발효(통밀, 시뮬용 과장 시간)',
                                                                 'duration': 7}, {'id': 'TSK_BW_PROOF2',
                                                                                  'tool': {'id': 'PROOFER',
                                                                                           'name': '발효기',
                                                                                           'description': '반죽 발효용 온습도 제어 설비'},
                                                                                  'seq': 4, 'name': '2차 발효',
                                                                                  'description': '발효기에서 2차 발효(통밀, 시뮬용 과장 시간)',
                                                                                  'duration': 6}, {'id': 'TSK_BW_SHAPE',
                                                                                                   'tool': {
                                                                                                       'id': 'PACK_TABLE',
                                                                                                       'name': '포장 작업대',
                                                                                                       'description': '완제품 수작업 포장 설비'},
                                                                                                   'seq': 3,
                                                                                                   'name': '분할/성형',
                                                                                                   'description': '통밀 반죽 분할 및 성형/팬닝',
                                                                                                   'duration': 2}]}}]}

    solver_job_scheduler(data)
