# APS Solution — Solver

OR-Tools CP-SAT으로 생산 일정을 계산하는 Python 최적화 서버입니다. Spring Boot API 서버가 시나리오를 넘기면, 최적 일정과 분석 지표를 돌려줍니다.

## 기술 스택

Python 3 · Flask · OR-Tools (CP-SAT)

## 모델링

| 현장 조건 | 제약 |
|---|---|
| 공정 순서 | 선후 관계(seq): 앞 공정이 끝난 뒤에 다음 공정 시작 |
| 설비·작업자 충돌 | `AddNoOverlap`, 설비·작업자별 `AddCumulative` |
| 주간·야간 배정 | 종료 시각이 06:00–18:00인지 Boolean 변수로 판별해 주간/야간 작업자 배정 |
| 목표 | Makespan(전체 완료 시각) 최소화 |

- 상태값: 시간 상한 안에 최적해를 증명하면 `OPTIMAL`, 해를 찾으면 `FEASIBLE`
- 해가 없으면 API 서버가 시나리오를 `FAILED`로 처리합니다.

## API

`POST /api/solve`

요청

```json
{
  "scenario": {},
  "scenarioProductList": [],
  "tools": [],
  "dayWorkers": [{ "id": "...", "name": "..." }],
  "nightWorkers": [{ "id": "...", "name": "..." }]
}
```

응답

```json
{
  "status": "OPTIMAL",
  "makespan": 0,
  "schedules": [],
  "analysis": {
    "bottleneckTool": "...",
    "workerUtilization": 0.0,
    "equipmentUtilization": 0.0,
    "peakConcurrentWorkers": 0,
    "averageIdleTimeBetweenTasks": 0.0,
    "bottleneckProcess": { "taskId": "...", "productId": "...", "duration": 0 }
  }
}
```

## 실행

```bash
pip install -r requirements.txt
python app.py          # http://0.0.0.0:5000
```

## 시간 상한

- `scheduler/solver.py`의 `max_time_in_seconds`로 조정합니다.
- 저장소 값(300초)은 시연·테스트용이고, 운영 기준은 43,200초(12시간)입니다.
- API 서버의 읽기 타임아웃(44,000초)은 이 값보다 길게 잡혀 있습니다.

## 관련 저장소

- Backend: https://github.com/aps-solution-project/apssolution-backend
- Frontend: https://github.com/aps-solution-project/apssolution-frontend
