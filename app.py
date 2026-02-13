import time
from flask import Flask, request
from scheduler.solver import solve_scenario

app = Flask(__name__)


@app.route('/api/solve', methods=['POST'])
def job_simulation_solver():
    data = request.get_json()
    print("\n📩 Solve request received")

    resp = solve_scenario(data)
    print("✅ Solve finished\n\n")
    return resp

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)