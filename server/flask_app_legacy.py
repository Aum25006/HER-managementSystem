from __future__ import annotations

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from server.hospital_environment import make_env, TASKS
from server.services.grader import grade

app = Flask(__name__, static_folder="../static", static_url_path="/static")
CORS(app)

# Global environment instance for simplicity
ENV_TASK = "easy"
env = make_env(task=ENV_TASK)
current_state = env.reset()
last_reward = 0.0
last_done = False


@app.route("/")
def index():
    # Serve the simple UI
    return send_from_directory("../static", "index.html")


@app.route("/state", methods=["GET"])
def get_state():
    global current_state, last_reward, last_done
    # Ensure state is current
    current_state = env.get_state()
    last_done = env.is_done()
    return jsonify(
        {
            "state": current_state,
            "reward": last_reward,
            "done": last_done,
        }
    )


@app.route("/reset", methods=["POST"])
def reset():
    global env, current_state, last_reward, last_done
    body = request.get_json(silent=True) or {}
    task = body.get("task", ENV_TASK)

    if task not in TASKS:
        task = ENV_TASK

    env = make_env(task=task)
    current_state = env.reset()
    last_reward = 0.0
    last_done = False

    return jsonify(
        {
            "state": current_state,
            "reward": last_reward,
            "done": last_done,
            "task": task,
        }
    )


@app.route("/step", methods=["POST"])
def step():
    global env, current_state, last_reward, last_done
    body = request.get_json(force=True)

    action_type = body.get("type")
    patient_id = body.get("patient_id")

    action = {"type": action_type, "patient_id": patient_id}
    state, reward, done, info = env.step(action)

    current_state = state
    last_reward = reward
    last_done = done

    return jsonify(
        {
            "state": state,
            "reward": reward,
            "done": done,
            "info": info,
            "action": action,
        }
    )


@app.route("/grade", methods=["POST"])
def grade_endpoint():
    """
    Optional endpoint to grade the current episode so far.
    Uses the environment's trajectory if available.
    """
    try:
        score = grade(getattr(env, "trajectory", []))
    except Exception:
        score = 0.0
    return jsonify({"score": score})


@app.route("/openenv", methods=["GET"])
def openenv_config():
    """
    Small helper endpoint to return core OpenEnv metadata.
    """
    from pathlib import Path
    import yaml

    config_path = Path("openenv.yaml")
    if not config_path.exists():
        return jsonify({"error": "openenv.yaml not found"}), 404

    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return jsonify(data)


if __name__ == "__main__":
    # Run the Flask app
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

