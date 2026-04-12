---
title: OpenEnv ER Decision Environment
emoji: 🌍
colorFrom: purple
colorTo: green
sdk: docker
pinned: false
---

# OpenEnv Environment: Emergency Decision Optimization

This project implements a **real-world OpenEnv-compatible environment** where an agent learns to make optimal decisions in a dynamic system using reward-based feedback.

The environment follows the standard OpenEnv interaction loop:

reset() → state → action → step() → reward → new state → done

---

## 🧠 Problem

In many real-world systems, decisions must be made under constraints (limited resources, time pressure, prioritization).

This environment simulates such a scenario, allowing an AI agent to learn how to optimize decisions over time.

---

## ⚙️ Environment Design

### 🔹 State (Observation)

The environment state includes:

* patients with:

  * severity (priority level)
  * wait_time
  * treated status
* available doctors
* ICU beds

---

### 🔹 Actions

The agent can take one of the following actions:

* `assign_doctor` → treat a patient
* `move_to_icu` → prioritize critical patient
* `wait` → delay all actions

---

### 🔹 Reward Function

The reward is designed to guide intelligent decision-making:

* Positive reward for treating high-severity patients
* Negative reward for:

  * delays (high wait_time)
  * incorrect prioritization
* Bonus for efficient resource usage

👉 This ensures the agent learns optimal prioritization strategies.

---

## 🎯 Objective

The goal of the agent is to:

> Maximize total reward by prioritizing critical tasks, minimizing delays, and using resources efficiently.

---

## 🔁 OpenEnv Workflow

The agent interacts with the environment as follows:

1. `reset()` → initialize environment
2. `state()` → observe current state
3. choose action
4. `step(action)` → receive reward and next state
5. repeat until `done`

---

## 🧪 Tasks

The environment defines three difficulty levels:

* **easy** → prioritize high-severity patients
* **medium** → manage resources effectively
* **hard** → optimize full system (priority + delay + efficiency)

Each task is evaluated using a grader that returns a score strictly between (0, 1).

---

## 🤖 Inference (Agent Execution)

To evaluate the agent's performance, ensure the environment server is running, then execute the inference script:

```bash
# Set your local or remote API URL
export OPENENV_URL="http://127.0.0.1:8000"
python inference.py
```

This script connects to the environment server via WebSockets, executes the agent's heuristic policy, and outputs a formatted trajectory for the grader.

---

## ⚠️ Notes

* This is a **production-spec OpenEnv environment**
* It uses a **FastAPI-based server** for agent interactions
* Supports standard `openenv validate` and `openenv run` commands

---

## 📦 Structure

* `hospital_env.py` → environment logic
* `grader.py` → evaluation
* `inference.py` → agent simulation
* `openenv.yaml` → environment specification
* `Dockerfile` → deployment

---

## 🏁 Goal

This project demonstrates how real-world decision systems can be modeled as learning environments for AI agents using reward-based optimization.
