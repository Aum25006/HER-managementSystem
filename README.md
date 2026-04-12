---
title: Hospital ER Management Environment
emoji: 🏥
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
---

# 🏥 Hospital ER Management Environment (OpenEnv)

A production-grade **decision-optimization environment** built for the OpenEnv benchmark. This repository simulates a high-pressure Emergency Room where an AI agent must manage limited resources (Doctors and ICU beds) to stabilize and treat patients based on medical priority.

---

## 🚀 Quick Start

### 1. Local Development
Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Start the Environment Server
The environment runs as a FastAPI-based server:
```bash
export PORT=8000
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### 3. Run the Evaluation Agent
In a separate terminal, execute the inference script to test the environment:
```bash
export OPENENV_URL="http://127.0.0.1:8000"
export HF_TOKEN="your_huggingface_token"
python inference.py
```

---

## 🧠 Core Logic & Simulation

The environment logic is encapsulated in `hospital_env.py`. It is designed as a **stochastic state machine** where time (steps) increases patient wait times and severity risk.

### 🔹 Resource Management
*   **Doctors**: Consumed by `assign_doctor` actions. Replenish over time as patients are treated.
*   **ICU Beds**: Mandatory for critical patients (Severity ≥ 8). Limited capacity requires strategic allocation.

### 🔹 Reward Function & Scoring
The agent is graded on a strict **(0, 1) scale** based on four weighted components:
1.  **Critical Life-Saving (40%)**: Percentage of severity 8-10 patients treated.
2.  **Wait Time Efficiency (25%)**: Minimizing the average time patients spend untreated.
3.  **Prioritization (20%)**: Penalties for treating low-severity cases while critical ones wait.
4.  **Resource Utilization (15%)**: Rewarding active interventions over idle 'wait' actions.

---

## 🛠 Repository Structure

```text
.
├── server/
│   ├── app.py              # FastAPI Server Entrypoint
│   └── openenv_wrapper/    # OpenEnv API Adapter
├── hospital_env.py         # The "Brain" (Core Simulation Logic)
├── grader.py               # Evaluation & Scoring Rubric
├── inference.py            # Phase 2 Agent (WebSocket + HF Token)
├── models.py               # Pydantic Schemas for Actions/Observations
├── openenv.yaml            # Environment Specification
└── Dockerfile              # Deployment Configuration
```

---

## 📡 Architecture: No-HTTP WebSocket Protocol

To ensure high performance and stateful persistence, this environment utilizes **WebSockets (WS)** for agent-environment interaction.
*   **Stateless Initialization**: The environment is configured via `openenv.yaml`.
*   **Persistent Episode**: Once a WebSocket connection is established, the session maintains state across all `step()` calls, satisfying the "No-Stateless-HTTP" requirement for Phase 2.

---

## 🎯 Evaluation Tasks

This environment provides three standardized tasks for automated benchmarking:
*   **Easy**: Standard prioritization of high-severity patients.
*   **Medium**: Resource-constrained management (Lower doctor/bed ratio).
*   **Hard**: High-traffic simulation requiring optimal wait-time balancing.

---

## 🏁 Goal
This project demonstrates the ability to model complex, real-world resource constraints as a standardized AI training environment. It is fully compliant with the **OpenEnv Phase 2** specification.
