---
title: Her Management Er
emoji: 🌍
colorFrom: purple
colorTo: green
sdk: docker
pinned: false
---

# Hospital ER Management System (OpenEnv)

This project implements a simple Hospital Emergency Room (ER) management environment for the OpenEnv workflow.
The environment exposes the standard OpenEnv-style loop: `reset()` -> `state()` -> `step(action)` and produces shaped rewards based on patient severity, wait time, prioritization, and resource usage.

## What’s included

- OpenEnv-compliant server (typed action/observation + OpenEnv FastAPI server)
  - `server/app.py`
  - `server/openenv_wrapper/hospital_openenv_wrapper.py`
  - `models.py`
- ER simulation logic
  - `server/hospital_environment.py`
- Reward grading
  - `server/services/grader.py`
- Baseline inference scripts
  - `client_notebooks/baseline_inference.py`
- Simple web UI (optional, for manual testing only)
  - `server/flask_app_legacy.py` + `static/index.html`

## Requirements

- Python 3.11+

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick start (OpenEnv)

### 1. Start the OpenEnv server (port 8000)

```bash
python -c "from server.app import main; main()"
```

### 2. Verify health

```powershell
(Invoke-WebRequest -UseBasicParsing -TimeoutSec 5 http://127.0.0.1:8000/health).Content
```

Expected:

```json
{"status":"healthy"}
```

### 3. Validate with OpenEnv CLI

```bash
openenv validate --verbose
```

### 4. Run baseline

```bash
python client_notebooks/inference.py
```

For manual baseline runs:

```bash
python client_notebooks/baseline_inference.py easy
python client_notebooks/baseline_inference.py medium
python client_notebooks/baseline_inference.py hard
```

## Optional: Run the simple UI (manual testing only)

```bash
python server/flask_app_legacy.py
```

Open: `http://127.0.0.1:5000`

## OpenEnv manifest

- `openenv.yaml`

Defines observation/action spaces and tasks (`easy`, `medium`, `hard`).
