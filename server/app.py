from __future__ import annotations

import os

from openenv.core.env_server.http_server import create_app

from models import HospitalAction, HospitalObservation
from server.openenv_wrapper.hospital_openenv_wrapper import HospitalEnvironment


def create_hospital_env() -> HospitalEnvironment:
    # Single task is configured via /reset task param for the Flask UI.
    # OpenEnv evaluation can use the same environment; reward shaping is in env.
    return HospitalEnvironment(task="easy")


app = create_app(
    create_hospital_env,
    HospitalAction,
    HospitalObservation,
    env_name="hospital-er-management",
)


def main() -> None:
    # Standard OpenEnv/docker entrypoint. We keep it compatible with
    # `openenv validate`, which expects a callable `main()` function.
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level=os.environ.get("LOG_LEVEL", "info"))


if __name__ == "__main__":
    main()

