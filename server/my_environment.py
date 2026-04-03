from __future__ import annotations

import uuid
from typing import Any, Dict, Optional, Tuple

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from hospital_env import HospitalEnv
from models import HospitalAction, HospitalObservation


class HospitalEnvironment(Environment[HospitalAction, HospitalObservation, State]):
    """
    OpenEnv wrapper around `HospitalEnv` so the environment is spec-compliant
    (typed Action/Observation models + OpenEnv server).
    """

    def __init__(self, task: str = "easy") -> None:
        super().__init__()
        self._task = task
        self._env = HospitalEnv(max_steps=50)
        self._state: State = State(episode_id=str(uuid.uuid4()), step_count=0)

    @property
    def state(self) -> State:
        return self._state

    def reset(
        self, seed: int | None = None, episode_id: str | None = None, **kwargs: Any
    ) -> HospitalObservation:
        if "task" in kwargs and isinstance(kwargs["task"], str):
            self._task = kwargs["task"]

        self._env = HospitalEnv(max_steps=self._env.max_steps, seed=seed)
        initial_state = self._env.reset()
        self._state = State(
            episode_id=episode_id or str(uuid.uuid4()), step_count=0
        )

        obs_dict = initial_state
        return HospitalObservation(
            patients=obs_dict.get("patients", []),
            available_doctors=obs_dict.get("available_doctors", 0),
            available_beds=obs_dict.get("available_beds", 0),
            step_count=obs_dict.get("step_count", 0),
            max_steps=obs_dict.get("max_steps", 50),
            reward=0.0,
            done=self._env.is_done(),
            metadata={"task": self._task},
        )

    def step(
        self, action: HospitalAction, timeout_s: float | None = None, **kwargs: Any
    ) -> HospitalObservation:
        action_dict = {"type": action.type, "patient_id": action.patient_id}
        next_state, reward, done, info = self._env.step(action_dict)
        self._state = State(
            episode_id=self._state.episode_id,
            step_count=self._env.step_count,
            **({"info": info} if info else {}),
        )

        return HospitalObservation(
            patients=next_state.get("patients", []),
            available_doctors=next_state.get("available_doctors", 0),
            available_beds=next_state.get("available_beds", 0),
            step_count=next_state.get("step_count", 0),
            max_steps=next_state.get("max_steps", 50),
            reward=reward,
            done=done,
            metadata={
                "task": self._task,
                "info": info,
                "action": action_dict,
            },
        )

