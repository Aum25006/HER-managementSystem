from __future__ import annotations

from typing import List, Optional, Literal, Dict, Any

from pydantic import Field
from openenv.core.env_server.types import Action, Observation


class PatientObservationItem(dict):
    """
    OpenEnv observation items are JSON-compatible dicts.
    We keep this wrapper minimal to avoid strict schema mismatch.
    """


class HospitalAction(Action):
    """
    ER action.
    """

    type: Literal["assign_doctor", "move_to_icu", "wait"] = Field(
        ..., description="Action type"
    )
    patient_id: Optional[int] = Field(
        None, description="Target patient id (null for 'wait')"
    )


class HospitalObservation(Observation):
    """
    Observable environment state.
    """

    patients: List[Dict[str, Any]] = Field(default_factory=list)
    available_doctors: int = Field(ge=0, default=0)
    available_beds: int = Field(ge=0, default=0)
    step_count: int = Field(ge=0, default=0)
    max_steps: int = Field(ge=1, default=50)


__all__ = ["HospitalAction", "HospitalObservation"]

