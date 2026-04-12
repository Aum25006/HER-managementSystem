from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Any, Optional
import random


@dataclass
class Patient:
    id: int
    severity: int  # 1 (low) - 10 (critical)
    wait_time: int = 0
    treated: bool = False
    in_icu: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HospitalEnv:
    """
    Simple hospital emergency room environment suitable for OpenEnv-style agents.
    """

    def __init__(
        self,
        min_patients: int = 5,
        max_patients: int = 8,
        min_doctors: int = 2,
        max_doctors: int = 3,
        min_icu_beds: int = 1,
        max_icu_beds: int = 2,
        max_steps: int = 50,
        seed: Optional[int] = None,
    ) -> None:
        self.min_patients = min_patients
        self.max_patients = max_patients
        self.min_doctors = min_doctors
        self.max_doctors = max_doctors
        self.min_icu_beds = min_icu_beds
        self.max_icu_beds = max_icu_beds
        self.max_steps = max_steps
        self.random = random.Random(seed)

        self.patients: List[Patient] = []
        self.total_doctors: int = 0
        self.available_doctors: int = 0
        self.total_beds: int = 0
        self.available_beds: int = 0
        self.step_count: int = 0

        # For reward shaping / grading
        self.trajectory: List[Dict[str, Any]] = []

    # ---- OpenEnv-style interface ----

    def reset(self) -> Dict[str, Any]:
        """Initialize a new episode."""
        num_patients = self.random.randint(self.min_patients, self.max_patients)
        self.patients = []
        for pid in range(num_patients):
            severity = self.random.randint(1, 10)
            self.patients.append(Patient(id=pid, severity=severity))

        self.total_doctors = self.random.randint(self.min_doctors, self.max_doctors)
        self.available_doctors = self.total_doctors

        self.total_beds = self.random.randint(self.min_icu_beds, self.max_icu_beds)
        self.available_beds = self.total_beds

        self.step_count = 0
        self.trajectory = []

        state = self.get_state()
        return state

    def get_state(self) -> Dict[str, Any]:
        """Return the current observable state as JSON-serializable dict."""
        return {
            "patients": [p.to_dict() for p in self.patients],
            "available_doctors": self.available_doctors,
            "available_beds": self.available_beds,
            "step_count": self.step_count,
            "max_steps": self.max_steps,
        }

    # alias to match the spec name
    state = get_state

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Perform a step in the environment.
        Action format:
        {
            "type": "assign_doctor" | "move_to_icu" | "wait",
            "patient_id": int | None
        }
        """
        self.step_count += 1
        action_type = action.get("type")
        patient_id = action.get("patient_id")

        reward = 0.0
        info: Dict[str, Any] = {"action_valid": True, "reason": ""}

        patient = None
        treated_this_step = False
        treated_patient_severity = None
        if patient_id is not None:
            patient = self._get_patient_by_id(patient_id)
            if patient is None:
                info["action_valid"] = False
                info["reason"] = "Invalid patient_id"

        # Apply action effects
        if action_type == "assign_doctor":
            reward, treated_this_step, treated_patient_severity = self._handle_assign_doctor(
                patient, info
            )
        elif action_type == "move_to_icu":
            reward, treated_this_step, treated_patient_severity = self._handle_move_to_icu(
                patient, info
            )
        elif action_type == "wait":
            reward += self._handle_wait()
        else:
            info["action_valid"] = False
            info["reason"] = "Unknown action type"
            # Small negative reward for invalid / idle actions
            reward -= 1.0

        # Reward / penalty for long waits (apply once)
        # Spec: -5 if any patient has wait_time > 5
        if any(p.wait_time > 5 for p in self.patients):
            reward -= 5.0

        # Wrong priority: low severity treated while a higher severity patient remains untreated.
        if treated_this_step and treated_patient_severity is not None and treated_patient_severity <= 3:
            if any((not p.treated) and p.severity >= 7 for p in self.patients):
                reward -= 10.0

        done = self._is_done()
        state = self.get_state()

        # Log trajectory step for grading
        self.trajectory.append(
            {
                "step": self.step_count,
                "action": action,
                "state": state,
                "reward": reward,
                "done": done,
            }
        )

        return state, reward, done, info

    # ---- Internal helpers ----

    def _get_patient_by_id(self, patient_id: int) -> Optional[Patient]:
        for p in self.patients:
            if p.id == patient_id:
                return p
        return None

    def _recover_resources(self) -> None:
        """
        Simple recovery model: capacity becomes available gradually over time.
        Recovery happens before the action is applied (so the UI sees the effect
        of using resources immediately).
        """
        if self.available_doctors < self.total_doctors:
            self.available_doctors += 1
        if self.available_beds < self.total_beds:
            self.available_beds += 1

    def _handle_assign_doctor(
        self, patient: Optional[Patient], info: Dict[str, Any]
    ) -> Tuple[float, bool, Optional[int]]:
        reward = 0.0
        if self.available_doctors <= 0:
            info["action_valid"] = False
            info["reason"] = "No available doctors"
            return -1.0, False, None

        if patient is None:
            return -1.0, False, None

        if patient.treated:
            return -1.0, False, None

        # Assign doctor and treat patient
        self.available_doctors -= 1
        patient.treated = True
        patient.in_icu = False

        # Track priority penalties based on this step's treatment.
        treated_this_step = True
        treated_patient_severity = patient.severity

        # Reward for treating based on severity
        if patient.severity >= 7:
            reward += 10.0
        elif 4 <= patient.severity <= 6:
            reward += 5.0
        else:
            # Low severity treatment gets no positive shaping here.
            reward += 0.0

        # Efficient use of resources
        reward += 2.0

        return reward, treated_this_step, treated_patient_severity

    def _handle_move_to_icu(
        self, patient: Optional[Patient], info: Dict[str, Any]
    ) -> Tuple[float, bool, Optional[int]]:
        reward = 0.0
        if self.available_beds <= 0:
            info["action_valid"] = False
            info["reason"] = "No available ICU beds"
            return -1.0, False, None

        if patient is None:
            return -1.0, False, None

        if patient.treated:
            return -1.0, False, None

        # Move to ICU and consider treated
        self.available_beds -= 1
        patient.in_icu = True
        patient.treated = True

        treated_this_step = True
        treated_patient_severity = patient.severity

        # Reward for treating based on severity
        if patient.severity >= 7:
            reward += 10.0
        elif 4 <= patient.severity <= 6:
            reward += 5.0
        else:
            # ICU misuse for low severity
            reward -= 2.0

        # Efficient use of resources
        reward += 2.0

        return reward, treated_this_step, treated_patient_severity

    def _handle_wait(self) -> float:
        # Increase wait time for all untreated patients
        for p in self.patients:
            if not p.treated:
                p.wait_time += 1

        # Waiting also replenishes some capacity (doctors/beds) over time.
        self._recover_resources()

        # Small negative reward for idle action
        return -1.0

    def _is_done(self) -> bool:
        all_treated = all(p.treated for p in self.patients)
        if all_treated:
            return True
        if self.step_count >= self.max_steps:
            return True
        return False

    def is_done(self) -> bool:
        """Public wrapper for current episode termination status."""
        return self._is_done()


# ---- Task definitions ----

TASKS: Dict[str, Dict[str, Any]] = {
    "easy": {
        "description": "Treat highest severity patients first.",
        "objective": "Always select the highest severity untreated patient.",
    },
    "medium": {
        "description": "Manage doctors and ICU beds effectively.",
        "objective": "Balance use of doctors and ICU beds while prioritizing critical patients.",
    },
    "hard": {
        "description": "Optimize wait time, prioritization, and resource efficiency.",
        "objective": "Minimize average wait time and penalties while maximizing rewards.",
    },
}


def make_env(task: str = "easy") -> HospitalEnv:
    """
    Factory function for creating the environment for a given task.
    Currently tasks share the same dynamics but can be extended.
    """
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task}")
    return HospitalEnv()

