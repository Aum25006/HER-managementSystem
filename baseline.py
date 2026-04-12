from __future__ import annotations

from typing import Dict, Any

from hospital_env import make_env


class BaselineAgent:
    """
    Simple baseline agent:
    - Always selects the highest severity untreated patient.
    - Uses doctors by default; if patient is very critical (severity >= 9),
      may choose ICU.
    """

    def __init__(self, task: str = "easy") -> None:
        self.env = make_env(task=task)
        self.state = self.env.reset()

    def select_action(self) -> Dict[str, Any]:
        patients = self.state.get("patients", [])
        untreated = [p for p in patients if not p.get("treated")]

        if not untreated:
            # Nothing left to do; wait
            return {"type": "wait", "patient_id": None}

        # Highest severity untreated patient
        target = max(untreated, key=lambda p: p.get("severity", 0))
        pid = target.get("id")
        severity = target.get("severity", 0)

        # If very critical and ICU capacity exists, move to ICU
        if severity >= 9 and self.state.get("available_beds", 0) > 0:
            return {"type": "move_to_icu", "patient_id": pid}

        # Otherwise assign a doctor if available
        if self.state.get("available_doctors", 0) > 0:
            return {"type": "assign_doctor", "patient_id": pid}

        # If no resources free, wait
        return {"type": "wait", "patient_id": None}

    def run_episode(self, render: bool = False) -> float:
        total_reward = 0.0
        done = False

        while not done:
            action = self.select_action()
            next_state, reward, done, info = self.env.step(action)
            total_reward += reward
            self.state = next_state

            if render:
                print("Action:", action)
                print("Reward:", reward, "Done:", done)
                print("State:", self.state)
                print("-" * 40)

        return total_reward


if __name__ == "__main__":
    agent = BaselineAgent(task="easy")
    total = agent.run_episode(render=True)
    print("Total reward:", total)

