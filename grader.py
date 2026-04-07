from __future__ import annotations

from typing import Dict, Any, List

EPS = 1e-6


def _extract_from_state(state: Dict[str, Any]) -> Dict[str, Any]:
    patients = state.get("patients", [])
    treated = [p for p in patients if p.get("treated")]
    untreated = [p for p in patients if not p.get("treated")]

    critical = [p for p in patients if p.get("severity", 0) >= 8]
    critical_treated = [p for p in critical if p.get("treated")]

    low_severity_treated = [p for p in treated if p.get("severity", 0) <= 3]

    return {
        "patients": patients,
        "treated": treated,
        "untreated": untreated,
        "critical": critical,
        "critical_treated": critical_treated,
        "low_severity_treated": low_severity_treated,
    }


def grade(result: Any) -> float:
    """
    Grade an episode trajectory or final state on a 0..1 scale.

    `result` can be:
      - a final state dict
      - a trajectory list of steps:
        [{"state": ..., "reward": ..., "done": ...}, ...]
    """
    trajectory: List[Dict[str, Any]] = []
    if isinstance(result, list):
        # Assume trajectory; use final state and whole trajectory
        if not result:
            return 0.5
        trajectory = result
        final_state = trajectory[-1].get("state", {})
    elif isinstance(result, dict):
        # Common wrapper formats from inference scripts / evaluators
        if "final_state" in result:
            final_state = result.get("final_state", {})
            maybe_traj = result.get("trajectory") or []
            if isinstance(maybe_traj, list):
                trajectory = maybe_traj
        elif "trajectory" in result and isinstance(result.get("trajectory"), list):
            maybe_traj = result.get("trajectory", [])
            trajectory = maybe_traj
            final_state = maybe_traj[-1].get("state", {}) if maybe_traj else {}
        elif "state" in result and isinstance(result.get("state"), dict):
            # Sometimes evaluators wrap as {"state": ..., "reward": ..., "done": ...}
            final_state = result.get("state", {})
        else:
            final_state = result
    else:
        raise ValueError("Unsupported result type for grading")

    extracted = _extract_from_state(final_state)
    patients = extracted["patients"]
    if not patients:
        return 0.5

    # ---- Component 1: % critical patients treated ----
    num_critical = len(extracted["critical"])
    num_critical_treated = len(extracted["critical_treated"])
    if num_critical > 0:
        critical_score = num_critical_treated / num_critical
    else:
        critical_score = 1.0  # no critical patients => trivially satisfied

    # ---- Component 2: Average wait time (lower is better) ----
    wait_times = [p.get("wait_time", 0) for p in patients]
    avg_wait = sum(wait_times) / len(wait_times)
    # Map average wait time to [0, 1]; 0 wait -> 1 score, >=10 -> 0
    wait_score = max(0.0, min(1.0, 1.0 - avg_wait / 10.0))

    # ---- Component 3: Prioritization correctness ----
    # Penalize low severity treated while higher severity untreated remain
    prioritization_penalty = 0.0
    untreated = extracted["untreated"]
    if untreated:
        max_untreated_severity = max(p.get("severity", 0) for p in untreated)
        for p in extracted["low_severity_treated"]:
            if p.get("severity", 0) < max_untreated_severity:
                prioritization_penalty += 0.2
    prioritization_score = max(0.0, 1.0 - prioritization_penalty)

    # ---- Component 4: Resource usage efficiency ----
    # If we have trajectory, look at how often actions were "wait" vs treating.
    if trajectory:
        total_steps = len(trajectory)
        wait_steps = 0
        treat_steps = 0
        for step in trajectory:
            action = step.get("action", {})
            action_type = action.get("type")
            if action_type == "wait":
                wait_steps += 1
            elif action_type in ("assign_doctor", "move_to_icu"):
                treat_steps += 1
        if total_steps > 0:
            # Encourage more treating steps than idle waits
            resource_score = max(0.0, min(1.0, treat_steps / max(1, wait_steps + treat_steps)))
        else:
            resource_score = 0.0
    else:
        resource_score = 0.5  # neutral if no trajectory provided

    # Weighted combination
    # Critical treatment is most important
    final_score = (
        0.4 * critical_score
        + 0.25 * wait_score
        + 0.2 * prioritization_score
        + 0.15 * resource_score
    )

    # Ensure score is strictly within (0, 1) per evaluator requirement.
    return max(EPS, min(1.0 - EPS, final_score))


__all__ = ["grade"]

