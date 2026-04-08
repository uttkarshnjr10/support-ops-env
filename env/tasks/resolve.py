"""Grader for the ticket-resolve (multi-turn) task."""

from __future__ import annotations

from typing import Any, Dict, List

from env.models import SupportAction

# Step-efficiency bonus coefficient (per unused step).
_STEP_EFFICIENCY_BONUS = 0.02


def grade_resolve_step(
    action: SupportAction,
    ticket: Dict[str, Any],
    step_index: int,
    history: List[Dict[str, object]],
    max_steps: int,
) -> Dict[str, Any]:
    """Score a single step within the multi-turn resolve task.

    Per-step scoring:
        - Following the correct resolution_path order: +0.20 per correct step
        - Penalise repetitive ask_clarification (same 2× in a row): -0.15

    Returns:
        A dict with ``reward`` (float) and ``final`` (bool) indicating whether
        this action concludes the episode (resolve / escalate as terminal).
    """
    score = 0.0
    resolution_path: List[str] = ticket.get("resolution_path", [])

    # --- Path-following reward ---
    if step_index < len(resolution_path):
        expected_action = resolution_path[step_index]
        if action.action_type == expected_action:
            score += 0.2

    # --- Repetitive ask_clarification penalty ---
    if action.action_type == "ask_clarification" and len(history) >= 1:
        last_entry = history[-1]
        if last_entry.get("action_type") == "ask_clarification":
            score -= 0.15

    # Terminal actions
    final = action.action_type in ("resolve", "escalate")

    return {"reward": score, "final": final}


def grade_resolve_final(
    ticket: Dict[str, Any],
    history: List[Dict[str, object]],
    cumulative_reward: float,
    max_steps: int,
) -> float:
    """Compute the normalised final score at episode end.

    Additional scoring at episode end:
        - Final action matches expected (resolve vs escalate): +0.40
        - Step-efficiency bonus: +0.02 per step saved (from _STEP_EFFICIENCY_BONUS)
        - Legacy efficiency bonus: +0.05 per step saved

    Returns:
        A float in [0.0, 1.0].
    """
    resolution_path: List[str] = ticket.get("resolution_path", [])
    expected_final = resolution_path[-1] if resolution_path else "resolve"

    score = cumulative_reward

    # --- Final action match ---
    if history:
        last_action = history[-1].get("action_type", "")
        if last_action == expected_final:
            score += 0.4

    # --- Efficiency bonus (original large bonus) ---
    steps_used = len(history)
    steps_saved = max_steps - steps_used
    if steps_saved > 0:
        score += 0.05 * steps_saved

    # --- Additional step-efficiency bonus (Task 2 requirement) ---
    if steps_saved > 0:
        score += _STEP_EFFICIENCY_BONUS * steps_saved

    # --- Tiny tiebreaker to avoid identical scores across different inputs ---
    # Based on path length and steps used; max contribution ≈ 0.006
    tiebreaker = (len(resolution_path) * 0.001) + (steps_used * 0.0005)
    score += min(tiebreaker, 0.009)

    # --- Normalise to [0, 1] ---
    return min(max(round(score, 4), 0.0), 1.0)
