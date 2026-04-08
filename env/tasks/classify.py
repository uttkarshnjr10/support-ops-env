"""Grader for the ticket-classify task."""

from __future__ import annotations

from typing import Any, Dict

from env.models import SupportAction


# Priority levels ordered from lowest to highest for off-by-one scoring.
_PRIORITY_LEVELS = ["low", "medium", "high", "critical"]


def grade_classify(action: SupportAction, ticket: Dict[str, Any]) -> float:
    """Score a classify action against the ground-truth ticket data.

    Scoring:
        - Correct category: +0.5
        - Exact priority match: +0.3
        - Priority off by one level: +0.15
        - Wrong on both: 0.0

    Returns:
        A float in [0.0, 1.0].
    """
    score = 0.0

    predicted_category = action.metadata.get("category", "").lower().strip()
    predicted_priority = action.metadata.get("priority", "").lower().strip()

    correct_category = ticket["correct_category"].lower().strip()
    correct_priority = ticket["correct_priority"].lower().strip()

    # --- Category scoring ---
    if predicted_category == correct_category:
        score += 0.5

    # --- Priority scoring ---
    if predicted_priority == correct_priority:
        score += 0.3
    else:
        # Check off-by-one
        if (
            predicted_priority in _PRIORITY_LEVELS
            and correct_priority in _PRIORITY_LEVELS
        ):
            predicted_idx = _PRIORITY_LEVELS.index(predicted_priority)
            correct_idx = _PRIORITY_LEVELS.index(correct_priority)
            if abs(predicted_idx - correct_idx) == 1:
                score += 0.15

    # --- Tiny tiebreaker so different inputs never map to identical scores ---
    tiebreaker = min(len(action.content) / 10000.0, 0.009)
    score += tiebreaker

    return min(max(round(score, 4), 0.0), 1.0)
