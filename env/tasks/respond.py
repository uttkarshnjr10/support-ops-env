"""Grader for the ticket-respond task."""

from __future__ import annotations

from typing import Any, Dict

from env.models import SupportAction

# Minimum keywords required for full marks by customer_tier + category combo.
# Gold-tier complaints require 3 keyword hits instead of 2 (10 % stricter).
_KEYWORD_THRESHOLD_DEFAULT = 2
_KEYWORD_THRESHOLD_STRICT = 3


def _is_strict_ticket(ticket: Dict[str, Any]) -> bool:
    """Return True if the ticket should be graded with stricter keyword requirements."""
    return (
        ticket.get("customer_tier", "").lower() == "gold"
        and ticket.get("correct_category", "").lower() == "complaint"
    )


def grade_respond(action: SupportAction, ticket: Dict[str, Any]) -> float:
    """Score a respond action against the ground-truth ticket data.

    Scoring (all partial, never binary):
        - Content length > 50 chars:         +0.10
        - Professional greeting present:      +0.10
        - Ticket subject acknowledged:        +0.15
        - Ideal keywords found:               +0.30 each (max varies by tier)
        - Contains "I don't know" / "cannot help": -0.20

    Gold-tier complaints are scored 10 % stricter:
        - Require 3 keyword matches (instead of 2) for the full +0.60
        - Each keyword is worth +0.20 (so 3 × 0.20 = 0.60)

    Returns:
        A float clamped to [0.0, 1.0].
    """
    score = 0.0
    content = action.content
    content_lower = content.lower()

    # --- Length check ---
    if len(content) > 50:
        score += 0.1

    # --- Professional greeting ---
    greetings = [
        "dear", "hello", "hi ", "good morning", "good afternoon",
        "good evening", "thank you for", "thanks for",
    ]
    if any(content_lower.startswith(g) or g in content_lower[:80] for g in greetings):
        score += 0.1

    # --- Subject acknowledgement ---
    subject_words = ticket["subject"].lower().split()
    # Consider acknowledged if at least half the meaningful words appear
    meaningful = [w for w in subject_words if len(w) > 3]
    if meaningful:
        matched = sum(1 for w in meaningful if w in content_lower)
        if matched >= max(len(meaningful) // 2, 1):
            score += 0.15

    # --- Ideal response keywords ---
    strict = _is_strict_ticket(ticket)
    threshold = _KEYWORD_THRESHOLD_STRICT if strict else _KEYWORD_THRESHOLD_DEFAULT
    per_keyword = 0.60 / threshold  # 0.20 if strict, 0.30 if normal

    keywords_found = 0
    for kw in ticket.get("ideal_response_keywords", []):
        if kw.lower() in content_lower:
            keywords_found += 1
            if keywords_found >= threshold:
                break
    score += per_keyword * keywords_found  # capped at 0.60

    # --- Penalties ---
    penalties = ["i don't know", "cannot help", "i can't help", "i am unable"]
    for phrase in penalties:
        if phrase in content_lower:
            score -= 0.2
            break

    # --- Tiny tiebreaker based on content length to avoid identical scores ---
    # Max tiebreaker is 0.009 so it never changes the score bracket.
    tiebreaker = min(len(content) / 10000.0, 0.009)
    score += tiebreaker

    return min(max(round(score, 4), 0.0), 1.0)
