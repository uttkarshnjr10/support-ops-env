"""Core SupportOpsEnv environment."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from env.models import EpisodeState, StepResult, SupportAction, TicketObservation
from env.tasks.classify import grade_classify
from env.tasks.resolve import grade_resolve_final, grade_resolve_step
from env.tasks.respond import grade_respond

_TASKS = {
    "ticket-classify": {"max_steps": 1, "difficulty": "easy"},
    "ticket-respond": {"max_steps": 3, "difficulty": "medium"},
    "ticket-resolve": {"max_steps": 6, "difficulty": "hard"},
}

_VALID_ACTIONS: Dict[str, List[str]] = {
    "ticket-classify": ["classify"],
    "ticket-respond": ["respond"],
    "ticket-resolve": ["ask_clarification", "respond", "escalate", "resolve"],
}

_DATA_PATH = Path(__file__).parent / "data" / "tickets.json"


class SupportOpsEnv:
    """OpenEnv-compliant customer-support ticket resolution environment."""

    def __init__(self, task_name: str) -> None:
        """Initialise the environment for a given task.

        Args:
            task_name: One of ``ticket-classify``, ``ticket-respond``,
                or ``ticket-resolve``.

        Raises:
            ValueError: If *task_name* is not a recognised task.
        """
        if task_name not in _TASKS:
            raise ValueError(
                f"Unknown task '{task_name}'. Choose from {list(_TASKS.keys())}"
            )
        self.task_name: str = task_name
        self._task_cfg: Dict[str, Any] = _TASKS[task_name]
        self._tickets: List[Dict[str, Any]] = self._load_tickets()

        # Episode state (populated on reset)
        self._ticket: Optional[Dict[str, Any]] = None
        self._step: int = 0
        self._done: bool = False
        self._history: List[Dict[str, object]] = []
        self._total_reward: float = 0.0
        self._cumulative_resolve_reward: float = 0.0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_tickets(self) -> List[Dict[str, Any]]:
        """Load tickets matching the current task difficulty."""
        with open(_DATA_PATH, encoding="utf-8") as fh:
            all_tickets: List[Dict[str, Any]] = json.load(fh)
        difficulty = self._task_cfg["difficulty"]
        filtered = [t for t in all_tickets if t["difficulty"] == difficulty]
        if not filtered:
            raise RuntimeError(
                f"No tickets found with difficulty='{difficulty}'"
            )
        return filtered

    def _make_observation(self) -> TicketObservation:
        """Build a ``TicketObservation`` from current state."""
        assert self._ticket is not None
        return TicketObservation(
            ticket_id=self._ticket["ticket_id"],
            subject=self._ticket["subject"],
            body=self._ticket["body"],
            customer_tier=self._ticket["customer_tier"],
            previous_messages=[
                {"role": str(h.get("role", "agent")), "content": str(h.get("content", ""))}
                for h in self._history
            ],
            current_step=self._step,
            max_steps=self._task_cfg["max_steps"],
            task_name=self.task_name,
            done=self._done,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> TicketObservation:
        """Start a new episode with a random ticket.

        Returns:
            The initial observation.
        """
        self._ticket = random.choice(self._tickets)
        self._step = 0
        self._done = False
        self._history = []
        self._total_reward = 0.0
        self._cumulative_resolve_reward = 0.0
        return self._make_observation()

    def step(self, action: SupportAction) -> StepResult:
        """Process one agent action.

        Args:
            action: The action submitted by the agent.

        Returns:
            A ``StepResult`` containing the new observation, reward,
            done flag, and info dict.
        """
        assert self._ticket is not None, "Call reset() before step()"

        info: Dict[str, object] = {}

        # --- Validate action type ---
        valid_types = _VALID_ACTIONS[self.task_name]
        if action.action_type not in valid_types:
            obs = self._make_observation()
            return StepResult(
                observation=obs,
                reward=0.0,
                done=False,
                info={"error": "invalid_action_type"},
            )

        self._step += 1
        reward = 0.0

        # --- Grade ---
        if self.task_name == "ticket-classify":
            reward = grade_classify(action, self._ticket)

        elif self.task_name == "ticket-respond":
            reward = grade_respond(action, self._ticket)

        elif self.task_name == "ticket-resolve":
            result = grade_resolve_step(
                action,
                self._ticket,
                step_index=self._step - 1,
                history=self._history,
                max_steps=self._task_cfg["max_steps"],
            )
            reward = result["reward"]
            self._cumulative_resolve_reward += reward

            if result["final"] or self._step >= self._task_cfg["max_steps"]:
                self._done = True

        # --- Log action ---
        self._history.append(
            {
                "step": self._step,
                "action_type": action.action_type,
                "content": action.content,
                "metadata": action.metadata,
                "reward": reward,
                "role": "agent",
            }
        )

        # --- Check termination ---
        if self._step >= self._task_cfg["max_steps"]:
            self._done = True

        # --- Finalise resolve score at episode end ---
        if self._done and self.task_name == "ticket-resolve":
            reward = grade_resolve_final(
                self._ticket,
                self._history,
                self._cumulative_resolve_reward,
                self._task_cfg["max_steps"],
            )
            # Replace step reward with final normalised score
            self._total_reward = reward
            info["final_score"] = reward
        else:
            self._total_reward += reward

        obs = self._make_observation()
        return StepResult(
            observation=obs,
            reward=round(reward, 4),
            done=self._done,
            info=info,
        )

    def state(self) -> EpisodeState:
        """Return a snapshot of the current episode.

        Returns:
            An ``EpisodeState`` instance.
        """
        return EpisodeState(
            task_name=self.task_name,
            ticket_id=self._ticket["ticket_id"] if self._ticket else "",
            step=self._step,
            total_reward=round(self._total_reward, 4),
            history=self._history,
        )
