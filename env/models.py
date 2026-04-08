"""Pydantic models for the SupportOpsEnv environment."""

from __future__ import annotations

from typing import Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field


class TicketObservation(BaseModel):
    """What the agent sees at each step — the current ticket context."""

    model_config = ConfigDict(extra="forbid")

    ticket_id: str
    subject: str
    body: str
    customer_tier: Literal["bronze", "silver", "gold"]
    previous_messages: List[Dict[str, str]] = Field(default_factory=list)
    current_step: int
    max_steps: int
    task_name: str
    done: bool


class SupportAction(BaseModel):
    """An action the agent can take on a support ticket."""

    model_config = ConfigDict(extra="forbid")

    action_type: Literal[
        "classify", "respond", "ask_clarification", "escalate", "resolve"
    ]
    content: str
    metadata: Dict[str, str] = Field(default_factory=dict)


class StepResult(BaseModel):
    """The result returned after the environment processes one step."""

    model_config = ConfigDict(extra="forbid")

    observation: TicketObservation
    reward: float
    done: bool
    info: Dict[str, object] = Field(default_factory=dict)


class EpisodeState(BaseModel):
    """Snapshot of the current episode state."""

    model_config = ConfigDict(extra="forbid")

    task_name: str
    ticket_id: str
    step: int
    total_reward: float
    history: List[Dict[str, object]] = Field(default_factory=list)
