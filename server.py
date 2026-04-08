"""FastAPI server exposing the SupportOpsEnv via HTTP."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from env.environment import SupportOpsEnv
from env.models import EpisodeState, StepResult, SupportAction, TicketObservation

app = FastAPI(
    title="SupportOpsEnv",
    description="OpenEnv-compliant customer support ticket resolution environment",
    version="1.0.0",
)

# Global environment instance (single-threaded demo).
_env: Optional[SupportOpsEnv] = None

# Path to the OpenEnv manifest.
_MANIFEST_PATH = Path(__file__).parent / "openenv.yaml"


# ------------------------------------------------------------------
# Request / response helpers
# ------------------------------------------------------------------

class ResetRequest(BaseModel):
    """Body for the ``/reset`` endpoint."""

    model_config = ConfigDict(extra="forbid")

    task_name: str = "ticket-classify"


class HealthResponse(BaseModel):
    """Health check response."""

    model_config = ConfigDict(extra="forbid")

    status: str = "ok"


class TaskInfo(BaseModel):
    """Single task entry returned by the ``/tasks`` endpoint."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    difficulty: str
    max_steps: int
    reward_range: List[float]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _load_manifest() -> Dict[str, Any]:
    """Load and cache the openenv.yaml manifest."""
    with open(_MANIFEST_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@app.post("/reset", response_model=TicketObservation)
def reset(body: ResetRequest) -> TicketObservation:
    """Reset the environment with a new task and return the first observation."""
    global _env
    try:
        _env = SupportOpsEnv(task_name=body.task_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _env.reset()


@app.post("/step", response_model=StepResult)
def step(action: SupportAction) -> StepResult:
    """Submit an action and receive the step result."""
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    return _env.step(action)


@app.get("/state", response_model=EpisodeState)
def state() -> EpisodeState:
    """Return the current episode state."""
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    return _env.state()


@app.get("/tasks", response_model=List[TaskInfo])
def tasks() -> List[TaskInfo]:
    """Return all available tasks from the openenv.yaml manifest."""
    manifest = _load_manifest()
    return [
        TaskInfo(
            name=t["name"],
            description=t["description"],
            difficulty=t["difficulty"],
            max_steps=t["max_steps"],
            reward_range=t["reward_range"],
        )
        for t in manifest.get("tasks", [])
    ]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Simple liveness probe."""
    return HealthResponse()


# ------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=7860, reload=False)
