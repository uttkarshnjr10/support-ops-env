"""Inference script — runs all three SupportOpsEnv tasks via an LLM."""

from __future__ import annotations

import json
import os
import re
import sys
import traceback
from typing import Any, Dict

import httpx
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration from environment variables
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
HF_TOKEN: str = os.getenv("HF_TOKEN", "")

ENV_URL: str = os.getenv("ENV_URL", "http://localhost:7860")
MAX_STEPS: int = int(os.getenv("MAX_STEPS", "8"))
TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.3"))

# ---------------------------------------------------------------------------
# System prompts per task
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS: Dict[str, str] = {
    "ticket-classify": (
        "You are a customer support AI. Your task is to classify a support "
        "ticket by category and priority.\n\n"
        "Categories: billing, technical, account, feature_request, complaint\n"
        "Priorities: low, medium, high, critical\n\n"
        "Respond with ONLY valid JSON matching this schema:\n"
        '{"action_type": "classify", "content": "<brief reasoning>", '
        '"metadata": {"category": "<category>", "priority": "<priority>"}}'
    ),
    "ticket-respond": (
        "You are a customer support AI. Your task is to draft a professional "
        "response to the customer's support ticket.\n\n"
        "Guidelines:\n"
        "- Start with a professional greeting\n"
        "- Acknowledge the customer's issue\n"
        "- Provide helpful, actionable information\n"
        "- Be empathetic and professional\n"
        "- Keep the response concise but thorough\n\n"
        "Respond with ONLY valid JSON matching this schema:\n"
        '{"action_type": "respond", "content": "<your full response to customer>", '
        '"metadata": {}}'
    ),
    "ticket-resolve": (
        "You are a customer support AI. Your task is to fully resolve a "
        "support case through multi-turn interaction.\n\n"
        "Available actions:\n"
        '- ask_clarification: Ask the customer for more details\n'
        '- respond: Send a helpful response\n'
        '- escalate: Escalate to a senior agent\n'
        '- resolve: Mark the ticket as resolved\n\n'
        "Choose the most appropriate action based on the ticket details and "
        "conversation history.\n\n"
        "Respond with ONLY valid JSON matching this schema:\n"
        '{"action_type": "<action>", "content": "<message>", "metadata": {}}'
    ),
}

TASKS = ["ticket-classify", "ticket-respond", "ticket-resolve"]

# Safe fallback actions when the model returns unparseable output.
_DEFAULT_ACTIONS: Dict[str, Dict[str, Any]] = {
    "ticket-classify": {
        "action_type": "classify",
        "content": "Unable to parse — defaulting",
        "metadata": {"category": "technical", "priority": "medium"},
    },
    "ticket-respond": {
        "action_type": "respond",
        "content": "Thank you for reaching out. We have received your request "
                   "and our team is looking into it. We will get back to you shortly.",
        "metadata": {},
    },
    "ticket-resolve": {
        "action_type": "respond",
        "content": "Thank you for the additional context. Let me investigate "
                   "this further and follow up with a resolution.",
        "metadata": {},
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_user_prompt(obs: Dict[str, Any]) -> str:
    """Build a user prompt from the observation dict."""
    lines = [
        f"Ticket ID: {obs['ticket_id']}",
        f"Subject: {obs['subject']}",
        f"Body: {obs['body']}",
        f"Customer Tier: {obs['customer_tier']}",
        f"Step: {obs['current_step']}/{obs['max_steps']}",
    ]
    if obs.get("previous_messages"):
        lines.append("Previous messages:")
        for msg in obs["previous_messages"]:
            lines.append(f"  [{msg.get('role', 'unknown')}]: {msg.get('content', '')}")
    return "\n".join(lines)


def parse_action(raw: str, task_name: str) -> Dict[str, Any]:
    """Extract a JSON action from the model's raw text response.

    Tries three strategies in order:
        1. Direct ``json.loads`` (fast path)
        2. Strip markdown code fences and retry
        3. Regex extraction of the first ``{...}`` block

    If all fail, returns the task-specific safe default action.
    """
    text = raw.strip()

    # --- Strategy 1: direct parse ---
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # --- Strategy 2: strip code fences ---
    if text.startswith("```"):
        stripped = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            return json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            pass

    # --- Strategy 3: regex extraction ---
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except (json.JSONDecodeError, ValueError):
            pass

    # --- Fallback: safe default ---
    print(
        f"  [WARN] Could not parse model response; using default action "
        f"for task={task_name}",
        file=sys.stderr,
    )
    return dict(_DEFAULT_ACTIONS[task_name])


def run_task(task_name: str, client: OpenAI) -> None:
    """Run a single task episode end-to-end.

    Guarantees a ``[START]`` line at the top and an ``[END]`` line at the
    bottom, even if an exception is thrown mid-episode.
    """
    print(f"[START] task={task_name} env=support-ops-env model={MODEL_NAME}")

    total_reward = 0.0
    steps = 0

    try:
        # Reset environment
        with httpx.Client(timeout=30.0) as http:
            resp = http.post(f"{ENV_URL}/reset", json={"task_name": task_name})
            resp.raise_for_status()
            obs = resp.json()

        done = False

        for step_num in range(1, MAX_STEPS + 1):
            if done:
                break

            user_prompt = build_user_prompt(obs)

            # Call the LLM
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    temperature=TEMPERATURE,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPTS[task_name]},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                raw_response = completion.choices[0].message.content or ""
            except Exception as llm_exc:
                print(
                    f"  [WARN] LLM call failed: {llm_exc}",
                    file=sys.stderr,
                )
                raw_response = ""

            # Parse action (never throws)
            action = parse_action(raw_response, task_name)

            # Ensure metadata exists
            if "metadata" not in action:
                action["metadata"] = {}

            # Submit action to env
            error = None
            with httpx.Client(timeout=30.0) as http:
                try:
                    resp = http.post(f"{ENV_URL}/step", json=action)
                    resp.raise_for_status()
                    result = resp.json()
                except (httpx.HTTPStatusError, httpx.ConnectError) as exc:
                    error = str(exc)
                    result = {"reward": 0.0, "done": True, "observation": obs}

            reward = result.get("reward", 0.0)
            done = result.get("done", False)
            total_reward += reward
            steps = step_num
            obs = result.get("observation", obs)

            action_summary = action.get("content", "")[:80]
            print(
                f"[STEP] step={step_num} "
                f"action={action_summary} "
                f"reward={reward:.2f} "
                f"done={'true' if done else 'false'} "
                f"error={error}"
            )

    except Exception as exc:
        print(
            f"  [ERROR] Unhandled exception in task={task_name}: {exc}",
            file=sys.stderr,
        )
        traceback.print_exc(file=sys.stderr)

    # --- Always emit [END] ---
    success = total_reward > 0
    print(
        f"[END] success={'true' if success else 'false'} "
        f"steps={steps} "
        f"score={total_reward:.3f} "
        f"rewards={total_reward:.2f}"
    )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run all tasks sequentially."""
    api_key = HF_TOKEN or os.getenv("OPENAI_API_KEY", "sk-placeholder")
    client = OpenAI(base_url=API_BASE_URL, api_key=api_key)

    for task in TASKS:
        run_task(task, client)


if __name__ == "__main__":
    main()
