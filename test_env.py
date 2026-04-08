"""Comprehensive stress-test suite for SupportOpsEnv.

Covers:
    1. HTTP endpoint tests (all tasks + invalid actions)
    2. Direct grader tests (perfect, blank, determinism)
    3. Inference log format validation
    4. Memory leak check (50 sequential resets via tracemalloc)

Usage:
    # Start the server first:
    python server.py
    # Then run tests:
    python test_env.py
"""

from __future__ import annotations

import re
import sys
import time
import tracemalloc
from typing import Any, Dict, List

import httpx

from env.environment import SupportOpsEnv
from env.models import SupportAction
from env.tasks.classify import grade_classify
from env.tasks.resolve import grade_resolve_final, grade_resolve_step
from env.tasks.respond import grade_respond

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

BASE = "http://localhost:7860"
PASSED = 0
FAILED = 0
ERRORS: List[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    """Record a single test result."""
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  ✓ {name}")
    else:
        FAILED += 1
        msg = f"  ✗ {name}" + (f" — {detail}" if detail else "")
        print(msg)
        ERRORS.append(msg)


# ===================================================================
# 1. HTTP ENDPOINT TESTS
# ===================================================================

OBSERVATION_FIELDS = {
    "ticket_id", "subject", "body", "customer_tier",
    "previous_messages", "current_step", "max_steps", "task_name", "done",
}

STEP_RESULT_FIELDS = {"observation", "reward", "done", "info"}

EPISODE_STATE_FIELDS = {"task_name", "ticket_id", "step", "total_reward", "history"}

VALID_STEP_ACTIONS: Dict[str, Dict[str, Any]] = {
    "ticket-classify": {
        "action_type": "classify",
        "content": "billing issue",
        "metadata": {"category": "billing", "priority": "high"},
    },
    "ticket-respond": {
        "action_type": "respond",
        "content": (
            "Hello, thank you for reaching out. I understand your concern "
            "about the API rate limit. We will review your usage and increase "
            "your limit within 24 hours."
        ),
        "metadata": {},
    },
    "ticket-resolve": {
        "action_type": "ask_clarification",
        "content": "Could you please provide your account ID and the exact error message?",
        "metadata": {},
    },
}

# Invalid actions: use an action_type that is NOT valid for that task.
INVALID_STEP_ACTIONS: Dict[str, Dict[str, Any]] = {
    "ticket-classify": {
        "action_type": "respond",
        "content": "This should fail",
        "metadata": {},
    },
    "ticket-respond": {
        "action_type": "classify",
        "content": "This should fail",
        "metadata": {},
    },
    "ticket-resolve": {
        "action_type": "classify",
        "content": "This should fail",
        "metadata": {},
    },
}


def test_endpoints() -> None:
    """Test every HTTP endpoint for each task."""
    print("\n" + "=" * 60)
    print("1. HTTP ENDPOINT TESTS")
    print("=" * 60)

    c = httpx.Client(base_url=BASE, timeout=15.0)

    # --- Health ---
    r = c.get("/health")
    check("GET /health returns 200", r.status_code == 200)
    check("GET /health body has status=ok", r.json().get("status") == "ok")

    # --- Tasks ---
    r = c.get("/tasks")
    check("GET /tasks returns 200", r.status_code == 200)
    tasks = r.json()
    check("GET /tasks returns 3 tasks", len(tasks) == 3)

    # --- Per-task tests ---
    for task_name in ["ticket-classify", "ticket-respond", "ticket-resolve"]:
        print(f"\n  --- {task_name} ---")

        # Reset
        r = c.post("/reset", json={"task_name": task_name})
        check(f"POST /reset {task_name} returns 200", r.status_code == 200)
        obs = r.json()
        missing = OBSERVATION_FIELDS - set(obs.keys())
        check(
            f"  observation has all fields",
            len(missing) == 0,
            f"missing: {missing}",
        )
        check(
            f"  customer_tier is valid",
            obs.get("customer_tier") in ("bronze", "silver", "gold"),
        )
        check(f"  current_step == 0", obs.get("current_step") == 0)
        check(f"  done == False", obs.get("done") is False)
        check(f"  task_name matches", obs.get("task_name") == task_name)

        # Valid step
        action = VALID_STEP_ACTIONS[task_name]
        r = c.post("/step", json=action)
        check(f"POST /step valid returns 200", r.status_code == 200)
        sr = r.json()
        sr_missing = STEP_RESULT_FIELDS - set(sr.keys())
        check(
            f"  StepResult has all fields",
            len(sr_missing) == 0,
            f"missing: {sr_missing}",
        )
        reward = sr.get("reward", -1)
        check(f"  reward in [0.0, 1.0]", 0.0 <= reward <= 1.0, f"got {reward}")
        check(f"  done is bool", isinstance(sr.get("done"), bool))

        # State
        r = c.get("/state")
        check(f"GET /state returns 200", r.status_code == 200)
        state = r.json()
        state_missing = EPISODE_STATE_FIELDS - set(state.keys())
        check(
            f"  EpisodeState has all fields",
            len(state_missing) == 0,
            f"missing: {state_missing}",
        )

        # --- Invalid action test (fresh episode) ---
        c.post("/reset", json={"task_name": task_name})
        inv_action = INVALID_STEP_ACTIONS[task_name]
        r = c.post("/step", json=inv_action)
        check(f"POST /step invalid returns 200", r.status_code == 200)
        inv_sr = r.json()
        check(
            f"  invalid action reward == 0.0",
            inv_sr.get("reward") == 0.0,
            f"got {inv_sr.get('reward')}",
        )
        check(
            f"  invalid action done == False",
            inv_sr.get("done") is False,
            f"got {inv_sr.get('done')}",
        )
        check(
            f"  invalid action info has error",
            "error" in inv_sr.get("info", {}),
        )

    c.close()


# ===================================================================
# 2. DIRECT GRADER TESTS
# ===================================================================

# A realistic "perfect" ticket for testing graders.
_PERFECT_TICKET = {
    "ticket_id": "TKT-TEST",
    "subject": "I was charged twice for my subscription",
    "body": "Duplicate charge on my account.",
    "customer_tier": "gold",
    "correct_category": "billing",
    "correct_priority": "high",
    "ideal_response_keywords": ["refund", "duplicate charge", "billing", "48 hours", "apologies"],
    "resolution_path": ["ask_clarification", "respond", "escalate"],
    "difficulty": "easy",
}


def test_graders() -> None:
    """Test each grader with perfect, blank, and determinism checks."""
    print("\n" + "=" * 60)
    print("2. DIRECT GRADER TESTS")
    print("=" * 60)

    # ------ Classify grader ------
    print("\n  --- classify grader ---")

    perfect_classify = SupportAction(
        action_type="classify",
        content="This is a billing issue with high priority",
        metadata={"category": "billing", "priority": "high"},
    )
    score = grade_classify(perfect_classify, _PERFECT_TICKET)
    check(f"perfect classify score >= 0.8", score >= 0.8, f"got {score}")

    blank_classify = SupportAction(
        action_type="classify", content="", metadata={},
    )
    score_blank = grade_classify(blank_classify, _PERFECT_TICKET)
    check(f"blank classify score <= 0.1", score_blank <= 0.1, f"got {score_blank}")

    # Determinism
    scores = [grade_classify(perfect_classify, _PERFECT_TICKET) for _ in range(5)]
    check(
        f"classify deterministic (5 calls)",
        len(set(scores)) == 1,
        f"got {scores}",
    )

    # ------ Respond grader ------
    print("\n  --- respond grader ---")

    perfect_respond = SupportAction(
        action_type="respond",
        content=(
            "Dear valued customer, thank you for reaching out. I sincerely "
            "apologize for the inconvenience. I can see there was a duplicate "
            "charge on your account and I will process your refund immediately. "
            "We are looking into the billing issue and you should see the refund "
            "within 48 hours. Our apologies for any trouble this has caused."
        ),
        metadata={},
    )
    score = grade_respond(perfect_respond, _PERFECT_TICKET)
    check(f"perfect respond score >= 0.8", score >= 0.8, f"got {score}")

    blank_respond = SupportAction(
        action_type="respond", content="", metadata={},
    )
    score_blank = grade_respond(blank_respond, _PERFECT_TICKET)
    check(f"blank respond score <= 0.1", score_blank <= 0.1, f"got {score_blank}")

    # Determinism
    scores = [grade_respond(perfect_respond, _PERFECT_TICKET) for _ in range(5)]
    check(
        f"respond deterministic (5 calls)",
        len(set(scores)) == 1,
        f"got {scores}",
    )

    # ------ Resolve grader ------
    print("\n  --- resolve grader ---")

    # Simulate a perfect multi-turn resolve episode
    ticket = _PERFECT_TICKET
    path = ticket["resolution_path"]  # ["ask_clarification", "respond", "escalate"]
    history: List[Dict[str, object]] = []
    cumulative = 0.0

    for i, expected_action in enumerate(path):
        act = SupportAction(
            action_type=expected_action,
            content=f"Step {i+1} action",
            metadata={},
        )
        result = grade_resolve_step(act, ticket, i, history, max_steps=6)
        cumulative += result["reward"]
        history.append({"action_type": expected_action, "step": i + 1})

    final_score = grade_resolve_final(ticket, history, cumulative, max_steps=6)
    check(f"perfect resolve final >= 0.8", final_score >= 0.8, f"got {final_score}")

    # Blank / worst-case resolve: wrong action at step 0, then immediate stop
    blank_act = SupportAction(
        action_type="resolve", content="", metadata={},
    )
    blank_result = grade_resolve_step(blank_act, ticket, 0, [], max_steps=6)
    blank_final = grade_resolve_final(
        ticket, [{"action_type": "resolve", "step": 1}],
        blank_result["reward"], max_steps=6,
    )
    check(f"blank resolve score <= 0.5", blank_final <= 0.5, f"got {blank_final}")

    # Determinism for resolve final
    scores = []
    for _ in range(5):
        h: List[Dict[str, object]] = []
        c_reward = 0.0
        for i, ea in enumerate(path):
            a = SupportAction(action_type=ea, content=f"Step {i+1} action", metadata={})
            r = grade_resolve_step(a, ticket, i, h, max_steps=6)
            c_reward += r["reward"]
            h.append({"action_type": ea, "step": i + 1})
        scores.append(grade_resolve_final(ticket, h, c_reward, max_steps=6))
    check(
        f"resolve deterministic (5 calls)",
        len(set(scores)) == 1,
        f"got {scores}",
    )


# ===================================================================
# 3. INFERENCE LOG FORMAT VALIDATION
# ===================================================================

# Regex patterns for each log line type.
_RE_START = re.compile(
    r"^\[START\] task=(\S+) env=(\S+) model=(\S+)$"
)
_RE_STEP = re.compile(
    r"^\[STEP\] step=(\d+) action=(.+?) reward=(\d+\.\d{2}) "
    r"done=(true|false) error=(.+)$"
)
_RE_END = re.compile(
    r"^\[END\] success=(true|false) steps=(\d+) "
    r"score=(\d+\.\d{3}) rewards=(\d+\.\d{2})$"
)


def validate_log_lines(lines: List[str]) -> None:
    """Validate inference log output against the required format."""
    print("\n" + "=" * 60)
    print("3. INFERENCE LOG FORMAT VALIDATION")
    print("=" * 60)

    # Filter to only [START], [STEP], [END] lines
    log_lines = [l for l in lines if l.startswith("[START]") or l.startswith("[STEP]") or l.startswith("[END]")]

    if not log_lines:
        check("log has at least 1 line", False, "no log lines found")
        return

    check(f"found {len(log_lines)} log lines", len(log_lines) > 0)

    start_count = 0
    end_count = 0
    step_count = 0

    for line in log_lines:
        if line.startswith("[START]"):
            m = _RE_START.match(line)
            check(
                f"[START] format valid",
                m is not None,
                f"line: {line[:100]}",
            )
            if m:
                check("  [START] has task= field", bool(m.group(1)))
                check("  [START] has env= field", bool(m.group(2)))
                check("  [START] has model= field", bool(m.group(3)))
            start_count += 1

        elif line.startswith("[STEP]"):
            m = _RE_STEP.match(line)
            check(
                f"[STEP] format valid",
                m is not None,
                f"line: {line[:120]}",
            )
            if m:
                check(
                    "  [STEP] reward has 2 decimals",
                    "." in m.group(3) and len(m.group(3).split(".")[1]) == 2,
                )
                check(
                    "  [STEP] done is lowercase bool",
                    m.group(4) in ("true", "false"),
                )
            step_count += 1

        elif line.startswith("[END]"):
            m = _RE_END.match(line)
            check(
                f"[END] format valid",
                m is not None,
                f"line: {line[:120]}",
            )
            if m:
                check(
                    "  [END] success is lowercase bool",
                    m.group(1) in ("true", "false"),
                )
                check(
                    "  [END] score has 3 decimals",
                    "." in m.group(3) and len(m.group(3).split(".")[1]) == 3,
                )
                check(
                    "  [END] rewards has 2 decimals",
                    "." in m.group(4) and len(m.group(4).split(".")[1]) == 2,
                )
            end_count += 1

    check(f"[START] count == [END] count", start_count == end_count,
          f"starts={start_count} ends={end_count}")
    check(f"at least 1 [STEP] line", step_count >= 1)


def test_log_format() -> None:
    """Simulate inference output and validate the log format.

    Instead of calling the real LLM, we simulate the log lines that
    inference.py would produce by running an episode locally and
    formatting the output identically.
    """
    import random

    lines: List[str] = []

    tasks = ["ticket-classify", "ticket-respond", "ticket-resolve"]
    model_name = "test-model"

    actions_by_task: Dict[str, List[Dict[str, Any]]] = {
        "ticket-classify": [
            {"action_type": "classify", "content": "billing high",
             "metadata": {"category": "billing", "priority": "high"}}
        ],
        "ticket-respond": [
            {"action_type": "respond",
             "content": "Hello, thank you for contacting support. We will investigate your rate limit issue and get back to you within 24 hours.",
             "metadata": {}}
        ],
        "ticket-resolve": [
            {"action_type": "ask_clarification",
             "content": "Could you share your account ID?", "metadata": {}},
            {"action_type": "respond",
             "content": "Thank you, we are investigating.", "metadata": {}},
            {"action_type": "escalate",
             "content": "Escalating to senior team.", "metadata": {}},
        ],
    }

    for task in tasks:
        env = SupportOpsEnv(task)
        env.reset()

        lines.append(f"[START] task={task} env=support-ops-env model={model_name}")

        total_reward = 0.0
        steps = 0
        done = False

        for step_num, action_dict in enumerate(actions_by_task[task], 1):
            if done:
                break
            action = SupportAction(**action_dict)
            result = env.step(action)
            reward = result.reward
            done = result.done
            total_reward += reward
            steps = step_num

            action_summary = action_dict["content"][:80]
            lines.append(
                f"[STEP] step={step_num} action={action_summary} "
                f"reward={reward:.2f} done={'true' if done else 'false'} "
                f"error=None"
            )

        success = total_reward > 0
        lines.append(
            f"[END] success={'true' if success else 'false'} "
            f"steps={steps} score={total_reward:.3f} rewards={total_reward:.2f}"
        )

    validate_log_lines(lines)


# ===================================================================
# 4. MEMORY LEAK CHECK
# ===================================================================

def test_memory() -> None:
    """Run 50 sequential reset() calls and check memory stays bounded."""
    print("\n" + "=" * 60)
    print("4. MEMORY LEAK CHECK (50 resets)")
    print("=" * 60)

    tracemalloc.start()

    env = SupportOpsEnv("ticket-classify")
    env.reset()

    # Baseline
    _, baseline_peak = tracemalloc.get_traced_memory()

    for i in range(50):
        env = SupportOpsEnv("ticket-classify")
        obs = env.reset()
        action = SupportAction(
            action_type="classify",
            content=f"run {i}",
            metadata={"category": "billing", "priority": "high"},
        )
        env.step(action)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Allow up to 10 MB growth (generous — should be < 1 MB)
    growth_mb = (peak - baseline_peak) / (1024 * 1024)
    check(
        f"memory growth < 10 MB after 50 resets",
        growth_mb < 10.0,
        f"grew {growth_mb:.2f} MB (baseline={baseline_peak/(1024*1024):.2f} MB, peak={peak/(1024*1024):.2f} MB)",
    )
    print(f"  ℹ Memory: current={current/(1024*1024):.2f} MB, peak={peak/(1024*1024):.2f} MB, growth={growth_mb:.2f} MB")


# ===================================================================
# 5. PERFORMANCE CHECK (env-only, no LLM)
# ===================================================================

def test_performance() -> None:
    """Ensure 100 full episodes complete in under 5 seconds (no LLM)."""
    print("\n" + "=" * 60)
    print("5. PERFORMANCE CHECK (100 local episodes)")
    print("=" * 60)

    start = time.perf_counter()

    for _ in range(100):
        env = SupportOpsEnv("ticket-classify")
        env.reset()
        env.step(SupportAction(
            action_type="classify", content="perf test",
            metadata={"category": "billing", "priority": "high"},
        ))

    for _ in range(100):
        env = SupportOpsEnv("ticket-respond")
        env.reset()
        env.step(SupportAction(
            action_type="respond",
            content="Hello, thank you for your message. We are on it.",
            metadata={},
        ))

    for _ in range(100):
        env = SupportOpsEnv("ticket-resolve")
        env.reset()
        for act_type in ["ask_clarification", "respond", "escalate"]:
            env.step(SupportAction(
                action_type=act_type,
                content=f"Resolve step: {act_type}",
                metadata={},
            ))

    elapsed = time.perf_counter() - start
    check(
        f"300 episodes in < 5s",
        elapsed < 5.0,
        f"took {elapsed:.2f}s",
    )
    print(f"  ℹ Elapsed: {elapsed:.3f}s ({300/elapsed:.0f} episodes/sec)")


# ===================================================================
# MAIN
# ===================================================================

def main() -> None:
    """Run all test suites."""
    print("╔══════════════════════════════════════════════════════════╗")
    print("║          SupportOpsEnv — Stress Test Suite              ║")
    print("╚══════════════════════════════════════════════════════════╝")

    test_endpoints()
    test_graders()
    test_log_format()
    test_memory()
    test_performance()

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total = PASSED + FAILED
    print(f"  Passed: {PASSED}/{total}")
    print(f"  Failed: {FAILED}/{total}")

    if ERRORS:
        print("\n  Failed tests:")
        for e in ERRORS:
            print(f"    {e}")

    if FAILED == 0:
        print("\n  🎉 ALL TESTS PASSED!")
    else:
        print(f"\n  ⚠ {FAILED} test(s) failed — see above for details.")

    sys.exit(0 if FAILED == 0 else 1)


if __name__ == "__main__":
    main()
