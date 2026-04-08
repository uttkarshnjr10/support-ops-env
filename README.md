---
title: Support Ops Env
emoji: 🎫
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - customer-support
---

# 🎫 SupportOpsEnv

**An OpenEnv-compliant environment that simulates real-world customer support
ticket resolution for AI agent training and evaluation.**

Customer support is one of the highest-volume, most repetitive workflows in
every SaaS company — yet it demands nuanced judgement: reading intent, applying
policy, and communicating empathetically.  SupportOpsEnv distils this into three
progressively harder tasks so researchers can benchmark LLM agents on the full
lifecycle of a support interaction: *classify → respond → resolve*.

---

## Why This Environment?

| Pain Point | What SupportOpsEnv Tests |
|---|---|
| Mis-routed tickets waste agent time | **ticket-classify** — category + priority in one shot |
| Template replies frustrate customers | **ticket-respond** — draft graded on empathy, accuracy & keywords |
| Complex cases need multi-turn reasoning | **ticket-resolve** — 6-step episode with escalation paths |

All rewards are **partial and continuous** (never binary), so gradient-based and
RL methods get a useful training signal at every step.

---

## Observation Space

Each observation is a JSON object with these fields:

| Field | Type | Description |
|-------|------|-------------|
| `ticket_id` | `str` | Unique ticket identifier (e.g. `TKT-014`) |
| `subject` | `str` | Ticket subject line |
| `body` | `str` | Full ticket body text |
| `customer_tier` | `"bronze"\|"silver"\|"gold"` | Customer plan tier — affects scoring |
| `previous_messages` | `List[{role, content}]` | Conversation history (multi-turn tasks) |
| `current_step` | `int` | Steps taken so far in this episode |
| `max_steps` | `int` | Maximum steps before forced termination |
| `task_name` | `str` | Active task name |
| `done` | `bool` | Whether the episode has ended |

## Action Space

Actions are JSON objects with three fields:

| Field | Type | Description |
|-------|------|-------------|
| `action_type` | `str` | One of `classify`, `respond`, `ask_clarification`, `escalate`, `resolve` |
| `content` | `str` | The agent's message or reasoning |
| `metadata` | `dict` | Extra fields — e.g. `{"category": "billing", "priority": "high"}` for classify |

**Valid action types per task:**

| Task | Valid Actions |
|------|-------------|
| `ticket-classify` | `classify` |
| `ticket-respond` | `respond` |
| `ticket-resolve` | `ask_clarification`, `respond`, `escalate`, `resolve` |

---

## Tasks

### 1. `ticket-classify` (Easy · 1 step)

Classify a support ticket by **category** (`billing`, `technical`, `account`,
`feature_request`, `complaint`) and **priority** (`low`, `medium`, `high`,
`critical`).

**Scoring:** correct category → +0.5 · exact priority → +0.3 · off-by-one
priority → +0.15.

### 2. `ticket-respond` (Medium · up to 3 steps)

Draft a professional customer-facing response. Partially graded on:

- Content length (> 50 chars): +0.10
- Professional greeting: +0.10
- Acknowledging the ticket subject: +0.15
- Ideal keyword matches (max 2): +0.30 each
- Penalty for "I don't know" / "cannot help": −0.20
- **Gold-tier complaints** are scored 10 % stricter on keyword matching

### 3. `ticket-resolve` (Hard · up to 6 steps)

Multi-turn resolution.  Follow the ideal resolution path (`ask_clarification` →
`respond` → `escalate`/`resolve`).

- Correct path step: +0.20
- Final action match: +0.40
- Efficiency bonus: +0.02 per step saved
- Repeated `ask_clarification` back-to-back: −0.15

---

## Baseline Scores

| Task | Model | Score |
|------|-------|-------|
| `ticket-classify` | Qwen2.5-72B | ~0.65 |
| `ticket-respond` | Qwen2.5-72B | ~0.45 |
| `ticket-resolve` | Qwen2.5-72B | ~0.30 |

---

## Setup

### Local

```bash
# Install dependencies
pip install -r requirements.txt

# Start the environment server
python server.py          # → http://localhost:7860

# In another terminal — run inference
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-3.5-turbo"
export HF_TOKEN="your-token"       # or set OPENAI_API_KEY
python inference.py
```

### Docker

```bash
docker build -t support-ops-env .
docker run -p 7860:7860 support-ops-env
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/reset` | Start a new episode |
| `POST` | `/step` | Submit an action |
| `GET`  | `/state` | Current episode state |
| `GET`  | `/tasks` | List all available tasks |
| `GET`  | `/health` | Liveness probe |

### Example `curl` Commands

**Reset (start a classify episode):**
```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name": "ticket-classify"}'
```

**Step (submit a classify action):**
```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "action_type": "classify",
    "content": "Billing issue with high urgency",
    "metadata": {"category": "billing", "priority": "high"}
  }'
```

**State (check current episode):**
```bash
curl http://localhost:7860/state
```

**List tasks:**
```bash
curl http://localhost:7860/tasks
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_BASE_URL` | `https://api.openai.com/v1` | LLM API endpoint |
| `MODEL_NAME` | `gpt-3.5-turbo` | Model to use for inference |
| `HF_TOKEN` | — | API key (falls back to `OPENAI_API_KEY`) |
| `ENV_URL` | `http://localhost:7860` | Environment server URL |
| `MAX_STEPS` | `8` | Maximum steps per episode |
| `TEMPERATURE` | `0.3` | LLM sampling temperature |

---

## Project Structure

```
support-ops-env/
├── env/
│   ├── models.py          # Pydantic data models
│   ├── environment.py     # Core environment logic
│   ├── tasks/
│   │   ├── classify.py    # Classification grader
│   │   ├── respond.py     # Response quality grader
│   │   └── resolve.py     # Multi-turn resolution grader
│   └── data/
│       └── tickets.json   # 30 synthetic support tickets
├── server.py              # FastAPI server
├── inference.py           # LLM inference loop
├── openenv.yaml           # OpenEnv manifest
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## License

MIT
