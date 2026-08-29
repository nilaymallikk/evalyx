# Evalyx

An AI evaluation and reliability platform for testing, observing, debugging, and regression-testing LLM applications and agents.

## Status: Phase 7 complete — background evaluation workers (Celery + Redis)

### Currently implemented

- Dockerized local infrastructure: PostgreSQL 16 (host port `5433` → container `5432`) and Redis 7 (`6379`) via `docker-compose.yml`
- Typed configuration layer (`src/evalyx/core/config.py`) — pydantic-settings, secrets as `SecretStr`, environment-driven
- Structured logging foundation (`src/evalyx/core/logging.py`) — structlog, level-configured
- PostgreSQL/Redis connection foundations (`src/evalyx/db/`) — async SQLAlchemy engine + Redis client, connectivity checks
- Domain model + Alembic migrations (`src/evalyx/db/models/`, `migrations/`): applications, application versions, datasets, dataset versions, test cases, evaluation runs, case results, guardrail results
- Repository layer (`src/evalyx/db/repositories/`) — async data access for the domain
- LLM provider abstraction (`src/evalyx/llm/`) — provider-neutral `LLMProvider` protocol + typed `LLMResponse`, async OpenRouter and Ollama implementations, bounded retries, typed error hierarchy, provider factory
- Evaluation engine (`src/evalyx/evaluation/`) — executes a run's pinned dataset version through the injected `LLMProvider`, persists per-case results (status `executed`/`error`), returns a typed run summary
- Guardrails (`src/evalyx/guardrails/`) — provider-neutral `Guardrail` abstraction; deterministic PII and prompt-injection indicators; LLM-judge evaluation of instruction following, hallucination/unsupported claims, and safety; a harness with per-guardrail failure isolation and idempotent persistence
- Scoring (`src/evalyx/evaluation/scoring.py`, `src/evalyx/evaluation/pipeline.py`) — combines guardrail verdicts into per-case outcomes (`executed → passed/failed`), keeps execution errors distinct, and produces accurate run summary counts; end-to-end orchestration via `EvaluationPipeline`
- Background workers (`src/evalyx/worker/`) — Celery + Redis(transport) orchestration around the existing pipeline: `run_evaluation` task receives a `run_id`, executes the async pipeline through a controlled event-loop bridge, and keeps PostgreSQL authoritative for all evaluation state
- Minimal API with health checks (`src/evalyx/api/app.py`): `GET /health` (liveness), `GET /health/ready` (dependency readiness)
- OpenRouter connectivity test for the selected free models (`test_openrouter.py`, run with `uv run python test_openrouter.py`)
- Project scaffolding: `uv`-managed Python 3.14 environment, `src/` layout (`src/evalyx/`)

### Planned / future

- Evaluation API (FastAPI) for applications, datasets, policies, and runs
- Regression detection, failure debugging
- Observability

See `project_context.md` for the full product context and phased implementation plan.

## Local development

```bash
docker compose up -d          # PostgreSQL (localhost:5433) + Redis (localhost:6379)
uv sync                       # install dependencies
cp .env.example .env          # then fill in OPENROUTER_API_KEY and EVALYX_SECRET_KEY
uv run python main.py         # start the API (http://127.0.0.1:8000)
uv run python test_openrouter.py
```

Health endpoints: `GET /health` (liveness), `GET /health/ready` (PostgreSQL + Redis readiness).

### Tests

```bash
uv run pytest                                  # unit tests (no network, no live services)
EVALYX_RUN_INTEGRATION_TESTS=1 uv run pytest   # + live PostgreSQL/Redis integration tests
```

### LLM providers

Evalyx calls models through a provider-agnostic interface
(`src/evalyx/llm/base.py`), selected via `LLM_PROVIDER` in `.env`:

| Provider | Status | Notes |
|---|---|---|
| `openrouter` | Implemented (`src/evalyx/llm/openrouter.py`) | Default; async `httpx`, bounded retries, typed errors |
| `ollama` | Implemented (`src/evalyx/llm/ollama.py`) | Optional/local; set `LLM_PROVIDER=ollama` and `OLLAMA_BASE_URL` |
| `azure_openai` | **Future** — not implemented | The interface is designed so it can be added without touching callers |

Defaults are the project's **free** OpenRouter models (`EVALYX_AGENT_MODEL`,
`EVALYX_JUDGE_MODEL` in `.env.example`); no paid-model fallback exists.

Provider unit tests are fully mocked (`uv run pytest` — no network). An
optional live OpenRouter test is double-gated:

```bash
EVALYX_RUN_INTEGRATION_TESTS=1 EVALYX_RUN_LLM_INTEGRATION_TESTS=1 uv run pytest
```

For local Ollama: install Ollama, pull a model, then point
`OLLAMA_BASE_URL=http://localhost:11434` and set `LLM_PROVIDER=ollama`.

### Evaluation engine

`EvaluationRunner` (in `src/evalyx/evaluation/`) executes one run against
the provider it is given (never a concrete provider directly). Flow:
pinned `DatasetVersion` → load `TestCase`s → `provider.complete(...)` using
the run's recorded `agent_model` and `configuration_snapshot`
(`temperature`, `max_tokens`, `system`) → persist one
`EvaluationCaseResult` per case → complete the run.

Semantics (Phase 5 — execution only):

- Result statuses are execution-honest: a case that produced a provider
  response is `executed`; provider failures are `error` with a safe
  provider-error category. `passed`/`failed` arrive with Phase 6 scoring.
- Individual case failures never stop the run; even if every case errors,
  the run `completes`. `failed` is reserved for catastrophic runner-level
  failures (e.g. persistence errors). Empty dataset versions complete with
  zero results.
- Each test case is executed exactly once per run (unique
  `(run, test_case)` constraint); re-executing a completed run raises.
- Execution is sequential by design (free models are rate-limited).

Programmatic example:

```python
from evalyx.core.config import get_settings
from evalyx.db.session import DatabaseManager
from evalyx.evaluation import EvaluationRunner
from evalyx.llm.factory import create_provider

settings = get_settings()
db = DatabaseManager(settings)
provider = create_provider(settings)   # OpenRouter (default)

runner = EvaluationRunner(provider, db.session_factory)
summary = await runner.run(
    application_id=...,
    dataset_version_id=...,
    agent_model=settings.evalyx_agent_model,
    judge_model=settings.evalyx_judge_model,
    configuration_snapshot={"temperature": 0.2, "max_tokens": 500},
)
```

Judge scoring and guardrails are implemented (Phase 6) and composed by
`EvaluationPipeline`; regression detection is a future phase.

### Guardrails & scoring (Phase 6)

Guardrails live in `src/evalyx/guardrails/` and depend only on the
provider-neutral `LLMProvider` — never on OpenRouter/Ollama directly. Each
guardrail returns a structured `GuardrailVerdict` (`name`, `type`, `passed`,
`score`, `reason`, `metadata`) which the harness persists as a
`GuardrailResult` row.

Two kinds of checks:

- **Deterministic indicators** (no LLM calls, regex-based):
  - `pii` — detects email / phone / SSN-like patterns in the model output.
    This is a *conservative indicator*, not a complete enterprise PII
    detector. Metadata records only categories and counts — matched values
    are never persisted.
  - `prompt_injection` — detects obvious instruction-override, system-prompt
    extraction, role-switch, and jailbreak patterns with normalized
    (case-insensitive, whitespace-collapsed) matching. An *indicator*, not
    proof of injection.
- **LLM-judge evaluations** (semantic, use the run's `judge_model`):
  - `instruction_following` — does the output follow the request's explicit
    instructions and format?
  - `hallucination` — are claims supported by the case's expected
    output/reference material? (LLM-judge heuristic, not perfect detection;
    skipped as an evaluation error when no reference material exists.)
  - `safety` — narrow, explainable policy (hate/harassment, violence/self-harm,
    explicit sexual content, illegal instructions, security bypass).

Judges receive the evaluated model output as clearly delimited **untrusted
data** (`<model_output>` tags) so a malicious output cannot rewrite the
evaluation instructions, and must return strict JSON
(`{"passed": bool, "score": 0.0–1.0, "reason": str}`). Malformed judge
output, out-of-range scores, and provider failures become **evaluation
errors** — they can never silently become a pass.

Execution order and isolation:

1. Deterministic checks run first (cheap, no LLM calls); judge checks after.
   All configured guardrails run — no short-circuiting — so results keep
   complete diagnostic information.
2. One guardrail failing or erroring never prevents the others from running.
3. Persistence is idempotent: a `(case_result, guardrail name)` is recorded
   once (DB-enforced unique constraint); repeated scoring never duplicates
   rows or re-invokes judges.

Scoring policy (documented in `src/evalyx/guardrails/policy.py`):

- **Critical guardrails** — `pii`, `safety`, `hallucination`. Any critical
  failure transitions the case `executed → failed`.
- **Non-critical guardrails** — `prompt_injection`, `instruction_following`.
  Their failures are persisted as guardrail failures (visible for debugging)
  but do not individually fail the case. The policy is config-driven;
  stricter policies can promote them.
- A guardrail **execution error** (judge timeout, invalid judge output) is
  *not* a model failure: the case stays `executed` and is surfaced as
  `evaluation_error_cases` in the summary. Execution failures (no model
  output) stay `error`.

Run summary counts are accurate and separated — e.g. 10 cases might yield
8 `passed`, 1 `failed`, 1 `error` (execution) — and execution errors are
never counted as semantic failures.

End-to-end example:

```python
from evalyx.evaluation import EvaluationPipeline

pipeline = EvaluationPipeline(provider=provider, session_factory=db.session_factory)
summary = await pipeline.run_and_score(
    application_id=...,
    dataset_version_id=...,
    agent_model=settings.evyx_agent_model,
    judge_model=settings.evyx_judge_model,
)
print(summary.passed_cases, summary.failed_cases, summary.error_cases)
```

Known limitations (honest): regex PII/injection checks are indicators with
known blind spots; judge verdicts are heuristic and model-dependent; guardrail
execution is sequential (deliberate — free models are rate-limited).

### Background workers (Phase 7)

Long evaluations run as Celery jobs so callers never block on them:

```text
Client / future API
       │ create EvaluationRun (pending)
       ▼
run_evaluation.delay(str(run_id))   ── only a run_id crosses the queue
       ▼
Redis (broker + result backend)     ── job delivery, operational state
       ▼
Celery worker                       ── thin orchestration only
       ▼
EvaluationPipeline (existing Phase 5/6 business logic, stays async)
       ▼
PostgreSQL                          ── authoritative run/case/guardrail state
```

Architecture rules the worker obeys:

- **Thin tasks.** The task receives a `run_id` string, creates its own
  database manager and provider per execution, calls
  `EvaluationPipeline.execute_and_score_existing_run(run_id)`, and returns a
  small JSON summary (`{"run_id", "status", "action", "passed_cases", ...}`).
  No evaluation logic is duplicated in the worker, and no secrets, ORM
  objects, or providers are ever serialized into task arguments.
- **Async stays async.** `EvaluationRunner`/`EvaluationPipeline`/`LLMProvider`
  remain async; the task bridges into asyncio with exactly one controlled
  event loop per invocation (never a loop per LLM call).
- **PostgreSQL is the source of truth.** Redis/Celery handle job delivery,
  retries, and operational task state (`PENDING/STARTED/RETRY/SUCCESS/FAILURE`).
  Run state (`pending/running/completed/failed/cancelled`), case results, and
  guardrail results live only in PostgreSQL — losing Redis loses queued jobs,
  never evaluation history.
- **Provider lifecycle.** The worker constructs the provider through the
  existing factory (never provider-specific HTTP logic) and always closes it —
  on success, failure, retry, and cancellation.

Idempotency, retries, and failure semantics:

- **Duplicate delivery** (Celery is at-least-once): a `completed` run is only
  re-scored (guardrails/scoring are idempotent; nothing re-executes); a
  `failed`/`cancelled` run is skipped; a `pending`/`running` run resumes via
  Phase 5 semantics — cases with existing results are skipped and the unique
  `(run, test_case)` constraint prevents duplicates. No distributed locking.
- **Two retry layers, no overlap.** Provider HTTP retries (429/5xx/timeouts)
  are Phase 4's job, and per-case provider errors already stay case-level —
  they never escalate to a task retry. Job-level retries cover only
  infrastructure failures (PostgreSQL/network), bounded at
  `WORKER_MAX_RETRIES` (default 3) with exponential backoff capped at
  `WORKER_RETRY_MAX_BACKOFF_SECONDS`. Permanent failures (bad run id, invalid
  state, configuration errors) mark the run `failed` and do not retry.
- **Time limits.** Soft/hard limits (`WORKER_SOFT/HARD_TIME_LIMIT_SECONDS`,
  default 3600/3900) are explicit and generous; the soft limit triggers
  cleanup, and Redis's visibility timeout is kept above the hard limit so an
  executing job is never redelivered early.
- **Reliability settings.** `task_acks_late` + `task_reject_on_worker_lost`
  (a worker crash redelivers the job instead of dropping it) with
  `worker_prefetch_multiplier=1` (fair dispatch). Worker interruption
  (shutdown, crash) is never conflated with user cancellation — the `cancelled`
  state stays reserved for application-level cancellation.

Free-model considerations: worker concurrency defaults to
`WORKER_CONCURRENCY=2` (multiple *runs* in parallel), case execution inside a
run stays sequential, and large datasets on free models legitimately take
time — the time limits exist for that reason.

Local development workflow:

```bash
# Terminal 1 — PostgreSQL + Redis
docker compose up -d

# Terminal 2 — Celery worker
uv run celery -A evalyx.worker.celery_app worker --loglevel=INFO

# Terminal 3 — submit an evaluation (programmatic submission example)
uv run python examples/submit_background_evaluation.py
```

The broker and result backend are derived from the existing `REDIS_URL`; no
separate Celery URLs are configured, and no additional Redis container is
needed.

### Database

Schema is managed with Alembic against the Docker PostgreSQL (`localhost:5433`):

```bash
uv run alembic upgrade head                              # apply all migrations
uv run alembic downgrade -1                              # revert last migration
uv run alembic revision --autogenerate -m "change"       # create a new migration
```

Domain overview: an `Application` has immutable `ApplicationVersion`s; a `Dataset`
has immutable integer `DatasetVersion`s (v1, v2, ...) containing `TestCase`s; an
`EvaluationRun` references exactly one dataset version (plus application
version) and snapshots the agent/judge models and full execution
configuration as JSONB at run time; each case executed produces an
`EvaluationCaseResult` (with input/expected snapshots, latency, metrics) and
multiple first-class `GuardrailResult`s. Migration commands are above;
models live in `src/evalyx/db/models/`, repositories in
`src/evalyx/db/repositories/`.
