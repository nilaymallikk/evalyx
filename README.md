# Evalyx

An AI evaluation and reliability platform for testing, observing, debugging, and regression-testing LLM applications and agents.

## Status: Phase 12 complete — reliability & failure analysis

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
- Regression detection & baseline comparison (`src/evalyx/evaluation/regression/`) — deterministic, LLM-free comparison of two completed runs (baseline vs current): pass/failure/error rates, per-guardrail failure rates, latency, case-level transitions, and typed threshold policy producing persisted, idempotent `RegressionComparison` artifacts
- REST API (`src/evalyx/api/`) — versioned FastAPI surface under `/api/v1` for applications, dataset/version management, test cases, asynchronous evaluation submission (Celery), run/case/guardrail inspection, and regression comparisons; centralized error mapping, pagination, typed request/response schemas, OpenAPI/Swagger UI
- Application-under-test integration (`src/evalyx/application/`) — evaluate external AI applications (not just raw model providers) over HTTP: typed `ApplicationTarget` protocol, secret-safe `HttpApplicationTarget` transport, a target registry, and a worker execution branch selected by the run's `application:<name>` model selector; the LLMProvider path is unchanged
- MLGPT reference demo (`examples/mlgpt_demo/`) — end-to-end demonstration driving MLGPT (the reference RAG chatbot) through the REST API: idempotent setup, baseline run, controlled reversible behavior change, second run, and a Phase 8 regression comparison
- Failure analysis (`src/evalyx/evaluation/failures.py`) — Phase 12 reliability layer: deterministic classification of *execution* failures (the application could not answer) into a small typed taxonomy, persisted on the case result and exposed through the API. Quality failures (the application answered, but the answer was bad) remain the domain of guardrails/scoring and are never conflated with infrastructure errors
- Minimal API with health checks (`src/evalyx/api/app.py`): `GET /health` (liveness), `GET /health/ready` (dependency readiness)
- OpenRouter connectivity test for the selected free models (`test_openrouter.py`, run with `uv run python test_openrouter.py`)
- Project scaffolding: `uv`-managed Python 3.14 environment, `src/` layout (`src/evalyx/`)

### Planned / future

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

### Regression detection & baseline comparison (Phase 8)

Phase 8 answers one question with evidence: *did the current version of the
AI application regress compared with the baseline?* The logic is **pure
Python + PostgreSQL** — it never calls an LLM, never touches OpenRouter,
never depends on Celery, and is fully deterministic: the same two runs with
the same policy always produce the same artifact byte-for-byte.

Concepts:

- **Baseline** — the reference `EvaluationRun` (older version, known-good
  state). **Current** — the run under evaluation. Both must be `completed`,
  belong to the **same application**, and reference versions of the **same
  dataset**. Runs are compared, never modified — history stays immutable.
- **Case matching** — same dataset version: cases are matched by
  `test_case_id`. Cross-version (e.g. v1 → v2 of one dataset): matched by
  test-case **name**, which must stay stable across versions (a documented
  requirement, since each version snapshots its own `TestCase` rows).
- **Metrics** — pass rate and failure rate over *evaluated* cases
  (passed + failed); execution-error rate (provider failures, status
  `error`), evaluation-error rate (cases whose evaluation could not
  complete), and a combined `error_rate` over *total* cases; average
  latency over non-null observations; per-guardrail failure rate over its
  passed+failed verdicts (guardrail errors/absence counted separately).
  Comparison metrics are computed over **matched cases only**, so new or
  removed cases never distort rates — they are reported separately.
- **Case-level detection** — a full transition matrix per matched case:
  `newly_failed`, `newly_errored`, `fixed`, `stable_failure`,
  `error_transition`, `recovered`, plus `new_cases`, `removed_cases`, and
  `missing_case_results` (dataset cases that produced no result — absence
  of evidence is surfaced explicitly, never invented into a pass or fail).

Threshold policy (typed, unit-explicit, frozen per artifact):

| Threshold | Unit | Default | Meaning |
|---|---|---|---|
| `max_pass_rate_drop_pp` | percentage points | `2.0` | pass rate fell by more than this |
| `max_error_rate_increase_pp` | percentage points | `2.0` | combined error rate rose by more than this |
| `max_guardrail_failure_rate_increase_pp` | percentage points | `2.0` | any guardrail's failure rate rose by more than this |
| `max_latency_increase_percent` | percent (relative) | `20.0` / disabled | average latency rose by more than this |

A violation requires the delta to be **strictly worse** than the threshold
(exactly equal → not a regression). Rates are percentages (0–100), deltas
in percentage points, latency delta in relative percent. The policy is
versioned (`comparison_version = "1"`) and fingerprinted (SHA-256 over the
canonical JSON of version + thresholds); the fingerprint is part of the
uniqueness key so re-running with a changed policy creates a new artifact
instead of overwriting history.

Decision: `REGRESSION_DETECTED` (≥1 violation), `NO_REGRESSION`, or
`NOT_COMPARABLE` (either side has no evaluated cases — no denominator).
Every artifact persists the threshold snapshot, metric deltas, case
findings, guardrail comparisons, configuration diff (secret-looking keys
sanitized), and the run context. Comparing a run with itself is rejected by
both the service and a database CHECK constraint; referenced runs cannot be
deleted while a comparison exists (FK RESTRICT).

Example — a regression detected through the service:

```python
from evalyx.db.session import DatabaseManager
from evalyx.evaluation.regression import RegressionService, RegressionThresholds

db = DatabaseManager(settings)
service = RegressionService(db.session_factory)

report = await service.compare_runs(baseline_run_id, current_run_id)  # defaults
if report.regression_detected:
    for v in report.threshold_violations:
        print(v.metric, v.delta, v.detail)      # e.g. pass_rate 2.777778 pp ...
    print("newly failed:", [f.name for f in report.newly_failed_cases])
    print("newly errored:", [f.name for f in report.newly_errored_cases])
print(report.result)  # REGRESSION_DETECTED | NO_REGRESSION | NOT_COMPARABLE
```

Re-running the same comparison returns the **same artifact** (same id, same
`created_at`) — idempotent by design; the persisted summary is recomputed
and verified identical. `list_for_run` finds every comparison a run
participated in, on either side.

Known limitations (deliberate): threshold-based only — a delta crossing a
threshold is *not* statistical significance testing (no sample-size model,
no noise floor); cross-version name matching trusts stable case names; the
combined error rate treats provider and evaluation errors as one budget;
JSONB summary storage means case findings are queried as JSON, not via a
dedicated relational table.

### REST API (Phase 9)

All resource endpoints live under the versioned prefix **`/api/v1`**; the
health endpoints stay outside the version. Swagger UI is served by FastAPI
itself (no custom documentation system):

```bash
docker compose up -d                  # PostgreSQL + Redis
uv run python main.py                 # API on http://127.0.0.1:8000
uv run celery -A evalyx.worker.celery_app worker --loglevel=INFO   # worker
# then open:
#   http://127.0.0.1:8000/docs          (Swagger UI)
#   http://127.0.0.1:8000/openapi.json  (OpenAPI schema)
```

Endpoint inventory:

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/v1/applications` | register an application (201; duplicate name → 409) |
| GET | `/api/v1/applications/{id}` | retrieve an application |
| GET | `/api/v1/applications/{id}/versions` | list immutable versions |
| POST | `/api/v1/applications/{id}/versions` | create a version (409 on duplicate label) |
| POST | `/api/v1/datasets` | create a dataset |
| GET | `/api/v1/datasets/{id}` | retrieve a dataset |
| GET | `/api/v1/datasets/{id}/versions` | list versions (ordered by version) |
| POST | `/api/v1/datasets/{id}/versions` | create an immutable version |
| POST | `/api/v1/datasets/{id}/versions/{v}/cases` | add a test case |
| GET | `/api/v1/datasets/{id}/versions/{v}/cases` | list test cases |
| POST | `/api/v1/evaluations` | submit a background evaluation (**202 Accepted**) |
| GET | `/api/v1/evaluations` | list run summaries (newest first) |
| GET | `/api/v1/evaluations/{run_id}` | run status, metadata, case counts |
| GET | `/api/v1/evaluations/{run_id}/results` | paginated case results + guardrails |
| GET | `/api/v1/evaluations/{run_id}/guardrails` | flat paginated guardrail results |
| GET | `/api/v1/evaluations/{run_id}/regressions` | comparisons involving the run |
| POST | `/api/v1/regressions` | compare two completed runs (see Phase 8) |
| GET | `/api/v1/regressions/{comparison_id}` | retrieve a persisted report |

Semantics and conventions:

- **Asynchronous evaluation**: `POST /api/v1/evaluations` persists the run
  (status `pending`), commits, then enqueues the existing
  `run_evaluation` Celery task with the run id, returning **202 Accepted**
  with `run_id`, `task_id`, and `status_url`. The worker owns execution;
  the API never calls the LLM. If enqueueing fails, the run is marked
  `failed` and **503** is returned — the API never claims a job was queued
  when it was not. Accepted consistency window: a crash after a successful
  enqueue but before the HTTP response leaves a queued job that still
  executes (an extra run, never a lost one).
- **Submission idempotency**: deliberately not idempotent — a retried HTTP
  request creates a new run. Treat any 2xx as success; a distributed
  idempotency-key mechanism is explicitly deferred (schema change).
- **PostgreSQL is authoritative**: run/case/guardrail state comes from
  PostgreSQL; Redis holds only operational Celery task state.
- **Errors**: one envelope, `{"error": {"code", "message"}}`; mapping —
  404 `not_found`, 409 `duplicate_version`/`conflict`, 400
  `invalid_comparison` (incompatible runs), 422 `validation_error`, 503
  `evaluation_enqueue_failed`, 500 `internal_error` (generic; details only
  in logs). A detected regression is a **200**, not an error.
- **Pagination**: `limit` (1–200, default 50) + `offset` on list endpoints
  with deterministic ordering and a stable `total`.
- **Configuration policy**: `configuration`/`configuration_snapshot`
  fields accept non-secret execution configuration only (temperature,
  max tokens, prompt template ids, ...). Secret-looking keys are stripped
  at the boundary *and* on read; provider credentials (`LLM_PROVIDER`,
  `OPENROUTER_API_KEY`) are server-side environment configuration and are
  never accepted from clients.
- **Dataset versions are immutable**: there is no update endpoint for
  version content — create a new version.

Example workflow (curl; no secrets involved):

```bash
# 1. Create an application and version
curl -s -X POST http://localhost:8000/api/v1/applications \
  -H 'Content-Type: application/json' \
  -d '{"name": "support-assistant", "description": "demo"}'
curl -s -X POST http://localhost:8000/api/v1/applications/<app_id>/versions \
  -H 'Content-Type: application/json' \
  -d '{"version": "v1", "configuration": {"prompt_template": "tpl-1"}}'

# 2. Create a dataset, version, and test cases
curl -s -X POST http://localhost:8000/api/v1/datasets \
  -H 'Content-Type: application/json' -d '{"name": "support-dataset"}'
curl -s -X POST http://localhost:8000/api/v1/datasets/<ds_id>/versions \
  -H 'Content-Type: application/json' -d '{"version": 1}'
curl -s -X POST http://localhost:8000/api/v1/datasets/<ds_id>/versions/1/cases \
  -H 'Content-Type: application/json' \
  -d '{"name": "greeting", "input": {"prompt": "Say hello in one sentence."}}'

# 3. Submit an evaluation (asynchronous; 202 Accepted)
curl -s -X POST http://localhost:8000/api/v1/evaluations \
  -H 'Content-Type: application/json' \
  -d '{
        "application_id": "<app_id>",
        "application_version_id": "<app_version_id>",
        "dataset_version_id": "<dsv_id>",
        "agent_model": "nvidia/nemotron-3-ultra-550b-a55b:free",
        "judge_model": "minimax/minimax-m3:free",
        "configuration_snapshot": {"temperature": 0.2, "max_tokens": 256}
      }'
# → {"run_id": "...", "status": "pending", "task_id": "...",
#    "status_url": "/api/v1/evaluations/..."}

# 4. Poll status, then inspect results and guardrails
curl -s http://localhost:8000/api/v1/evaluations/<run_id>
curl -s http://localhost:8000/api/v1/evaluations/<run_id>/results
curl -s http://localhost:8000/api/v1/evaluations/<run_id>/guardrails

# 5. Compare against a baseline run
curl -s -X POST http://localhost:8000/api/v1/regressions \
  -H 'Content-Type: application/json' \
  -d '{"baseline_run_id": "<baseline>", "current_run_id": "<current>"}'
# 200 with {"result": "regression_detected" | "no_regression" | "not_comparable", ...}
```

Security limitations (explicit, intentional for this phase):

- **Authentication/authorization is not implemented yet.** The API is for
  local development and portfolio demonstration; do not expose it to
  untrusted networks. JWT/OAuth/API-key auth and RBAC are future
  hardening.
- No rate limiting and no CORS (permissive origins are not enabled).
- Secrets are never accepted in request bodies; error responses never
  include stack traces, SQL, filesystem paths, or provider payloads;
  guardrail metadata contains categories/counts, never raw PII matches.

### Observability (Phase 10)

Evalyx is observable through **structured logs** and a **lightweight
in-process metrics registry** — no external telemetry stack (no Prometheus,
Grafana, OpenTelemetry, Sentry, Datadog, Azure SDKs) and no new runtime
dependencies. Everything is cross-cutting infrastructure: domain logic
contains no logging/formatting concerns beyond typed events.

**Request correlation (request IDs).** Every HTTP response carries an
`X-Request-ID` header. Send your own (letters/digits/`.`/`_`/`-`, max 128
chars) and it is reused; send nothing and a UUID4 is generated; send
something unsafe and it is replaced — the rejected value is never logged
(only `reason` + `length`). Example:

```
Request:   POST /api/v1/evaluations
Response:  X-Request-ID: test-observability-123   (client id reused)
Logs:      http_request_completed request_id=test-observability-123
```

**Correlation strategy.** `request_id`, `run_id`, and `task_id` live in
context variables (`structlog.contextvars`, never mutable globals) and are
merged into *every* structured log event automatically. The chain for an
evaluation is:

```
request_id (HTTP middleware)
    → run_id   (persisted EvaluationRun; the API logs evaluation_submitted
                with run_id + task_id)
    → task_id  (reused from Celery's request context; the worker binds it
                so all pipeline/runner/guardrail logs carry it)
```

IDs are **log correlation fields, never metric labels** — the metrics
registry *rejects* label keys like `request_id`/`run_id`/`task_id` because
identifiers would create unbounded label cardinality.

**Structured logging.** Phase 2's setup is extended, not replaced:
human-readable console output in development, JSON lines in production.
Key events:

```json
{"event": "http_request_completed", "request_id": "...", "method": "POST",
 "route": "/api/v1/evaluations", "status_code": 202, "duration_ms": 18}
```

```json
{"event": "evaluation_run_completed", "run_id": "...", "task_id": "...",
 "executed_cases": 2, "duration_ms": 842}
```

Other events: `http_request_started` (debug), `http_request_slow` (warning,
when a request exceeds `SLOW_REQUEST_THRESHOLD_MS`, default 1000),
`request_id_rejected`, `evaluation_submitted` (request→run→task link),
`evaluation_run_started/completed/failed/cancelled` (with
`application_id`, `dataset_version_id`, durations), `task_received/
retrying/completed/failed`, `provider_retry_scheduled` (provider, model,
attempt, error type — never response bodies), `regression_comparison_
started/completed` (ids + bounded result + duration), and
`readiness_check_failed` (dependency name only).

**Safe logging.** Observability code explicitly selects safe fields.
Never logged: request bodies, `Authorization`/cookies/headers, query
strings, prompts, model outputs, raw PII, provider response bodies,
connection strings, or secrets. 422 responses deliberately exclude
offending input values; 500 responses are generic (tracebacks go to logs
only); worker failure events log the exception *type*, never its message.

**Metrics.** `src/evalyx/core/metrics.py` provides a thread-safe,
async-safe, in-process registry (counters + timing observations) with a
deterministic `snapshot()` for tests/debugging and a test-only `reset()`.
Recorded metrics (all labels bounded):

| Metric | Labels |
|---|---|
| `http_requests_total` | method, route template, status |
| `http_request_duration_ms` | method, route template |
| `evaluation_runs_total` | status (completed/failed/cancelled) |
| `evaluation_run_duration_ms` | status |
| `evaluation_cases_total` | status (executed/passed/failed/error) |
| `evaluation_case_errors_total` | status (error) |
| `guardrail_evaluations_total` | guardrail name (configured set), status |
| `worker_tasks_total` | task name, outcome (success/failed/retry) |
| `worker_task_failures_total` | task name |
| `regression_comparisons_total` | result (bounded enum) |

Route labels are **templates** (`/api/v1/evaluations/{run_id}`), never
concrete paths; unmatched requests are bucketed as `/unmatched`. There is
no public `/metrics` endpoint by design — a future Prometheus/OpenTelemetry
phase can export the same registry without touching instrumented code.

**Health observability.** `/health` and `/health/ready` behavior is
unchanged; a failing readiness dependency additionally logs
`readiness_check_failed {dependency}` (name only, no connection strings).

**Development vs production.** Development stays human-readable console
output; production emits JSON lines on stdout. Because everything is
structured, correlation-aware, and secret-free, logs and metrics can later
be shipped to **Azure Monitor / Application Insights** (or any other
backend) with a log forwarder — no domain-code changes and no vendor SDKs
required.

**Configuration.** One new setting: `SLOW_REQUEST_THRESHOLD_MS` (default
`1000`). Metrics are always enabled in-process (no setting needed); they
cost only in-memory dict updates behind a lock.

## Evalyx × MLGPT reference demo

MLGPT — a production RAG chatbot (FastAPI + Streamlit, Pinecone/Jina retrieval,
Redis memory, OpenRouter generation) living in a **separate repository** — is
Evalyx's official reference application-under-test. The demo (`examples/mlgpt_demo/`)
shows the full reliability workflow a recruiter can follow in ~20 minutes.

```
┌─────────────────────────────┐         ┌──────────────────────────────────────┐
│  MLGPT (system under test)  │         │  Evalyx (evaluation platform)        │
│                             │         │                                      │
│  Streamlit UI   :8501       │         │  REST API        :8000               │
│  FastAPI RAG API:8001 ◄─────┼─────────┼── Celery worker (evaluation target)  │
│   · retrieval + rerank      │  HTTP   │   · guardrails + scoring             │
│   · Redis conversation memory│        │   · PostgreSQL (state of record)     │
│   · OpenRouter (free model) │         │   · regression engine (LLM-free)     │
└─────────────────────────────┘         └──────────────────────────────────────┘
```

**Separation of concerns.** MLGPT is the *system under test*: Evalyx never imports
MLGPT code, never touches its database, and never pretends it is a model provider.
Evalyx is the *platform*: it drives MLGPT's HTTP API (`POST /v1/chat`) through the
typed `ApplicationTarget` adapter, judges the answers with its own LLM judges, and
produces evidence. A run selects the demo target with `agent_model="application:mlgpt"`;
a normal model run (`agent_model="<model>"`) uses the unchanged `LLMProvider` path.

**The demo dataset** (8 cases, `examples/mlgpt_demo/dataset.py`) covers ordinary
grounded questions, instruction-following constraints, prompt-injection attempts,
fake-PII echo probes (`john.doe@example.test` — clearly fake), a safety-sensitive
request, an out-of-domain hallucination trap, and a degenerate edge case.

**The controlled regression** changes *MLGPT's behavior*, never Evalyx's results:
the demo temporarily replaces MLGPT's `prompts/rag_prompt.txt` with a deliberately
weakened variant (MLGPT reads that file from disk on every request). The original
prompt is restored automatically (`try/finally` plus `--restore`); no MLGPT source
code is modified.

```bash
# Prerequisites: PostgreSQL + Redis (docker compose up -d), Evalyx API (:8000),
# Evalyx Celery worker, and MLGPT running (uv run uvicorn src.api.main:app --port 8001)

uv run python examples/mlgpt_demo/run_demo.py             # full demo
uv run python examples/mlgpt_demo/run_demo.py --resume    # reuse the completed baseline run
                                                          # from the state file and only execute
                                                          # the degraded evaluation (quota-friendly)
uv run python examples/mlgpt_demo/run_demo.py --restore   # restore MLGPT prompt only
```

The script: registers the MLGPT application + two versions (idempotent), creates the
dataset (idempotent, stable case names), submits the baseline run, polls to completion,
applies the temporary prompt degradation, runs the second evaluation, restores the
prompt, then compares the runs and prints a summary — case names, guardrail findings,
and threshold violations only (never prompts, outputs, PII, or keys).

## Reliability & failure analysis (Phase 12)

The Phase 11 demo hit real OpenRouter instability (upstream 502s, then daily
free-tier quota exhaustion). That exposed a gap: Evalyx could say a case was
`error`, but not *why*. Phase 12 answers three questions:

1. **Did the application produce a bad answer?** — a *quality failure*
   (guardrails failed it). Unchanged; lives in guardrail/scoring results.
2. **Did the application fail to produce an answer?** — an *execution
   failure* (case status `error`).
3. **For execution failures: why?** — deterministic classification into a
   small typed taxonomy.

### Failure taxonomy

`src/evalyx/evaluation/failures.py` classifies from **exception types and
HTTP status codes only** — never an LLM, never response-body sniffing.
Ambiguous evidence becomes `unknown` (no invented root causes):

| Category | Meaning | Retryable |
|---|---|---|
| `provider_unavailable` | Provider 5xx (502/503/504...) | yes |
| `rate_limited` | HTTP 429, quota not exhausted | yes |
| `quota_exhausted` | Daily free-model quota gone (429) | **no** |
| `timeout` | No response within the timeout | yes |
| `connection_error` | Could not reach the application/provider | yes |
| `malformed_response` | Provider answered, payload unusable | no |
| `application_http_error` | Application under test returned 5xx | no* |
| `application_response_invalid` | Application rejected the request (4xx) | no |
| `authentication` | Credentials rejected | no |
| `unknown` | Insufficient evidence | no |

\* application-boundary 5xx is not retried by the evaluation layer because the
HTTP transport already exhausted its own bounded retries before surfacing the
error (`attempts` records how many). For MLGPT, Evalyx records only what is
observable at the application boundary — an MLGPT 500 caused by *its* upstream
provider is still `application_http_error`; Evalyx never invents internal
MLGPT knowledge.

### Where failure information appears

- **Persistence** — each errored case's `metrics["failure"]` holds
  `{category, reason, retryable, http_status, attempts}` (JSONB; **zero
  migrations**). Reasons are classifier-authored templates, never exception
  bodies — nothing secret or prompt-like can leak.
- **API** — `GET /api/v1/evaluations/{run_id}/results` now includes an
  optional, backwards-compatible `failure` object per case (`null` for
  passed/failed cases and pre-Phase 12 rows).
- **Reliability summary** — `GET /api/v1/evaluations/{run_id}/reliability`
  returns the run's execution-failure breakdown by category:
  `{"total_cases": 8, "errored_cases": 6, "classified_failures": 5,
  "unclassified_execution_failures": 1, "retryable_failures": 4,
  "failure_breakdown": {"provider_unavailable": 4, "rate_limited": 1}}`.
- **Regression reports** — `newly_errored_cases` findings carry
  `current_failure_category`, so an error-rate spike is immediately
  triageable (a `provider_unavailable` spike is an outage, not an app bug).
  Regression *metrics and thresholds are unchanged* — an outage can never
  become a hallucination or instruction-following failure.

### Retry behavior

Application-target invocations retry only clearly transient failures
(connection errors, timeouts, 502/503/504) with bounded exponential backoff
(3 attempts, 1s/4s) — the same conservative philosophy as the Phase 4
provider layer, compatible with Phase 7's sequential, rate-limit-conscious
execution. A retried case still produces exactly **one** case result (the
final outcome plus failure metadata). 4xx, malformed responses, and quota
exhaustion are never retried.

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
