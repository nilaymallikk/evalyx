# Evalyx

An AI evaluation and reliability platform for testing, observing, debugging, and regression-testing LLM applications and agents.

## Status: pre-implementation (Phase 1 — architecture/audit)

### Currently implemented

- Dockerized local infrastructure: PostgreSQL 16 (host port `5433` → container `5432`) and Redis 7 (`6379`) via `docker-compose.yml`
- Typed configuration layer (`src/evalyx/core/config.py`) — pydantic-settings, secrets as `SecretStr`, environment-driven
- Structured logging foundation (`src/evalyx/core/logging.py`) — structlog, level-configured
- PostgreSQL/Redis connection foundations (`src/evalyx/db/`) — async SQLAlchemy engine + Redis client, connectivity checks
- Domain model + Alembic migrations (`src/evalyx/db/models/`, `migrations/`): applications, application versions, datasets, dataset versions, test cases, evaluation runs, case results, guardrail results
- Repository layer (`src/evalyx/db/repositories/`) — async data access for the domain
- Minimal API with health checks (`src/evalyx/api/app.py`): `GET /health` (liveness), `GET /health/ready` (dependency readiness)
- OpenRouter connectivity test for the selected free models (`test_openrouter.py`, run with `uv run python test_openrouter.py`)
- Project scaffolding: `uv`-managed Python 3.14 environment, `src/` layout (`src/evalyx/`)

### Planned / future

- Evaluation API (FastAPI), versioned datasets, evaluation runs
- Deterministic guardrails and LLM-as-a-judge scoring
- Regression detection, failure debugging, structured results
- Background evaluation workers, observability

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
