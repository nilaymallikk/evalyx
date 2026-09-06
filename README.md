# Evalyx

**Terminal-first platform for evaluating LLM applications and agents.**
Register your app, run versioned test datasets against it over HTTP, grade
answers with guardrails, and detect regressions between runs.

Version `0.9.0b1` (Public Beta)

## Run it

Prerequisites: Docker, [`uv`](https://docs.astral.sh/uv), Python 3.14.

```bash
git clone https://github.com/nilaymallikk/evalyx
cd evalyx

docker compose up -d          # PostgreSQL (:5433) + Redis (:6379)
uv sync
cp .env.example .env          # fill in OPENROUTER_API_KEY + EVALYX_SECRET_KEY

uv run python main.py         # API on http://127.0.0.1:8000 (terminal 1)
uv run celery -A evalyx.worker.celery_app worker --loglevel=INFO  # worker (terminal 2)
```

Dev login (no Clerk needed locally):

```bash
evalyx login --org org_quickstart
```

Production is a Compose stack (API + worker + Postgres + Redis + nginx):
see [docs/deployment.md](docs/deployment.md).

## Connect an external app

Evalyx never imports your code — it calls your app's HTTP endpoint.

> **Reading the commands below.** Anything in `<angle brackets>` is a
> placeholder you replace with a real value:
>
> - `<app_id>` — the id printed when you create the app, e.g.
>   `evalyx app create support-assistant` prints something like
>   `Application created: support-assistant (48a5bcc2-…)`. Copy that long
>   id (or its first 8 characters) wherever you see `<app_id>`.
> - `https://your-app.example.com/v1/chat` — the real public URL of **your**
>   AI app's chat endpoint, e.g. `https://api.mycompany.com/v1/chat`.
> - `--auth bearer` — how your endpoint authenticates: `none` (open),
>   `bearer` (needs a token, which you store separately via
>   `evalyx app secret <app_id>`), or `api_key`.

```bash
# 1. Register the app (Evalyx prints its <app_id>)
evalyx app create support-assistant --type http

# 2. Tell Evalyx how to call it (saved permanently on version "v1")
evalyx app version <app_id> v1 \
  --endpoint https://your-app.example.com/v1/chat \
  --auth bearer --input-field question --response-path answer

# 3. Check the connection
evalyx app test <app_id>
```

What happens on each call: Evalyx sends `POST {"question": "<test input>"}`
to your endpoint and reads the answer from the field you named in
`--response-path` (here: `{"answer": "…"}`). Secrets are encrypted at rest
and never shown back. Only public endpoints are accepted (SSRF-protected),
so `localhost` URLs are rejected — use a reachable URL or tunnel.

## Evaluate it

> **More placeholders, same idea:**
>
> - `<ds_id>` — the dataset id printed by `evalyx dataset create`, e.g.
>   `Dataset created: support-dataset (94927eac-…)`.
> - `<dsv_id>` — the *dataset version* id. Datasets are versioned, so a run
>   points at a version, not the dataset itself. Create one with:
>   `curl -s -X POST http://127.0.0.1:8000/api/v1/datasets/<ds_id>/versions -H 'Content-Type: application/json' -d '{"version": 1}'`
>   and copy the `"id"` from the response.
> - `<run_id>` — the run id printed by `eval run`, e.g. `Run: ee648d90-…`.
> - `application:support-assistant` — the literal word `application:` plus
>   the **name** you gave your app (not its id). This tells Evalyx to call
>   your app instead of a raw model.

```bash
# 1. Dataset + test cases
evalyx dataset create support-dataset
evalyx dataset add-case <ds_id> 1 --name greeting \
  --input '{"prompt":"Say hello in one sentence."}' \
  --expected '{"answer":"Hello!"}'

# 2. Submit a run (async) and wait
evalyx eval run --application <app_id> --dataset-version <dsv_id> \
  --agent-model application:support-assistant --wait

# 3. Inspect results
evalyx eval results <run_id>       # per-case outcomes
evalyx eval guardrails <run_id>    # PII, injection, safety, hallucination, instruction-following
evalyx eval reliability <run_id>   # why cases failed to execute (timeout, rate limit, …)

# 4. Compare with a baseline
evalyx regression run --baseline <old_run_id> --current <run_id>
```

Every command supports `--json` for CI. Exit `0` from
`eval run --wait --json` means no quality failures. Full command list:
`evalyx --help`. Raw HTTP API: Swagger at `http://127.0.0.1:8000/docs`.

## Architecture

```text
evalyx CLI/TUI ──REST──► FastAPI ──202──► Redis ⇄ Celery worker
                                                      │
                              ┌───────────────────────┼───────────────────────┐
                              ▼                       ▼                       ▼
                       EvaluationRunner          Guardrails               Scoring
                    (your app over HTTP     (deterministic PII/     (executed →
                     or an LLM provider,     injection + LLM-judge   passed/failed;
                     one per run)            safety/hallucination/   errors stay
                                             instruction-following)  distinct)
                              └───────────────────────┼───────────────────────┘
                                                      ▼
                                    PostgreSQL (state of record)
                                                      ▼
                                   Regression engine (LLM-free):
                                   baseline vs current → verdict
```

Rules the codebase enforces:

- **PostgreSQL is authoritative.** Redis/Celery carry task state only.
- **Quality ≠ execution failure.** A bad answer (`failed`) is distinct
  from no answer (`error`, with a typed reason like `timeout`).
- **One target per run.** `agent_model` is a model name or
  `application:<name>`; both share runner, guardrails, and scoring.
- **Multi-tenant.** Clerk organizations bound every query; other tenants
  read as 404.
- **No web dashboard.** Terminal + API only.

Tests: `uv run pytest` (unit) · `EVALYX_RUN_INTEGRATION_TESTS=1 uv run pytest`
(live PG/Redis) · `uv run ruff check src tests` · `uv run mypy src`.

License: see [LICENSE](LICENSE).
