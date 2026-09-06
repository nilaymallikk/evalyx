# Evalyx

Test your AI app before and after every change. Evalyx sends test questions
to your app over HTTP, grades the answers, and tells you what got worse.

```bash
curl -fsSL https://raw.githubusercontent.com/nilaymallikk/evalyx/main/install.sh | bash
```

Or manually:

```bash
git clone https://github.com/nilaymallikk/evalyx
cd evalyx
docker compose up -d
uv sync
cp .env.example .env          # add OPENROUTER_API_KEY + EVALYX_SECRET_KEY
uv run python main.py         # terminal 1: API on http://127.0.0.1:8000
uv run celery -A evalyx.worker.celery_app worker --loglevel=INFO  # terminal 2: worker
```

## Evaluate your app in 5 minutes

```bash
uv run evalyx quickstart
```

Answer the questions it asks (your app's URL, a few sample questions).
It registers the app, checks the connection, runs an evaluation, and
prints the score. Done.

## Do it step by step

Replace `<app_id>`, `<ds_id>`, `<dsv_id>`, `<run_id>` with the ids each
command prints.

```bash
# 1. Register your app
uv run evalyx app create myapp --type http

# 2. Tell Evalyx how to call it
uv run evalyx app version <app_id> v1 \
  --endpoint https://your-app.com/chat \
  --auth none --input-field question --response-path answer

# 3. Check it works
uv run evalyx app test <app_id>

# 4. Make a dataset
uv run evalyx dataset create mydata
# create version 1:
curl -s -X POST http://127.0.0.1:8000/api/v1/datasets/<ds_id>/versions \
  -H 'Content-Type: application/json' -d '{"version": 1}'
uv run evalyx dataset add-case <ds_id> 1 --name q1 \
  --input '{"prompt":"Say hello."}'

# 5. Run the evaluation
uv run evalyx eval run --application <app_id> --dataset-version <dsv_id> \
  --agent-model application:myapp --wait

# 6. See the results
uv run evalyx eval results <run_id>
uv run evalyx eval guardrails <run_id>

# 7. After changing your app, run again and compare
uv run evalyx regression run --baseline <old_run_id> --current <run_id>
```

Your endpoint gets `POST {"question": "..."}` and should reply with the
answer in the field you named (`{"answer": "..."}`).

## App runs on your own machine?

Public URLs work by default. For `localhost`, add this to `.env` and
restart the API and worker:

```
EVALYX_ALLOW_PRIVATE_ENDPOINTS=1
```

Then register the localhost URL normally. (Never enable this in
production. A local **MLGPT** is the exception — use
`evalyx app create mlgpt --type mlgpt` with
`MLGPT_BASE_URL=http://127.0.0.1:8002`, no flag needed.)

## No app to test yet?

Evaluate a model directly — no app, no endpoint:

```bash
uv run evalyx eval run --application <app_id> --dataset-version <dsv_id> \
  --agent-model nvidia/nemotron-3-ultra-550b-a55b:free --wait
```

## Commands

```bash
uv run evalyx --help            # everything
uv run evalyx whoami            # check the setup
uv run evalyx                   # interactive terminal UI
```

Every command takes `--json` for scripts. API docs: http://127.0.0.1:8000/docs

## How it works

```
CLI ──REST──► API ──202──► Redis ⇄ Worker ──HTTP──► your app
                                          │
                              guardrails + scoring
                                          ▼
                              PostgreSQL ──► regression compare
```

Tests: `uv run pytest`
