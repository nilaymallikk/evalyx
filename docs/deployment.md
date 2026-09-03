# Evalyx production deployment & operations (Phase 17)

Reference deployment: Docker Compose. Portable — no Kubernetes, no
Terraform, no cloud-specific dependencies. The CLI/TUI stays an API client
(`evalyx --api-url https://api.example.com`).

```text
                    Internet
                       │
                       ▼
               ┌──────────────┐
               │ Reverse Proxy│  nginx: TLS, HTTP→HTTPS, security headers
               └───────┬──────┘
                       │  frontend network
                       ▼
               ┌──────────────┐
               │  Evalyx API  │  FastAPI (uvicorn, 1 worker by default)
               └──────┬───────┘
                      │  internal network (internal: true)
         ┌────────────┼────────────┐
         ▼            ▼            ▼
   PostgreSQL       Redis       Celery worker
   state store      broker      (same image, no ports)
   (private)        (private)
```

`docker-compose.yml` stays the local-development stack (published infra
ports, dev defaults). `docker-compose.production.yml` is the production
reference: `api`, `worker`, `postgres`, `redis`, one-shot `migrate`, and
`reverse-proxy`. Postgres/Redis/worker expose no public ports; the API is
reachable only through the proxy; only the proxy publishes 80/443.

## 1. Required production environment

Copy `.env.production.example` to `.env.production` (gitignored) and fill
every value from your secret store. The compose file refuses to start when
a required value is missing (`${VAR:?…}` guards).

| Variable | Required | Notes |
|---|---|---|
| `APP_ENV=production` | yes | Enables production validators |
| `AUTH_REQUIRED=1` | yes | Startup fails otherwise |
| `EVALYX_SECRET_KEY` | yes | Fresh random value |
| `EVALYX_ENCRYPTION_KEY` | yes | urlsafe base64 of 32 bytes; back up with the DB |
| `CLERK_SECRET_KEY` | yes | **Production** Clerk instance, never dev keys |
| `CLERK_JWKS_URL` | yes | Production instance `/.well-known/jwks.json` |
| `CLERK_AUTHORIZED_PARTIES` | no | Comma-separated accepted origins, or empty |
| `DATABASE_URL` | yes | `postgresql+asyncpg://USER:PASS@postgres:5432/evalyx` |
| `REDIS_URL` | yes | `redis://:PASS@redis:6379/0` |
| `POSTGRES_USER/PASSWORD/DB` | yes | Strong password, URL-encoded into `DATABASE_URL` |
| `REDIS_PASSWORD` | yes | Also wired into `REDIS_URL` |
| `OPENROUTER_API_KEY` | yes | Only if the deployment runs evaluations |
| `CORS_ALLOWED_ORIGINS` | no | Empty = disabled (correct for CLI/TUI); never `*` |

Startup fails fast with a named error (never the secret value): `Clerk
configuration missing`, `EVALYX_ENCRYPTION_KEY must be set`, or
`AUTH_REQUIRED cannot be disabled`. `AUTH_REQUIRED=0` is rejected in
production.

## 2. Secrets

Deliver secrets via environment injection or container secrets (Docker
secrets, Vault, cloud secret manager → env). Never in `Dockerfile`,
`docker-compose.*.yml`, `README`, CI logs, or the built image. The image
contains no `.env` files (see `.dockerignore`). Logs redact
secret/token/password/authorization-shaped keys; health, readiness, and
metrics responses never carry credentials.

## 3. Clerk production configuration

Use a dedicated Clerk **production** instance (separate from development).
Set `CLERK_SECRET_KEY` (server-side only) and `CLERK_JWKS_URL` from that
instance; tokens are verified locally against the JWKS (no per-request
Clerk round-trip). Optional `CLERK_AUTHORIZED_PARTIES` restricts accepted
token audiences. Rotate by updating the secret store and recreating the
`api`/`worker` containers — no image rebuild needed.

## 4. Encryption key handling

`EVALYX_ENCRYPTION_KEY` encrypts application credentials (AES-256-GCM,
versioned envelope). It is **never generated at startup** — auto-generation
would orphan existing ciphertext. Missing/invalid key fails startup with a
named error. Store it where the database backups are documented (losing it
makes stored credentials undecryptable); rotation of the master key with
re-encryption is not implemented (documented limitation — same as Phase 15).

## 5. Build / start / migrate

```bash
cp .env.production.example .env.production   # fill from secret store
docker compose -f docker-compose.production.yml --env-file .env.production build
docker compose -f docker-compose.production.yml --env-file .env.production run --rm migrate
docker compose -f docker-compose.production.yml --env-file .env.production up -d
docker compose -f docker-compose.production.yml ps
curl -k https://<host>/health/ready   # remove -k once TLS is provisioned
```

Migration workflow (never auto-run from API replicas):

```text
backup → alembic upgrade head (migrate service) → start new app version
```

`alembic upgrade head` runs via the one-shot `migrate` service. Downgrades
are never automatic; verify `upgrade → downgrade → upgrade` on a staging
copy before touching production.

## 6. Health & readiness

- `GET /health` — liveness only (process alive). No dependency checks, no
  secrets. Used by the container `HEALTHCHECK` and load balancers.
- `GET /health/ready` — readiness: PostgreSQL + Redis checks. `200 {"status":
  "ok"}` serves traffic; `503 {"status": "degraded"}` does not (dependency
  names only, never connection strings).
- `GET /api/v1/metrics` — authenticated operational snapshot (counters +
  timing aggregates; bounded labels only — no ids, URLs, payloads, or
  secrets). Worker metrics live in worker logs.

## 7. Graceful shutdown

- API: SIGTERM stops the listener, in-flight requests drain (30 s grace),
  then the lifespan handler disposes the DB engine and closes Redis.
  `stop_grace_period: 45s` in compose.
- Worker: SIGTERM is a Celery warm shutdown — no new tasks, the current
  task finishes within the soft/hard time limits (`task_acks_late` +
  `task_reject_on_worker_lost` redeliver on crash). `stop_grace_period:
  120s`. Never `kill -9` a busy worker; a redelivered job resumes
  idempotently (completed runs re-score, never re-execute).

## 8. Resource limits & tuning

Defaults (override via env): API 1 CPU / 1 GiB, worker 2 CPU / 2 GiB,
`WORKER_CONCURRENCY=2`, Redis `maxmemory 256mb` (`noeviction`), Postgres
`max_connections 100` with app pool `DB_POOL_SIZE=5` + `DB_MAX_OVERFLOW=10`,
request bodies capped at 1 MiB, pages at 200 items. Tune Postgres pool ≤
`max_connections / (api_workers × processes + workers)`; raise worker
concurrency only with provider quota headroom (free models are
rate-limited; per-run case execution stays sequential by design).

## 9. Rate limiting & evaluation bounds

In-memory fixed-window limiter (per API process): 120 req/min default, 20
evaluation submissions/min, 10 connection tests/min, per-IP health-endpoint
baseline. Oversized bodies get 413; over-limit gets 429 + `Retry-After`.
Evaluations are additionally bounded by `MAX_CASES_PER_EVALUATION` (default
500 → 422 `evaluation_too_large`). One org cannot trivially exhaust workers.
Limitation: per-process state — correct for the single-worker reference
compose; multi-replica needs a Redis-backed limiter (Phase 18).

## 10. Outbound networking (Phase 15 protections intact)

SSRF checks, public-destination DNS validation, redirect re-validation,
`trust_env=False` (ambient proxies ignored), connect/read timeouts, 2 MiB
response cap, bounded retries (transient only). The production environment
must not set proxy env vars that bypass these; the app target ignores them
by construction.

## 11. Redis vs PostgreSQL roles

Redis is a **transient broker/cache** (Celery transport + task metadata with
TTL). PostgreSQL is the **authoritative state store** (runs, cases,
guardrails, comparisons). Losing Redis loses queued jobs, never evaluation
history. Redis runs with a password, no public port, bounded memory;
persistence (`appendonly`/`save`) stays off by default — enable it only if
your queue-durability posture requires it, and document that choice.

## 12. Backup & restore

Backups are **operator-managed** (not automatic in the reference compose):

- Backup: `POSTGRES_USER=… POSTGRES_DB=… ./scripts/backup.sh ./backups`
  (`pg_dump -Fc` + SHA-256). Store off-host, encrypted; recommended daily
  with 30-day retention (adjust to your policy).
- Restore: `./scripts/restore.sh <dump>` after `sha256sum -c`, then run
  migrations, restart the app, and verify (record counts, latest run,
  `/health/ready`). Prove recoverability with a restore drill:

```text
backup → restore to scratch → migrate → start → verify data
```

Keep the encryption key with the backups — without it, restored
application credentials cannot be decrypted.

## 13. Deployment & rollback

```text
1. build image → 2. run tests → 3. push image → 4. backup database
5. run migrations → 6. start new API → 7. start/update workers
8. check readiness → 9. smoke test → 10. monitor logs/metrics
```

Keep the old containers until readiness + smoke test pass
(`docker compose up -d` replaces only after the new image is healthy).
Rollback = redeploy the previous image tag (application rollback only).
**Never auto-downgrade the database** — migrations must be
backwards-compatible (expand-then-contract); if a schema change cannot roll
back, restore from the pre-deploy backup instead (data loss window
documented to stakeholders).

## 14. Smoke test

```bash
EVALYX_API_URL=https://api.example.com EVALYX_TOKEN="$(cat token.txt)" \
  python scripts/smoke_test.py
# With one evaluation round-trip:
EVALYX_SMOKE_SUBMIT=1 EVALYX_APPLICATION_ID=… EVALYX_DATASET_VERSION_ID=… \
  python scripts/smoke_test.py
# Against a local AUTH_REQUIRED=0 server (dev header instead of a token):
EVALYX_API_URL=http://127.0.0.1:8000 EVALYX_ORG=org_smoketest \
  python scripts/smoke_test.py
```

Covers `/health`, `/health/ready`, authenticated `/me`, application and
dataset listings, and (opt-in) evaluation submit → status → reliability.
Credentials come from the environment; nothing is committed.

## 15. CLI against production

```bash
evalyx --api-url https://api.example.com login   # Clerk session token via prompt/stdin
evalyx --api-url https://api.example.com whoami
evalyx --api-url https://api.example.com app list
```

Or export `EVALYX_API_URL`. HTTPS only in production; auth, error-code, and
JSON-mode behavior are unchanged. The CLI never connects to the database.

## 16. Container security

Non-root `evalyx` user (uid 10001), no privileged containers, no host
networking, no host mounts except the read-only nginx config + TLS files,
minimal exposed ports (proxy 80/443 only), health checks + restart policies
everywhere. Postgres/Redis/worker/API communicate over a private internal
network (`internal: true`).

## 17. Logging & metrics inventory

Structured JSON logs in production (console locally): `http_request_*`,
`evaluation_submitted/started/completed/failed`, `task_received/retrying/
completed/failed`, `provider_retry_scheduled`, `regression_comparison_*`,
`readiness_check_failed`, `rate_limit_exceeded` — every event carries
`request_id`/`run_id`/`task_id` correlation; secret-shaped fields are
redacted. Metrics: `http_requests_total`, `http_request_duration_ms`,
`evaluation_*`, `application_requests_total` + latency/retries, connection
tests — bounded labels only (the registry raises on id/URL/payload labels).

## 18. HTTPS provisioning

TLS is operator-provisioned (certbot, cloud load balancer, internal CA).
Place `fullchain.pem`/`privkey.pem` at `TLS_CERT_PATH`/`TLS_KEY_PATH`, set
the `server_name` in `deploy/nginx.conf` to the public hostname, and keep
port 80 only for the HTTP→HTTPS redirect. The FastAPI port is never public.

## 19. Known limitations (Phase 18 candidates)

Per-process rate limits (no shared limiter), no per-org quotas or billing,
no multi-replica metrics aggregation, no automated backups, no master-key
rotation with re-encryption, DNS-check/TCP-connect TOCTOU window (Phase 15),
threshold-based (not statistical) regression. None of these block a safe
single-site production deployment.
