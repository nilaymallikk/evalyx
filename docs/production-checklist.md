# Production checklist (Phase 17; Phase 18 additions marked ★)

## Before deployment

- [ ] Production Clerk instance created (not development keys)
- [ ] `AUTH_REQUIRED=1` in `.env.production`
- [ ] `EVALYX_SECRET_KEY` set (fresh random value)
- [ ] `EVALYX_ENCRYPTION_KEY` set and backed up with the DB backups
- [ ] `POSTGRES_PASSWORD` strong, URL-encoded into `DATABASE_URL`
- [ ] `REDIS_PASSWORD` set and wired into `REDIS_URL`
- [ ] `DATABASE_URL` / `REDIS_URL` point at the `postgres` / `redis` services
- [ ] TLS certificate + key in place (`TLS_CERT_PATH`, `TLS_KEY_PATH`)
- [ ] `server_name` in `deploy/nginx.conf` set to the public hostname
- [ ] Database backup taken (`./scripts/backup.sh`)
- [ ] Migrations reviewed (`alembic upgrade head --sql` on staging first)
- [ ] Image built reproducibly (`docker build -t evalyx:<tag> .`)
- [ ] Unit + integration tests green, mypy + ruff clean
- [ ] ★ Quota defaults reviewed (`QUOTA_*`); overrides set if needed
- [ ] ★ `RATE_LIMIT_ON_REDIS_ERROR` policy chosen deliberately (allow/deny)
- [ ] ★ Encryption rotation state known: pending count via dry-run is zero
      unless a rotation is in progress (history key present only then)
- [ ] ★ Audit retention scheduled (`AUDIT_RETENTION_DAYS`, cleanup SQL)

## Deploy

- [ ] `run --rm migrate` applied cleanly
- [ ] New API container healthy (`/health`, `/health/ready` → 200)
- [ ] Worker running (`celery inspect ping` → pong)
- [ ] Reverse proxy serving HTTPS, HTTP → HTTPS redirect verified
- [ ] Old deployment retained until smoke test passes

## After deployment

- [ ] `GET /health` → `{"status":"ok"}`
- [ ] `GET /health/ready` → `{"status":"ok"}`
- [ ] Authenticated `GET /api/v1/me` works (401 without a token)
- [ ] Application connection test succeeds
- [ ] Evaluation submission accepted (202) and processed by the worker
- [ ] Logs are structured JSON with request IDs and no secrets
- [ ] Authenticated `GET /api/v1/metrics` returns operational counters
- [ ] `docker scout` / image inspection shows no `.env` or secrets
- [ ] Rollback image tag recorded
- [ ] ★ Over-quota probe denied cleanly (`429 quota_exceeded`, no leakage)
- [ ] ★ Cross-tenant probe returns uniform 404s (no existence signals)
