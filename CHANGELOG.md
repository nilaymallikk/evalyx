# Evalyx Changelog

All notable changes to Evalyx. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
principles; versions are PEP 440 (`pyproject.toml` is authoritative,
`evalyx --version` prints the CLI build).

## [0.9.0b1] — Public Beta (Phase 19)

First public beta release. An independent user can deploy Evalyx with Docker
Compose, register their own HTTP AI application, and run the full
evaluate → guardrail → regression workflow from the CLI/TUI.

### Added

- Beta end-to-end coverage: `tests/integration/test_phase19_beta.py` drives
  register → version → dataset → cases → submit → execute → guardrails →
  scoring → reliability → regression comparison over the real HTTP surface.
- CLI/TUI beta tests: `tests/unit/test_phase19_beta.py` (dev-org names,
  metadata-only versions, headless TUI navigation across all views).
- Minimal CI: `.github/workflows/ci.yml` (`uv sync`, `ruff check`, `mypy`,
  unit tests, `alembic check` against a service database).

### Fixed (release blockers found by the Phase 19 audit)

- Evaluation submission without `judge_model` no longer produces a
  completed-but-never-scored run: the configured `EVALYX_JUDGE_MODEL`
  default applies server-side (explicit values unchanged).
- `evalyx app version` accepts reference (`mlgpt`) applications without
  `--endpoint` (metadata-only version); generic `http` apps without an
  endpoint fail fast with a usage error instead of a server rejection.
- Dev-mode organization header accepts `org_<letters/digits/_/->`
  (e.g. `org_beta_e2e`, `org_dev_default`); injection payloads still 401.
- `EVALYX_ENCRYPTION_KEY` accepts the documented unpadded
  `secrets.token_urlsafe(32)` form as well as padded base64.
- `supports_unicode()` no longer crashes under captured/redirected stdout
  (broke the TUI dashboard under test pilots).
- Production Compose CPU defaults no longer carry quotes
  (`docker compose config` failed to parse `"1.0"`).
- Model/migration drift: `applications.connection_type` index declared in
  the model (`alembic check` is clean again).

### Verified for this release

- `docker build` reproduces `evalyx:phase19-beta`; image imports, runs as
  non-root `evalyx`, ships no `.env`/credentials.
- Production Compose validates; API boots with `APP_ENV=production` +
  `AUTH_REQUIRED=1` (liveness + readiness `ok`); worker boots and registers
  `run_evaluation`; both shut down gracefully.
- Migrations: fresh-database `upgrade head`, single-step
  downgrade → upgrade, and no-op upgrade on the existing database.
- Backup (`pg_dump -Fc` + SHA-256) → restore to scratch → migrate →
  connect, with matching record counts.
- Smoke test incl. evaluation round-trip: all checks pass.
- Phase 18 security suites re-run green (auth, RBAC, tenant isolation,
  rate limits, quotas, encryption/rotation, audit, SSRF).
- Beta-scale sanity: 10 concurrent submissions admitted/limited correctly
  (`quota_exceeded` 429s on concurrent-evaluation overflow), worker drains
  bounded by free-model judge latency, readiness stays `ok`.

### Known limitations (unchanged, see README)

- No web dashboard (terminal-first by design), no billing, no Kubernetes.
- Threshold-based (not statistical) regression; quota overrides and audit
  reads via SQL; per-IP (not per-org-tier) rate limits.
- Master encryption-key rotation with re-encryption is workflow-supported
  (`scripts/reencrypt_credentials.py`) — see `docs/security.md`.

## Earlier phases (summary)

- **Phase 1** — Repository audit & architecture decision.
- **Phase 2** — Typed configuration, Docker Compose, PostgreSQL + Redis,
  health/readiness probes, secret handling.
- **Phase 3** — SQLAlchemy domain model, Alembic migrations, repositories.
- **Phase 4** — Provider-neutral LLM layer (OpenRouter default, Ollama
  optional), bounded retries, typed errors.
- **Phase 5** — Evaluation runner: pinned dataset versions, per-case
  results, execution-honest statuses.
- **Phase 6** — Hybrid guardrails (deterministic PII/injection indicators +
  LLM-judge safety/hallucination/instruction-following) and scoring policy.
- **Phase 7** — Celery + Redis background workers (thin tasks, idempotent,
  PostgreSQL authoritative).
- **Phase 8** — Deterministic LLM-free regression engine (baselines,
  thresholds, case-level findings, persisted artifacts).
- **Phase 9** — Versioned FastAPI surface (`/api/v1`), async 202
  submissions, pagination, error envelope, OpenAPI.
- **Phase 10** — Structured logging, request IDs, in-process metrics.
- **Phase 11** — MLGPT reference demo application (separate repo).
- **Phase 12** — Failure analysis taxonomy (execution vs quality failures).
- **Phase 13** — Documentation & recruiter demo polish.
- **Phase 14** — Clerk authentication + organization multi-tenancy
  (uniform 404s, RBAC).
- **Phase 15** — Generic HTTP application connections (validated configs,
  AES-GCM credentials, SSRF protection).
- **Phase 16** — CLI (`evalyx`) + Textual TUI (REST-only client, strict
  `--json`, meaningful exit codes).
- **Phase 17** — Production deployment (multi-stage Dockerfile,
  production Compose, nginx, backup/restore/smoke scripts).
- **Phase 18** — Security hardening: Redis-backed rate limits, race-safe
  quotas, audit log, keyring + rotation, IP-pinning SSRF transport.
