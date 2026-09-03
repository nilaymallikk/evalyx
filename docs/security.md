# Evalyx security, quotas & hardening (Phase 18)

Companion to `deployment.md` (reference deployment) and
`production-checklist.md`. This document covers the multi-tenant
hardening: distributed rate limiting, organization quotas, RBAC/tenant
rules, encryption rotation, audit logging, SSRF, and residual limitations.

## 1. Distributed rate limiting

One Redis `INCR` per request decides admission atomically — the returned
count is the linearization point, so N replicas share one budget exactly.
Buckets (per minute, per IP): default 120, evaluation submission 20,
connection test 10, health baseline 120. Over-limit → `429 rate_limited`
with an accurate `Retry-After` (seconds left in the 60 s window).

Keys are `evalyx:rl:{bucket}:{sanitized-ip}:{window}` with a 60 s TTL:
bounded bucket set, charset-restricted identifier (≤ 64 chars), no payloads.

Redis unavailable → the configured `RATE_LIMIT_ON_REDIS_ERROR` policy
(`allow` default, or `deny` → `503 rate_limiter_unavailable`), always with
a warning log and the `rate_limiter_errors_total{policy}` metric. There is
deliberately **no** silent per-process fallback: per-process state would
under-enforce across replicas while looking enforced. (`allow` keeps
availability during a Redis blip; readiness already excludes a
Redis-less replica from rotation, so the window is small either way.)

Multi-replica API deployments are now supported: raise `API_WORKERS` /
replicas without breaking limits. The per-process caveat in
`deployment.md §9` no longer applies.

## 2. Organization quotas

Server-side admission in PostgreSQL, independent from billing (there is
none). Dimensions (env defaults, per-org overridable via the
`organization_quota_overrides` table):

| Dimension | Default | Denial |
|---|---|---|
| applications | 50 | `429 quota_exceeded` |
| datasets | 50 | `429 quota_exceeded` |
| cases per dataset version | 5000 | `429 quota_exceeded` |
| evaluations per UTC day | 100 | `429 quota_exceeded` |
| connection tests per UTC day | 200 | `429 quota_exceeded` |
| concurrent evaluations | 5 | `429 quota_exceeded` |

Race safety: admission runs under `SELECT … FOR UPDATE` on the
organization row in the same transaction as the admitted insert —
concurrent admissions serialize and re-read committed counts (tested with
10 concurrent submissions against a quota of 2 → exactly 2 admitted).

Capacity release: the concurrency quota counts runs in `pending`/`running`
created within `QUOTA_STALE_RUN_SECONDS` (default 7200 s, must exceed the
worker hard limit). Every terminal transition (completed/failed/cancelled —
including enqueue-failure marking and duplicate-delivery rescore/skip)
releases capacity automatically; stuck runs age out. There is no release
call to forget.

Connection-test budget is counted from admitted audit rows (same
transaction commits the admission row before the outbound call, so
in-flight tests count). Disabling the audit log disables this one
dimension loudly (one warning); all other quotas are audit-independent.

## 3. Tenant isolation & roles

Rules (tested by a full cross-tenant matrix):

- Every tenant-owned read/write/delete is scoped by `organization_id` at
  the repository boundary; foreign resources read as missing (uniform 404,
  existence never leaked) — applications, versions, secrets, datasets,
  cases, runs, results, guardrails, reliability, regressions, listings.
- Identity and roles come only from the verified Clerk `AuthContext`.
  Request schemas carry no `organization_id`/`role`/`user_id` fields, so
  forged tenant fields are ignored (pydantic drops them; scoping never
  reads them).
- Unknown Clerk roles map to `None` (least privilege) and can never
  satisfy admin guards (`require_role` → 403 `insufficient_role`).
- The `X-Dev-Organization-Id` header is honored only by the local-dev
  verifier, which production can never select (`AUTH_REQUIRED=0` fails
  startup when `APP_ENV=production`).
- Authenticated-but-orgless users get `403 organization_required` plus a
  durable audit row; anonymous callers get `401` (log + metric only — no
  unauthenticated DB writes).

Deliberate non-change: member vs admin endpoint requirements were left as
in Phase 14–17 (no endpoint currently gates on admin). Tightening
destructive operations to admin-only is a policy decision for a later
phase, not taken silently here.

## 4. Encryption key rotation

Envelopes: `v1:<nonce>:<ct>` (legacy, readable) and
`v2:<key-id>:<nonce>:<ct>` (current writes). Key id = truncated SHA-256 of
the raw key (deterministic, non-sensitive). The encryptor holds a keyring:
current key encrypts; current + `EVALYX_PREVIOUS_ENCRYPTION_KEYS` decrypt.

Rotation procedure:

```bash
# 1. deploy new key as current, old key as history (reads keep working)
EVALYX_ENCRYPTION_KEY=<new> EVALYX_PREVIOUS_ENCRYPTION_KEYS=<old>
# 2. dry-run, then apply (idempotent, resumable, per-row commits)
uv run python scripts/reencrypt_credentials.py --dry-run
uv run python scripts/reencrypt_credentials.py --apply
# 3. dry-run again (expect zero remaining), drop the old key from history
```

Reports carry counts and application ids only — never plaintext,
ciphertext, or keys. Undecryptable rows are listed by id for investigation.
`secret_metadata` records `key_version`/`key_id` (safe) for rotation
auditing; API responses still expose only `secret_configured`.

## 5. Audit logging

Durable `audit_events` rows (organization, actor, action, resource, result,
request id, timestamp, sanitized details) written in the action's own
transaction; denials commit immediately before raising. Covered actions:
application create/update/delete/version/secret-rotate/connection-test,
dataset create/version/case-add, evaluation submit, organization-required
denials, quota denials. 401s and role denials without a session go to
structured logs + `auth_denied_total{reason}` instead (no unauthenticated
DB writes).

Details sanitizer drops secret-shaped keys (secret/token/password/
prompt/response/…), truncates strings (500), bounds keys (20) — defense in
depth; callers already pass minimal facts (names, counts, ids — never
descriptions, inputs, or outputs).

Retention is operator-managed (`AUDIT_RETENTION_DAYS`, default 90):

```sql
DELETE FROM audit_events WHERE created_at < now() - make_interval(days => 90);
```

There is no API read surface for audit events in this phase (operated via
SQL). Connection-test quota and retention both rely on this table — do not
drop it while quotas are enabled.

## 6. SSRF hardening

Phase 15 protections preserved (DNS validation, redirect re-validation,
private/loopback/link-local/metadata blocking, `trust_env=False`, 2 MiB
cap, timeouts, bounded retries). Phase 18 additions:

- Configuration-time recognition of obfuscated numeric literals
  (`2130706433`, `0x7f000001`, `0177.0.0.1`, `127.1`, …) — the OS resolver
  accepts these even though `ipaddress` does not.
- Out-of-range ports now rejected cleanly (previously escaped as a raw
  `ValueError`).
- **TOCTOU closure**: the connector's HTTP client now uses an IP-pinning
  transport that resolves + validates *inside* the request path and
  connects only to a validated address, preserving `Host` and TLS SNI —
  the validated address is the connected address. Per-hop re-validation
  stays as defense in depth.

Residual risk (honest): a hostile DNS server chooses *which public*
address is returned, but every candidate is validated public before use —
steering cannot reach private targets. The MLGPT reference target
(server-side `MLGPT_BASE_URL`, not user input) intentionally stays on the
plain transport.

## 7. Observability across replicas

- `/metrics` responses carry an `instance` id; operators aggregate by
  summing counters across replicas (no shared metric state needed).
- New bounded series: `rate_limited_total{bucket}`,
  `rate_limiter_errors_total{policy}`, `auth_denied_total{reason}`,
  `quota_denied_total{resource}`, `audit_events_total{action,result}`.
- Database outages surface as `503 database_unavailable` (OperationalError,
  pool timeouts, connection-refused errnos, asyncio connect aggregates) —
  never DSNs, addresses, or ports. Liveness stays dependency-free.

## 8. Residual limitations (Phase 19+ candidates)

Per-org quota overrides have no admin API yet (set via SQL/service);
audit events have no API read surface; rate limits are per-IP (no
per-org/per-user tiers); connection-test quota needs the audit log;
regression comparison is unaudited (read-mostly artifact creation);
stuck-run aging is time-based (no active reaper). None block a safe
multi-tenant deployment at the documented defaults.
