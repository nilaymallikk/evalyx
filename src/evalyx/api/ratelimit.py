"""Distributed rate limiting (Phase 18).

Redis-backed, atomic fixed-window counters shared across all API replicas —
the Phase 17 process-local limiter is gone:

- per-IP limits for unauthenticated infrastructure endpoints (/health*)
- authenticated API default limit
- tighter limits for evaluation submissions and connection tests

Design:

- One Redis ``INCR`` per request decides admission atomically (the returned
  count is the linearization point, so concurrent replicas cannot jointly
  exceed the limit). A TTL is set when the counter is created
  (``count == 1``); the window is anchored at first hit.
- Keys are bounded: ``{prefix}:{bucket}:{identifier}:{window}`` where the
  bucket is a fixed set, the identifier is a sanitized IP (bounded length,
  restricted charset), and the window is ``epoch // 60``. TTLs (60 s) bound
  storage; no key ever carries user payloads.
- Exceeding the limit returns 429 with an accurate ``Retry-After`` (seconds
  remaining in the window) and a safe envelope — never secrets, never
  request bodies.
- Redis unavailable: the configured policy (``RATE_LIMIT_ON_REDIS_ERROR``,
  ``allow``/``deny``) applies **loudly** — a warning log plus the
  ``rate_limiter_errors_total`` metric. There is deliberately no silent
  per-process fallback: per-process state would under-enforce limits across
  replicas while looking enforced.

The :class:`RateLimitBackend` protocol keeps the middleware testable: unit
tests inject an in-memory fake (an explicit test double, never a
production fallback).
"""

from __future__ import annotations

import re
import time
from typing import Final, Protocol

import structlog
from fastapi.responses import JSONResponse

from evalyx.core.config import Settings
from evalyx.core.metrics import metrics

logger = structlog.get_logger(__name__)

WINDOW_SECONDS: Final[int] = 60

_EVAL_PATHS: Final[tuple[str, ...]] = ("/api/v1/evaluations",)
_TEST_SUFFIX: Final[str] = "/test"
_HEALTH_PREFIXES: Final[tuple[str, ...]] = ("/health",)

_BUCKETS: Final[frozenset[str]] = frozenset({"health", "eval", "test", "default"})

_MAX_IDENTIFIER_LENGTH: Final[int] = 64
_SAFE_IDENTIFIER_RE: Final[re.Pattern[str]] = re.compile(r"[^A-Za-z0-9._-]")


def _client_ip(scope: dict) -> str:
    client = scope.get("client")
    if isinstance(client, (list, tuple)) and client:
        return str(client[0])
    return "unknown"


def _path(scope: dict) -> str:
    return str(scope.get("path", "") or "")


def _method(scope: dict) -> str:
    return str(scope.get("method", "") or "").upper()


def _bucket_for(path: str, method: str) -> str:
    if path in _HEALTH_PREFIXES or any(
        path == p or path.startswith(p + "/") for p in _HEALTH_PREFIXES
    ):
        return "health"
    if method == "POST" and path in _EVAL_PATHS:
        return "eval"
    if method == "POST" and path.endswith(_TEST_SUFFIX):
        return "test"
    return "default"


def sanitize_identifier(raw: str) -> str:
    """Bound an identifier for use in a Redis key.

    Anything outside ``[A-Za-z0-9._-]`` becomes ``_`` and the result is
    truncated — IPv4/IPv6 literals pass through; hostile input cannot grow
    or poison the keyspace.
    """
    return _SAFE_IDENTIFIER_RE.sub("_", raw)[:_MAX_IDENTIFIER_LENGTH] or "unknown"


def rate_limit_key(prefix: str, bucket: str, identifier: str, now: float) -> str:
    """Build the bounded Redis key for one window."""
    window = int(now // WINDOW_SECONDS)
    return f"{prefix}:{bucket}:{sanitize_identifier(identifier)}:{window}"


def retry_after_seconds(now: float) -> int:
    """Seconds remaining in the current window (≥ 1, for Retry-After)."""
    return max(1, WINDOW_SECONDS - int(now % WINDOW_SECONDS))


class RateLimitBackend(Protocol):
    """Atomic fixed-window admission check."""

    async def check_and_increment(
        self, key: str, limit: int, window_seconds: int
    ) -> tuple[bool, int]:
        """Atomically increment ``key``; return ``(allowed, retry_after)``."""
        ...


class RedisRateLimitBackend:
    """Redis backend: one atomic INCR decides admission.

    The TTL is set when the counter is created. A crash between INCR and
    EXPIRE can leave a TTL-less key behind; the next window uses a different
    key (the window number is part of it), so the damage is one stale key,
    reaped by Redis eviction — never incorrect admission.
    """

    def __init__(self, redis_client) -> None:
        self._redis = redis_client

    async def check_and_increment(
        self, key: str, limit: int, window_seconds: int
    ) -> tuple[bool, int]:
        count = int(await self._redis.incr(key))
        if count == 1:
            await self._redis.expire(key, window_seconds)
        now = time.time()
        if count <= limit:
            return True, 0
        return False, retry_after_seconds(now)


class InMemoryRateLimitBackend:
    """Explicit test double (unit tests only — never a production fallback).

    Same fixed-window semantics as the Redis backend, guarded by an asyncio
    lock so concurrent-admission tests are meaningful.
    """

    def __init__(self) -> None:
        import asyncio

        self._lock = asyncio.Lock()
        self._counts: dict[str, int] = {}

    async def check_and_increment(
        self, key: str, limit: int, window_seconds: int
    ) -> tuple[bool, int]:
        async with self._lock:
            count = self._counts.get(key, 0) + 1
            self._counts[key] = count
        if count <= limit:
            return True, 0
        return False, retry_after_seconds(time.time())


class RateLimiter:
    """Fixed-window limiter over an injected backend (one key per call)."""

    def __init__(self, settings: Settings, backend: RateLimitBackend) -> None:
        self._settings = settings
        self._backend = backend

    def limit_for(self, bucket: str) -> int:
        if bucket == "eval":
            return self._settings.rate_limit_eval_per_minute
        if bucket == "test":
            return self._settings.rate_limit_test_per_minute
        return self._settings.rate_limit_per_minute

    async def allowed(self, bucket: str, identifier: str) -> tuple[bool, int]:
        """Return ``(allowed, retry_after_seconds)`` for one request."""
        key = rate_limit_key(
            self._settings.rate_limit_redis_prefix, bucket, identifier, time.time()
        )
        return await self._backend.check_and_increment(
            key, self.limit_for(bucket), WINDOW_SECONDS
        )


class RateLimitMiddleware:
    """ASGI middleware enforcing distributed per-IP fixed-window limits."""

    def __init__(
        self,
        app,
        settings: Settings,
        redis_client=None,
        backend: RateLimitBackend | None = None,
    ) -> None:
        self.app = app
        self.settings = settings
        if backend is not None:
            self._backend: RateLimitBackend | None = backend
        elif redis_client is not None:
            self._backend = RedisRateLimitBackend(redis_client)
        else:
            logger.error("rate_limiter_no_backend")
            self._backend = None
        self._limiter: RateLimiter | None = (
            RateLimiter(settings, self._backend) if self._backend is not None else None
        )

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        # Bounded request bodies (DoS guard): reject oversized Content-Length
        # before routing. Chunked bodies without a length header are bounded
        # downstream by JSON parsing + page-size limits.
        for key, value in scope.get("headers", []) or []:
            if key == b"content-length":
                try:
                    if int(value.decode("latin-1")) > self.settings.max_request_body_bytes:
                        response = JSONResponse(
                            status_code=413,
                            content={
                                "error": {
                                    "code": "payload_too_large",
                                    "message": "Request body exceeds the size limit.",
                                }
                            },
                        )
                        await response(scope, receive, send)
                        return
                except ValueError:
                    pass
                break
        if not self.settings.rate_limit_enabled:
            await self.app(scope, receive, send)
            return
        if self._limiter is None:
            await self._degraded(scope, receive, send, reason="no_backend")
            return
        bucket = _bucket_for(_path(scope), _method(scope))
        identifier = _client_ip(scope)
        try:
            allowed, retry_after = await self._limiter.allowed(bucket, identifier)
        except Exception as exc:  # noqa: BLE001 — Redis outage path is specified
            await self._degraded(scope, receive, send, reason=type(exc).__name__)
            return
        if not allowed:
            logger.warning(
                "rate_limit_exceeded",
                bucket=bucket,
                method=_method(scope),
                route=_path(scope),
            )
            metrics.increment("rate_limited_total", {"bucket": bucket})
            response = JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "code": "rate_limited",
                        "message": "Rate limit exceeded; retry later.",
                    }
                },
                headers={"Retry-After": str(retry_after)},
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)

    async def _degraded(self, scope, receive, send, *, reason: str) -> None:
        """Redis (or backend) unavailable: apply the configured policy loudly.

        ``allow`` keeps serving (availability) while ``deny`` sheds load
        (strictness); both emit a warning log and a metric, so the outage
        is visible instead of silently changing enforcement.
        """
        logger.warning("rate_limiter_degraded", reason=reason)
        metrics.increment(
            "rate_limiter_errors_total",
            {"policy": self.settings.rate_limit_on_redis_error},
        )
        if self.settings.rate_limit_on_redis_error == "deny":
            response = JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "code": "rate_limiter_unavailable",
                        "message": "Rate limiting is temporarily unavailable.",
                    }
                },
                headers={"Retry-After": str(WINDOW_SECONDS)},
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)
