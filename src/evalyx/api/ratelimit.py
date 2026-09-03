"""Production baseline rate limiting (Phase 17).

Simple in-memory fixed-window limiter, per API process. Deliberately small:

- per-IP limits for unauthenticated infrastructure endpoints (/health*)
- authenticated API default limit
- tighter limits for evaluation submission and connection tests

Limits come from Settings (RATE_LIMIT_*). Exceeding the limit returns 429
with a safe envelope — never secrets, never request bodies.

Limitation (documented): in-memory state is per-process, so multi-replica or
multi-worker deployments need an external (Redis-backed) limiter. The
default single-worker production compose keeps this correct.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Final

import structlog
from fastapi.responses import JSONResponse

from evalyx.core.config import Settings

logger = structlog.get_logger(__name__)

_RETRY_AFTER_SECONDS: Final[int] = 60

_EVAL_PATHS: Final[tuple[str, ...]] = ("/api/v1/evaluations",)
_TEST_SUFFIX: Final[str] = "/test"
_HEALTH_PREFIXES: Final[tuple[str, ...]] = ("/health",)


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


class RateLimiter:
    """Fixed-window in-memory rate limiter."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._hits: dict[tuple[str, str], list[float]] = defaultdict(list)

    def limit_for(self, bucket: str) -> int:
        if bucket == "eval":
            return self._settings.rate_limit_eval_per_minute
        if bucket == "test":
            return self._settings.rate_limit_test_per_minute
        return self._settings.rate_limit_per_minute

    def allowed(self, key: tuple[str, str], bucket: str, now: float) -> bool:
        window_start = now - 60.0
        hits = [t for t in self._hits[key] if t > window_start]
        self._hits[key] = hits
        if len(hits) >= self.limit_for(bucket):
            return False
        hits.append(now)
        return True


class RateLimitMiddleware:
    """ASGI middleware enforcing per-IP fixed-window rate limits."""

    def __init__(self, app, settings: Settings) -> None:
        self.app = app
        self.settings = settings
        self.limiter = RateLimiter(settings)

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
        path = _path(scope)
        method = _method(scope)
        bucket = _bucket_for(path, method)
        ip = _client_ip(scope)
        # Health bucket is per-IP (unauthenticated abuse baseline);
        # API buckets are per-IP as a coarse baseline (Phase 18: per-org).
        key = (bucket, ip)
        if not self.limiter.allowed(key, bucket, time.monotonic()):
            logger.warning(
                "rate_limit_exceeded", bucket=bucket, method=method, route=path
            )
            response = JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "code": "rate_limited",
                        "message": "Rate limit exceeded; retry later.",
                    }
                },
                headers={"Retry-After": str(_RETRY_AFTER_SECONDS)},
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)
