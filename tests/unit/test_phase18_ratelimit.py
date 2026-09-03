"""Phase 18 distributed rate-limit tests (hermetic + live Redis).

Hermetic tests use the explicit in-memory fake backend (a test double, never
a production fallback). Live-Redis tests verify cross-"replica" atomicity:
two limiter instances sharing one Redis cannot jointly exceed the limit.
"""

from __future__ import annotations

import asyncio
import time

import pytest
import redis.asyncio as redis_asyncio
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from evalyx.api.ratelimit import (
    InMemoryRateLimitBackend,
    RateLimiter,
    RedisRateLimitBackend,
    _bucket_for,
    rate_limit_key,
    retry_after_seconds,
    sanitize_identifier,
)

# Async tests run under asyncio auto mode (see pyproject); live-Redis tests
# carry the `integration` mark and are skipped without live services.


def _settings(**overrides):
    from evalyx.core.config import Settings

    defaults = {"evalyx_secret_key": "placeholder", "auth_required": False}
    return Settings(_env_file=None, **{**defaults, **overrides})


class TestKeyHygiene:
    def test_buckets_stable(self):
        assert _bucket_for("/health", "GET") == "health"
        assert _bucket_for("/health/ready", "GET") == "health"
        assert _bucket_for("/api/v1/evaluations", "POST") == "eval"
        assert _bucket_for("/api/v1/applications/1/test", "POST") == "test"
        assert _bucket_for("/api/v1/me", "GET") == "default"

    def test_identifier_sanitized_and_bounded(self):
        assert sanitize_identifier("127.0.0.1") == "127.0.0.1"
        assert sanitize_identifier("::1") == "__1"
        assert sanitize_identifier("a" * 200) == "a" * 64
        assert sanitize_identifier("") == "unknown"
        hostile = "../../../etc:passwd\n"
        clean = sanitize_identifier(hostile)
        assert "/" not in clean and "\n" not in clean and ":" not in clean

    def test_key_shape_bounded(self):
        key = rate_limit_key("evalyx:rl", "default", "::ffff:10.0.0.1", 1_700_000_000.0)
        assert key.startswith("evalyx:rl:default:")
        assert len(key) < 200
        assert "\n" not in key and " " not in key

    def test_retry_after_positive(self):
        assert 1 <= retry_after_seconds(time.time()) <= 60


class TestFakeBackendSemantics:
    async def test_fixed_window(self):
        limiter = RateLimiter(
            _settings(rate_limit_per_minute=2), InMemoryRateLimitBackend()
        )
        assert await limiter.allowed("default", "1.1.1.1") == (True, 0)
        assert await limiter.allowed("default", "1.1.1.1") == (True, 0)
        allowed, retry_after = await limiter.allowed("default", "1.1.1.1")
        assert allowed is False and retry_after >= 1

    async def test_concurrent_admission_respects_limit(self):
        """N concurrent callers against one backend admit at most `limit`."""
        limiter = RateLimiter(
            _settings(rate_limit_per_minute=5), InMemoryRateLimitBackend()
        )
        results = await asyncio.gather(
            *(limiter.allowed("default", "9.9.9.9") for _ in range(20))
        )
        assert sum(1 for allowed, _ in results if allowed) == 5

    async def test_buckets_have_independent_budgets(self):
        limiter = RateLimiter(
            _settings(rate_limit_per_minute=1, rate_limit_eval_per_minute=1),
            InMemoryRateLimitBackend(),
        )
        assert (await limiter.allowed("default", "1.2.3.4"))[0] is True
        assert (await limiter.allowed("eval", "1.2.3.4"))[0] is True
        assert (await limiter.allowed("default", "1.2.3.4"))[0] is False


class _FailingBackend:
    async def check_and_increment(self, key, limit, window_seconds):
        raise ConnectionError("redis down")


def _middleware_app(settings, backend):
    from evalyx.api.ratelimit import RateLimitMiddleware

    inner = FastAPI()

    @inner.get("/health")
    async def health():
        return JSONResponse({"status": "ok"})

    return RateLimitMiddleware(inner, settings, backend=backend)


def _call(app, path="/health"):
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "query_string": b"",
        "headers": [],
        "client": ("10.0.0.9", 1234),
        "server": ("test", 80),
    }
    messages = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)

    import anyio

    anyio.run(app, scope, receive, send)
    statuses = [m["status"] for m in messages if m["type"] == "http.response.start"]
    return statuses[0] if statuses else None


class TestMiddlewareOutagePolicy:
    def test_redis_outage_fail_open_is_loud_but_serving(self):
        settings = _settings(
            rate_limit_per_minute=1, rate_limit_on_redis_error="allow"
        )
        app = _middleware_app(settings, _FailingBackend())
        assert _call(app) == 200  # serving, not silently limited

    def test_redis_outage_deny_sheds_load(self):
        settings = _settings(
            rate_limit_per_minute=1000, rate_limit_on_redis_error="deny"
        )
        app = _middleware_app(settings, _FailingBackend())
        assert _call(app) == 503

    def test_no_backend_degrades_explicitly(self):
        from evalyx.api.ratelimit import RateLimitMiddleware

        inner = FastAPI()

        @inner.get("/health")
        async def health():
            return JSONResponse({"status": "ok"})

        app = RateLimitMiddleware(inner, _settings(), backend=None, redis_client=None)
        assert _call(app) == 200


class TestLiveRedisBackend:
    """Live-Redis atomicity: two limiter instances, one shared budget."""

    @pytest.mark.integration
    async def test_shared_budget_across_replicas(self):
        import uuid

        settings = _settings(
            rate_limit_per_minute=7,
            rate_limit_redis_prefix=f"evalyx:test18-rl:{uuid.uuid4().hex[:8]}",
        )
        client = redis_asyncio.Redis.from_url("redis://localhost:6379/0")
        try:
            replica_a = RateLimiter(settings, RedisRateLimitBackend(client))
            replica_b = RateLimiter(settings, RedisRateLimitBackend(client))
            calls = []
            for i in range(10):
                calls.append(replica_a.allowed("default", "203.0.113.7"))
                calls.append(replica_b.allowed("default", "203.0.113.7"))
            results = await asyncio.gather(*calls)
            assert sum(1 for allowed, _ in results if allowed) == 7
        finally:
            await client.aclose()

    @pytest.mark.integration
    async def test_window_key_has_ttl(self):
        import uuid

        prefix = f"evalyx:test18-ttl:{uuid.uuid4().hex[:8]}"
        client = redis_asyncio.Redis.from_url(
            "redis://localhost:6379/0", decode_responses=True
        )
        try:
            backend = RedisRateLimitBackend(client)
            key = rate_limit_key(prefix, "default", "198.51.100.3", time.time())
            allowed, _ = await backend.check_and_increment(key, 100, 60)
            assert allowed is True
            ttl = await client.ttl(key)
            assert 1 <= ttl <= 60
        finally:
            await client.aclose()

    def test_testclient_429_envelope(self):
        from evalyx.api.app import create_app

        settings = _settings(rate_limit_per_minute=1)
        app = create_app(settings, rate_limit_backend=InMemoryRateLimitBackend())
        client = TestClient(app)
        assert client.get("/health").status_code == 200
        response = client.get("/health")
        assert response.status_code == 429
        assert response.json()["error"]["code"] == "rate_limited"
        assert "Retry-After" in response.headers
