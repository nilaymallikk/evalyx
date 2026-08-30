"""Observability unit tests: correlation context, metrics registry, request-id
resolution, provider retry logging, and worker/task correlation.

Hermetic: no database, no Redis, no HTTP server. Sensitive values used here
are fake by construction (e.g. "fake-secret-never-log").
"""

import asyncio
import threading
import uuid

import httpx
import pytest
import structlog
from structlog.testing import capture_logs

from evalyx.core import context as correlation
from evalyx.core.metrics import MetricsRegistry, metrics
from evalyx.llm.base import RetryPolicy, send_with_retries

# -- correlation context -------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_context():
    yield
    correlation.clear_correlation_context()


def test_context_set_and_get_roundtrip():
    correlation.set_request_id("req-1")
    correlation.set_run_id("run-1")
    correlation.set_task_id("task-1")
    assert correlation.get_request_id() == "req-1"
    assert correlation.get_run_id() == "run-1"
    assert correlation.get_task_id() == "task-1"


def test_context_defaults_to_none():
    correlation.clear_correlation_context()
    assert correlation.get_request_id() is None
    assert correlation.get_run_id() is None
    assert correlation.get_task_id() is None


def test_context_clear_removes_all_ids():
    correlation.set_request_id("a")
    correlation.set_run_id("b")
    correlation.set_task_id("c")
    correlation.clear_correlation_context()
    assert correlation.get_request_id() is None
    assert correlation.get_run_id() is None
    assert correlation.get_task_id() is None


def test_context_bound_values_flow_into_structured_logs():
    """The first processor in the configured chain is merge_contextvars;
    bound ids must therefore appear in every event dict rendered from a
    request/task context. (capture_logs bypasses processors, so the
    processor is exercised directly here.)"""
    correlation.set_request_id("req-42")
    correlation.set_run_id("run-42")
    merged = structlog.contextvars.merge_contextvars(
        None, "info", {"event": "some_event", "extra": "safe-field"}
    )
    assert merged == {
        "event": "some_event",
        "extra": "safe-field",
        "request_id": "req-42",
        "run_id": "run-42",
    }


def test_context_does_not_leak_across_async_tasks():
    """Explicit leak test: two concurrent coroutines must not see each
    other's correlation context (§31/§32)."""
    seen: dict[str, str | None] = {}

    async def worker(name: str) -> None:
        correlation.set_request_id(name)
        await asyncio.sleep(0.01)  # yield mid-task: context switch happens
        seen[name] = correlation.get_request_id()

    async def main() -> None:
        await asyncio.gather(worker("alpha"), worker("beta"))

    asyncio.run(main())
    assert seen == {"alpha": "alpha", "beta": "beta"}


def test_context_does_not_leak_across_threads():
    """Worker threads reuse a thread pool; context must be thread-local."""
    results: dict[str, str | None] = {}
    # Both threads must have bound their own id before either reads.
    barrier = threading.Barrier(2, timeout=5)

    def worker(name: str) -> None:
        correlation.set_request_id(name)
        barrier.wait()  # both threads bound now
        results[name] = correlation.get_request_id()

    threads = [threading.Thread(target=worker, args=(n,)) for n in ("t1", "t2")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
    assert results == {"t1": "t1", "t2": "t2"}


# -- metrics registry ----------------------------------------------------------


@pytest.fixture
def registry() -> MetricsRegistry:
    return MetricsRegistry()


def test_counter_starts_at_zero(registry: MetricsRegistry):
    snap = registry.snapshot()
    assert snap == {}


def test_counter_increment_and_accumulate(registry: MetricsRegistry):
    registry.increment("http_requests_total", {"method": "GET", "route": "/x", "status": "200"})
    registry.increment("http_requests_total", {"method": "GET", "route": "/x", "status": "200"})
    registry.increment("http_requests_total", {"method": "GET", "route": "/x", "status": "200"})
    snap = registry.snapshot()
    assert snap["http_requests_total"] == [
        {"labels": {"method": "GET", "route": "/x", "status": "200"}, "value": 3.0}
    ]


def test_counter_labels_deterministic_order(registry: MetricsRegistry):
    registry.increment("m", {"b": "2", "a": "1"})
    snap = registry.snapshot()
    assert snap["m"][0]["labels"] == {"a": "1", "b": "2"}  # sorted by key


def test_snapshot_is_deterministic(registry: MetricsRegistry):
    registry.increment("m", {"b": "2", "a": "1"})
    registry.increment("m", {"a": "9"})
    registry.observe("t", 12.5, {"route": "/x"})
    first = registry.snapshot()
    second = registry.snapshot()
    assert first == second
    assert list(first.keys()) == sorted(first.keys())


def test_timing_observation_stats(registry: MetricsRegistry):
    registry.observe("evaluation_run_duration_ms", 100.0, {"status": "completed"})
    registry.observe("evaluation_run_duration_ms", 300.0, {"status": "completed"})
    snap = registry.snapshot()
    entry = snap["evaluation_run_duration_ms"][0]
    assert entry["count"] == 2
    assert entry["total_ms"] == 400.0
    assert entry["max_ms"] == 300.0
    assert entry["avg_ms"] == 200.0


def test_registry_separate_label_series(registry: MetricsRegistry):
    registry.increment("worker_tasks_total", {"task": "run_evaluation", "outcome": "success"})
    registry.increment("worker_tasks_total", {"task": "run_evaluation", "outcome": "failed"})
    snap = registry.snapshot()
    values = {e["labels"]["outcome"]: e["value"] for e in snap["worker_tasks_total"]}
    assert values == {"success": 1.0, "failed": 1.0}


def test_registry_rejects_correlation_ids_as_labels(registry: MetricsRegistry):
    """§38: request_id/run_id/task_id must never be metric labels."""
    for forbidden in ("request_id", "run_id", "task_id", "case_id", "comparison_id"):
        with pytest.raises(ValueError, match="forbidden"):
            registry.increment("some_metric", {forbidden: "abc"})
        with pytest.raises(ValueError, match="forbidden"):
            registry.observe("some_metric", 1.0, {forbidden: "abc"})
    snap = registry.snapshot()
    assert snap == {}  # nothing was recorded


def test_registry_reset_for_tests(registry: MetricsRegistry):
    registry.increment("m", {"a": "1"})
    registry.observe("t", 5.0)
    registry.reset()
    assert registry.snapshot() == {}


def test_registry_thread_safety(registry: MetricsRegistry):
    """Concurrent increments from many threads must not lose updates."""

    def hammer() -> None:
        for _ in range(500):
            registry.increment("thread_metric", {"worker": "w"})
            registry.observe("thread_timing", 1.0, {"worker": "w"})

    threads = [threading.Thread(target=hammer) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    snap = registry.snapshot()
    assert snap["thread_metric"][0]["value"] == 8 * 500
    assert snap["thread_timing"][0]["count"] == 8 * 500


def test_global_metrics_registry_available():
    """The shared registry is importable from both API and worker code."""
    assert isinstance(metrics, MetricsRegistry)


# -- request id resolution -----------------------------------------------------


def test_resolve_request_id_generates_uuid4_when_absent():
    from evalyx.api.middleware import resolve_request_id

    request_id, origin = resolve_request_id(None)
    assert origin == "generated"
    assert uuid.UUID(request_id).version == 4


def test_resolve_request_id_reuses_valid_client_id():
    from evalyx.api.middleware import resolve_request_id

    request_id, origin = resolve_request_id("test-observability-123")
    assert request_id == "test-observability-123"
    assert origin == "client"


def test_resolve_request_id_rejects_oversized_without_logging_value(capsys):
    from evalyx.api.middleware import resolve_request_id

    evil = "x" * 500
    with capture_logs() as logs:
        request_id, origin = resolve_request_id(evil)
    assert origin == "generated"
    assert uuid.UUID(request_id).version == 4
    # The invalid value itself is never logged.
    assert all(evil not in str(entry) for entry in logs)
    reasons = [entry.get("reason") for entry in logs if entry["event"] == "request_id_rejected"]
    assert reasons == ["oversized"]


def test_resolve_request_id_rejects_invalid_characters():
    from evalyx.api.middleware import resolve_request_id

    request_id, origin = resolve_request_id("bad id with spaces/../etc")
    assert origin == "generated"
    assert request_id != "bad id with spaces/../etc"


def test_route_template_uses_matched_route_not_concrete_path():
    from evalyx.api.middleware import UNMATCHED_ROUTE, route_template

    class FakeRoute:
        path = "/api/v1/evaluations/{run_id}"

    matched = route_template({"route": FakeRoute()})
    assert matched == "/api/v1/evaluations/{run_id}"  # template, not a UUID path
    assert route_template({}) == UNMATCHED_ROUTE
    assert route_template({"route": object()}) == UNMATCHED_ROUTE


# -- provider retry logging (safe fields only) ---------------------------------


async def test_provider_retry_log_contains_safe_fields_only():
    """§28: retries log provider/model/attempt/error type — never bodies."""
    policy = RetryPolicy(max_retries=1, backoff_seconds=0.0)
    calls = {"n": 0}

    async def send() -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("connection refused to fake-host")
        return httpx.Response(200, json={"ok": True})

    with capture_logs() as logs:
        response, _ = await send_with_retries(
            send, policy, provider="openrouter", model="fake/model"
        )
    assert response.status_code == 200
    retries = [e for e in logs if e["event"] == "provider_retry_scheduled"]
    assert len(retries) == 1
    event = retries[0]
    assert event["provider"] == "openrouter"
    assert event["model"] == "fake/model"
    assert event["attempt"] == 1
    assert event["max_attempts"] == 2
    assert event["error_type"] == "ConnectError"
    # No raw exception text (could embed response bodies) in the retry event.
    assert "connection refused to fake-host" not in str(event)


async def test_provider_retry_honors_retry_after_header():
    """429 with Retry-After surfaces as retry_after_seconds (numeric only)."""
    policy = RetryPolicy(max_retries=1, backoff_seconds=0.0)
    calls = {"n": 0}

    async def send() -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(200, json={"ok": True})

    with capture_logs() as logs:
        await send_with_retries(send, policy, provider="openrouter", model="fake/model")
    event = next(e for e in logs if e["event"] == "provider_retry_scheduled")
    assert event["error_type"] == "http_429"
    assert event["retry_after_seconds"] == 0.0


@pytest.fixture
def clean_metrics():
    metrics.reset()
    yield
    metrics.reset()
