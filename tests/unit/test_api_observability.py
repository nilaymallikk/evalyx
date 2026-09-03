"""API observability unit tests over the real app (no live infrastructure).

Covers: X-Request-ID generation/reuse/rejection, lifecycle logging, HTTP
metrics with bounded route-template labels, slow-request warnings, and the
500 path. Sensitive values in assertions are fake by construction.
"""

import uuid

from fastapi.testclient import TestClient
from structlog.testing import capture_logs

from evalyx.api.app import create_app
from evalyx.api.auth import AuthContext, OrganizationRole
from evalyx.api.dependencies import get_session, require_organization
from evalyx.core.config import Settings
from evalyx.core.metrics import metrics
from evalyx.db.models import Organization


def build_client(settings: Settings | None = None) -> TestClient:
    # No lifespan execution: no database/Redis connections are opened.
    # Tenant authentication is bypassed (observability is auth-agnostic);
    # Clerk behavior has its own suite (tests/unit/test_auth_api.py).
    # Explicit in-memory rate-limit backend: hermetic tests never touch
    # real Redis (production always uses the Redis backend).
    from evalyx.api.ratelimit import InMemoryRateLimitBackend

    app = create_app(
        settings or Settings(auth_required=False),
        rate_limit_backend=InMemoryRateLimitBackend(),
    )
    fake_context = (
        AuthContext(
            clerk_user_id="unit-test-user",
            clerk_organization_id="org_unit_test",
            organization_role=OrganizationRole.ADMIN,
        ),
        Organization(name="Unit Test Org"),
    )
    app.dependency_overrides[require_organization] = lambda: fake_context
    return TestClient(app)


def setup_function(_: object) -> None:
    metrics.reset()


def teardown_function(_: object) -> None:
    metrics.reset()


# -- X-Request-ID --------------------------------------------------------------


def test_response_always_contains_request_id():
    client = build_client()
    response = client.get("/health")
    assert response.status_code == 200
    request_id = response.headers["X-Request-ID"]
    assert uuid.UUID(request_id).version == 4


def test_each_request_gets_a_distinct_request_id():
    client = build_client()
    first = client.get("/health").headers["X-Request-ID"]
    second = client.get("/health").headers["X-Request-ID"]
    assert first != second


def test_valid_client_request_id_is_preserved():
    client = build_client()
    response = client.get("/health", headers={"X-Request-ID": "test-observability-123"})
    assert response.headers["X-Request-ID"] == "test-observability-123"


def test_oversized_client_request_id_is_replaced():
    client = build_client()
    response = client.get("/health", headers={"X-Request-ID": "x" * 500})
    request_id = response.headers["X-Request-ID"]
    assert request_id != "x" * 500
    assert uuid.UUID(request_id).version == 4


def test_invalid_client_request_id_is_replaced():
    client = build_client()
    response = client.get("/health", headers={"X-Request-ID": "bad id with spaces/../"})
    request_id = response.headers["X-Request-ID"]
    assert request_id != "bad id with spaces/../"
    assert uuid.UUID(request_id).version == 4


def test_error_responses_carry_request_id():
    client = build_client()
    not_found = client.get("/api/v1/applications/not-a-uuid")  # UUID validation
    assert not_found.status_code == 422
    assert "X-Request-ID" in not_found.headers
    validation = client.post("/api/v1/applications", json={"name": ""})
    assert validation.status_code == 422
    assert "X-Request-ID" in validation.headers


def test_unexpected_500_response_still_carries_request_id():
    """§35/§58: the client gets a generic 500 but the correlation header."""
    from evalyx.api.ratelimit import InMemoryRateLimitBackend

    app = create_app(Settings(), rate_limit_backend=InMemoryRateLimitBackend())

    async def boom_session() -> object:
        raise RuntimeError("internal probe failure")
        yield  # pragma: no cover

    app.dependency_overrides[get_session] = boom_session
    client = TestClient(app, raise_server_exceptions=False)  # server-like behavior
    try:
        response = client.get("/api/v1/evaluations")
        assert response.status_code == 500
        assert response.json()["error"]["code"] == "internal_error"
        assert response.json()["error"]["message"] == "An unexpected internal error occurred."
        assert "X-Request-ID" in response.headers
        assert "RuntimeError" not in response.text
        assert "internal probe failure" not in response.text
    finally:
        app.dependency_overrides.clear()


# -- structured request logging -------------------------------------------------


def test_request_lifecycle_logged_with_route_template():
    client = build_client()
    with capture_logs() as logs:
        response = client.get("/health", headers={"X-Request-ID": "log-check-1"})
    assert response.status_code == 200
    events = [e for e in logs if e["event"] == "http_request_completed"]
    assert len(events) == 1
    event = events[0]
    assert event["request_id"] == "log-check-1"
    assert event["method"] == "GET"
    assert event["route"] == "/health"
    assert event["status_code"] == 200
    assert isinstance(event["duration_ms"], float)
    assert event["duration_ms"] >= 0


def test_unmatched_route_logged_as_unmatched_not_concrete_path():
    client = build_client()
    with capture_logs() as logs:
        client.get("/api/v1/does-not-exist/3f2c1a6b-1234-5678-90ab-cdef12345678")
    events = [e for e in logs if e["event"] == "http_request_completed"]
    assert events[0]["route"] == "/unmatched"
    assert "3f2c1a6b" not in events[0]["route"]


def test_request_logs_never_contain_secrets_or_bodies():
    """§39/§40: observability selects safe fields — bodies/headers/PII never
    appear even when the client sends them."""
    client = build_client()
    fake_pii = "jane.doe@example.com"
    fake_secret = "fake-secret-never-log"
    with capture_logs() as logs:
        client.post(
            "/api/v1/applications",
            json={"name": fake_pii, "description": fake_secret},
            headers={"Authorization": f"Bearer {fake_secret}", "X-Request-ID": "sec-check-1"},
        )
    serialized = str(logs)
    assert fake_pii not in serialized
    assert fake_secret not in serialized
    assert "Bearer" not in serialized
    assert all(e["event"] != "http_request_body" for e in logs)


# -- HTTP metrics ----------------------------------------------------------------


def test_http_metrics_recorded_with_bounded_labels():
    client = build_client()
    client.get("/health")
    client.get("/health")
    client.post("/api/v1/applications", json={})  # 422
    snap = metrics.snapshot()
    counter = snap["http_requests_total"]
    labels_seen = [e["labels"] for e in counter]
    assert {"method": "GET", "route": "/health", "status": "200"} in labels_seen
    assert {"method": "POST", "route": "/api/v1/applications", "status": "422"} in labels_seen
    total = sum(e["value"] for e in counter)
    assert total == 3.0
    timing = snap["http_request_duration_ms"]
    assert all("method" in e["labels"] and "route" in e["labels"] for e in timing)


def test_http_metrics_never_use_ids_as_labels():
    """§38/§19: no request_id (or any correlation id) in metric labels."""
    client = build_client()
    with capture_logs():
        client.get("/health", headers={"X-Request-ID": "metric-check-1"})
        client.get("/api/v1/applications/not-a-uuid")
    snap = metrics.snapshot()
    serialized = str(snap)
    assert "metric-check-1" not in serialized
    for forbidden in ("request_id", "run_id", "task_id"):
        assert f"'{forbidden}'" not in serialized
        assert f'"{forbidden}"' not in serialized


def test_http_metrics_use_route_templates_for_dynamic_paths():
    client = build_client()
    client.get("/api/v1/applications/3f2c1a6b-1234-5678-90ab-cdef12345678")  # 422
    snap = metrics.snapshot()
    routes = {e["labels"]["route"] for e in snap["http_requests_total"]}
    # The matched template is bounded; no UUID ever becomes a label.
    assert routes == {"/api/v1/applications/{application_id}"}
    assert "3f2c1a6b" not in str(snap)


# -- slow request threshold --------------------------------------------------------


def test_slow_request_warning_emitted_above_threshold():
    settings = Settings(slow_request_threshold_ms=1)
    client = build_client(settings)
    with capture_logs() as logs:
        client.get("/health")
    warnings = [e for e in logs if e["event"] == "http_request_slow"]
    assert len(warnings) == 1
    assert warnings[0]["threshold_ms"] == 1
    assert warnings[0]["duration_ms"] >= 1


def test_fast_request_does_not_warn():
    settings = Settings(slow_request_threshold_ms=60_000)
    client = build_client(settings)
    with capture_logs() as logs:
        client.get("/health")
    assert all(e["event"] != "http_request_slow" for e in logs)


# -- health/readiness observability ------------------------------------------------


def test_health_endpoints_preserved():
    client = build_client()
    assert client.get("/health").json() == {"status": "ok"}
    # /health/ready would attempt connections; without lifespan (no real
    # clients) it still must return a structured payload.
    response = client.get("/health/ready")
    assert response.status_code in (200, 503)
    body = response.json()
    assert set(body["dependencies"]) == {"database", "redis"}


def test_request_started_logged_at_debug_level():
    client = build_client(Settings(log_level="DEBUG"))
    with capture_logs() as logs:
        client.get("/health")
    assert any(e["event"] == "http_request_started" for e in logs)
