"""API tests: the ``GET /api/v1/me`` local workspace endpoint (offline)."""

from fastapi.testclient import TestClient

from evalyx.api.app import create_app
from evalyx.api.me import describe_caller
from evalyx.core.config import Settings


def _client() -> TestClient:
    from evalyx.api.ratelimit import InMemoryRateLimitBackend

    app = create_app(
        Settings(),
        rate_limit_backend=InMemoryRateLimitBackend(),
    )
    return TestClient(app)


def test_me_endpoint_returns_local_workspace():
    response = _client().get("/api/v1/me")
    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == "local"
    assert body["active_organization"]["organization_id"] == "local"
    assert body["active_organization"]["name"] == "local"


def test_me_response_never_contains_secret_like_fields():
    import asyncio

    body = asyncio.run(describe_caller()).model_dump()
    flattened = str(body).lower()
    assert "token" not in flattened
    assert "secret" not in flattened
    assert "authorization" not in flattened
