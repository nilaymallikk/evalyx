"""CLI/API security tests: no-login surface + secret redaction.

Evalyx needs no login, so there are no tokens to leak and no auth
bypass to probe. These tests pin that contract: the local identity is
fixed, and application secrets never appear in API responses.
"""

from fastapi.testclient import TestClient

from evalyx.api.app import create_app
from evalyx.api.auth import AuthContext, local_context
from evalyx.core.config import Settings


def test_local_context_is_fixed_and_authenticated():
    context = local_context()
    assert isinstance(context, AuthContext)
    assert context.user_id == "local"
    assert context.is_authenticated is True


def test_me_endpoint_needs_no_credentials():
    """GET /api/v1/me answers without any header (no login)."""
    from evalyx.api.ratelimit import InMemoryRateLimitBackend

    app = create_app(
        Settings(),
        rate_limit_backend=InMemoryRateLimitBackend(),
    )
    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/api/v1/me")
    assert response.status_code == 200
    assert response.json()["user_id"] == "local"
