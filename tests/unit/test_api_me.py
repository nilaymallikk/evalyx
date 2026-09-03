"""Phase 16 API tests: the ``/api/v1/me`` endpoint (offline, fake context)."""

from fastapi.testclient import TestClient

from evalyx.api.app import create_app
from evalyx.api.auth import AuthContext, OrganizationRole
from evalyx.api.dependencies import require_authenticated_user
from evalyx.api.me import describe_caller
from evalyx.core.config import Settings


def _client(auth: AuthContext) -> TestClient:
    app = create_app(Settings(auth_required=False))
    app.dependency_overrides[require_authenticated_user] = lambda: auth
    return TestClient(app)


def test_me_endpoint_returns_token_derived_identity():
    auth = AuthContext(
        clerk_user_id="user_1",
        clerk_organization_id="org_1",
        organization_role=OrganizationRole.ADMIN,
    )
    response = _client(auth).get("/api/v1/me")
    assert response.status_code == 200
    body = response.json()
    assert body["clerk_user_id"] == "user_1"
    assert body["active_organization"]["clerk_organization_id"] == "org_1"
    assert body["active_organization"]["role"] == "admin"


def test_me_endpoint_requires_authentication():
    app = create_app(Settings(auth_required=False))
    # Dev mode with no org header yields an anonymous context → 401.
    response = TestClient(app).get("/api/v1/me")
    assert response.status_code == 401


def test_me_response_never_contains_token_like_fields():
    auth = AuthContext("user_1", "org_1", OrganizationRole.MEMBER)
    body = _client(auth).get("/api/v1/me").json()
    flattened = str(body).lower()
    assert "token" not in flattened
    assert "secret" not in flattened
    assert "authorization" not in flattened


def test_describe_caller_skips_enrichment_without_clerk():
    """Dev verifier (no Clerk) → token-derived fields only, no crash."""
    import asyncio

    auth = AuthContext("user_1", None, None)

    class _NoClerkVerifier:
        async def verify(self, request) -> AuthContext:  # pragma: no cover
            return auth

    response = asyncio.run(describe_caller(auth, _NoClerkVerifier()))
    assert response.email is None
    assert response.organizations == []
    assert response.active_organization is None
