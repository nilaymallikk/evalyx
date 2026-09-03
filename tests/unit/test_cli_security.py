"""Phase 16 security tests: dev-mode organization context + leak guards.

All credential-looking strings are fake placeholders built from derived
fragments (never real secrets), mirroring the Phase 14 test suite's
convention.
"""

import pytest
from fastapi.testclient import TestClient

from evalyx.api.app import create_app
from evalyx.api.dependencies import DevOrganizationContext
from evalyx.core.config import Settings

# Derived fake placeholders (same convention as test_auth_api.py) —
# concatenated so they can never match a real credential pattern.
_FAKE_CLERK_SECRET = "sk_test_placeholder_" + "not-a-real-key"
_FAKE_JWKS_URL = "https://instance" + ".clerk.accounts.dev/.well-known/jwks.json"


class _FakeRequest:
    def __init__(self, headers: dict[str, str]):
        self.headers = {k.lower(): v for k, v in headers.items()}


@pytest.mark.asyncio
async def test_dev_context_accepts_bounded_org_header():
    context = await DevOrganizationContext().verify(
        _FakeRequest({"X-Dev-Organization-Id": "org_abc123"})
    )
    assert context.clerk_organization_id == "org_abc123"
    assert context.clerk_user_id == "dev-cli"


@pytest.mark.asyncio
async def test_dev_context_rejects_malformed_header():
    for bad in ("", "drop table", "org_" + "x" * 200, "org_$evil", "other-org"):
        context = await DevOrganizationContext().verify(
            _FakeRequest({"X-Dev-Organization-Id": bad})
        )
        assert context.clerk_organization_id is None, bad


def test_dev_context_is_never_used_in_production_mode():
    """AUTH_REQUIRED=1 constructs ClerkTokenVerifier, never the dev context."""
    from evalyx.api.auth import create_token_verifier

    settings = Settings(
        auth_required=True,
        evalyx_secret_key="x" * 20,
        clerk_secret_key=_FAKE_CLERK_SECRET,
        clerk_jwks_url=_FAKE_JWKS_URL,
    )
    assert type(create_token_verifier(settings)).__name__ == "ClerkTokenVerifier"


def test_dev_org_requests_reach_authenticated_routes():
    """A dev-org request hitting an authenticated route flows through the
    real auth dependency chain (the verifier swap happens in get_token_verifier)."""
    from evalyx.api.ratelimit import InMemoryRateLimitBackend

    app = create_app(
        Settings(auth_required=False),
        rate_limit_backend=InMemoryRateLimitBackend(),
    )
    client = TestClient(app, raise_server_exceptions=False)
    # /api/v1/me is authenticated-user-only (works without an org).
    response = client.get(
        "/api/v1/me", headers={"X-Dev-Organization-Id": "org_devtest"}
    )
    assert response.status_code == 200
    assert response.json()["clerk_user_id"] == "dev-cli"
