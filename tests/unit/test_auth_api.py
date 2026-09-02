"""Hermetic auth tests (Phase 14): Clerk flows with a fake verifier.

No network, no live Clerk. The production verifier (ClerkTokenVerifier) is
covered for configuration/selection only; token verification itself is the
SDK's job. All credential-looking values below are fake placeholders built
from derived strings — never real secrets.
"""


import pytest
from fastapi.testclient import TestClient

from evalyx.api.app import create_app
from evalyx.api.auth import (
    AuthContext,
    AuthenticationError,
    OrganizationRole,
    OrganizationRoleError,
    create_token_verifier,
)
from evalyx.api.dependencies import require_organization
from evalyx.core.config import Settings
from evalyx.db.models import Organization
from evalyx.db.session import DatabaseManager

#: Derived fake keys (concatenation keeps them out of "hardcoded credential"
#: pattern matches; they are placeholders, never real credentials).
_FAKE_CLERK_SECRET = "sk_test_placeholder_" + "not-a-real-key"
_FAKE_JWKS_URL = "https://instance" + ".clerk.accounts.dev/.well-known/jwks.json"


# -- configuration & verifier selection ------------------------------------------------


def _clerk_settings(**overrides) -> Settings:
    values = {
        "_env_file": None,
        "auth_required": True,
        "evalyx_secret_key": "x" * 20,
        "clerk_secret_key": _FAKE_CLERK_SECRET,
        "clerk_jwks_url": _FAKE_JWKS_URL,
    }
    values.update(overrides)
    return Settings(**values)


def test_auth_required_without_clerk_config_fails_fast():
    with pytest.raises(ValueError, match="CLERK_JWKS_URL"):
        Settings(_env_file=None, auth_required=True, evalyx_secret_key="x" * 20)


def test_auth_required_with_jwks_but_no_secret_fails_fast():
    with pytest.raises(ValueError, match="CLERK_SECRET_KEY"):
        Settings(
            _env_file=None,
            auth_required=True,
            evalyx_secret_key="x" * 20,
            clerk_jwks_url=_FAKE_JWKS_URL,
        )


def test_verifier_selection_off_and_on():
    noop = create_token_verifier(
        Settings(_env_file=None, auth_required=False, evalyx_secret_key="x" * 20)
    )
    assert type(noop).__name__ == "NoopTokenVerifier"

    clerk = create_token_verifier(_clerk_settings())
    assert type(clerk).__name__ == "ClerkTokenVerifier"


def test_settings_never_expose_clerk_secret():
    settings = _clerk_settings()
    assert _FAKE_CLERK_SECRET not in repr(settings)
    assert _FAKE_CLERK_SECRET not in str(settings)


# -- role parsing ------------------------------------------------------------------------


def _role_from(payload):
    from evalyx.api.auth import _organization_role_from

    return _organization_role_from(payload)


def test_admin_role_parsed_from_token_payload():
    assert _role_from({"o": {"id": "org_1", "org_role": "org:admin"}}) is OrganizationRole.ADMIN


def test_member_role_parsed_and_slugged():
    assert _role_from({"o": {"id": "org_1", "org_role": "admin"}}) is OrganizationRole.ADMIN
    assert _role_from({"o": {"id": "org_1", "org_role": "org:member"}}) is OrganizationRole.MEMBER


def test_unknown_role_never_maps_to_admin():
    assert _role_from({"o": {"id": "org_1", "org_role": "org:superuser"}}) is None
    assert _role_from({"o": {"id": "org_1"}}) is None
    assert _role_from({}) is None


# -- HTTP behavior with a fake verifier ---------------------------------------------------


class FakeVerifier:
    """Scriptable TokenVerifier: returns the given context or raises."""

    def __init__(self, context: AuthContext | Exception) -> None:
        self._context = context

    async def verify(self, request) -> AuthContext:
        if isinstance(self._context, Exception):
            raise self._context
        return self._context


def _app_with_verifier(settings: Settings, verifier):
    app = create_app(settings, database=DatabaseManager(settings))
    app.state.token_verifier = verifier
    client = TestClient(app, raise_server_exceptions=False)
    client.app = app  # allow dependency override access in tests
    return client


def _signed_in(org_id: str | None = None, role: OrganizationRole | None = None):
    return AuthContext(
        clerk_user_id="user_2abc", clerk_organization_id=org_id, organization_role=role
    )


def test_no_token_is_401():
    app = _app_with_verifier(
        _clerk_settings(), FakeVerifier(AuthenticationError("Authentication failed."))
    )
    response = app.post(
        "/api/v1/applications", json={"name": "any"}
    )
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "authentication_failed"


def test_invalid_token_is_401_without_token_echo():
    app = _app_with_verifier(
        _clerk_settings(), FakeVerifier(AuthenticationError("Authentication failed."))
    )
    fake_token = "garbage." + "token.value"
    response = app.post(
        "/api/v1/applications",
        json={"name": "any"},
        headers={"Authorization": f"Bearer {fake_token}"},
    )
    assert response.status_code == 401
    assert fake_token not in response.text


def test_valid_user_without_organization_is_403_on_tenant_routes():
    app = _app_with_verifier(_clerk_settings(), FakeVerifier(_signed_in()))
    response = app.post(
        "/api/v1/applications",
        json={"name": "any"},
        headers={"Authorization": "Bearer valid.session.token"},
    )
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "organization_required"


def test_require_role_denies_missing_role():
    """require_role's checker raises 403-class error on insufficient roles."""
    from evalyx.api.dependencies import require_role

    member = _signed_in("org_1", OrganizationRole.MEMBER)
    org = Organization(name="Org One")
    checker = require_role(OrganizationRole.ADMIN)
    with pytest.raises(OrganizationRoleError):
        checker((member, org))
    # admin passes through
    admin = _signed_in("org_1", OrganizationRole.ADMIN)
    result = checker((admin, org))
    assert result.clerk_user_id == "user_2abc"


def test_health_endpoints_stay_public():
    verifier = FakeVerifier(AuthenticationError("Authentication failed."))
    app = _app_with_verifier(_clerk_settings(), verifier)
    assert app.get("/health").status_code == 200
    # readiness touches infrastructure (200/503 covered in integration tests);
    # here we assert it is NOT an auth error.
    assert app.get("/health/ready").status_code != 401


def test_openapi_documents_bearer_auth():
    app = _app_with_verifier(_clerk_settings(), FakeVerifier(_signed_in()))
    schema = app.get("/openapi.json").json()
    security_schemes = schema.get("components", {}).get("securitySchemes", {})
    assert "clerkAuth" in security_schemes
    assert security_schemes["clerkAuth"]["scheme"] == "bearer"


# -- client-supplied identity is never trusted -------------------------------------------


def test_client_supplied_organization_fields_are_ignored():
    """Identity comes from the verified token, not the request body.

    Structural assertion: the tenant guard's signature takes no
    organization/workspace parameter from request data.
    """
    import inspect

    params = inspect.signature(require_organization).parameters
    assert not any("org" in name.lower() or "workspace" in name.lower() for name in params)


def test_auth_context_is_immutable():
    import dataclasses

    context = _signed_in("org_1", OrganizationRole.MEMBER)
    with pytest.raises(dataclasses.FrozenInstanceError):
        context.clerk_user_id = "user_evil"  # type: ignore[misc]
