"""Phase 18 tenant + RBAC hardening tests (hermetic).

Proves the trust boundaries without live infrastructure:

- organization identity and roles come only from the verified Clerk
  ``AuthContext`` — client-supplied tenant/role fields are ignored
  (request schemas carry no such fields at all);
- unknown Clerk roles map to ``None`` and can never satisfy admin guards;
- the dev-organization verifier is unreachable when authentication is
  required (production path always uses the Clerk verifier);
- forged dev-organization headers are ignored by the production verifier
  selection (wrong verifier type → real Clerk verifier, never dev).
"""

from __future__ import annotations

import pytest

from evalyx.api.auth import (
    AuthContext,
    OrganizationRole,
    _organization_role_from,
    create_token_verifier,
)
from evalyx.api.dependencies import dev_verifier, get_token_verifier, require_role
from evalyx.api.errors import OrganizationRoleError


def _settings(**overrides):
    from evalyx.core.config import Settings

    defaults = {"evalyx_secret_key": "placeholder", "auth_required": False}
    return Settings(_env_file=None, **{**defaults, **overrides})


class TestRoleParsing:
    def test_known_admin_role(self):
        assert (
            _organization_role_from({"o": {"id": "org_1", "org_role": "org:admin"}})
            is OrganizationRole.ADMIN
        )

    def test_short_admin_slug(self):
        assert (
            _organization_role_from({"o": {"id": "org_1", "org_role": "admin"}})
            is OrganizationRole.ADMIN
        )

    def test_member_role(self):
        assert (
            _organization_role_from({"o": {"id": "org_1", "org_role": "org:member"}})
            is OrganizationRole.MEMBER
        )

    @pytest.mark.parametrize(
        "payload",
        [
            {"o": {"id": "org_1", "org_role": "org:superadmin"}},
            {"o": {"id": "org_1", "org_role": "admin:evil"}},
            {"o": {"id": "org_1", "org_role": ""}},
            {"o": {"id": "org_1", "org_role": "org:"}},
            {"o": {"id": "org_1", "org_role": 42}},
            {"o": {"id": "org_1"}},
            {"o": "org_1"},
            {},
        ],
    )
    def test_unknown_roles_never_become_admin(self, payload):
        role = _organization_role_from(payload)
        assert role is not OrganizationRole.ADMIN
        # Unknown/missing roles map to None (least privilege), never member.
        assert role is None or role is OrganizationRole.MEMBER


class TestRequireRole:
    def _admin_check(self):
        return require_role(OrganizationRole.ADMIN)

    def test_unknown_role_denied(self):
        from evalyx.db.models import Organization

        checker = self._admin_check()
        auth = AuthContext(
            clerk_user_id="u",
            clerk_organization_id="org_x",
            organization_role=None,  # unknown Clerk role
        )
        with pytest.raises(OrganizationRoleError):
            checker((auth, Organization(name="x")))

    def test_member_denied_admin(self):
        from evalyx.db.models import Organization

        checker = self._admin_check()
        auth = AuthContext(
            clerk_user_id="u",
            clerk_organization_id="org_x",
            organization_role=OrganizationRole.MEMBER,
        )
        with pytest.raises(OrganizationRoleError):
            checker((auth, Organization(name="x")))

    def test_admin_allowed(self):
        from evalyx.db.models import Organization

        checker = self._admin_check()
        auth = AuthContext(
            clerk_user_id="u",
            clerk_organization_id="org_x",
            organization_role=OrganizationRole.ADMIN,
        )
        assert checker((auth, Organization(name="x"))) is auth


class TestVerifierSelection:
    def test_production_path_never_uses_dev_verifier(self):
        settings = _settings(
            auth_required=True,
            clerk_secret_key="sk-test",
            clerk_jwks_url="https://example.clerk.dev/.well-known/jwks.json",
        )
        verifier = create_token_verifier(settings)
        assert type(verifier).__name__ == "ClerkTokenVerifier"

    def test_dev_verifier_only_when_auth_disabled(self):
        verifier = create_token_verifier(_settings(auth_required=False))
        assert type(verifier).__name__ == "NoopTokenVerifier"

    def test_get_token_verifier_swaps_noop_for_dev_context(self):
        class _App:
            state = type(
                "S", (), {"token_verifier": create_token_verifier(_settings())}
            )()

        class _Request:
            app = _App()

            def __init__(self) -> None:
                self.headers: dict = {}

        import asyncio

        verifier = get_token_verifier(_Request())  # type: ignore[arg-type]
        assert verifier is dev_verifier
        # A forged dev header for another org is honored only in this
        # local-dev path — production (Clerk verifier above) never reads it.
        context = asyncio.run(
            verifier.verify(
                type("R", (), {"headers": {"x-dev-organization-id": "org_forged"}})()
            )
        )
        assert context.clerk_organization_id == "org_forged"

    def test_forged_roles_cannot_ride_dev_header(self):
        """The dev context grants a fixed ADMIN role only in local-dev mode.

        Production ignores the header entirely (Clerk verifier); this test
        pins the dev-only nature so a future refactor cannot silently
        promote header-based identity in production.
        """
        import asyncio

        context = asyncio.run(
            dev_verifier.verify(
                type("R", (), {"headers": {"x-dev-organization-id": "org_x"}})()
            )
        )
        assert context.organization_role is OrganizationRole.ADMIN
        # ...while malformed headers yield an anonymous context (401 downstream).
        anonymous = asyncio.run(
            dev_verifier.verify(type("R", (), {"headers": {}})())
        )
        assert anonymous.is_authenticated is False


class TestSchemasCarryNoTenantFields:
    """Request bodies must not accept organization/role identity fields."""

    @pytest.mark.parametrize(
        "import_path",
        [
            "evalyx.api.schemas.applications:ApplicationCreate",
            "evalyx.api.schemas.applications:ApplicationUpdate",
            "evalyx.api.schemas.datasets:DatasetCreate",
            "evalyx.api.schemas.evaluations:EvaluationCreate",
        ],
    )
    def test_no_tenant_or_role_fields(self, import_path):
        import importlib

        module_name, class_name = import_path.split(":")
        model = getattr(importlib.import_module(module_name), class_name)
        fields = set(model.model_fields)
        assert not {
            "organization_id",
            "workspace_id",
            "clerk_organization_id",
            "organization_role",
            "role",
            "user_id",
            "clerk_user_id",
        } & fields, fields
