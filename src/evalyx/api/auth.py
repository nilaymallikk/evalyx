"""Authentication: Clerk-backed identity and tenant context (Phase 14).

Clerk owns authentication, user identity, organizations, memberships, and
roles. Evalyx owns domain data and tenant scoping. This module is the only
place that knows *how* a Clerk token is verified; the rest of Evalyx sees
only the immutable :class:`AuthContext`.

Verification uses the official Clerk backend SDK's ``authenticate_request``
(local JWKS public-key verification — no Clerk API round-trip per request,
no token content ever logged or returned).
"""

import enum
from dataclasses import dataclass
from typing import Any, Protocol

from clerk_backend_api import Clerk
from clerk_backend_api.security import AuthenticateRequestOptions
from clerk_backend_api.security.types import AuthStatus, Requestish
from fastapi import Request

from evalyx.core.config import Settings


class AuthenticationError(Exception):
    """A request failed authentication (401).

    Safe by construction: the message names the failure *kind* only — never
    token contents, Clerk responses, or credentials.
    """


class OrganizationRequiredError(Exception):
    """An authenticated user has no active organization (403)."""


class OrganizationRoleError(Exception):
    """The active organization role is insufficient for the operation (403)."""


class OrganizationRole(str, enum.Enum):
    """Clerk organization roles Evalyx distinguishes (minimized set)."""

    ADMIN = "org:admin"  # Clerk's built-in admin role key
    MEMBER = "org:member"


#: Privileged operations (workspace bootstrap / deletion) require this role.
PRIVILEGED_ROLES = frozenset({OrganizationRole.ADMIN})


@dataclass(frozen=True)
class AuthContext:
    """The immutable authentication/tenant context for one request.

    Everything downstream (services, repositories, authorization) consumes
    this — never raw Clerk types, never client-supplied identity fields.
    """

    clerk_user_id: str
    clerk_organization_id: str | None
    organization_role: OrganizationRole | None

    @property
    def is_authenticated(self) -> bool:
        return bool(self.clerk_user_id)

    @property
    def is_admin(self) -> bool:
        return self.organization_role in PRIVILEGED_ROLES


class TokenVerifier(Protocol):
    """Verifies one request and returns an :class:`AuthContext`.

    Production implementation wraps the Clerk SDK; tests substitute fakes —
    nothing else in Evalyx depends on Clerk specifics.
    """

    async def verify(self, request: Request) -> AuthContext: ...


def _organization_role_from(payload: dict[str, Any]) -> OrganizationRole | None:
    """Extract the active-organization role from a verified Clerk payload.

    Clerk session tokens carry the active organization under ``o`` (with
    role fields ``org_role``/``org:role`` depending on version); the legacy
    ``org_roles`` may also be present. Any unrecognized role value maps to
    MEMBER-adjacent behavior via :class:`OrganizationRole` lookups — never
    to admin.
    """
    organization = payload.get("o")
    if not isinstance(organization, dict):
        return None
    for key in ("org_role", "org:role"):
        raw = organization.get(key)
        if isinstance(raw, str):
            # Clerk may send "org:admin" or "admin" depending on token version.
            slug = raw.split(":", 1)[-1]
            try:
                return OrganizationRole(f"org:{slug}")
            except ValueError:
                return None
    return None


class ClerkTokenVerifier:
    """Production :class:`TokenVerifier` backed by the official Clerk SDK.

    Verification is local (JWKS public key via ``jwt_key``); the Clerk
    secret key is held only inside the SDK client and is never exposed.
    """

    def __init__(self, settings: Settings) -> None:
        self._clerk = Clerk(bearer_auth=settings.clerk_secret_key.get_secret_value())
        secret_key_value = settings.clerk_secret_key.get_secret_value()
        self._options = AuthenticateRequestOptions(
            jwt_key=settings.clerk_jwks_url,
            authorized_parties=[
                party.strip()
                for party in settings.clerk_authorized_parties.split(",")
                if party.strip()
            ]
            or None,
            secret_key=secret_key_value,
        )

    async def verify(self, request: Request) -> AuthContext:
        # Starlette requests satisfy the SDK's Requestish protocol
        # (headers + method + url); httpx/Starlette both work.
        starlette_request: Requestish = request  # type: ignore[assignment]
        state = await self._clerk.authenticate_request_async(
            starlette_request, self._options
        )
        if state.status is not AuthStatus.SIGNED_IN or not isinstance(
            state.payload, dict
        ):
            raise AuthenticationError("Authentication failed.")
        payload = state.payload
        user_id = payload.get("sub")
        if not isinstance(user_id, str) or not user_id:
            raise AuthenticationError("Authentication failed.")
        organization = payload.get("o")
        organization_id = (
            organization.get("id")
            if isinstance(organization, dict) and isinstance(organization.get("id"), str)
            else None
        )
        return AuthContext(
            clerk_user_id=user_id,
            clerk_organization_id=organization_id,
            organization_role=_organization_role_from(payload),
        )


class NoopTokenVerifier:
    """Development verifier used when AUTH_REQUIRED=0 (no Clerk configured).

    Exists so the dependency graph is unchanged in unauthenticated local
    mode — every request gets an anonymous context and tenant-scoped
    endpoints still require an organization id via the dev bootstrap.
    """

    async def verify(self, request: Request) -> AuthContext:
        return AuthContext(
            clerk_user_id="dev-anonymous",
            clerk_organization_id=None,
            organization_role=None,
        )


def create_token_verifier(settings: Settings) -> TokenVerifier:
    """Select the verifier from configuration (single wiring point)."""
    if settings.auth_required:
        return ClerkTokenVerifier(settings)
    return NoopTokenVerifier()


__all__ = [
    "PRIVILEGED_ROLES",
    "AuthContext",
    "AuthenticationError",
    "ClerkTokenVerifier",
    "NoopTokenVerifier",
    "OrganizationRequiredError",
    "OrganizationRole",
    "OrganizationRoleError",
    "TokenVerifier",
    "create_token_verifier",
]
