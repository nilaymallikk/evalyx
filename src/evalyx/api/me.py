"""Caller-identity enrichment for ``GET /api/v1/me``.

The verified session token answers *who* (user id, active organization,
role) but not *how to address the human* (email) nor *what else they belong
to* (other organizations). This module asks the Clerk Backend API via the
official SDK, using the server-side Clerk secret key held inside
:class:`~evalyx.api.auth.ClerkTokenVerifier`. Failure handling is
display-only by design: if Clerk lookups fail, ``/me`` still answers with
token-derived fields and omits the enrichment (the CLI shows fewer lines
rather than failing authentication that actually succeeded).
"""

import structlog

from evalyx.api.auth import (
    AuthContext,
    ClerkTokenVerifier,
    OrganizationRole,
    TokenVerifier,
)
from evalyx.api.schemas.me import MeResponse, OrganizationSummary

logger = structlog.get_logger(__name__)


def _role_slug(role: OrganizationRole | None) -> str | None:
    return role.value.split(":", 1)[-1] if role is not None else None


async def describe_caller(auth: AuthContext, verifier: TokenVerifier) -> MeResponse:
    """Build the ``/me`` response: token-derived facts + best-effort Clerk enrichment."""
    email: str | None = None
    organizations: list[OrganizationSummary] = []

    clerk = _clerk_client(verifier)
    if clerk is not None:
        try:
            user = await clerk.users.get_async(user_id=auth.clerk_user_id)
            primary = user.primary_email_address_id
            email = next(
                (
                    entry.email_address
                    for entry in user.email_addresses
                    if entry.id == primary
                ),
                None,
            ) or (user.email_addresses[0].email_address if user.email_addresses else None)
        except Exception as exc:  # noqa: BLE001 — enrichment is display-only
            logger.info("me_user_lookup_unavailable", error=type(exc).__name__)

        if auth.clerk_organization_id is None:
            try:
                memberships = await clerk.users.get_organization_memberships_async(
                    user_id=auth.clerk_user_id, limit=20
                )
                organizations = [
                    OrganizationSummary(
                        clerk_organization_id=membership.organization.id,
                        name=membership.organization.name,
                        role=_role_slug(
                            OrganizationRole(membership.role)
                            if membership.role in {"org:admin", "org:member"}
                            else None
                        ),
                    )
                    for membership in memberships.data or []
                ]
            except Exception as exc:  # noqa: BLE001 — enrichment is display-only
                logger.info(
                    "me_memberships_lookup_unavailable", error=type(exc).__name__
                )

    active = None
    if auth.clerk_organization_id is not None:
        active = OrganizationSummary(
            clerk_organization_id=auth.clerk_organization_id,
            name=auth.clerk_organization_id,
            role=_role_slug(auth.organization_role),
        )
        organizations = [active] + organizations

    return MeResponse(
        clerk_user_id=auth.clerk_user_id,
        email=email,
        active_organization=active,
        organizations=organizations,
    )


def _clerk_client(verifier: TokenVerifier):
    """The underlying Clerk SDK client when the verifier is Clerk-backed.

    Returns ``None`` for the dev verifier (no Clerk configured) — callers
    skip enrichment entirely.
    """
    if isinstance(verifier, ClerkTokenVerifier):
        return verifier.clerk
    return None
