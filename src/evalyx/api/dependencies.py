"""FastAPI dependencies: authentication, database sessions, and services.

The engine and session factory are owned by the single ``DatabaseManager``
stored on ``app.state`` at startup (created once — never per request). The
session dependency opens one transactional session per request and closes
it afterwards; commits are performed by repositories.

Authentication (Phase 14): a :class:`TokenVerifier` stored on ``app.state``
turns the request's Clerk session token into an :class:`AuthContext`.
Tenant-scoped endpoints go through :func:`require_organization`, which maps
the verified Clerk organization to the local tenant row and raises 401/403
per the authentication behavior contract. Identity is *never* read from
request body/query fields.
"""

import typing
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession

from evalyx.api.auth import (
    AuthContext,
    OrganizationRequiredError,
    OrganizationRole,
    OrganizationRoleError,
    TokenVerifier,
)
from evalyx.api.schemas.common import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE
from evalyx.api.services import EvaluationService
from evalyx.core.config import Settings
from evalyx.db.models import Organization
from evalyx.db.session import DatabaseManager
from evalyx.db.tenancy import require_organization as require_organization_row
from evalyx.evaluation.regression.service import RegressionService


def get_settings(request: Request) -> Settings:
    """Application settings carried on app state (set at startup)."""
    return request.app.state.settings


def get_database(request: Request) -> DatabaseManager:
    """The single DatabaseManager owned by the application."""
    return request.app.state.database


def get_token_verifier(request: Request) -> TokenVerifier:
    """The Clerk token verifier wired at startup (``app.state``)."""
    return request.app.state.token_verifier


async def get_auth_context(
    request: Request,
    verifier: Annotated[TokenVerifier, Depends(get_token_verifier)],
) -> AuthContext:
    """Verify the request's Clerk token into an immutable AuthContext.

    Raises :class:`AuthenticationError` (mapped to 401 by the error
    handlers) for missing/invalid/expired tokens; error messages never
    include token contents.
    """
    return await verifier.verify(request)


async def require_authenticated_user(
    auth: Annotated[AuthContext, Depends(get_auth_context)],
) -> AuthContext:
    """Authenticated user endpoint guard (no organization required)."""
    if not auth.is_authenticated:
        from evalyx.api.auth import AuthenticationError

        raise AuthenticationError("Authentication failed.")
    return auth


async def require_organization(
    session: Annotated[AsyncSession, Depends(get_session)],
    auth: Annotated[AuthContext, Depends(require_authenticated_user)],
) -> tuple[AuthContext, Organization]:
    """Tenant-scoped endpoint guard.

    Requires an authenticated user with an active Clerk organization and
    maps it to the local tenant row (auto-provisioned on first use). The
    returned ``organization.id`` is the only tenant id endpoints may use —
    client-supplied organization/workspace fields are never trusted.
    """
    if auth.clerk_organization_id is None:
        raise OrganizationRequiredError(
            "An active organization is required for this operation."
        )
    organization = await require_organization_row(
        session, auth.clerk_organization_id
    )
    return auth, organization


def require_role(
    *allowed: OrganizationRole,
) -> typing.Callable[[tuple[AuthContext, Organization]], AuthContext]:
    """Privileged-operation guard: the active role must be in ``allowed``."""

    def _check(
        context: Annotated[tuple[AuthContext, Organization], Depends(require_organization)],
    ) -> AuthContext:
        auth, _organization = context
        if auth.organization_role not in allowed:
            raise OrganizationRoleError(
                "Your organization role does not permit this operation."
            )
        return auth

    return _check


require_admin = require_role(OrganizationRole.ADMIN)


async def get_session(
    database: Annotated[DatabaseManager, Depends(get_database)],
) -> AsyncIterator[AsyncSession]:
    """One ``AsyncSession`` per request, opened and closed around the handler."""
    async with database.session() as session:
        yield session


async def get_regression_service(
    database: Annotated[DatabaseManager, Depends(get_database)],
) -> RegressionService:
    """RegressionService bound to the application's session factory."""
    return RegressionService(database.session_factory)


def get_evaluation_service(request: Request) -> EvaluationService:
    """EvaluationService bound to the application's session factory."""
    return EvaluationService(request.app.state.database.session_factory)


def pagination_params(
    limit: int = Query(
        default=DEFAULT_PAGE_SIZE,
        ge=1,
        le=MAX_PAGE_SIZE,
        description=f"Page size (1–{MAX_PAGE_SIZE}).",
    ),
    offset: int = Query(
        default=0,
        ge=0,
        description="Number of items to skip (stable, deterministic ordering).",
    ),
) -> tuple[int, int]:
    """Validated ``limit``/``offset`` pair shared by all list endpoints."""
    return limit, offset

