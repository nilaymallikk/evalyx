"""FastAPI dependencies: database sessions and services.

The engine and session factory are owned by the single ``DatabaseManager``
stored on ``app.state`` at startup (created once — never per request). The
session dependency opens one transactional session per request and closes
it afterwards; commits are performed by repositories.

Identity: Evalyx is local-first with a single workspace. ``require_organization``
resolves (auto-provisioning) that workspace; every resource read/write is
still filtered by ``organization_id`` at the repository boundary.
"""

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession

from evalyx.api.auth import AuthContext, local_context
from evalyx.api.schemas.common import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE
from evalyx.api.services import EvaluationService
from evalyx.core.config import Settings
from evalyx.db.models import Organization
from evalyx.db.session import DatabaseManager
from evalyx.db.tenancy import require_local_organization
from evalyx.evaluation.regression.service import RegressionService
from evalyx.quotas import QuotaService


def get_settings(request: Request) -> Settings:
    """Application settings carried on app state (set at startup)."""
    return request.app.state.settings


def get_database(request: Request) -> DatabaseManager:
    """The single DatabaseManager owned by the application."""
    return request.app.state.database


async def require_authenticated_user() -> AuthContext:
    """Local operator context (no login required)."""
    return local_context()


async def require_organization(
    session: Annotated[AsyncSession, Depends(get_session)],
) -> tuple[AuthContext, Organization]:
    """Workspace-scoped endpoint guard.

    Resolves (auto-provisioning) the single local workspace. The returned
    ``organization.id`` is the only tenant id endpoints may use.
    """
    organization = await require_local_organization(session)
    return local_context(), organization


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
    return EvaluationService(
        request.app.state.database.session_factory,
        settings=request.app.state.settings,
    )


def get_quota_service(request: Request) -> QuotaService:
    """QuotaService bound to the session factory and settings."""
    return QuotaService(
        request.app.state.database.session_factory,
        request.app.state.settings,
    )


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

