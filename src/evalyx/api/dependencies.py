"""FastAPI dependencies: database sessions and application services.

The engine and session factory are owned by the single ``DatabaseManager``
stored on ``app.state`` at startup (created once — never per request). The
session dependency opens one transactional session per request and closes
it afterwards; commits are performed by repositories.
"""

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession

from evalyx.api.schemas.common import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE
from evalyx.api.services import EvaluationService
from evalyx.core.config import Settings
from evalyx.db.session import DatabaseManager
from evalyx.evaluation.regression.service import RegressionService


def get_settings(request: Request) -> Settings:
    """Application settings carried on app state (set at startup)."""
    return request.app.state.settings


def get_database(request: Request) -> DatabaseManager:
    """The single DatabaseManager owned by the application."""
    return request.app.state.database


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
