"""Shared fixtures for integration tests (live PostgreSQL on localhost:5433)."""

from collections.abc import AsyncIterator

import pytest
from sqlalchemy import text

from evalyx.core.config import Settings
from evalyx.db.session import DatabaseManager

#: All Evalyx domain tables, children first, for test cleanup.
DOMAIN_TABLES = (
    "guardrail_results",
    "evaluation_case_results",
    "evaluation_runs",
    "test_cases",
    "dataset_versions",
    "datasets",
    "application_versions",
    "applications",
)


@pytest.fixture
def settings() -> Settings:
    """Real application settings; the Evalyx database lives on localhost:5433."""
    return Settings()


@pytest.fixture
async def db_manager(settings: Settings):
    manager = DatabaseManager(settings)
    yield manager
    await manager.dispose()


@pytest.fixture
async def db_session(db_manager: DatabaseManager):
    """Provide a session with a clean domain schema (truncate before test)."""
    async with db_manager.engine.begin() as conn:
        await conn.execute(
            text(f"TRUNCATE {', '.join(DOMAIN_TABLES)} RESTART IDENTITY CASCADE")
        )
    async with db_manager.session() as session:
        yield session


@pytest.fixture
async def clean_db(db_manager: DatabaseManager) -> AsyncIterator[DatabaseManager]:
    """DatabaseManager with a truncated domain schema (clean slate per test)."""
    async with db_manager.engine.begin() as conn:
        await conn.execute(
            text(f"TRUNCATE {', '.join(DOMAIN_TABLES)} RESTART IDENTITY CASCADE")
        )
    yield db_manager
