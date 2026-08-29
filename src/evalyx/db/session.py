"""PostgreSQL connection foundation.

Only connection infrastructure lives here for now: an injectable
``DatabaseManager`` that owns the async SQLAlchemy engine and session
factory. Domain models and Alembic migrations are introduced in Phase 3.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from evalyx.core.config import Settings


class DatabaseManager:
    """Owns the async engine and session factory for PostgreSQL."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._engine: AsyncEngine = create_async_engine(
            settings.database_url,
            # Connection pooling tuning belongs to a later phase.
            pool_pre_ping=True,
        )
        self._session_factory = async_sessionmaker(
            self._engine,
            expire_on_commit=False,
        )

    @property
    def engine(self) -> AsyncEngine:
        return self._engine

    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        return self._session_factory

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Provide a transactional session scope."""
        async with self._session_factory() as session:
            yield session

    async def check(self) -> bool:
        """Return True when PostgreSQL answers a trivial query."""
        try:
            async with self._engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    async def dispose(self) -> None:
        await self._engine.dispose()
