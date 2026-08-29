"""Minimal Evalyx application with health-check foundations.

Phase 2 scope: liveness (`/health`) and dependency readiness
(`/health/ready` for PostgreSQL and Redis). The full API layer arrives in a
later phase; this module intentionally stays small.
"""

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI, Response
from redis.asyncio import Redis

from evalyx.core.config import Settings, get_settings
from evalyx.core.logging import configure_logging
from evalyx.db.redis import check_redis, create_redis_client
from evalyx.db.session import DatabaseManager


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create the Evalyx FastAPI application.

    Accepting ``settings`` explicitly keeps the app testable and avoids
    hidden global configuration.
    """
    settings = settings or get_settings()
    configure_logging(settings)

    database = DatabaseManager(settings)
    redis_client: Redis = create_redis_client(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        await database.dispose()
        await redis_client.aclose()

    app = FastAPI(title="Evalyx", lifespan=lifespan)
    app.state.settings = settings
    app.state.database = database
    app.state.redis = redis_client

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Liveness: the application process is running."""
        return {"status": "ok"}

    @app.get("/health/ready")
    async def readiness(response: Response) -> dict[str, object]:
        """Readiness: application dependencies are healthy."""
        database_ok = await database.check()
        redis_ok = await check_redis(redis_client)

        dependencies = {
            "database": "ok" if database_ok else "error",
            "redis": "ok" if redis_ok else "error",
        }
        all_ok = database_ok and redis_ok
        if not all_ok:
            response.status_code = 503

        return {"status": "ok" if all_ok else "degraded", "dependencies": dependencies}

    return app


# Module-level instance for `uvicorn evalyx.api.app:app`.
app = create_app()
