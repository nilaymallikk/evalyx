"""Minimal Evalyx application with health checks and the versioned REST API.

- ``/health`` and ``/health/ready`` (Phase 2 behavior, unchanged) live
  outside the API version.
- ``/api/v1/...`` carries the resource API (Phase 9): applications,
  datasets, evaluations, regressions.

The HTTP layer is orchestration only: routers delegate to repositories and
domain services; no LLM, guardrail, scoring, regression, or Celery
execution logic lives here. PostgreSQL/Redis connections are owned by the
single ``DatabaseManager``/redis client created at startup (one engine,
never one per request).
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Response
from redis.asyncio import Redis

from evalyx.api.errors import register_error_handlers
from evalyx.api.middleware import ObservabilityMiddleware
from evalyx.api.routers import api_router
from evalyx.core.config import Settings, get_settings
from evalyx.core.logging import configure_logging
from evalyx.db.redis import check_redis, create_redis_client
from evalyx.db.session import DatabaseManager

logger = structlog.get_logger(__name__)

API_VERSION = "1.0.0"


def create_app(
    settings: Settings | None = None,
    *,
    database: DatabaseManager | None = None,
    redis_client: Redis | None = None,
) -> FastAPI:
    """Create the Evalyx FastAPI application.

    Accepting ``settings`` (and optionally pre-built infrastructure) keeps
    the app testable and avoids hidden global configuration. By default the
    engine, session factory, and Redis client are created here exactly once
    per application.
    """
    settings = settings or get_settings()
    configure_logging(settings)

    database = database or DatabaseManager(settings)
    redis_client = redis_client or create_redis_client(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        await database.dispose()
        await redis_client.aclose()

    app = FastAPI(
        title="Evalyx",
        version=API_VERSION,
        description=(
            "AI evaluation and reliability platform: register applications, "
            "version datasets, submit background evaluations, inspect "
            "results, and compare runs for regressions. All resource "
            "endpoints live under the `/api/v1` prefix. "
            "Authentication/authorization is **not implemented yet** — this "
            "API is intended for local development and portfolio use."
        ),
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.database = database
    app.state.redis = redis_client

    register_error_handlers(app)
    # Outermost user middleware: request correlation, structured lifecycle
    # logging and bounded-label HTTP metrics for every request.
    app.add_middleware(
        ObservabilityMiddleware,
        slow_request_threshold_ms=settings.slow_request_threshold_ms,
    )
    app.include_router(api_router)

    @app.get("/health", tags=["health"])
    async def health() -> dict[str, str]:
        """Liveness: the application process is running."""
        return {"status": "ok"}

    @app.get("/health/ready", tags=["health"])
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
            # Safe readiness observability: dependency name only — never
            # connection strings, URLs, or credentials.
            for dependency, state in dependencies.items():
                if state != "ok":
                    logger.warning(
                        "readiness_check_failed", dependency=dependency
                    )

        return {"status": "ok" if all_ok else "degraded", "dependencies": dependencies}

    return app


# Module-level instance for `uvicorn evalyx.api.app:app`.
app = create_app()
