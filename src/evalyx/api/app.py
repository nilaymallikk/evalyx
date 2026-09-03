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

from evalyx.api.auth import create_token_verifier
from evalyx.api.errors import register_error_handlers
from evalyx.api.middleware import ObservabilityMiddleware
from evalyx.api.ratelimit import RateLimitMiddleware
from evalyx.api.routers import api_router
from evalyx.api.security_headers import SecurityHeadersMiddleware
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
        logger.info(
            "api_startup",
            app_env=settings.app_env,
            api_workers=settings.api_workers,
        )
        yield
        # Graceful shutdown: stop accepting new work (server closes the
        # listener first), then release pooled resources.
        logger.info("api_shutdown")
        await database.dispose()
        await redis_client.aclose()
        logger.info("api_shutdown_complete")

    app = FastAPI(
        title="Evalyx",
        version=API_VERSION,
        description=(
            "AI evaluation and reliability platform: register applications, "
            "version datasets, submit background evaluations, inspect "
            "results, and compare runs for regressions. All resource "
            "endpoints live under the `/api/v1` prefix and are "
            "authenticated with Clerk (bearer session token); resources are "
            "scoped to the caller's Clerk organization. Health endpoints "
            "are public."
        ),
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.database = database
    app.state.redis = redis_client
    app.state.token_verifier = create_token_verifier(settings)

    # OpenAPI: document Clerk bearer authentication (the future CLI/TUI
    # sends `Authorization: Bearer <Clerk session token>`).
    def _decorate_openapi() -> None:
        def _openapi_with_auth() -> dict:
            schema = FastAPI.openapi(app)
            schema.setdefault("components", {})["securitySchemes"] = {
                "clerkAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "bearerFormat": "JWT",
                    "description": "Clerk session token (Bearer).",
                }
            }
            for path_item in schema.get("paths", {}).values():
                for operation in path_item.values():
                    if isinstance(operation, dict):
                        operation.setdefault("security", [{"clerkAuth": []}])
            return schema

        # Standard FastAPI customization pattern (docs: overriding openapi()).
        app.openapi = _openapi_with_auth  # type: ignore[method-assign]

    _decorate_openapi()

    register_error_handlers(app)
    # Restrictive CORS: disabled unless CORS_ALLOWED_ORIGINS names explicit
    # origins (never "*"; the CLI/TUI/API clients do not need CORS).
    if settings.cors_origins:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["GET", "POST", "PATCH", "DELETE"],
            allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
            allow_credentials=False,
            max_age=600,
        )
    # Middleware order: last added runs first (outermost). Rate limiting is
    # outermost (reject abuse before correlation work), then observability,
    # then security headers closest to the router.
    app.add_middleware(SecurityHeadersMiddleware)
    # Outermost user middleware: request correlation, structured lifecycle
    # logging and bounded-label HTTP metrics for every request.
    app.add_middleware(
        ObservabilityMiddleware,
        slow_request_threshold_ms=settings.slow_request_threshold_ms,
    )
    app.add_middleware(RateLimitMiddleware, settings=settings)
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
