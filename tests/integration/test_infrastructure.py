"""Integration tests: live PostgreSQL and Redis connectivity.

Run with:

    EVALYX_RUN_INTEGRATION_TESTS=1 uv run pytest
"""

import pytest

from evalyx.core.config import Settings
from evalyx.db.redis import check_redis, create_redis_client
from evalyx.db.session import DatabaseManager


@pytest.fixture
def settings() -> Settings:
    # Reads real env vars / .env so we validate the actual local setup.
    return Settings()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_postgresql_connectivity(settings: Settings):
    manager = DatabaseManager(settings)
    try:
        assert await manager.check() is True
    finally:
        await manager.dispose()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_redis_connectivity(settings: Settings):
    client = create_redis_client(settings)
    try:
        assert await check_redis(client) is True
    finally:
        await client.aclose()


@pytest.mark.integration
def test_readiness_endpoint_reports_healthy_dependencies(settings: Settings):
    from fastapi.testclient import TestClient

    from evalyx.api.app import create_app

    app = create_app(settings)
    with TestClient(app) as client:
        response = client.get("/health/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["dependencies"] == {"database": "ok", "redis": "ok"}
