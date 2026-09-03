"""Phase 18 infrastructure failure tests (live PostgreSQL + Redis).

Redis outage (rate-limiter policies), PostgreSQL outage (readiness +
honest 503s, never connection strings), worker duplicate delivery
(idempotent rescore, no quota impact), and recovery behavior. The
quality-vs-execution distinction is preserved throughout (existing
failure-taxonomy suites keep passing unchanged).
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from evalyx.api.app import create_app
from evalyx.api.auth import AuthContext, OrganizationRole
from evalyx.api.dependencies import require_organization
from evalyx.core.config import Settings
from evalyx.db.models import RunStatus
from evalyx.db.session import DatabaseManager
from evalyx.worker.execution import decide_action

pytestmark = pytest.mark.integration


def _hermetic_settings(**overrides):
    defaults = {"evalyx_secret_key": "placeholder", "auth_required": False}
    return Settings(_env_file=None, **{**defaults, **overrides})


class TestDuplicateDeliverySemantics:
    """Pure eligibility decisions: duplicates never re-execute, and terminal
    runs release quota capacity by construction (no active state)."""

    def test_failed_and_cancelled_skip(self):
        assert decide_action(RunStatus.FAILED) == "skip"
        assert decide_action(RunStatus.CANCELLED) == "skip"

    def test_completed_rescores_without_reexecution(self):
        assert decide_action(RunStatus.COMPLETED) == "rescore"

    def test_pending_and_running_execute(self):
        assert decide_action(RunStatus.PENDING) == "execute"
        assert decide_action(RunStatus.RUNNING) == "execute"


class TestRedisOutage:
    def _app_with_broken_redis(self, **overrides):
        from evalyx.api.ratelimit import InMemoryRateLimitBackend  # noqa: F401
        from evalyx.db.redis import create_redis_client

        settings = _hermetic_settings(**overrides)
        broken = create_redis_client(
            settings.model_copy(update={"redis_url": "redis://localhost:1/0"})
        )
        return create_app(settings, redis_client=broken)

    def test_allow_policy_serves_through_outage(self):
        app = self._app_with_broken_redis(rate_limit_on_redis_error="allow")
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/health")
        assert response.status_code == 200

    def test_deny_policy_sheds_load(self):
        app = self._app_with_broken_redis(rate_limit_on_redis_error="deny")
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/health")
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "rate_limiter_unavailable"

    def test_degraded_requests_are_metered(self):
        from evalyx.core.metrics import metrics

        app = self._app_with_broken_redis(rate_limit_on_redis_error="allow")
        client = TestClient(app, raise_server_exceptions=False)
        before = metrics.snapshot().get("rate_limiter_errors_total", [])
        client.get("/health")
        after = metrics.snapshot().get("rate_limiter_errors_total", [])
        total = lambda series: sum(entry["value"] for entry in series)
        assert total(after) > total(before)


class TestPostgresOutage:
    async def test_submit_maps_outage_to_503_without_details(
        self, clean_db: DatabaseManager, settings: Settings
    ):
        """A dead database surfaces 503 database_unavailable — never DSNs."""
        bad = settings.model_copy(
            update={"database_url": "postgresql+asyncpg://localhost:1/db"}
        )
        app = create_app(bad, database=DatabaseManager(bad))

        async def _resolve():
            from evalyx.db.tenancy import require_organization as resolve_row

            async with clean_db.session() as s:
                organization = await resolve_row(s, "org_outage")
            return (
                AuthContext(
                    clerk_user_id="u",
                    clerk_organization_id="org_outage",
                    organization_role=OrganizationRole.ADMIN,
                ),
                organization,
            )

        app.dependency_overrides[require_organization] = _resolve
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            response = await client.post(
                "/api/v1/evaluations",
                json={
                    "application_id": str(uuid.uuid4()),
                    "dataset_version_id": str(uuid.uuid4()),
                    "agent_model": "m",
                },
            )
            assert response.status_code == 503
            body = response.json()
            assert body["error"]["code"] == "database_unavailable"
            assert "localhost:1" not in response.text
            assert "postgresql" not in response.text

    async def test_readiness_reports_degraded(
        self, clean_db: DatabaseManager, settings: Settings
    ):
        bad = settings.model_copy(
            update={"database_url": "postgresql+asyncpg://localhost:1/db"}
        )
        app = create_app(bad, database=DatabaseManager(bad))
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            response = await client.get("/health/ready")
            assert response.status_code == 503
            body = response.json()
            assert body["status"] == "degraded"
            assert body["dependencies"]["database"] == "error"
            # Liveness is independent of dependencies.
            assert (await client.get("/health")).status_code == 200


class TestRecovery:
    async def test_quota_counts_read_committed_state_after_failure(
        self, clean_db: DatabaseManager, settings: Settings
    ):
        """After an enqueue failure the run is failed → a retry is admittable.

        Exercises the full submit path twice against a broker that fails
        once: no stuck capacity, no manual release.
        """
        from evalyx.api.services import EvaluationService

        org = f"org_recovery_{uuid.uuid4().hex[:8]}"
        tight = settings.model_copy(
            update={
                "quota_max_concurrent_evaluations": 1,
                "quota_max_evaluations_per_day": 1000,
                "rate_limit_redis_prefix": f"evalyx:test18-rec:{uuid.uuid4().hex[:8]}",
            }
        )
        app = create_app(tight, database=clean_db)

        async def _resolve():
            from evalyx.db.tenancy import require_organization as resolve_row

            async with clean_db.session() as s:
                organization = await resolve_row(s, org)
            return (
                AuthContext(
                    clerk_user_id="recovery-user",
                    clerk_organization_id=org,
                    organization_role=OrganizationRole.ADMIN,
                ),
                organization,
            )

        app.dependency_overrides[require_organization] = _resolve
        calls = {"n": 0}

        def flaky_enqueue(run_id):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ConnectionError("broker unreachable")
            return "task-recovered"

        from evalyx.api.dependencies import get_evaluation_service

        app.dependency_overrides[get_evaluation_service] = lambda: EvaluationService(
            clean_db.session_factory, enqueue=flaky_enqueue, settings=tight
        )
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            app_row = (
                await client.post("/api/v1/applications", json={"name": "rec-app"})
            ).json()
            ds = (
                await client.post("/api/v1/datasets", json={"name": "rec-ds"})
            ).json()
            ver = (
                await client.post(
                    f"/api/v1/datasets/{ds['id']}/versions", json={"version": 1}
                )
            ).json()
            await client.post(
                f"/api/v1/datasets/{ds['id']}/versions/1/cases",
                json={"name": "c1", "input": {"prompt": "x"}},
            )
            payload = {
                "application_id": app_row["id"],
                "dataset_version_id": ver["id"],
                "agent_model": "m",
            }
            first = await client.post("/api/v1/evaluations", json=payload)
            assert first.status_code == 503  # enqueue failed, run failed
            second = await client.post("/api/v1/evaluations", json=payload)
            assert second.status_code == 202, second.text  # capacity free
