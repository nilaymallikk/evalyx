"""Phase 18 quota integration tests (live PostgreSQL + Redis).

Admission, denial envelopes, per-organization overrides, race safety under
concurrent submissions, capacity release on terminal transitions, stale-run
aging, and audit trails for quota decisions. Quotas stay independent from
billing (there is none).
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select, text

from evalyx.api.app import create_app
from evalyx.api.auth import AuthContext, OrganizationRole
from evalyx.api.dependencies import require_organization
from evalyx.core.config import Settings
from evalyx.db.models import EvaluationRun, Organization, RunStatus
from evalyx.db.session import DatabaseManager

pytestmark = pytest.mark.integration


def _tight_settings(base: Settings, **overrides) -> Settings:
    return base.model_copy(update=overrides)


async def _api(db: DatabaseManager, settings: Settings, clerk_org_id: str):
    app = create_app(settings, database=db)

    async def _resolve():
        from evalyx.db.tenancy import require_organization as resolve_row

        async with db.session() as s:
            organization = await resolve_row(s, clerk_org_id)
        return (
            AuthContext(
                clerk_user_id="quota-user",
                clerk_organization_id=clerk_org_id,
                organization_role=OrganizationRole.ADMIN,
            ),
            organization,
        )

    app.dependency_overrides[require_organization] = _resolve
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://testserver"), app


async def _org_id(db: DatabaseManager, clerk_org_id: str) -> uuid.UUID:
    async with db.session() as session:
        org = (
            await session.scalars(
                select(Organization).filter_by(clerk_organization_id=clerk_org_id)
            )
        ).first()
        assert org is not None
        return org.id


async def _seed_app(client: AsyncClient, name: str) -> dict:
    response = await client.post("/api/v1/applications", json={"name": name})
    assert response.status_code == 201, response.text
    return response.json()


async def _seed_dataset(client: AsyncClient, name: str, cases: list[str]) -> dict:
    response = await client.post("/api/v1/datasets", json={"name": name})
    assert response.status_code == 201, response.text
    dataset = response.json()
    response = await client.post(
        f"/api/v1/datasets/{dataset['id']}/versions", json={"version": 1}
    )
    assert response.status_code == 201, response.text
    version = response.json()
    for case in cases:
        response = await client.post(
            f"/api/v1/datasets/{dataset['id']}/versions/1/cases",
            json={"name": case, "input": {"prompt": case}},
        )
        assert response.status_code == 201, response.text
    return {"dataset": dataset, "version": version}


async def _submit(
    client: AsyncClient, app_id: str, version_id: str
) -> tuple[int, dict]:
    response = await client.post(
        "/api/v1/evaluations",
        json={
            "application_id": app_id,
            "dataset_version_id": version_id,
            "agent_model": "quota-probe-model",
        },
    )
    try:
        body = response.json()
    except ValueError:
        body = {}
    return response.status_code, body if isinstance(body, dict) else {}


class TestApplicationQuota:
    async def test_second_application_denied_with_counts(
        self, clean_db: DatabaseManager, settings: Settings
    ):
        org = f"org_quota_app_{uuid.uuid4().hex[:8]}"
        tight = _tight_settings(settings, quota_max_applications=1)
        client, _ = await _api(clean_db, tight, org)
        async with client:
            assert (await _seed_app(client, "first"))["name"] == "first"
            response = await client.post("/api/v1/applications", json={"name": "second"})
            assert response.status_code == 429
            body = response.json()
            assert body["error"]["code"] == "quota_exceeded"
            assert "1/1" in body["error"]["message"]
            assert "Retry-After" in response.headers

    async def test_override_raises_ceiling(
        self, clean_db: DatabaseManager, settings: Settings
    ):
        from evalyx.quotas import QuotaService

        org = f"org_quota_override_{uuid.uuid4().hex[:8]}"
        tight = _tight_settings(settings, quota_max_applications=1)
        client, app = await _api(clean_db, tight, org)
        async with client:
            await _seed_app(client, "first")
            response = await client.post("/api/v1/applications", json={"name": "second"})
            assert response.status_code == 429
            quotas = QuotaService(
                app.state.database.session_factory, app.state.settings
            )
            async with clean_db.session() as session:
                organization_id = await _org_id(clean_db, org)
                await quotas.set_overrides(
                    session, organization_id, max_applications=2
                )
            response = await client.post("/api/v1/applications", json={"name": "second"})
            assert response.status_code == 201, response.text


class TestDatasetAndCaseQuotas:
    async def test_dataset_quota(self, clean_db: DatabaseManager, settings: Settings):
        org = f"org_quota_ds_{uuid.uuid4().hex[:8]}"
        tight = _tight_settings(settings, quota_max_datasets=1)
        client, _ = await _api(clean_db, tight, org)
        async with client:
            await _seed_dataset(client, "d1", [])
            response = await client.post("/api/v1/datasets", json={"name": "d2"})
            assert response.status_code == 429
            assert response.json()["error"]["code"] == "quota_exceeded"

    async def test_case_quota(self, clean_db: DatabaseManager, settings: Settings):
        org = f"org_quota_case_{uuid.uuid4().hex[:8]}"
        tight = _tight_settings(settings, quota_max_cases_per_dataset_version=2)
        client, _ = await _api(clean_db, tight, org)
        async with client:
            data = await _seed_dataset(client, "d1", ["c1", "c2"])
            dataset_id = data["dataset"]["id"]
            response = await client.post(
                f"/api/v1/datasets/{dataset_id}/versions/1/cases",
                json={"name": "c3", "input": {"prompt": "x"}},
            )
            assert response.status_code == 429
            assert response.json()["error"]["code"] == "quota_exceeded"


class TestEvaluationQuotas:
    async def test_concurrent_quota_race_safe(
        self, clean_db: DatabaseManager, settings: Settings
    ):
        """10 concurrent submissions against a quota of 2 admit exactly 2."""
        org = f"org_quota_race_{uuid.uuid4().hex[:8]}"
        tight = _tight_settings(
            settings, quota_max_concurrent_evaluations=2, quota_max_evaluations_per_day=1000
        )
        client, _ = await _api(clean_db, tight, org)
        async with client:
            app = await _seed_app(client, "race-app")
            data = await _seed_dataset(client, "race-ds", ["c1"])
            results = await asyncio.gather(
                *(
                    _submit(client, app["id"], data["version"]["id"])
                    for _ in range(10)
                )
            )
            admitted = [r for r in results if r[0] == 202]
            denied = [r for r in results if r[0] == 429]
            assert len(admitted) == 2
            assert len(denied) == 8
            assert all(r[1].get("error", {}).get("code") == "quota_exceeded" for r in denied)

    async def test_capacity_released_on_terminal_transition(
        self, clean_db: DatabaseManager, settings: Settings
    ):
        org = f"org_quota_release_{uuid.uuid4().hex[:8]}"
        tight = _tight_settings(
            settings, quota_max_concurrent_evaluations=1, quota_max_evaluations_per_day=1000
        )
        client, _ = await _api(clean_db, tight, org)
        async with client:
            app = await _seed_app(client, "rel-app")
            data = await _seed_dataset(client, "rel-ds", ["c1"])
            status, body = await _submit(client, app["id"], data["version"]["id"])
            assert status == 202, body
            # Quota full while the run is live...
            status, _ = await _submit(client, app["id"], data["version"]["id"])
            assert status == 429
            # ...released when the run reaches a terminal state (worker
            # completion, failure, or duplicate-delivery rescore all land
            # here — no explicit release call exists to forget).
            run_id = body["run_id"]
            async with clean_db.session() as session:
                run = await session.get(EvaluationRun, uuid.UUID(run_id))
                assert run is not None
                run.status = RunStatus.COMPLETED
                run.completed_at = datetime.now(UTC)
                await session.commit()
            status, _ = await _submit(client, app["id"], data["version"]["id"])
            assert status == 202

    async def test_failed_runs_release_capacity(
        self, clean_db: DatabaseManager, settings: Settings
    ):
        """Enqueue failure marks the run failed → capacity is not stuck."""
        org = f"org_quota_fail_{uuid.uuid4().hex[:8]}"
        tight = _tight_settings(
            settings, quota_max_concurrent_evaluations=1, quota_max_evaluations_per_day=1000
        )
        client, _ = await _api(clean_db, tight, org)
        async with client:
            app_data = await _seed_app(client, "fail-app")
            data = await _seed_dataset(client, "fail-ds", ["c1"])
            # Fill the single slot, then fail the run like the worker would.
            status, body = await _submit(client, app_data["id"], data["version"]["id"])
            assert status == 202, body
            async with clean_db.session() as session:
                run = await session.get(EvaluationRun, uuid.UUID(body["run_id"]))
                assert run is not None
                run.status = RunStatus.FAILED
                run.completed_at = datetime.now(UTC)
                await session.commit()
            status, _ = await _submit(client, app_data["id"], data["version"]["id"])
            assert status == 202

    async def test_stale_runs_stop_holding_capacity(
        self, clean_db: DatabaseManager, settings: Settings
    ):
        org = f"org_quota_stale_{uuid.uuid4().hex[:8]}"
        tight = _tight_settings(
            settings,
            quota_max_concurrent_evaluations=1,
            quota_max_evaluations_per_day=1000,
            quota_stale_run_seconds=600,
        )
        client, _ = await _api(clean_db, tight, org)
        async with client:
            app = await _seed_app(client, "stale-app")
            data = await _seed_dataset(client, "stale-ds", ["c1"])
            status, body = await _submit(client, app["id"], data["version"]["id"])
            assert status == 202, body
            # Simulate a worker lost mid-run: still `running`, but older
            # than the staleness horizon → excluded from the count.
            async with clean_db.engine.begin() as conn:
                await conn.execute(
                    text(
                        "UPDATE evaluation_runs SET status = 'running', "
                        "created_at = :cutoff WHERE id = :run_id"
                    ),
                    {
                        "cutoff": datetime.now(UTC) - timedelta(seconds=3600),
                        "run_id": uuid.UUID(body["run_id"]),
                    },
                )
            status, _ = await _submit(client, app["id"], data["version"]["id"])
            assert status == 202

    async def test_daily_quota(self, clean_db: DatabaseManager, settings: Settings):
        org = f"org_quota_daily_{uuid.uuid4().hex[:8]}"
        tight = _tight_settings(
            settings, quota_max_concurrent_evaluations=100, quota_max_evaluations_per_day=2
        )
        client, _ = await _api(clean_db, tight, org)
        async with client:
            app = await _seed_app(client, "daily-app")
            data = await _seed_dataset(client, "daily-ds", ["c1"])
            for _ in range(2):
                status, _ = await _submit(client, app["id"], data["version"]["id"])
                assert status == 202
            status, body = await _submit(client, app["id"], data["version"]["id"])
            assert status == 429
            assert body["error"]["code"] == "quota_exceeded"


class TestConnectionTestQuota:
    async def test_daily_connection_test_budget(
        self, clean_db: DatabaseManager, settings: Settings, monkeypatch
    ):
        from evalyx.api.routers import applications as applications_router
        from evalyx.application.base import ApplicationInvocationError

        class _FailingTarget:
            closed = False

            async def invoke(self, prompt: str):
                raise ApplicationInvocationError(
                    "Could not reach application.", category="connection_error"
                )

            async def close(self):
                self.closed = True

        monkeypatch.setattr(
            applications_router, "build_http_target", lambda *a, **k: _FailingTarget()
        )
        org = f"org_quota_ct_{uuid.uuid4().hex[:8]}"
        tight = _tight_settings(settings, quota_max_connection_tests_per_day=1)
        client, _ = await _api(clean_db, tight, org)
        async with client:
            response = await client.post(
                "/api/v1/applications",
                json={
                    "name": "ct-app",
                    "connection_type": "http",
                    "secret": "ct-secret-value",
                },
            )
            assert response.status_code == 201, response.text
            app_id = response.json()["id"]
            response = await client.post(
                f"/api/v1/applications/{app_id}/versions",
                json={
                    "version": "v1",
                    "connection": {
                        "endpoint": "https://93.184.216.34/v1/chat",
                        "method": "POST",
                        "auth": {"type": "bearer"},
                        "request": {"input_field": "question"},
                        "response_path": "answer",
                    },
                },
            )
            assert response.status_code == 201, response.text
            # First call admitted, fails with a classified failure
            # (HTTP 200, success=false); the second exceeds the daily
            # budget of 1 and is denied before any outbound call.
            first = await client.post(
                f"/api/v1/applications/{app_id}/test", json={"prompt": "hi"}
            )
            assert first.status_code == 200, first.text
            assert first.json()["success"] is False
            second = await client.post(
                f"/api/v1/applications/{app_id}/test", json={"prompt": "hi"}
            )
            assert second.status_code == 429
            assert second.json()["error"]["code"] == "quota_exceeded"
