"""Multi-tenancy integration tests (Phase 14): live PostgreSQL, two tenants.

Clerk is bypassed with fixed AuthContexts per organization; the tests
exercise the part Evalyx owns — tenant scoping at the data/API boundary and
cross-tenant rejection (IDOR).
"""

import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from evalyx.api.app import create_app
from evalyx.api.auth import AuthContext, OrganizationRole
from evalyx.api.dependencies import require_organization
from evalyx.db.models import CaseStatus
from evalyx.db.repositories import EvaluationRepository
from evalyx.db.session import DatabaseManager
from evalyx.db.tenancy import require_organization as resolve_row

pytestmark = pytest.mark.integration

ORG_A = "org_test_a"
ORG_B = "org_test_b"


def _tenant_override(db: DatabaseManager, clerk_org_id: str):
    """Dependency override: fixed Clerk identity → local organization row."""

    async def _resolve():
        async with db.session() as session:
            organization = await resolve_row(session, clerk_org_id)
        return (
            AuthContext(
                clerk_user_id=f"user_{clerk_org_id}",
                clerk_organization_id=clerk_org_id,
                organization_role=OrganizationRole.ADMIN,
            ),
            organization,
        )

    return _resolve


@pytest.fixture
async def clean_db(db_manager: DatabaseManager):
    from conftest import DOMAIN_TABLES
    from sqlalchemy import text

    async with db_manager.engine.begin() as conn:
        # Fixed identifier list from conftest (never external input).
        statement = text(
            "TRUNCATE " + ", ".join(DOMAIN_TABLES) + " RESTART IDENTITY CASCADE"
        )
        await conn.execute(statement)
    yield db_manager


async def _make_client(db: DatabaseManager, clerk_org_id: str) -> AsyncClient:
    from evalyx.core.config import Settings

    app = create_app(Settings(auth_required=False), database=db)
    app.dependency_overrides[require_organization] = _tenant_override(db, clerk_org_id)
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://testserver")


async def _seed_run(db: DatabaseManager, *, organization_id, name: str) -> uuid.UUID:
    from evalyx.db.repositories import ApplicationRepository, DatasetRepository

    apps, datasets, evaluations = (
        ApplicationRepository(),
        DatasetRepository(),
        EvaluationRepository(),
    )
    async with db.session() as session:
        app = await apps.create(
            session, organization_id=organization_id, name=f"{name}-app"
        )
        dataset = await datasets.create(
            session, organization_id=organization_id, name=f"{name}-ds"
        )
        version = await datasets.create_version(session, dataset_id=dataset.id, version=1)
        case = await datasets.add_test_case(
            session, dataset_version_id=version.id, name="c0", input={"prompt": "x"}
        )
        run = await evaluations.create_run(
            session,
            organization_id=organization_id,
            application_id=app.id,
            dataset_version_id=version.id,
            agent_model="agent-model:free",
        )
        await evaluations.add_case_result(
            session,
            evaluation_run_id=run.id,
            test_case_id=case.id,
            input=case.input,
            actual_output="answer",
            status=CaseStatus.EXECUTED,
        )
        return run.id


# -- cross-tenant isolation at the API boundary ------------------------------------------


async def test_cross_tenant_reads_are_404(clean_db: DatabaseManager, settings):
    async with clean_db.session() as session:
        org_a = await resolve_row(session, ORG_A)

    run_a = await _seed_run(clean_db, organization_id=org_a.id, name="tenant-a")

    client_a = await _make_client(clean_db, ORG_A)
    client_b = await _make_client(clean_db, ORG_B)
    try:
        # Org A sees its own resources everywhere.
        own = await client_a.get(f"/api/v1/evaluations/{run_a}")
        assert own.status_code == 200

        # Org B cannot read A's run / results / guardrails / reliability / regressions.
        for path in (
            f"/api/v1/evaluations/{run_a}",
            f"/api/v1/evaluations/{run_a}/results",
            f"/api/v1/evaluations/{run_a}/guardrails",
            f"/api/v1/evaluations/{run_a}/reliability",
            f"/api/v1/evaluations/{run_a}/regressions",
        ):
            response = await client_b.get(path)
            assert response.status_code == 404, path

        # A's run list does not leak into B's list.
        listed = await client_b.get("/api/v1/evaluations")
        assert listed.status_code == 200
        assert all(item["id"] != str(run_a) for item in listed.json()["items"])
    finally:
        await client_a.aclose()
        await client_b.aclose()


async def test_cross_tenant_comparison_is_rejected(clean_db: DatabaseManager, settings):
    async with clean_db.session() as session:
        org_a = await resolve_row(session, ORG_A)
        org_b = await resolve_row(session, ORG_B)

    run_a1 = await _seed_run(clean_db, organization_id=org_a.id, name="cmp-a1")
    run_b1 = await _seed_run(clean_db, organization_id=org_b.id, name="cmp-b1")

    # A requests a comparison involving B's run → the foreign run reads as
    # missing (typed NotFoundError), not a cross-tenant data leak.
    from evalyx.db.repositories.errors import NotFoundError
    from evalyx.evaluation.regression.service import RegressionService

    service = RegressionService(clean_db.session_factory)
    with pytest.raises(NotFoundError):
        await service.compare_runs(run_a1, run_b1, organization_id=org_a.id)


async def test_single_tenant_full_lifecycle_still_works(clean_db: DatabaseManager, settings):
    """Phase 1–13 flows work unchanged for an authenticated tenant."""
    client = await _make_client(clean_db, ORG_A)
    try:
        app_resp = await client.post(
            "/api/v1/applications", json={"name": f"app-{uuid.uuid4().hex[:8]}"}
        )
        assert app_resp.status_code == 201

        ds_resp = await client.post(
            "/api/v1/datasets", json={"name": f"ds-{uuid.uuid4().hex[:8]}"}
        )
        assert ds_resp.status_code == 201
        dataset = ds_resp.json()

        version_resp = await client.post(
            f"/api/v1/datasets/{dataset['id']}/versions", json={"version": 1}
        )
        assert version_resp.status_code == 201
    finally:
        await client.aclose()
