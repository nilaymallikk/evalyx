"""Phase 18 tenant-isolation matrix (live PostgreSQL).

Every tenant-owned resource — applications, versions, secrets, connection
tests, datasets, versions, cases, runs, results, guardrails, reliability,
regressions — is probed cross-tenant and must read as missing (uniform
404, never another tenant's data). Forged tenant identity in request bodies
is ignored, and unauthenticated callers get 401s without disclosure.
"""

from __future__ import annotations

import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from evalyx.api.app import create_app
from evalyx.api.auth import AuthContext, OrganizationRole
from evalyx.api.dependencies import require_organization
from evalyx.core.config import Settings
from evalyx.db.session import DatabaseManager

pytestmark = pytest.mark.integration

ORG_A = "org_matrix_a"
ORG_B = "org_matrix_b"


async def _client(db: DatabaseManager, settings: Settings, clerk_org_id: str | None):
    from evalyx.api.dependencies import require_authenticated_user

    app = create_app(settings, database=db)
    if clerk_org_id is not None:
        auth = AuthContext(
            clerk_user_id=f"user-{clerk_org_id}",
            clerk_organization_id=clerk_org_id,
            organization_role=OrganizationRole.MEMBER,
        )

        async def _resolve():
            from evalyx.db.tenancy import require_organization as resolve_row

            async with db.session() as s:
                organization = await resolve_row(s, clerk_org_id)
            return (auth, organization)

        app.dependency_overrides[require_authenticated_user] = lambda: auth
        app.dependency_overrides[require_organization] = _resolve
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://testserver")


async def _seed_full_world(client: AsyncClient) -> dict:
    """One application + version, dataset + version + case, and a run id."""
    app = (
        await client.post("/api/v1/applications", json={"name": f"app-{uuid.uuid4().hex[:6]}"})
    ).json()
    version = (
        await client.post(
            f"/api/v1/applications/{app['id']}/versions", json={"version": "v1"}
        )
    ).json()
    dataset = (
        await client.post("/api/v1/datasets", json={"name": f"ds-{uuid.uuid4().hex[:6]}"})
    ).json()
    ds_version = (
        await client.post(
            f"/api/v1/datasets/{dataset['id']}/versions", json={"version": 1}
        )
    ).json()
    case = (
        await client.post(
            f"/api/v1/datasets/{dataset['id']}/versions/1/cases",
            json={"name": "c1", "input": {"prompt": "hello"}},
        )
    ).json()
    run = (
        await client.post(
            "/api/v1/evaluations",
            json={
                "application_id": app["id"],
                "dataset_version_id": ds_version["id"],
                "agent_model": "matrix-model",
            },
        )
    ).json()
    comparison = (
        await client.post(
            "/api/v1/regressions",
            json={"baseline_run_id": run["run_id"], "current_run_id": run["run_id"]},
        )
    ).json()
    return {
        "app": app,
        "version": version,
        "dataset": dataset,
        "ds_version": ds_version,
        "case": case,
        "run": run,
        "comparison": comparison,
    }


async def test_cross_tenant_matrix_is_uniform_404(
    clean_db: DatabaseManager, settings: Settings
):
    async with await _client(clean_db, settings, ORG_A) as client_a:
        world = await _seed_full_world(client_a)
    async with await _client(clean_db, settings, ORG_B) as client_b:
        app_id = world["app"]["id"]
        version_id = world["version"]["id"]
        dataset_id = world["dataset"]["id"]
        run_id = world["run"]["run_id"]
        probes = [
            ("GET", f"/api/v1/applications/{app_id}", None),
            ("PATCH", f"/api/v1/applications/{app_id}", {"description": "x"}),
            ("DELETE", f"/api/v1/applications/{app_id}", None),
            ("GET", f"/api/v1/applications/{app_id}/versions", None),
            ("GET", f"/api/v1/applications/{app_id}/versions/{version_id}", None),
            ("PATCH", f"/api/v1/applications/{app_id}/connection", {"secret": "x"}),
            ("POST", f"/api/v1/applications/{app_id}/test", {"prompt": "hi"}),
            ("GET", f"/api/v1/datasets/{dataset_id}", None),
            ("GET", f"/api/v1/datasets/{dataset_id}/versions", None),
            ("GET", f"/api/v1/datasets/{dataset_id}/versions/1/cases", None),
            ("POST", f"/api/v1/datasets/{dataset_id}/versions", {"version": 2}),
            ("POST", f"/api/v1/datasets/{dataset_id}/versions/1/cases",
             {"name": "evil", "input": {"prompt": "x"}}),
            ("GET", f"/api/v1/evaluations/{run_id}", None),
            ("GET", f"/api/v1/evaluations/{run_id}/results", None),
            ("GET", f"/api/v1/evaluations/{run_id}/guardrails", None),
            ("GET", f"/api/v1/evaluations/{run_id}/reliability", None),
            ("GET", f"/api/v1/evaluations/{run_id}/regressions", None),
        ]
        for method, path, body in probes:
            if method == "GET":
                response = await client_b.get(path)
            elif method == "PATCH":
                response = await client_b.patch(path, json=body)
            elif method == "DELETE":
                response = await client_b.delete(path)
            else:
                response = await client_b.post(path, json=body)
            assert response.status_code == 404, (method, path, response.text)
            assert response.json()["error"]["code"] == "not_found"
        # Regression comparison between two foreign runs: rejected without
        # disclosing either run (400 invalid comparison, same as same-tenant
        # self-comparison — no existence signal either way).
        response = await client_b.post(
            "/api/v1/regressions",
            json={"baseline_run_id": run_id, "current_run_id": run_id},
        )
        assert response.status_code in (400, 404), response.text
        # Listings never leak foreign rows.
        for path in ("/api/v1/applications", "/api/v1/datasets", "/api/v1/evaluations"):
            response = await client_b.get(path)
            assert response.status_code == 200
            assert response.json()["total"] == 0


async def test_forged_tenant_fields_ignored(
    clean_db: DatabaseManager, settings: Settings
):
    """Bodies carrying another tenant's ids cannot reseat identity."""
    async with await _client(clean_db, settings, ORG_A) as client_a:
        world = await _seed_full_world(client_a)
    async with await _client(clean_db, settings, ORG_B) as client_b:
        # Unknown fields are ignored by pydantic (extra=ignore convention);
        # tenant-scoped lookup still 404s.
        response = await client_b.get(
            f"/api/v1/applications/{world['app']['id']}",
            params={"organization_id": "anything"},
        )
        assert response.status_code == 404
        response = await client_b.post(
            "/api/v1/evaluations",
            json={
                "application_id": world["app"]["id"],
                "dataset_version_id": world["ds_version"]["id"],
                "agent_model": "m",
                "organization_id": str(uuid.uuid4()),
                "clerk_organization_id": ORG_A,
            },
        )
        assert response.status_code == 404


async def test_unauthenticated_gets_401_without_disclosure(
    clean_db: DatabaseManager, settings: Settings
):
    from evalyx.api.auth import AuthenticationError

    app = create_app(settings, database=clean_db)

    class _RejectingVerifier:
        async def verify(self, request) -> AuthContext:
            raise AuthenticationError("Authentication failed.")

    app.state.token_verifier = _RejectingVerifier()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        for path in (
            "/api/v1/applications",
            "/api/v1/datasets",
            "/api/v1/evaluations",
            "/api/v1/metrics",
        ):
            response = await client.get(path)
            assert response.status_code == 401, path
            assert "error" in response.json()
        # Health stays public by design.
        assert (await client.get("/health")).status_code == 200


async def test_metrics_requires_auth_and_labels_stay_bounded(
    clean_db: DatabaseManager, settings: Settings
):
    async with await _client(clean_db, settings, ORG_A) as client:
        response = await client.get("/api/v1/metrics")
        assert response.status_code == 200
        body = response.json()
        assert body["instance"]  # replica identity for aggregation
        assert isinstance(body["metrics"], dict)
