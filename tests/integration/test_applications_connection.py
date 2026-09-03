"""Generic application connection integration tests (Phase 15).

Live PostgreSQL (localhost:5433), no external network: Clerk is bypassed
with fixed AuthContexts per organization; the connection test endpoint's
outbound HTTP is mocked at the target boundary. Covers CRUD, tenant
isolation (IDOR), secret lifecycle, connection testing, and worker target
resolution.
"""

import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from evalyx.api.app import create_app
from evalyx.api.auth import AuthContext, AuthenticationError, OrganizationRole
from evalyx.api.dependencies import require_organization
from evalyx.core.config import Settings
from evalyx.core.encryption import (
    SecretEncryptor,
)
from evalyx.db.repositories import (
    ApplicationRepository,
    DatasetRepository,
    EvaluationRepository,
)
from evalyx.db.session import DatabaseManager

pytestmark = pytest.mark.integration

ORG_A = "org_test_a"
ORG_B = "org_test_b"

SECRET_A = "sk-app-" + "tenant-a-credential"
SECRET_B = "sk-app-" + "tenant-b-credential"

CONNECTION = {
    "endpoint": "https://93.184.216.34/v1/chat",
    "method": "POST",
    "auth": {"type": "bearer"},
    "request": {"input_field": "question"},
    "response_path": "answer",
}


def _tenant_override(db: DatabaseManager, clerk_org_id: str):
    async def _resolve():
        from evalyx.db.tenancy import require_organization as resolve_row

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
        statement = text(
            "TRUNCATE " + ", ".join(DOMAIN_TABLES) + " RESTART IDENTITY CASCADE"
        )
        await conn.execute(statement)
    yield db_manager


async def _client(db: DatabaseManager, clerk_org_id: str | None) -> AsyncClient:
    settings = Settings(auth_required=False)
    app = create_app(settings, database=db)
    if clerk_org_id is not None:
        app.dependency_overrides[require_organization] = _tenant_override(
            db, clerk_org_id
        )
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://testserver")


async def _unauthenticated_client(db: DatabaseManager) -> AsyncClient:
    """A client whose verifier rejects every request (401 behavior)."""

    class _RejectingVerifier:
        async def verify(self, request) -> AuthContext:
            raise AuthenticationError("Authentication failed.")

    settings = Settings(auth_required=False)
    app = create_app(settings, database=db)
    app.state.token_verifier = _RejectingVerifier()
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://testserver")


async def _create_http_application(
    client: AsyncClient,
    *,
    name: str,
    secret: str | None = None,
    connection: dict | None = None,
) -> dict:
    payload = {"name": name, "connection_type": "http"}
    if secret is not None:
        payload["secret"] = secret
    response = await client.post("/api/v1/applications", json=payload)
    assert response.status_code == 201, response.text
    application = response.json()
    if connection is not None:
        version_response = await client.post(
            f"/api/v1/applications/{application['id']}/versions",
            json={"version": "v1", "connection": connection},
        )
        assert version_response.status_code == 201, version_response.text
    return application


# -- unauthenticated access ------------------------------------------------------


async def test_unauthenticated_creation_is_401(clean_db):
    client = await _unauthenticated_client(clean_db)
    response = await client.post(
        "/api/v1/applications", json={"name": "nope", "connection_type": "http"}
    )
    assert response.status_code == 401


# -- CRUD ---------------------------------------------------------------------


async def test_create_list_get_patch_delete(clean_db):
    client = await _client(clean_db, ORG_A)
    created = await _create_http_application(
        client, name="crud-app", secret=SECRET_A, connection=CONNECTION
    )
    assert created["connection_type"] == "http"
    assert created["secret_configured"] is True

    listed = await client.get("/api/v1/applications")
    assert listed.status_code == 200
    assert listed.json()["total"] == 1
    assert listed.json()["items"][0]["name"] == "crud-app"

    fetched = await client.get(f"/api/v1/applications/{created['id']}")
    assert fetched.status_code == 200
    assert fetched.json()["secret_configured"] is True

    patched = await client.patch(
        f"/api/v1/applications/{created['id']}", json={"description": "updated"}
    )
    assert patched.status_code == 200
    assert patched.json()["description"] == "updated"

    versions = await client.get(f"/api/v1/applications/{created['id']}/versions")
    assert versions.status_code == 200
    assert (
        versions.json()["items"][0]["connection"]["endpoint"] == CONNECTION["endpoint"]
    )

    version_id = versions.json()["items"][0]["id"]
    single = await client.get(
        f"/api/v1/applications/{created['id']}/versions/{version_id}"
    )
    assert single.status_code == 200

    deleted = await client.delete(f"/api/v1/applications/{created['id']}")
    assert deleted.status_code == 204
    gone = await client.get(f"/api/v1/applications/{created['id']}")
    assert gone.status_code == 404


async def test_duplicate_name_conflicts(clean_db):
    client = await _client(clean_db, ORG_A)
    await _create_http_application(client, name="dup-app")
    response = await client.post(
        "/api/v1/applications", json={"name": "dup-app", "connection_type": "http"}
    )
    assert response.status_code == 409


async def test_connection_rejected_on_reference_application(clean_db):
    client = await _client(clean_db, ORG_A)
    created = await client.post(
        "/api/v1/applications", json={"name": "mlgpt-ref", "connection_type": "mlgpt"}
    )
    app_id = created.json()["id"]
    response = await client.post(
        f"/api/v1/applications/{app_id}/versions",
        json={"version": "v1", "connection": CONNECTION},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_connection"


async def test_invalid_connection_rejected_and_secret_not_echoed(clean_db):
    client = await _client(clean_db, ORG_A)
    created = await _create_http_application(client, name="bad-conn", secret=SECRET_A)
    response = await client.post(
        f"/api/v1/applications/{created['id']}/versions",
        json={
            "version": "v1",
            "connection": {
                "endpoint": "http://localhost/steal",
                "response_path": "answer",
            },
        },
    )
    assert response.status_code == 422
    assert SECRET_A not in response.text


# -- tenant isolation (IDOR) -----------------------------------------------------


async def test_cross_tenant_reads_are_404(clean_db):
    client_a = await _client(clean_db, ORG_A)
    client_b = await _client(clean_db, ORG_B)
    created = await _create_http_application(client_a, name="a-app", secret=SECRET_A)

    fetched = await client_b.get(f"/api/v1/applications/{created['id']}")
    assert fetched.status_code == 404

    listed = await client_b.get("/api/v1/applications")
    assert listed.status_code == 200
    assert listed.json()["total"] == 0  # B never sees A's applications


async def test_cross_tenant_modify_and_delete_are_404(clean_db):
    client_a = await _client(clean_db, ORG_A)
    client_b = await _client(clean_db, ORG_B)
    created = await _create_http_application(client_a, name="a-app", secret=SECRET_A)
    app_id = created["id"]

    patched = await client_b.patch(
        f"/api/v1/applications/{app_id}", json={"description": "hijacked"}
    )
    assert patched.status_code == 404

    deleted = await client_b.delete(f"/api/v1/applications/{app_id}")
    assert deleted.status_code == 404

    # Still intact for the owner.
    still_there = await client_a.get(f"/api/v1/applications/{app_id}")
    assert still_there.status_code == 200


async def test_cross_tenant_versions_and_rotation_are_404(clean_db):
    client_a = await _client(clean_db, ORG_A)
    client_b = await _client(clean_db, ORG_B)
    created = await _create_http_application(
        client_a, name="a-app", secret=SECRET_A, connection=CONNECTION
    )
    app_id = created["id"]

    versions = await client_b.get(f"/api/v1/applications/{app_id}/versions")
    assert versions.status_code == 404

    rotated = await client_b.patch(
        f"/api/v1/applications/{app_id}/connection", json={"secret": SECRET_B}
    )
    assert rotated.status_code == 404


async def test_cross_tenant_connection_test_is_404(clean_db):
    client_a = await _client(clean_db, ORG_A)
    client_b = await _client(clean_db, ORG_B)
    created = await _create_http_application(
        client_a, name="a-app", secret=SECRET_A, connection=CONNECTION
    )
    tested = await client_b.post(
        f"/api/v1/applications/{created['id']}/test",
        json={"prompt": "hi"},
    )
    assert tested.status_code == 404


async def test_cross_tenant_evaluation_submission_is_404(clean_db):
    """Organization B cannot run evaluations against Organization A's
    generic application (uniform 404, no existence leak)."""
    client_a = await _client(clean_db, ORG_A)
    client_b = await _client(clean_db, ORG_B)
    created = await _create_http_application(
        client_a, name="a-app", secret=SECRET_A, connection=CONNECTION
    )

    # Seed a dataset + version for B directly (tenant-scoped repository).
    from evalyx.db.tenancy import get_organization_by_clerk_id

    # First API call auto-provisions B's local organization row.
    await client_b.get("/api/v1/applications")
    async with clean_db.session() as session:
        org_b = await get_organization_by_clerk_id(session, ORG_B)
        assert org_b is not None
        dataset = await DatasetRepository().create(
            session, organization_id=org_b.id, name="b-dataset"
        )
        version = await DatasetRepository().create_version(
            session, dataset_id=dataset.id, version=1
        )
        await DatasetRepository().add_test_case(
            session, dataset_version_id=version.id, name="c0", input={"prompt": "x"}
        )
        dataset_version_id = version.id

    submitted = await client_b.post(
        "/api/v1/evaluations",
        json={
            "application_id": created["id"],
            "dataset_version_id": str(dataset_version_id),
            "agent_model": "application:generic",
            "judge_model": None,
        },
    )
    assert submitted.status_code == 404


# -- secret lifecycle ------------------------------------------------------------


async def test_secret_never_returned_from_api(clean_db):
    client = await _client(clean_db, ORG_A)
    created = await _create_http_application(
        client, name="secret-app", secret=SECRET_A, connection=CONNECTION
    )
    app_id = created["id"]

    for endpoint in (f"/api/v1/applications/{app_id}", "/api/v1/applications"):
        response = await client.get(endpoint)
        assert response.status_code == 200
        assert SECRET_A not in response.text
        assert f'"secret": "{SECRET_A}"' not in response.text

    versions = await client.get(f"/api/v1/applications/{app_id}/versions")
    assert versions.status_code == 200
    assert SECRET_A not in versions.text


async def test_plaintext_secret_never_persisted(clean_db, db_manager):
    client = await _client(clean_db, ORG_A)
    created = await _create_http_application(
        client, name="encrypted-app", secret=SECRET_A, connection=CONNECTION
    )

    async with db_manager.session() as session:
        row = await ApplicationRepository().get_in_organization(
            session, uuid.UUID(created["id"]), organization_id=await _org_id(session)
        )
        assert row is not None
        assert row.encrypted_secret is not None
        assert SECRET_A not in (row.encrypted_secret or "")
        assert SECRET_A not in (row.secret_metadata or {})
        # The envelope decrypts back to the original value (proves the stored
        # ciphertext is the credential, just never in plaintext).
        settings = Settings(auth_required=False)
        encryptor = SecretEncryptor.from_settings(settings)
        assert encryptor.decrypt(row.encrypted_secret) == SECRET_A


async def test_secret_rotation_replaces_ciphertext(clean_db, db_manager):
    client = await _client(clean_db, ORG_A)
    created = await _create_http_application(
        client, name="rotate-app", secret=SECRET_A, connection=CONNECTION
    )
    app_id = created["id"]

    rotated = await client.patch(
        f"/api/v1/applications/{app_id}/connection", json={"secret": SECRET_B}
    )
    assert rotated.status_code == 200
    assert rotated.json()["secret_configured"] is True
    assert SECRET_A not in rotated.text
    assert SECRET_B not in rotated.text

    async with db_manager.session() as session:
        row = await ApplicationRepository().get_in_organization(
            session, uuid.UUID(app_id), organization_id=await _org_id(session)
        )
        assert row is not None
        assert row.encrypted_secret is not None
        assert SECRET_A not in (row.encrypted_secret or "")
        assert SECRET_B not in (row.encrypted_secret or "")
        settings = Settings(auth_required=False)
        encryptor = SecretEncryptor.from_settings(settings)
        assert encryptor.decrypt(row.encrypted_secret) == SECRET_B


async def test_test_endpoint_without_configured_secret_is_409(clean_db):
    client = await _client(clean_db, ORG_A)
    created = await _create_http_application(client, name="no-secret-app", connection=CONNECTION)
    response = await client.post(
        f"/api/v1/applications/{created['id']}/test", json={"prompt": "hi"}
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "connection_not_ready"


async def _org_id(session):
    from evalyx.db.tenancy import get_organization_by_clerk_id

    org = await get_organization_by_clerk_id(session, ORG_A)
    assert org is not None
    return org.id


# -- connection test endpoint ------------------------------------------------------


class FakeApplicationTarget:
    """Deterministic stand-in for the outbound HTTP target (no network)."""

    def __init__(self, response=None, error=None) -> None:
        self._response = response
        self._error = error
        self.closed = False

    async def invoke(self, prompt: str):
        if self._error is not None:
            raise self._error
        return self._response

    async def close(self) -> None:
        self.closed = True


async def test_connection_test_success(clean_db, monkeypatch):
    from evalyx.api.routers import applications as applications_router
    from evalyx.application.base import ApplicationResponse

    fake = FakeApplicationTarget(
        ApplicationResponse(content="the answer preview", latency_ms=42, status_code=200)
    )
    monkeypatch.setattr(applications_router, "build_http_target", lambda *a, **k: fake)

    client = await _client(clean_db, ORG_A)
    created = await _create_http_application(
        client, name="test-ok-app", secret=SECRET_A, connection=CONNECTION
    )
    response = await client.post(
        f"/api/v1/applications/{created['id']}/test", json={"prompt": "hi"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["latency_ms"] == 42
    assert body["http_status"] == 200
    assert body["preview"].startswith("the answer preview")
    assert fake.closed is True


async def test_connection_test_failure_classified(clean_db, monkeypatch):
    from evalyx.api.routers import applications as applications_router
    from evalyx.application.base import ApplicationInvocationError

    fake = FakeApplicationTarget(
        error=ApplicationInvocationError(
            "Application timed out.", category="timeout", attempts=3
        )
    )
    monkeypatch.setattr(applications_router, "build_http_target", lambda *a, **k: fake)

    client = await _client(clean_db, ORG_A)
    created = await _create_http_application(
        client, name="test-fail-app", secret=SECRET_A, connection=CONNECTION
    )
    response = await client.post(
        f"/api/v1/applications/{created['id']}/test", json={"prompt": "hi"}
    )
    assert response.status_code == 200  # failures are structured results, not HTTP errors
    body = response.json()
    assert body["success"] is False
    assert body["failure"]["category"] == "timeout"
    assert body["failure"]["attempts"] == 3
    assert SECRET_A not in response.text
    assert fake.closed is True


# -- worker target resolution (DB-driven) ------------------------------------------


async def test_resolve_run_target_builds_http_target(clean_db, db_manager):
    from evalyx.application.generic_http import HTTPApplicationTarget
    from evalyx.application.resolve import resolve_run_target

    client = await _client(clean_db, ORG_A)
    created = await _create_http_application(
        client, name="resolve-app", secret=SECRET_A, connection=CONNECTION
    )
    version = (
        await client.get(f"/api/v1/applications/{created['id']}/versions")
    ).json()["items"][0]

    # Seed a dataset version + run (tenant-scoped repositories).
    async with db_manager.session() as session:
        org = await _org_id(session)
        dataset = await DatasetRepository().create(
            session, organization_id=org, name="resolve-ds"
        )
        ds_version = await DatasetRepository().create_version(
            session, dataset_id=dataset.id, version=1
        )
        run = await EvaluationRepository().create_run(
            session,
            organization_id=org,
            application_id=uuid.UUID(created["id"]),
            application_version_id=uuid.UUID(version["id"]),
            dataset_version_id=ds_version.id,
            agent_model="application:generic",
            configuration_snapshot={},
        )
        run_id = run.id

    settings = Settings(auth_required=False)
    async with db_manager.session() as session:
        from evalyx.db.repositories import EvaluationRepository as EvalRepo

        loaded_run = await EvalRepo().get_run(session, run_id)
        assert loaded_run is not None
        target = await resolve_run_target(session, loaded_run, settings)
        assert isinstance(target, HTTPApplicationTarget)
        await target.close()