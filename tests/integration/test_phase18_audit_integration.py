"""Phase 18 audit-log integration tests (live PostgreSQL).

End-to-end audit rows for mutations, secret rotation, evaluation
submission, authorization failures, and quota denials — with actor,
organization, action, resource, result, request id, and timestamp — plus
the secret-freedom property of every stored row and the retention cleanup.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select, text

from evalyx.api.app import create_app
from evalyx.api.auth import AuthContext, OrganizationRole
from evalyx.api.dependencies import require_organization
from evalyx.core.config import Settings
from evalyx.db.models import Organization
from evalyx.db.models.governance import AuditEvent
from evalyx.db.session import DatabaseManager

pytestmark = pytest.mark.integration

ORG = "org_audit_e2e"


async def _api(db: DatabaseManager, settings: Settings):
    app = create_app(settings, database=db)

    async def _resolve():
        from evalyx.db.tenancy import require_organization as resolve_row

        async with db.session() as s:
            organization = await resolve_row(s, ORG)
        return (
            AuthContext(
                clerk_user_id="audit-user",
                clerk_organization_id=ORG,
                organization_role=OrganizationRole.ADMIN,
            ),
            organization,
        )

    app.dependency_overrides[require_organization] = _resolve
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://testserver"), app


async def _rows(db: DatabaseManager, action: str | None = None) -> list[AuditEvent]:
    async with db.session() as session:
        query = select(AuditEvent).order_by(AuditEvent.created_at)
        if action is not None:
            query = query.where(AuditEvent.action == action)
        return list((await session.execute(query)).scalars().all())


async def test_mutations_leave_audit_trail(clean_db: DatabaseManager, settings: Settings):
    client, _ = await _api(clean_db, settings)
    async with client:
        created = (
            await client.post(
                "/api/v1/applications",
                json={"name": "audited-app", "secret": "top-secret-value"},
            )
        ).json()
        await client.patch(
            f"/api/v1/applications/{created['id']}",
            json={"description": "new description"},
        )
        await client.post(
            "/api/v1/applications",
            json={"name": "gone-app"},
        )
        gone_id = (
            await client.get("/api/v1/applications", params={"limit": 100})
        ).json()["items"][-1]["id"]
        await client.delete(f"/api/v1/applications/{gone_id}")

    events = await _rows(clean_db)
    by_action = {}
    for event in events:
        by_action.setdefault(event.action, []).append(event)
    assert set(by_action) >= {
        "application.create",
        "application.update",
        "application.delete",
    }
    created_event = by_action["application.create"][0]
    assert created_event.result == "allowed"
    assert created_event.clerk_user_id == "audit-user"
    assert created_event.organization_id is not None
    assert created_event.request_id  # correlated with the HTTP request
    assert isinstance(created_event.created_at, datetime)
    assert created_event.details.get("name") == "audited-app"
    # Secret-freedom across every stored row: no credential material, no
    # descriptions (free-form user text that could hide pasted secrets).
    for event in events:
        blob = str(event.details) + (event.resource_id or "")
        assert "top-secret-value" not in blob
        assert "new description" not in blob


async def test_secret_rotation_audited_without_secret(
    clean_db: DatabaseManager, settings: Settings
):
    client, _ = await _api(clean_db, settings)
    async with client:
        created = (
            await client.post(
                "/api/v1/applications",
                json={"name": "rot-app", "secret": "first-secret"},
            )
        ).json()
        rotated = await client.patch(
            f"/api/v1/applications/{created['id']}/connection",
            json={"secret": "second-secret"},
        )
        assert rotated.status_code == 200

    events = await _rows(clean_db, "application.secret.rotate")
    assert len(events) == 1
    (event,) = events
    assert event.result == "allowed"
    assert event.resource_id == created["id"]
    assert "first-secret" not in str(event.details)
    assert "second-secret" not in str(event.details)
    assert event.details.get("key_version") == "v2"
    assert event.details.get("key_id")


async def test_evaluation_submit_audited(
    clean_db: DatabaseManager, settings: Settings
):
    client, _ = await _api(clean_db, settings)
    async with client:
        app = (
            await client.post("/api/v1/applications", json={"name": "eval-audit-app"})
        ).json()
        dataset = (
            await client.post("/api/v1/datasets", json={"name": "eval-audit-ds"})
        ).json()
        version = (
            await client.post(
                f"/api/v1/datasets/{dataset['id']}/versions", json={"version": 1}
            )
        ).json()
        await client.post(
            f"/api/v1/datasets/{dataset['id']}/versions/1/cases",
            json={"name": "c1", "input": {"prompt": "secret-prompt-must-not-audit"}},
        )
        response = await client.post(
            "/api/v1/evaluations",
            json={
                "application_id": app["id"],
                "dataset_version_id": version["id"],
                "agent_model": "audit-model",
            },
        )
        assert response.status_code == 202, response.text

    events = await _rows(clean_db, "evaluation.submit")
    assert len(events) == 1
    (event,) = events
    assert event.resource_id == response.json()["run_id"]
    assert event.details.get("agent_model") == "audit-model"
    assert "secret-prompt-must-not-audit" not in str(event.details)


async def test_quota_denial_audited(
    clean_db: DatabaseManager, settings: Settings
):
    tight = settings.model_copy(update={"quota_max_applications": 1})
    client, _ = await _api(clean_db, tight)
    async with client:
        assert (await client.post("/api/v1/applications", json={"name": "a1"})).status_code == 201
        denied = await client.post("/api/v1/applications", json={"name": "a2"})
        assert denied.status_code == 429

    events = await _rows(clean_db, "quota.exceeded")
    assert len(events) == 1
    (event,) = events
    assert event.result == "denied"
    assert event.resource_id == "applications"
    assert event.details.get("limit") == 1


async def test_organization_required_denial_audited(
    clean_db: DatabaseManager, settings: Settings
):
    """Authenticated user without an active org → 403 + durable audit row."""
    from evalyx.api.dependencies import require_authenticated_user

    app = create_app(settings, database=clean_db)
    app.dependency_overrides[require_authenticated_user] = lambda: AuthContext(
        clerk_user_id="orgless-user",
        clerk_organization_id=None,
        organization_role=None,
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/api/v1/applications")
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "organization_required"

    events = await _rows(clean_db, "auth.organization_required")
    assert len(events) == 1
    (event,) = events
    assert event.result == "denied"
    assert event.clerk_user_id == "orgless-user"
    assert event.organization_id is None


async def test_retention_cleanup_deletes_only_expired(
    clean_db: DatabaseManager, settings: Settings
):
    async with clean_db.session() as session:
        org = (
            await session.scalars(
                select(Organization).filter_by(clerk_organization_id=ORG)
            )
        ).first()
        if org is None:
            org = Organization(clerk_organization_id=ORG, name="retention org")
            session.add(org)
            await session.commit()
            await session.refresh(org)
        old = AuditEvent(
            organization_id=org.id,
            clerk_user_id="u",
            action="application.create",
            result="allowed",
            details={},
        )
        fresh = AuditEvent(
            organization_id=org.id,
            clerk_user_id="u",
            action="application.create",
            result="allowed",
            details={},
        )
        session.add_all([old, fresh])
        await session.commit()
        await session.refresh(old)
        await session.refresh(fresh)
        # Backdate the old row past the retention horizon (documented SQL
        # uses the same predicate operators run in production).
        await session.execute(
            text("UPDATE audit_events SET created_at = :cutoff WHERE id = :id"),
            {
                "cutoff": datetime.now(UTC) - timedelta(days=settings.audit_retention_days + 1),
                "id": old.id,
            },
        )
        await session.commit()
        deleted = await session.execute(
            text(
                "DELETE FROM audit_events WHERE created_at < :cutoff"
            ),
            {"cutoff": datetime.now(UTC) - timedelta(days=settings.audit_retention_days)},
        )
        await session.commit()
        assert deleted.rowcount == 1
        remaining = (
            await session.execute(select(func.count()).select_from(AuditEvent))
        ).scalar_one()
        assert remaining == 1
        survivor = (await session.execute(select(AuditEvent))).scalars().one()
        assert survivor.id == fresh.id
