"""API integration tests: full HTTP surface against live PostgreSQL (5433).

No live LLM and no real worker: ``run_evaluation.delay`` is monkeypatched
(a dedicated worker-path integration test already covers the complete
pipeline in ``test_worker_jobs.py``). Redis is never contacted — the ASGI
transport does not run lifespan events.
"""

import uuid
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text

from evalyx.api.app import create_app
from evalyx.api.auth import AuthContext, OrganizationRole
from evalyx.api.dependencies import get_evaluation_service, require_organization
from evalyx.api.services import EvaluationService
from evalyx.core.config import Settings
from evalyx.db.models import CaseStatus, GuardrailStatus, RunStatus
from evalyx.db.repositories import EvaluationRepository
from evalyx.db.session import DatabaseManager
from evalyx.worker import tasks as worker_tasks

pytestmark = pytest.mark.integration


@pytest.fixture
async def api(clean_db: DatabaseManager, settings: Settings):
    """HTTP client wired to the truncated test database (no lifespan).

    Tenant authentication is bypassed with a fixed organization; the
    dedicated multi-tenancy suite (test_api_multi_tenant.py) exercises two
    organizations and cross-tenant rejection.
    """
    app = create_app(settings, database=clean_db)
    app.dependency_overrides[require_organization] = _fake_require_organization(
        clean_db, clerk_org_id="org_integration_test"
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client, app


def _fake_require_organization(db: DatabaseManager, *, clerk_org_id: str):
    """Dependency override resolving the fake Clerk org to a real local row."""

    async def _resolve():
        from evalyx.db.tenancy import require_organization as resolve_row

        async with db.session() as s:
            organization = await resolve_row(s, clerk_org_id)
        return (
            AuthContext(
                clerk_user_id="integration-user",
                clerk_organization_id=clerk_org_id,
                organization_role=OrganizationRole.ADMIN,
            ),
            organization,
        )

    return _resolve


async def seed_application(client: AsyncClient, name: str) -> dict:
    response = await client.post(
        "/api/v1/applications", json={"name": name, "description": "api test"}
    )
    assert response.status_code == 201, response.text
    return response.json()


async def seed_dataset(client: AsyncClient, name: str, case_names: list[str]) -> dict:
    response = await client.post("/api/v1/datasets", json={"name": name})
    assert response.status_code == 201, response.text
    dataset = response.json()
    response = await client.post(
        f"/api/v1/datasets/{dataset['id']}/versions", json={"version": 1}
    )
    assert response.status_code == 201, response.text
    version = response.json()
    for case_name in case_names:
        response = await client.post(
            f"/api/v1/datasets/{dataset['id']}/versions/1/cases",
            json={"name": case_name, "input": {"prompt": f"input for {case_name}"}},
        )
        assert response.status_code == 201, response.text
    return {"dataset": dataset, "version": version}


async def seed_case_ids(db: DatabaseManager, dataset_version_id: uuid.UUID) -> dict[str, uuid.UUID]:
    async with db.session() as session:
        from sqlalchemy import select

        from evalyx.db.models import TestCase

        result = await session.execute(
            select(TestCase).where(TestCase.dataset_version_id == dataset_version_id)
        )
        return {case.name: case.id for case in result.scalars().all()}


async def seed_completed_run(
    db: DatabaseManager,
    *,
    application_id: uuid.UUID,
    dataset_version_id: uuid.UUID,
    outcomes: dict[str, tuple[CaseStatus, int]],
    case_metrics: dict[str, dict] | None = None,
    case_ids: dict[str, uuid.UUID],
    guardrails: dict[str, list[tuple[str, GuardrailStatus]]] | None = None,
    judge_model: str | None = None,
) -> uuid.UUID:
    """Seed one completed run through repositories (worker-equivalent state)."""
    evaluations = EvaluationRepository()
    async with db.session() as session:
        from evalyx.db.tenancy import require_organization as resolve_row

        organization = await resolve_row(session, "org_integration_test")
        run = await evaluations.create_run(
            session,
            organization_id=organization.id,
            application_id=application_id,
            dataset_version_id=dataset_version_id,
            agent_model="agent-a",
            judge_model=judge_model,
            configuration_snapshot={"temperature": 0.2, "max_tokens": 512},
        )
        for name, (status, latency) in outcomes.items():
            case_result = await evaluations.add_case_result(
                session,
                evaluation_run_id=run.id,
                test_case_id=case_ids[name],
                input={"prompt": name},
                status=status,
                actual_output=None if status is CaseStatus.ERROR else f"reply:{name}",
                latency_ms=latency if status is not CaseStatus.ERROR else None,
                error="provider exploded" if status is CaseStatus.ERROR else None,
                metrics=(case_metrics or {}).get(name),
            )
            for guardrail_name, verdict in (guardrails or {}).get(name, []):
                await evaluations.add_guardrail_result(
                    session,
                    evaluation_case_result_id=case_result.id,
                    name=guardrail_name,
                    passed=verdict is GuardrailStatus.PASSED,
                    status=verdict,
                    metadata={"categories": [], "count": 0},
                )
        await evaluations.update_status(session, run, RunStatus.COMPLETED)
    return run.id


async def evaluation_snapshot(db: DatabaseManager) -> tuple:
    async with db.engine.connect() as conn:
        runs = (
            await conn.execute(text(
                "SELECT id, status, agent_model, configuration_snapshot, started_at, "
                "completed_at FROM evaluation_runs ORDER BY id"
            ))
        ).all()
        cases = (
            await conn.execute(text(
                "SELECT id, evaluation_run_id, test_case_id, status, latency_ms, actual_output "
                "FROM evaluation_case_results ORDER BY id"
            ))
        ).all()
        guardrails = (
            await conn.execute(text(
                "SELECT id, evaluation_case_result_id, name, status, passed "
                "FROM guardrail_results ORDER BY id"
            ))
        ).all()
    return runs, cases, guardrails


# -- applications -----------------------------------------------------------------


async def test_application_lifecycle(api):
    client, _ = api
    created = await seed_application(client, "support-bot")

    fetched = await client.get(f"/api/v1/applications/{created['id']}")
    assert fetched.status_code == 200
    assert fetched.json()["name"] == "support-bot"

    version_response = await client.post(
        f"/api/v1/applications/{created['id']}/versions",
        json={"version": "v1", "configuration": {"prompt_template": "tpl-1"}},
    )
    assert version_response.status_code == 201
    assert version_response.json()["configuration"] == {"prompt_template": "tpl-1"}

    versions = await client.get(f"/api/v1/applications/{created['id']}/versions")
    assert versions.status_code == 200
    assert versions.json()["total"] == 1


async def test_duplicate_application_name_is_conflict(api):
    client, _ = api
    await seed_application(client, "dup-app")
    response = await client.post("/api/v1/applications", json={"name": "dup-app"})
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "conflict"


async def test_missing_application_is_404(api):
    client, _ = api
    response = await client.get(f"/api/v1/applications/{uuid.uuid4()}")
    assert response.status_code == 404
    body = response.json()
    assert body["error"]["code"] == "not_found"
    assert "Application" in body["error"]["message"]


async def test_duplicate_application_version_is_conflict(api):
    client, _ = api
    app_data = await seed_application(client, "versioned-app")
    payload = {"version": "v1"}
    first = await client.post(f"/api/v1/applications/{app_data['id']}/versions", json=payload)
    assert first.status_code == 201
    second = await client.post(f"/api/v1/applications/{app_data['id']}/versions", json=payload)
    assert second.status_code == 409
    assert second.json()["error"]["code"] == "duplicate_version"


async def test_application_version_configuration_strips_secrets(api):
    client, _ = api
    fake_marker = "fake-" + "do-not-persist"  # fake secret-shaped value
    app_data = await seed_application(client, "secret-config-app")
    response = await client.post(
        f"/api/v1/applications/{app_data['id']}/versions",
        json={
            "version": "v1",
            "configuration": {
                "temperature": 0.2,
                "api_key": fake_marker,
            },
        },
    )
    assert response.status_code == 201
    assert response.json()["configuration"] == {"temperature": 0.2}
    assert fake_marker not in response.text


# -- datasets & test cases ----------------------------------------------------------


async def test_dataset_lifecycle_and_immutable_versions(api):
    client, _ = api
    created = await seed_dataset(client, "support-dataset", ["c1", "c2", "c3"])

    fetched = await client.get(f"/api/v1/datasets/{created['dataset']['id']}")
    assert fetched.status_code == 200

    cases = await client.get(
        f"/api/v1/datasets/{created['dataset']['id']}/versions/1/cases"
    )
    assert cases.status_code == 200
    assert cases.json()["total"] == 3
    assert [case["name"] for case in cases.json()["items"]] == ["c1", "c2", "c3"]

    duplicate = await client.post(
        f"/api/v1/datasets/{created['dataset']['id']}/versions", json={"version": 1}
    )
    assert duplicate.status_code == 409
    assert duplicate.json()["error"]["code"] == "duplicate_version"


async def test_test_case_metadata_strips_secrets(api):
    client, _ = api
    created = await seed_dataset(client, "metadata-dataset", [])
    response = await client.post(
        f"/api/v1/datasets/{created['dataset']['id']}/versions/1/cases",
        json={
            "name": "case-with-metadata",
            "input": {"prompt": "hi"},
            "metadata": {"owner": "team-a", "auth_token": "tok-leak"},
        },
    )
    assert response.status_code == 201
    assert response.json()["metadata"] == {"owner": "team-a"}
    assert "tok-leak" not in response.text


async def test_cases_for_missing_version_are_404(api):
    client, _ = api
    created = await seed_dataset(client, "ghost-version-dataset", [])
    response = await client.get(
        f"/api/v1/datasets/{created['dataset']['id']}/versions/99/cases"
    )
    assert response.status_code == 404


# -- evaluation submission ------------------------------------------------------------


async def test_submit_evaluation_enqueues_existing_celery_task(api, monkeypatch):
    client, _ = api
    app_data = await seed_application(client, "submit-app")
    data = await seed_dataset(client, "submit-dataset", ["c1"])
    payload = {
        "application_id": app_data["id"],
        "dataset_version_id": data["version"]["id"],
        "agent_model": "nvidia/nemotron-3-ultra-550b-a55b:free",
        "configuration_snapshot": {"temperature": 0.2, "max_tokens": 512},
    }

    delay_calls: list[str] = []

    def fake_delay(run_id: str) -> SimpleNamespace:
        delay_calls.append(run_id)
        return SimpleNamespace(id="task-xyz-789")

    monkeypatch.setattr(worker_tasks.run_evaluation, "delay", fake_delay)

    response = await client.post("/api/v1/evaluations", json=payload)
    assert response.status_code == 202, response.text
    body = response.json()
    assert body["status"] == "pending"
    assert body["task_id"] == "task-xyz-789"
    assert body["status_url"] == f"/api/v1/evaluations/{body['run_id']}"
    # The existing task receives exactly the persisted run id.
    assert delay_calls == [body["run_id"]]

    status_response = await client.get(body["status_url"])
    assert status_response.status_code == 200
    summary = status_response.json()
    assert summary["status"] == "pending"
    assert summary["counts"] == {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "error": 0,
        "executed": 0,
    }
    assert summary["agent_model"] == payload["agent_model"]
    assert summary["configuration_snapshot"] == {"temperature": 0.2, "max_tokens": 512}


async def test_submit_evaluation_with_unknown_references_is_404(api):
    client, _ = api
    response = await client.post(
        "/api/v1/evaluations",
        json={
            "application_id": str(uuid.uuid4()),
            "dataset_version_id": str(uuid.uuid4()),
            "agent_model": "m",
        },
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


async def test_submit_evaluation_with_foreign_application_version_is_404(api):
    client, _ = api
    app_a = await seed_application(client, "foreign-app-a")
    app_b = await seed_application(client, "foreign-app-b")
    version_b = await client.post(
        f"/api/v1/applications/{app_b['id']}/versions", json={"version": "v1"}
    )
    data = await seed_dataset(client, "foreign-dataset", ["c1"])

    response = await client.post(
        "/api/v1/evaluations",
        json={
            "application_id": app_a["id"],
            "application_version_id": version_b.json()["id"],
            "dataset_version_id": data["version"]["id"],
            "agent_model": "m",
        },
    )
    assert response.status_code == 404


async def test_submit_evaluation_enqueue_failure_marks_run_failed(api):
    """Broker failure → 503 and the run is never left pretending to be queued."""
    client, app = api
    app_data = await seed_application(client, "enqueue-fail-app")
    data = await seed_dataset(client, "enqueue-fail-dataset", ["c1"])
    payload = {
        "application_id": app_data["id"],
        "dataset_version_id": data["version"]["id"],
        "agent_model": "m",
    }

    def broken_enqueue(run_id: uuid.UUID) -> str:
        raise ConnectionError("broker unreachable")

    # Real service, real PostgreSQL — only the broker call fails.
    app.dependency_overrides[get_evaluation_service] = lambda: EvaluationService(
        app.state.database.session_factory, enqueue=broken_enqueue
    )

    response = await client.post("/api/v1/evaluations", json=payload)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "evaluation_enqueue_failed"

    listing = await client.get("/api/v1/evaluations")
    runs = listing.json()["items"]
    assert len(runs) == 1
    assert runs[0]["status"] == RunStatus.FAILED.value
    app.dependency_overrides.clear()


# -- run listing / status --------------------------------------------------------------


async def test_list_evaluations_is_newest_first(api, monkeypatch):
    client, _ = api
    app_data = await seed_application(client, "list-app")
    data = await seed_dataset(client, "list-dataset", ["c1"])
    payload_base = {
        "application_id": app_data["id"],
        "dataset_version_id": data["version"]["id"],
        "agent_model": "m",
    }
    monkeypatch.setattr(
        worker_tasks.run_evaluation,
        "delay",
        lambda run_id: SimpleNamespace(id=f"task-{run_id[:8]}"),
        raising=False,
    )
    first = await client.post("/api/v1/evaluations", json=payload_base)
    second = await client.post("/api/v1/evaluations", json=payload_base)
    assert first.status_code == 202 and second.status_code == 202

    listing = await client.get("/api/v1/evaluations")
    assert listing.status_code == 200
    body = listing.json()
    assert body["total"] == 2
    assert [item["id"] for item in body["items"]] == [
        second.json()["run_id"],
        first.json()["run_id"],
    ]


async def test_missing_run_is_404(api):
    client, _ = api
    response = await client.get(f"/api/v1/evaluations/{uuid.uuid4()}")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


# -- results & guardrails ---------------------------------------------------------------


async def test_evaluation_results_with_guardrails_and_pagination(api, clean_db):
    client, _ = api
    app_data = await seed_application(client, "results-app")
    data = await seed_dataset(client, "results-dataset", ["ok", "bad", "boom"])
    case_ids = await seed_case_ids(clean_db, data["version"]["id"])
    run_id = await seed_completed_run(
        clean_db,
        application_id=uuid.UUID(app_data["id"]),
        dataset_version_id=uuid.UUID(data["version"]["id"]),
        case_ids=case_ids,
        outcomes={
            "ok": (CaseStatus.PASSED, 100),
            "bad": (CaseStatus.FAILED, 120),
            "boom": (CaseStatus.ERROR, 0),
        },
        guardrails={
            "ok": [("pii", GuardrailStatus.PASSED), ("safety", GuardrailStatus.PASSED)],
            "bad": [("pii", GuardrailStatus.FAILED), ("safety", GuardrailStatus.PASSED)],
        },
    )

    results = await client.get(f"/api/v1/evaluations/{run_id}/results")
    assert results.status_code == 200
    body = results.json()
    assert body["total"] == 3
    assert len(body["items"]) == 3
    # Index by the seeded input prompt (error cases have no actual_output).
    by_name = {item["input"]["prompt"]: item for item in body["items"]}
    assert by_name["bad"]["status"] == CaseStatus.FAILED.value
    guardrail_names = [g["name"] for g in by_name["bad"]["guardrail_results"]]
    assert guardrail_names == ["pii", "safety"]
    assert by_name["bad"]["guardrail_results"][0]["status"] == GuardrailStatus.FAILED.value
    assert by_name["boom"]["status"] == CaseStatus.ERROR.value
    assert by_name["boom"]["error"] == "provider exploded"
    assert by_name["ok"]["status"] == CaseStatus.PASSED.value

    # Pagination: stable, capped, total preserved.
    page = await client.get(
        f"/api/v1/evaluations/{run_id}/results", params={"limit": 1, "offset": 1}
    )
    assert page.status_code == 200
    assert page.json()["total"] == 3
    assert len(page.json()["items"]) == 1
    too_big = await client.get(
        f"/api/v1/evaluations/{run_id}/results", params={"limit": 201}
    )
    assert too_big.status_code == 422

    guardrails = await client.get(f"/api/v1/evaluations/{run_id}/guardrails")
    assert guardrails.status_code == 200
    assert guardrails.json()["total"] == 4
    assert all(
        set(g["metadata"]) == {"categories", "count"} for g in guardrails.json()["items"]
    )


async def test_summary_endpoint_omits_case_outputs(api, clean_db):
    client, _ = api
    app_data = await seed_application(client, "summary-app")
    data = await seed_dataset(client, "summary-dataset", ["ok"])
    case_ids = await seed_case_ids(clean_db, data["version"]["id"])
    run_id = await seed_completed_run(
        clean_db,
        application_id=uuid.UUID(app_data["id"]),
        dataset_version_id=uuid.UUID(data["version"]["id"]),
        case_ids=case_ids,
        outcomes={"ok": (CaseStatus.PASSED, 100)},
    )
    response = await client.get(f"/api/v1/evaluations/{run_id}")
    summary = response.json()
    assert summary["counts"]["total"] == 1 and summary["counts"]["passed"] == 1
    assert "items" not in summary  # no case results in the summary
    assert "actual_output" not in response.text


# -- regression API -----------------------------------------------------------------------


async def seed_regression_pair(clean_db: DatabaseManager, client: AsyncClient) -> tuple[uuid.UUID, uuid.UUID]:
    app_data = await seed_application(client, "regression-app")
    data = await seed_dataset(client, "regression-dataset", ["ok0", "ok1", "to-fix"])
    case_ids = await seed_case_ids(clean_db, data["version"]["id"])
    application_id = uuid.UUID(app_data["id"])
    dataset_version_id = uuid.UUID(data["version"]["id"])
    baseline = await seed_completed_run(
        clean_db,
        application_id=application_id,
        dataset_version_id=dataset_version_id,
        case_ids=case_ids,
        outcomes={
            "ok0": (CaseStatus.PASSED, 100),
            "ok1": (CaseStatus.PASSED, 110),
            "to-fix": (CaseStatus.PASSED, 120),
        },
        guardrails={"ok0": [("pii", GuardrailStatus.PASSED)]},
    )
    current = await seed_completed_run(
        clean_db,
        application_id=application_id,
        dataset_version_id=dataset_version_id,
        case_ids=case_ids,
        outcomes={
            "ok0": (CaseStatus.PASSED, 140),
            "ok1": (CaseStatus.FAILED, 150),
            "to-fix": (CaseStatus.ERROR, 0),
        },
        guardrails={"ok0": [("pii", GuardrailStatus.PASSED)]},
    )
    return baseline, current


async def test_regression_endpoint_detects_and_persists(api, clean_db):
    client, _ = api
    baseline_id, current_id = await seed_regression_pair(clean_db, client)
    before = await evaluation_snapshot(clean_db)

    response = await client.post(
        "/api/v1/regressions",
        json={"baseline_run_id": str(baseline_id), "current_run_id": str(current_id)},
    )
    assert response.status_code == 200  # a detected regression is not an error
    body = response.json()
    # Enums serialize to their domain values (consistent with every other
    # status string in the API, e.g. run status "pending").
    assert body["result"] == "regression_detected"
    assert body["regression_detected"] is True
    assert body["comparison_id"] is not None
    assert body["matched_cases"] == 3
    metrics = {v["metric"] for v in body["threshold_violations"]}
    assert "pass_rate" in metrics and "error_rate" in metrics
    assert [f["name"] for f in body["newly_failed_cases"]] == ["ok1"]
    assert [f["name"] for f in body["newly_errored_cases"]] == ["to-fix"]

    # Historical evaluation data untouched by the comparison.
    after = await evaluation_snapshot(clean_db)
    assert before == after

    # Idempotent: same pair + policy returns the original artifact.
    repeat = await client.post(
        "/api/v1/regressions",
        json={"baseline_run_id": str(baseline_id), "current_run_id": str(current_id)},
    )
    assert repeat.json()["comparison_id"] == body["comparison_id"]
    assert repeat.json()["created_at"] == body["created_at"]

    fetched = await client.get(f"/api/v1/regressions/{body['comparison_id']}")
    assert fetched.status_code == 200
    assert fetched.json()["result"] == "regression_detected"

    per_run = await client.get(f"/api/v1/evaluations/{current_id}/regressions")
    assert per_run.status_code == 200
    assert per_run.json()["total"] == 1
    assert per_run.json()["items"][0]["comparison_id"] == body["comparison_id"]


async def test_regression_detected_not_comparable_and_invalid_requests(api, clean_db):
    client, _ = api
    baseline_id, current_id = await seed_regression_pair(clean_db, client)

    # Same run on both sides → 400 invalid_comparison (typed service error).
    same = await client.post(
        "/api/v1/regressions",
        json={"baseline_run_id": str(baseline_id), "current_run_id": str(baseline_id)},
    )
    assert same.status_code == 400
    assert same.json()["error"]["code"] == "invalid_comparison"

    # Unknown run → 404.
    missing = await client.post(
        "/api/v1/regressions",
        json={"baseline_run_id": str(uuid.uuid4()), "current_run_id": str(current_id)},
    )
    assert missing.status_code == 404

    # Non-completed run → 400 with a clear message.

    evaluations = EvaluationRepository()
    async with clean_db.session() as session:
        baseline_run = await evaluations.get_run(session, baseline_id)
        assert baseline_run is not None
        pending_run = await evaluations.create_run(
            session,
            organization_id=baseline_run.organization_id,
            application_id=baseline_run.application_id,
            dataset_version_id=baseline_run.dataset_version_id,
            agent_model="m",
        )
    pending = await client.post(
        "/api/v1/regressions",
        json={"baseline_run_id": str(pending_run.id), "current_run_id": str(current_id)},
    )
    assert pending.status_code == 400
    assert "completed" in pending.json()["error"]["message"]

    unknown_comparison = await client.get(f"/api/v1/regressions/{uuid.uuid4()}")
    assert unknown_comparison.status_code == 404


async def test_regression_custom_thresholds_create_distinct_artifacts(api, clean_db):
    client, _ = api
    baseline_id, current_id = await seed_regression_pair(clean_db, client)

    strict = await client.post(
        "/api/v1/regressions",
        json={"baseline_run_id": str(baseline_id), "current_run_id": str(current_id)},
    )
    lax = await client.post(
        "/api/v1/regressions",
        json={
            "baseline_run_id": str(baseline_id),
            "current_run_id": str(current_id),
            "thresholds": {
                "max_pass_rate_drop_pp": 99.0,
                "max_error_rate_increase_pp": 99.0,
                "max_guardrail_failure_rate_increase_pp": 99.0,
                "max_latency_increase_percent": None,
            },
        },
    )
    assert strict.json()["result"] == "regression_detected"
    assert lax.json()["result"] == "no_regression"
    assert lax.json()["comparison_id"] != strict.json()["comparison_id"]
    # Metric deltas present and signed (current - baseline): the 100% → 50%
    # pass-rate drop appears as a negative delta even under the lax policy.
    assert lax.json()["deltas"]["pass_rate_pp"] == pytest.approx(-50.0)


# -- security ---------------------------------------------------------------------------


async def test_no_secrets_or_pii_in_run_responses(api, clean_db, monkeypatch):
    client, _ = api
    app_data = await seed_application(client, "security-app")
    data = await seed_dataset(client, "security-dataset", ["ok"])
    case_ids = await seed_case_ids(clean_db, data["version"]["id"])
    run_id = await seed_completed_run(
        clean_db,
        application_id=uuid.UUID(app_data["id"]),
        dataset_version_id=uuid.UUID(data["version"]["id"]),
        case_ids=case_ids,
        outcomes={"ok": (CaseStatus.FAILED, 100)},
        guardrails={"ok": [("pii", GuardrailStatus.FAILED)]},
    )

    # Seed a fake secret into the run's configuration snapshot via SQL
    # (simulating a compromised snapshot): the API must not surface it.
    async with clean_db.engine.begin() as conn:
        await conn.execute(
            text(
                "UPDATE evaluation_runs SET configuration_snapshot = "
                "configuration_snapshot || '{\"api_key\": \"sk-fake-seed-value\"}'::jsonb "
                "WHERE id = :id"
            ),
            {"id": str(run_id)},
        )

    monkeypatch.setattr(
        worker_tasks.run_evaluation,
        "delay",
        lambda run_id: SimpleNamespace(id="task-sec"),
        raising=False,
    )

    for path in (
        f"/api/v1/evaluations/{run_id}",
        f"/api/v1/evaluations/{run_id}/results",
        f"/api/v1/evaluations/{run_id}/guardrails",
    ):
        response = await client.get(path)
        assert response.status_code == 200
        assert "sk-fake-seed-value" not in response.text
        assert "OPENROUTER_API_KEY" not in response.text
        assert "Authorization" not in response.text


# -- Phase 12: failure info & reliability summary ----------------------------------------


async def test_case_results_expose_failure_info_and_reliability_summary(api, clean_db):
    """Phase 12 surface: errored cases carry a typed `failure` object; the
    reliability endpoint breaks execution failures down by category."""
    client, _ = api
    app_data = await seed_application(client, "reliability-app")
    data = await seed_dataset(client, "reliability-dataset", ["ok", "outage", "limited", "old"])
    case_ids = await seed_case_ids(clean_db, data["version"]["id"])
    run_id = await seed_completed_run(
        clean_db,
        application_id=uuid.UUID(app_data["id"]),
        dataset_version_id=uuid.UUID(data["version"]["id"]),
        case_ids=case_ids,
        outcomes={
            "ok": (CaseStatus.PASSED, 100),
            "outage": (CaseStatus.ERROR, 0),
            "limited": (CaseStatus.ERROR, 0),
            # "old" errors but has no classified failure (pre-Phase 12 row)
            "old": (CaseStatus.ERROR, 0),
        },
        case_metrics={
            "outage": {
                "failure": {
                    "category": "provider_unavailable",
                    "reason": "Provider returned HTTP 502.",
                    "retryable": True,
                    "http_status": 502,
                    "attempts": 3,
                }
            },
            "limited": {
                "failure": {
                    "category": "rate_limited",
                    "reason": "Provider rate limit reached (HTTP 429).",
                    "retryable": True,
                    "http_status": 429,
                }
            },
        },
    )

    results = await client.get(f"/api/v1/evaluations/{run_id}/results")
    assert results.status_code == 200
    by_name = {i["input"]["prompt"]: i for i in results.json()["items"]}
    assert by_name["outage"]["failure"]["category"] == "provider_unavailable"
    assert by_name["outage"]["failure"]["attempts"] == 3
    assert by_name["limited"]["failure"]["retryable"] is True
    # quality outcomes and pre-Phase 12 rows have no failure object
    assert by_name["ok"]["failure"] is None
    assert by_name["old"]["failure"] is None

    reliability = await client.get(f"/api/v1/evaluations/{run_id}/reliability")
    assert reliability.status_code == 200
    report = reliability.json()
    assert report["total_cases"] == 4
    assert report["errored_cases"] == 3
    assert report["classified_failures"] == 2
    assert report["unclassified_execution_failures"] == 1
    assert report["retryable_failures"] == 2
    # most frequent first; "old" is unclassified so it is absent here
    assert list(report["failure_breakdown"].items()) == [
        ("provider_unavailable", 1),
        ("rate_limited", 1),
    ]

    # quality failures must not appear in the reliability breakdown
    assert "failed" not in str(report["failure_breakdown"])
