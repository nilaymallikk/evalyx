"""API observability integration tests (live PostgreSQL; no live LLM).

Covers: X-Request-ID over the real app, readiness failure logging, the
request → run → task correlation chain for evaluation submission, PII/secret
safety of captured logs, bounded metric labels over real routes, and the
full pipeline's evaluation/guardrail metrics with a fake provider.
"""

import uuid
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from structlog.testing import capture_logs
from test_runner import DOMAIN_TABLES, FakeProvider, seed

from evalyx.core.metrics import metrics
from evalyx.evaluation.pipeline import EvaluationPipeline
from evalyx.worker import tasks as worker_tasks


@pytest.fixture
async def clean_db(db_manager):
    async with db_manager.engine.begin() as conn:
        await conn.execute(
            text(f"TRUNCATE {', '.join(DOMAIN_TABLES)} RESTART IDENTITY CASCADE")
        )
    yield db_manager


@pytest.fixture
async def api(clean_db, settings):
    from evalyx.api.app import create_app

    app = create_app(settings, database=clean_db)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client, app


@pytest.fixture(autouse=True)
def _clean_metrics():
    metrics.reset()
    yield
    metrics.reset()


async def _seed_via_api(client: AsyncClient, name: str) -> dict:
    app_resp = await client.post("/api/v1/applications", json={"name": name})
    assert app_resp.status_code == 201
    app_body = app_resp.json()
    version_resp = await client.post(
        f"/api/v1/applications/{app_body['id']}/versions",
        json={"version": "v1", "configuration": {"temperature": 0.2}},
    )
    assert version_resp.status_code == 201
    dataset_resp = await client.post("/api/v1/datasets", json={"name": f"ds-{name}"})
    assert dataset_resp.status_code == 201
    dataset = dataset_resp.json()
    dsv_resp = await client.post(
        f"/api/v1/datasets/{dataset['id']}/versions", json={"version": 1}
    )
    assert dsv_resp.status_code == 201
    dsv_id = dsv_resp.json()["id"]
    return {"app": app_body, "dataset": dataset, "dataset_version_id": dsv_id}


# -- request IDs over the real app ------------------------------------------------


async def test_real_app_generates_and_preserves_request_ids(api):
    client, _ = api
    generated = await client.get("/health")
    assert uuid.UUID(generated.headers["X-Request-ID"]).version == 4

    preserved = await client.get("/health", headers={"X-Request-ID": "test-observability-123"})
    assert preserved.headers["X-Request-ID"] == "test-observability-123"

    replaced = await client.get("/health", headers={"X-Request-ID": "x" * 999})
    assert uuid.UUID(replaced.headers["X-Request-ID"]).version == 4


async def test_readiness_failure_logs_dependency_name_only(settings, clean_db):
    """§24/§26: dependency failures log a bounded name — never URLs or
    credentials."""
    from evalyx.api.app import create_app
    from evalyx.db.redis import create_redis_client

    broken_redis = create_redis_client(
        settings.model_copy(update={"redis_url": "redis://localhost:1/0"})
    )
    app = create_app(settings, database=clean_db, redis_client=broken_redis)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        with capture_logs() as logs:
            response = await client.get("/health/ready")
    assert response.status_code == 503
    assert response.json()["dependencies"]["redis"] == "error"
    failures = [e for e in logs if e["event"] == "readiness_check_failed"]
    assert failures == [
        {"event": "readiness_check_failed", "dependency": "redis", "log_level": "warning"}
    ]
    assert "redis://localhost:1" not in str(logs)


# -- request → run → task correlation ----------------------------------------------


async def test_evaluation_submission_correlation_chain(api, monkeypatch):
    """§57: logs allow following request_id → run_id → task_id."""
    client, _ = api
    seeded = await _seed_via_api(client, "corr-app")

    delay_calls: list[str] = []

    def fake_delay(run_id: str) -> SimpleNamespace:
        delay_calls.append(run_id)
        return SimpleNamespace(id="fake-celery-task-id-123")

    monkeypatch.setattr(worker_tasks.run_evaluation, "delay", fake_delay)

    with capture_logs() as logs:
        response = await client.post(
            "/api/v1/evaluations",
            headers={"X-Request-ID": "corr-request-1"},
            json={
                "application_id": seeded["app"]["id"],
                "dataset_version_id": seeded["dataset_version_id"],
                "agent_model": "agent-model:free",
                "judge_model": "judge-model:free",
            },
        )
    assert response.status_code == 202, response.text
    body = response.json()
    assert delay_calls == [body["run_id"]]

    submitted = [e for e in logs if e["event"] == "evaluation_submitted"]
    assert len(submitted) == 1
    assert submitted[0]["run_id"] == body["run_id"]
    assert submitted[0]["task_id"] == "fake-celery-task-id-123"
    assert submitted[0]["application_id"] == seeded["app"]["id"]
    assert submitted[0]["dataset_version_id"] == seeded["dataset_version_id"]

    completed = [e for e in logs if e["event"] == "http_request_completed"]
    assert completed[0]["request_id"] == "corr-request-1"
    assert completed[0]["route"] == "/api/v1/evaluations"
    assert completed[0]["status_code"] == 202


# -- PII / secret safety ------------------------------------------------------------


async def test_submission_logs_never_contain_pii_or_secrets(api, monkeypatch):
    """§40: fake PII in payloads never reaches any log event."""
    client, _ = api
    fake_pii_email = "jane.doe@example.com"
    fake_pii_phone = "+1-555-0100"
    fake_secret = "sk-fake-secret-value"

    with capture_logs() as logs:
        app_resp = await client.post(
            "/api/v1/applications",
            json={"name": "pii-app", "description": f"contact {fake_pii_email}"},
            headers={"Authorization": f"Bearer {fake_secret}"},
        )
        assert app_resp.status_code == 201
        ds_resp = await client.post("/api/v1/datasets", json={"name": "pii-ds"})
        assert ds_resp.status_code == 201
        dsv = await client.post(
            f"/api/v1/datasets/{ds_resp.json()['id']}/versions", json={"version": 1}
        )
        assert dsv.status_code == 201
        case = await client.post(
            f"/api/v1/datasets/{ds_resp.json()['id']}/versions/1/cases",
            json={
                "name": "pii-case",
                "input": {
                    "prompt": f"Customer {fake_pii_email} phone {fake_pii_phone} asks for help"
                },
            },
        )
        assert case.status_code == 201

    serialized = str(logs)
    assert fake_pii_email not in serialized
    assert fake_pii_phone not in serialized
    assert fake_secret not in serialized
    assert "Bearer" not in serialized
    # The case input content itself is never logged (only ids and statuses).
    assert all(e["event"] != "test_case_input" for e in logs)


# -- metric cardinality over real routes --------------------------------------------


async def test_metric_labels_stay_bounded_over_real_requests(api):
    client, _ = api
    with capture_logs():
        await client.get("/health", headers={"X-Request-ID": "card-check-1"})
        await client.get("/api/v1/applications", headers={"X-Request-ID": "card-check-2"})
        await client.get("/api/v1/does-not-exist/3f2c1a6b-1234-5678-90ab-cdef12345678")
    snap = metrics.snapshot()
    serialized = str(snap)
    for forbidden in (
        "card-check-1",
        "card-check-2",
        "3f2c1a6b",
        "request_id",
        "run_id",
        "task_id",
    ):
        assert forbidden not in serialized
    routes = {e["labels"]["route"] for e in snap["http_requests_total"]}
    assert "/unmatched" in routes  # arbitrary 404 paths stay bounded


# -- pipeline metrics with a fake provider -------------------------------------------


async def test_pipeline_lifecycle_metrics_and_logs(clean_db):
    """Full pipeline over real PostgreSQL with a fake provider: run/case
    metrics use bounded labels; correlation ids and prompts never appear."""
    app_id, run_id, dsv_id, _case_ids = await seed(
        clean_db, case_inputs=[{"prompt": "hello"}, {"prompt": "goodbye"}]
    )
    provider = FakeProvider()
    pipeline = EvaluationPipeline(provider=provider, session_factory=clean_db.session_factory)
    with capture_logs() as logs:
        summary = await pipeline.execute_and_score_existing_run(run_id)
    assert summary.status.value == "completed"

    snap = metrics.snapshot()
    runs = {e["labels"]["status"]: e["value"] for e in snap["evaluation_runs_total"]}
    assert runs == {"completed": 1.0}
    cases = {e["labels"]["status"]: e["value"] for e in snap["evaluation_cases_total"]}
    assert cases == {"executed": 2.0}  # no judge configured → cases stay executed
    # Regex-based guardrails ran; labels are the bounded configured names.
    guardrail_series = snap["guardrail_evaluations_total"]
    names = {e["labels"]["name"] for e in guardrail_series}
    assert names <= {"pii", "prompt_injection", "hallucination", "instruction_following", "safety"}
    assert all(e["labels"]["status"] in {"passed", "failed", "error"} for e in guardrail_series)

    serialized = str(snap)
    assert str(run_id) not in serialized
    assert str(app_id) not in serialized

    run_events = [e for e in logs if e["event"] == "evaluation_run_started"]
    assert len(run_events) == 1
    assert run_events[0]["run_id"] == str(run_id)
    assert run_events[0]["application_id"] == str(app_id)
    assert run_events[0]["dataset_version_id"] == str(dsv_id)
    assert run_events[0]["total_cases"] == 2
    # Prompts never appear in lifecycle logs.
    assert "hello" not in str(logs)
    assert "goodbye" not in str(logs)


# -- regression comparison observability ---------------------------------------------


async def test_regression_comparison_logs_and_metrics(clean_db):
    """§41: comparisons emit started/completed events with safe fields only
    (ids, bounded result enum, duration) — never the full report."""
    from test_regression import seed_scenario

    from evalyx.evaluation.regression.models import RegressionThresholds
    from evalyx.evaluation.regression.service import RegressionService

    baseline_id, current_id = await seed_scenario(clean_db)
    service = RegressionService(clean_db.session_factory)

    with capture_logs() as logs:
        report = await service.compare_runs(baseline_id, current_id)
    assert report.result.value == "regression_detected"

    started = [e for e in logs if e["event"] == "regression_comparison_started"]
    completed = [e for e in logs if e["event"] == "regression_comparison_completed"]
    assert len(started) == 1 and len(completed) == 1
    assert started[0]["baseline_run_id"] == str(baseline_id)
    assert started[0]["current_run_id"] == str(current_id)
    assert completed[0]["comparison_id"] == str(report.comparison_id)
    assert completed[0]["result"] == "regression_detected"
    assert completed[0]["regression_detected"] is True
    assert isinstance(completed[0]["duration_ms"], float)
    # The full report (per-case findings) is never dumped into logs.
    assert all("newly_failed_cases" not in e for e in logs)
    assert all("thresholds" not in e for e in logs)

    snap = metrics.snapshot()
    comparisons = {
        e["labels"]["result"]: e["value"] for e in snap["regression_comparisons_total"]
    }
    assert comparisons == {"regression_detected": 1.0}
    assert str(baseline_id) not in str(snap)  # ids never become labels

    # Second comparison (idempotent reuse) still logs + counts.
    with capture_logs() as logs2:
        await service.compare_runs(
            baseline_id, current_id, thresholds=RegressionThresholds()
        )
    reused = [e for e in logs2 if e["event"] == "regression_comparison_completed"]
    assert reused[0]["reused"] is True
    assert reused[0]["comparison_id"] == str(report.comparison_id)
