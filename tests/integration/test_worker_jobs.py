"""Worker integration tests — live PostgreSQL (5433) + Redis (6379).

No network calls: the LLM provider is a deterministic fake injected through
the worker's provider factory. The Celery task runs eagerly in-process
(``.apply``), so no worker process is required; the real broker path is
exercised by the manual smoke test documented in the README.
"""

import uuid

import pytest
from sqlalchemy import text
from tenant_helpers import integration_organization_id

from evalyx.core.config import Settings
from evalyx.db.models import CaseStatus, RunStatus
from evalyx.db.redis import check_redis, create_redis_client
from evalyx.db.repositories import EvaluationRepository
from evalyx.db.session import DatabaseManager
from evalyx.llm.base import LLMResponse, TokenUsage
from evalyx.llm.factory import create_provider as real_create_provider
from evalyx.worker.execution import PermanentEvaluationError
from evalyx.worker.tasks import run_evaluation

pytestmark = pytest.mark.integration

#: All Evalyx domain tables, children first (mirrors tests/integration/conftest.py).
DOMAIN_TABLES = (
    "guardrail_results",
    "evaluation_case_results",
    "evaluation_runs",
    "test_cases",
    "dataset_versions",
    "datasets",
    "application_versions",
    "applications",
    "audit_events",
    "organization_quota_overrides",
)

JUDGE_SYSTEM_PREFIX = "You are an evaluation judge for Evalyx"


class SmartFakeProvider:
    """Deterministic provider: real output for the agent, valid judge JSON
    for judge calls. Records every call."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.closed = False

    async def complete(self, prompt, *, model, temperature=0.2, max_tokens=512, system=None):
        self.calls.append({"prompt": prompt, "model": model, "system": system})
        if system is not None and system.startswith(JUDGE_SYSTEM_PREFIX):
            content = '{"passed": true, "score": 1.0, "reason": "ok"}'
        else:
            content = f"reply:{prompt}"
        return LLMResponse(
            content=content,
            model=model,
            latency_ms=7,
            usage=TokenUsage(prompt_tokens=3, completion_tokens=5, total_tokens=8),
            finish_reason="stop",
        )

    async def close(self) -> None:
        self.closed = True


async def seed(db_manager: DatabaseManager, *, case_inputs: list):
    """Create application + dataset + version + cases + a pending run."""
    from evalyx.db.repositories import ApplicationRepository, DatasetRepository

    async with db_manager.session() as session:
        apps, datasets, evaluations = (
            ApplicationRepository(),
            DatasetRepository(),
            EvaluationRepository(),
        )
        org = await integration_organization_id(session)
        app = await apps.create(
            session, organization_id=org, name=f"app-{uuid.uuid4().hex[:8]}"
        )
        dataset = await datasets.create(
            session, organization_id=org, name=f"dataset-{uuid.uuid4().hex[:8]}"
        )
        version = await datasets.create_version(session, dataset_id=dataset.id, version=1)
        case_ids = []
        for index, case_input in enumerate(case_inputs):
            case = await datasets.add_test_case(
                session,
                dataset_version_id=version.id,
                name=f"case-{index}",
                input=case_input,
                expected_output={"must_contain": "expected marker"},
            )
            case_ids.append(case.id)
        run = await evaluations.create_run(
            session,
            organization_id=org,
            application_id=app.id,
            dataset_version_id=version.id,
            agent_model="agent-model:free",
            judge_model="judge-model:free",
        )
    return run.id, case_ids


@pytest.fixture
async def clean_db(db_manager: DatabaseManager):
    async with db_manager.engine.begin() as conn:
        await conn.execute(
            text(f"TRUNCATE {', '.join(DOMAIN_TABLES)} RESTART IDENTITY CASCADE")
        )
    yield db_manager


@pytest.fixture
def fake_provider_factory(monkeypatch):
    """Route the worker's provider factory to a SmartFakeProvider instance."""
    provider = SmartFakeProvider()
    monkeypatch.setattr(
        "evalyx.worker.execution.create_provider", lambda settings: provider
    )
    return provider


async def count_rows(db_manager: DatabaseManager, table: str, run_id: uuid.UUID) -> int:
    async with db_manager.engine.connect() as conn:
        result = await conn.execute(
            text(
                f"SELECT COUNT(*) FROM {table} WHERE evaluation_run_id = :run_id"
                if table == "evaluation_case_results"
                else "SELECT COUNT(*) FROM guardrail_results g "
                "JOIN evaluation_case_results c ON g.evaluation_case_result_id = c.id "
                "WHERE c.evaluation_run_id = :run_id"
            ),
            {"run_id": run_id},
        )
        return int(result.scalar_one())


async def get_run_status(db_manager: DatabaseManager, run_id: uuid.UUID) -> RunStatus:
    async with db_manager.session() as session:
        run = await EvaluationRepository().get_run(session, run_id)
        assert run is not None
        return run.status


async def get_case_results(db_manager: DatabaseManager, run_id: uuid.UUID):
    async with db_manager.session() as session:
        return await EvaluationRepository().list_case_results(session, run_id)


async def test_redis_is_reachable(settings: Settings):
    client = create_redis_client(settings)
    try:
        assert await check_redis(client) is True
    finally:
        await client.aclose()


async def test_task_executes_run_end_to_end(clean_db, fake_provider_factory):
    run_id, case_ids = await seed(clean_db, case_inputs=["one", "two"])

    result = run_evaluation.apply(args=[str(run_id)]).get()

    assert result["run_id"] == str(run_id)
    assert result["action"] == "executed"
    assert result["status"] == "completed"
    assert result["total_cases"] == 2
    assert result["passed_cases"] == 2
    assert result["failed_cases"] == 0
    assert result["error_cases"] == 0

    assert await get_run_status(clean_db, run_id) is RunStatus.COMPLETED
    case_results = await get_case_results(clean_db, run_id)
    assert {r.test_case_id for r in case_results} == set(case_ids)
    assert all(r.status is CaseStatus.PASSED for r in case_results)
    # Guardrail results exist for every executed case.
    assert await count_rows(clean_db, "evaluation_case_results", run_id) == 2
    assert await count_rows(clean_db, "guardrail_results", run_id) == 10  # 5 per case


async def test_duplicate_delivery_does_not_duplicate_results(
    clean_db, fake_provider_factory
):
    run_id, _ = await seed(clean_db, case_inputs=["a", "b"])

    first = run_evaluation.apply(args=[str(run_id)]).get()
    case_rows_after_first = _case_rows(await get_case_results(clean_db, run_id))
    guardrails_after_first = await count_rows(clean_db, "guardrail_results", run_id)
    calls_after_first = len(fake_provider_factory.calls)

    second = run_evaluation.apply(args=[str(run_id)]).get()

    assert first["action"] == "executed"
    assert second["action"] == "rescored"  # never blindly re-executed
    assert second["passed_cases"] == first["passed_cases"]
    assert _case_rows(await get_case_results(clean_db, run_id)) == case_rows_after_first
    assert await count_rows(clean_db, "guardrail_results", run_id) == guardrails_after_first
    assert len(fake_provider_factory.calls) == calls_after_first  # no re-execution
    assert await get_run_status(clean_db, run_id) is RunStatus.COMPLETED


def _case_rows(case_results):
    return sorted((str(r.test_case_id), r.status.value, r.actual_output) for r in case_results)


async def test_infrastructure_failure_retries_then_completes(clean_db, monkeypatch):
    run_id, case_ids = await seed(clean_db, case_inputs=["a", "b"])

    # First attempt: PostgreSQL connectivity dies before any work is done.
    attempts = {"count": 0}

    def failing_factory(settings):
        attempts["count"] += 1
        from sqlalchemy.exc import OperationalError

        raise OperationalError("connection refused", {}, Exception())

    monkeypatch.setattr(
        "evalyx.worker.execution.create_provider", failing_factory
    )
    with pytest.raises(Exception) as excinfo:
        run_evaluation.apply(args=[str(run_id)], throw=True)
    from celery.exceptions import Retry

    assert isinstance(excinfo.value, Retry)
    # The failed attempt persisted nothing; the run is still executable.
    assert await get_case_results(clean_db, run_id) == []
    assert await get_run_status(clean_db, run_id) is RunStatus.PENDING

    # Second attempt: infrastructure recovered; the run completes.
    provider = SmartFakeProvider()
    monkeypatch.setattr(
        "evalyx.worker.execution.create_provider", lambda settings: provider
    )
    result = run_evaluation.apply(args=[str(run_id)]).get()

    assert result["action"] == "executed"
    assert result["passed_cases"] == 2
    assert await get_run_status(clean_db, run_id) is RunStatus.COMPLETED
    assert {r.test_case_id for r in await get_case_results(clean_db, run_id)} == set(case_ids)


async def test_resume_after_simulated_worker_death(clean_db, fake_provider_factory):
    """A run interrupted mid-way resumes: completed cases are not re-run."""
    run_id, case_ids = await seed(clean_db, case_inputs=["first", "second"])

    # Simulate a worker that died after completing case 1: its result exists,
    # the run looks running, and no completion was ever recorded.
    async with clean_db.session() as session:
        await EvaluationRepository().add_case_result(
            session,
            evaluation_run_id=run_id,
            test_case_id=case_ids[0],
            input={"pre": "existing"},
            status=CaseStatus.EXECUTED,
        )
        run = await EvaluationRepository().get_run(session, run_id)
        await EvaluationRepository().update_status(session, run, RunStatus.RUNNING)
    agent_calls_before = len(fake_provider_factory.calls)

    result = run_evaluation.apply(args=[str(run_id)]).get()

    assert result["action"] == "executed"
    assert result["status"] == "completed"
    assert result["total_cases"] == 2
    case_results = await get_case_results(clean_db, run_id)
    assert len(case_results) == 2  # resumed, not duplicated
    assert {r.test_case_id for r in case_results} == set(case_ids)
    resumed = next(r for r in case_results if r.test_case_id == case_ids[0])
    assert resumed.input == {"pre": "existing"}  # original result preserved
    # Only the missing case was executed through the provider.
    new_calls = fake_provider_factory.calls[agent_calls_before:]
    assert len([c for c in new_calls if c["system"] is None]) == 1


async def test_cancelled_run_is_not_executed(clean_db, monkeypatch):
    run_id, _ = await seed(clean_db, case_inputs=["a"])

    def factory(settings):
        raise AssertionError("provider must not be created for a cancelled run")

    monkeypatch.setattr("evalyx.worker.execution.create_provider", factory)
    async with clean_db.session() as session:
        run = await EvaluationRepository().get_run(session, run_id)
        await EvaluationRepository().update_status(session, run, RunStatus.CANCELLED)

    result = run_evaluation.apply(args=[str(run_id)]).get()

    assert result["action"] == "skipped"
    assert result["status"] == "cancelled"
    assert await get_case_results(clean_db, run_id) == []


async def test_missing_run_fails_permanently(clean_db):
    with pytest.raises(PermanentEvaluationError):
        run_evaluation.apply(args=[str(uuid.uuid4())], throw=True)


async def test_real_provider_factory_still_constructs_openrouter(settings: Settings):
    """The default factory is untouched; the worker uses it as-is."""
    from evalyx.llm.openrouter import OpenRouterProvider

    provider = real_create_provider(settings)
    try:
        assert isinstance(provider, OpenRouterProvider)
    finally:
        await provider.close()
