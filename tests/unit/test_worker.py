"""Worker unit tests — no live services, no Celery worker required.

The Celery app is importable and the task body is exercised through injected
fakes (fake DatabaseManager / provider / pipeline), so these tests never
touch PostgreSQL, Redis, or OpenRouter.
"""

import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from sqlalchemy.exc import OperationalError
from structlog.testing import capture_logs

import evalyx
from evalyx.core import context as correlation
from evalyx.core.config import Settings
from evalyx.core.metrics import metrics
from evalyx.db.models import RunStatus
from evalyx.evaluation.runner import EvaluationSummary, RunnerError
from evalyx.llm.errors import LLMConfigurationError, LLMRateLimitError
from evalyx.worker.celery_app import celery_app, create_celery_app
from evalyx.worker.execution import (
    PermanentEvaluationError,
    decide_action,
    execute_evaluation,
    is_retryable_infrastructure_error,
)
from evalyx.worker.tasks import _coerce_run_id, _retry_countdown, run_evaluation

WORKER_PACKAGE_DIR = Path(evalyx.__file__).parent / "worker"

FORBIDDEN_MODULES = ("openrouter", "ollama", "httpx")


def make_summary(run_id: uuid.UUID, status: RunStatus = RunStatus.COMPLETED) -> EvaluationSummary:
    return EvaluationSummary(
        run_id=run_id,
        status=status,
        total_cases=10,
        executed_cases=9,
        error_cases=1,
        failed_cases=1,
        passed_cases=8,
        evaluation_error_cases=1,
    )


class FakeRun:
    def __init__(self, status: RunStatus) -> None:
        self.status = status


class FakeSession:
    def __init__(self, run: FakeRun | None) -> None:
        self._run = run

    async def get(self, model, pk):
        return self._run


class FakeDBManager:
    """Stands in for DatabaseManager; records disposal."""

    session_factory = None  # fakes never touch real sessions

    def __init__(self, run: FakeRun | None = None) -> None:
        self.run = run
        self.disposed = False

    @asynccontextmanager
    async def session(self):
        yield FakeSession(self.run)

    async def dispose(self) -> None:
        self.disposed = True


class FakeProvider:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class FakePipeline:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.execute_calls = 0
        self.score_calls = 0
        self._error = error

    async def execute_and_score_existing_run(self, run_id):
        self.execute_calls += 1
        if self._error is not None:
            raise self._error
        return make_summary(run_id)

    async def score_existing_run(self, run_id):
        self.score_calls += 1
        return make_summary(run_id)


def summary_dict(run_id, action="executed"):
    return {
        "run_id": str(run_id),
        "status": "completed",
        "action": action,
        "total_cases": 10,
        "executed_cases": 9,
        "error_cases": 1,
        "passed_cases": 8,
        "failed_cases": 1,
        "evaluation_error_cases": 1,
    }


# -- registration & configuration -------------------------------------------


def test_task_is_importable_and_registered():
    import evalyx.worker.tasks  # noqa: F401 — registration side effect

    assert "evalyx.worker.run_evaluation" in celery_app.tasks


def test_worker_package_never_imports_concrete_providers():
    """The worker must not contain provider-specific HTTP logic."""
    offenders: list[str] = []
    for source_file in WORKER_PACKAGE_DIR.glob("*.py"):
        for line in source_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")) and any(
                module in stripped for module in FORBIDDEN_MODULES
            ):
                offenders.append(f"{source_file.name}: {stripped}")
    assert offenders == []


def test_celery_app_reliability_configuration():
    settings = Settings()
    app = create_celery_app(settings)
    conf = app.conf
    assert conf.broker_url == settings.redis_url  # derived from REDIS_URL
    assert conf.result_backend == settings.redis_url
    assert conf.task_serializer == "json"
    assert conf.accept_content == ["json"]
    # Reliability: do not lose evaluation jobs silently.
    assert conf.task_acks_late is True
    assert conf.task_reject_on_worker_lost is True
    assert conf.worker_prefetch_multiplier == 1
    assert conf.worker_concurrency == settings.worker_concurrency
    # Explicit, generous-but-bounded time limits.
    assert conf.task_soft_time_limit == settings.worker_soft_time_limit_seconds
    assert conf.task_time_limit == settings.worker_hard_time_limit_seconds
    # Visibility timeout must exceed the hard time limit.
    assert conf.broker_transport_options["visibility_timeout"] >= (
        2 * settings.worker_hard_time_limit_seconds
    )


def test_settings_derive_broker_from_redis_url():
    settings = Settings()
    assert settings.celery_broker_url == settings.redis_url
    assert settings.celery_result_backend == settings.redis_url


# -- serialization ----------------------------------------------------------


def test_coerce_run_id_accepts_string_and_uuid():
    value = uuid.uuid4()
    assert _coerce_run_id(str(value)) == value
    assert _coerce_run_id(value) == value


def test_coerce_run_id_rejects_garbage():
    with pytest.raises((ValueError, PermanentEvaluationError)):
        _coerce_run_id("not-a-uuid")


def test_invalid_run_id_is_a_permanent_task_failure():
    with pytest.raises(PermanentEvaluationError):
        run_evaluation.apply(args=["not-a-uuid"], throw=True)


# -- retry classification ----------------------------------------------------


def test_infrastructure_errors_are_retryable():
    assert is_retryable_infrastructure_error(OperationalError("db down", {}, Exception()))
    assert is_retryable_infrastructure_error(ConnectionError("refused"))


def test_wrapped_persistence_error_is_retryable():
    wrapped = RunnerError("run failed")
    wrapped.__cause__ = OperationalError("db down", {}, Exception())
    assert is_retryable_infrastructure_error(wrapped)


def test_permanent_errors_are_not_retryable():
    assert not is_retryable_infrastructure_error(ValueError("bad state"))
    assert not is_retryable_infrastructure_error(RunnerError("bad state"))
    assert not is_retryable_infrastructure_error(LLMConfigurationError("no key"))
    # Case-level provider errors are Phase 5's concern, never task retries.
    assert not is_retryable_infrastructure_error(LLMRateLimitError("429"))
    assert not is_retryable_infrastructure_error(PermanentEvaluationError("nope"))


def test_retry_countdown_is_exponential_and_capped():
    settings = Settings()
    assert _retry_countdown(settings, 0) == settings.worker_retry_backoff_seconds
    assert _retry_countdown(settings, 1) == settings.worker_retry_backoff_seconds * 2
    assert _retry_countdown(settings, 20) == settings.worker_retry_max_backoff_seconds


def test_retryable_task_failure_triggers_bounded_retry(monkeypatch):
    async def boom(run_id, settings, **kwargs):
        raise ConnectionError("postgres unreachable")

    monkeypatch.setattr("evalyx.worker.tasks.execute_evaluation", boom)
    with pytest.raises(Exception) as excinfo:
        run_evaluation.apply(args=[str(uuid.uuid4())], throw=True)
    from celery.exceptions import Retry

    assert isinstance(excinfo.value, Retry)


def test_permanent_task_failure_does_not_retry(monkeypatch):
    async def bad(run_id, settings, **kwargs):
        raise PermanentEvaluationError("run does not exist")

    monkeypatch.setattr("evalyx.worker.tasks.execute_evaluation", bad)
    with pytest.raises(PermanentEvaluationError):
        run_evaluation.apply(args=[str(uuid.uuid4())], throw=True)


# -- eligibility & execution -------------------------------------------------


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (RunStatus.PENDING, "execute"),
        (RunStatus.RUNNING, "execute"),  # interrupted execution → resume
        (RunStatus.COMPLETED, "rescore"),
        (RunStatus.FAILED, "skip"),
        (RunStatus.CANCELLED, "skip"),
    ],
)
def test_decide_action_maps_run_status(status, expected):
    assert decide_action(status) == expected


async def test_missing_run_is_a_permanent_error():
    provider = FakeProvider()

    def provider_factory(settings):
        raise AssertionError("provider must not be created for a missing run")

    with pytest.raises(PermanentEvaluationError):
        await execute_evaluation(
            uuid.uuid4(),
            Settings(),
            db_manager_factory=lambda s: FakeDBManager(None),
            provider_factory=provider_factory,
        )
    assert not provider.closed  # never created


@pytest.mark.parametrize("status", [RunStatus.FAILED, RunStatus.CANCELLED])
async def test_terminal_failed_runs_are_never_reexecuted(status):
    provider_created = []

    def provider_factory(settings):
        provider_created.append(True)
        return FakeProvider()

    run_id = uuid.uuid4()
    result = await execute_evaluation(
        run_id,
        Settings(),
        db_manager_factory=lambda s: FakeDBManager(FakeRun(status)),
        provider_factory=provider_factory,
    )

    assert result["action"] == "skipped"
    assert result["status"] == status.value
    assert provider_created == []


async def test_completed_run_is_rescored_not_reexecuted():
    pipeline = FakePipeline()
    run_id = uuid.uuid4()

    result = await execute_evaluation(
        run_id,
        Settings(),
        db_manager_factory=lambda s: FakeDBManager(FakeRun(RunStatus.COMPLETED)),
        provider_factory=lambda s: FakeProvider(),
        pipeline_factory=lambda provider, sf: pipeline,
    )

    assert pipeline.execute_calls == 0  # never blindly re-executed
    assert pipeline.score_calls == 1
    assert result["action"] == "rescored"
    assert result == summary_dict(run_id, action="rescored")


async def test_valid_run_invokes_pipeline_exactly_once():
    pipeline = FakePipeline()
    run_id = uuid.uuid4()

    result = await execute_evaluation(
        run_id,
        Settings(),
        db_manager_factory=lambda s: FakeDBManager(FakeRun(RunStatus.PENDING)),
        provider_factory=lambda s: FakeProvider(),
        pipeline_factory=lambda provider, sf: pipeline,
    )

    assert pipeline.execute_calls == 1
    assert result["action"] == "executed"


async def test_task_result_is_json_serializable_and_small():
    run_id = uuid.uuid4()
    result = await execute_evaluation(
        run_id,
        Settings(),
        db_manager_factory=lambda s: FakeDBManager(FakeRun(RunStatus.PENDING)),
        provider_factory=lambda s: FakeProvider(),
        pipeline_factory=lambda provider, sf: FakePipeline(),
    )

    encoded = json.dumps(result)
    assert json.loads(encoded) == summary_dict(run_id)
    assert len(encoded) < 300


# -- provider lifecycle -------------------------------------------------------


async def test_provider_is_closed_on_success():
    provider = FakeProvider()

    await execute_evaluation(
        uuid.uuid4(),
        Settings(),
        db_manager_factory=lambda s: FakeDBManager(FakeRun(RunStatus.PENDING)),
        provider_factory=lambda s: provider,
        pipeline_factory=lambda p, sf: FakePipeline(),
    )

    assert provider.closed


async def test_provider_is_closed_on_pipeline_failure():
    provider = FakeProvider()
    pipeline = FakePipeline(error=RuntimeError("pipeline exploded"))

    with pytest.raises(RuntimeError):
        await execute_evaluation(
            uuid.uuid4(),
            Settings(),
            db_manager_factory=lambda s: FakeDBManager(FakeRun(RunStatus.RUNNING)),
            provider_factory=lambda s: provider,
            pipeline_factory=lambda p, sf: pipeline,
        )

    assert provider.closed


async def test_provider_is_closed_on_provider_creation_failure():
    def provider_factory(settings):
        # Simulate a partially-constructed provider raising mid-setup.
        raise LLMConfigurationError("no api key")

    db = FakeDBManager(FakeRun(RunStatus.PENDING))
    with pytest.raises(LLMConfigurationError):
        await execute_evaluation(
            uuid.uuid4(),
            Settings(),
            db_manager_factory=lambda s: db,
            provider_factory=provider_factory,
        )
    assert db.disposed


async def test_database_is_disposed_on_success_and_failure():
    db = FakeDBManager(FakeRun(RunStatus.PENDING))
    await execute_evaluation(
        uuid.uuid4(),
        Settings(),
        db_manager_factory=lambda s: db,
        provider_factory=lambda s: FakeProvider(),
        pipeline_factory=lambda p, sf: FakePipeline(),
    )
    assert db.disposed

    db2 = FakeDBManager(FakeRun(RunStatus.PENDING))
    with pytest.raises(RuntimeError):
        await execute_evaluation(
            uuid.uuid4(),
            Settings(),
            db_manager_factory=lambda s: db2,
            provider_factory=lambda s: FakeProvider(),
            pipeline_factory=lambda p, sf: FakePipeline(error=RuntimeError("x")),
        )
    assert db2.disposed


# -- race guard -----------------------------------------------------------------


async def test_runner_error_with_now_completed_run_rescores_instead():
    """If another execution completed the run mid-flight, re-score, don't fail."""
    run_id = uuid.uuid4()
    pipeline = FakePipeline(error=RunnerError("already completed"))
    db = FakeDBManager(FakeRun(RunStatus.PENDING))
    # After the failure, the status check must see the completed run.
    db.run = FakeRun(RunStatus.PENDING)

    class StatusFlippingDB(FakeDBManager):
        @asynccontextmanager
        async def session(self):
            self.run = FakeRun(RunStatus.COMPLETED)  # flip after first read
            yield FakeSession(self.run)

    result = await execute_evaluation(
        run_id,
        Settings(),
        db_manager_factory=lambda s: StatusFlippingDB(FakeRun(RunStatus.PENDING)),
        provider_factory=lambda s: FakeProvider(),
        pipeline_factory=lambda p, sf: pipeline,
    )

    assert result["action"] == "rescored"
    assert pipeline.score_calls == 1


# -- observability: correlation & metrics (Phase 10) --------------------------


def test_task_logs_carry_task_id_and_run_id(monkeypatch):
    """§36: worker lifecycle events include task_id and run_id; no secrets,
    prompts, outputs, or Authorization headers ever appear."""
    run_id = uuid.uuid4()

    async def ok(run_id_arg, settings, **kwargs):
        return summary_dict(run_id_arg)

    monkeypatch.setattr("evalyx.worker.tasks.execute_evaluation", ok)
    metrics.reset()
    with capture_logs() as logs:
        run_evaluation.apply(args=[str(run_id)], throw=True)

    received = [e for e in logs if e["event"] == "task_received"]
    completed = [e for e in logs if e["event"] == "task_completed"]
    assert len(received) == 1 and len(completed) == 1
    assert received[0]["run_id"] == str(run_id)
    assert completed[0]["run_id"] == str(run_id)
    assert received[0]["task_id"]  # eager mode still assigns an id
    assert received[0]["task_id"] == completed[0]["task_id"]

    serialized = str(logs)
    for forbidden in ("OPENROUTER_API_KEY", "fake-api-key", "Authorization", "Bearer "):
        assert forbidden not in serialized


def test_task_failure_logs_error_type_not_message(monkeypatch):
    run_id = uuid.uuid4()

    async def boom(run_id_arg, settings, **kwargs):
        raise PermanentEvaluationError(
            f"secret probe value {run_id_arg} - details must not leak"
        )

    monkeypatch.setattr("evalyx.worker.tasks.execute_evaluation", boom)
    metrics.reset()
    with capture_logs() as logs, pytest.raises(PermanentEvaluationError):
        run_evaluation.apply(args=[str(run_id)], throw=True)
    failed = [e for e in logs if e["event"] == "task_failed"]
    assert len(failed) == 1
    assert failed[0]["error"] == "PermanentEvaluationError"  # type only
    assert "secret probe value" not in str(logs)  # message never logged
    metrics.reset()


def test_task_metrics_recorded_per_outcome(monkeypatch):
    run_id = uuid.uuid4()

    async def ok(run_id_arg, settings, **kwargs):
        return summary_dict(run_id_arg)

    monkeypatch.setattr("evalyx.worker.tasks.execute_evaluation", ok)
    metrics.reset()
    run_evaluation.apply(args=[str(run_id)], throw=True)
    snap = metrics.snapshot()
    outcomes = {e["labels"]["outcome"]: e["value"] for e in snap["worker_tasks_total"]}
    assert outcomes == {"success": 1.0}
    failures = snap.get("worker_task_failures_total", [])
    assert failures == []  # success produces no failure counter
    # Correlation ids are never metric labels (§38).
    assert "task_id" not in str(snap["worker_tasks_total"][0]["labels"])
    assert str(run_id) not in str(snap)
    metrics.reset()


def test_task_failure_metrics_recorded(monkeypatch):
    run_id = uuid.uuid4()

    async def boom(run_id_arg, settings, **kwargs):
        raise PermanentEvaluationError("nope")

    monkeypatch.setattr("evalyx.worker.tasks.execute_evaluation", boom)
    metrics.reset()
    with capture_logs(), pytest.raises(PermanentEvaluationError):
        run_evaluation.apply(args=[str(run_id)], throw=True)
    snap = metrics.snapshot()
    outcomes = {e["labels"]["outcome"]: e["value"] for e in snap["worker_tasks_total"]}
    assert outcomes == {"failed": 1.0}
    assert snap["worker_task_failures_total"][0]["value"] == 1.0
    assert snap["worker_task_failures_total"][0]["labels"] == {"task": "run_evaluation"}
    metrics.reset()


def test_task_context_cleared_after_execution(monkeypatch):
    """§31: the correlation context never leaks into the next task."""
    run_id = uuid.uuid4()

    async def ok(run_id_arg, settings, **kwargs):
        return summary_dict(run_id_arg)

    monkeypatch.setattr("evalyx.worker.tasks.execute_evaluation", ok)
    run_evaluation.apply(args=[str(run_id)], throw=True)
    assert correlation.get_run_id() is None
    assert correlation.get_task_id() is None
