"""Application-target evaluation integration tests.

Two gates:

- ``EVALYX_RUN_INTEGRATION_TESTS=1`` — pipeline runs against a fake
  application target with live PostgreSQL (hermetic otherwise).
- ``EVALYX_RUN_MLGPT_INTEGRATION_TESTS=1`` — live calls against a running
  MLGPT instance (``MLGPT_BASE_URL``, default ``http://127.0.0.1:8001``).
  MLGPT must be started separately; see README "Evalyx × MLGPT".
"""

import os

import anyio
import httpx
import pytest
import sqlalchemy as sa
from sqlalchemy import text
from structlog.testing import capture_logs
from test_runner import DOMAIN_TABLES, FakeProvider, seed

from evalyx.application.base import (
    ApplicationInvocationError,
    ApplicationResponse,
)
from evalyx.application.http import HttpApplicationTarget
from evalyx.db.models import CaseStatus, RunStatus
from evalyx.db.repositories import EvaluationRepository
from evalyx.evaluation.pipeline import EvaluationPipeline


@pytest.fixture
async def clean_db(db_manager):
    async with db_manager.engine.begin() as conn:
        await conn.execute(
            text(f"TRUNCATE {', '.join(DOMAIN_TABLES)} RESTART IDENTITY CASCADE")
        )
    yield db_manager


class FakeApplication:
    """Deterministic application target; records prompts, never logs them."""

    def __init__(self, *, fail_on: str | None = None) -> None:
        self.prompts: list[str] = []
        self.closed = False
        self._fail_on = fail_on

    async def invoke(self, prompt: str) -> ApplicationResponse:
        self.prompts.append(prompt)
        if self._fail_on is not None and self._fail_on in prompt:
            raise ApplicationInvocationError(
                "Application 'fake-app' returned HTTP 500."
            )
        return ApplicationResponse(
            content=f"app-answer for case {len(self.prompts)}",
            latency_ms=11,
            status_code=200,
            metadata={"application": "fake-app", "sources_count": 3},
        )

    async def close(self) -> None:
        self.closed = True


# -- pipeline with an application target (live PostgreSQL) ------------------------


async def test_application_target_run_end_to_end(clean_db):
    """A run whose selector is an application target executes cases against
    the application, persists outputs + metadata, and scores normally."""
    _app_id, run_id, _dsv_id, _case_ids = await seed(
        clean_db, case_inputs=[{"prompt": "question one"}, {"prompt": "question two"}]
    )
    # Re-point the run at the application selector (seed defaults to a model).
    async with clean_db.session() as session:
        run = await EvaluationRepository().get_run(session, run_id)
        await session.execute(
            sa.update(type(run))
            .where(type(run).id == run_id)
            .values(agent_model="application:fake-app")
        )

    target = FakeApplication()
    judge = FakeProvider()
    pipeline = EvaluationPipeline(
        provider=judge,
        session_factory=clean_db.session_factory,
        application_target=target,
    )
    with capture_logs() as logs:
        summary = await pipeline.execute_and_score_existing_run(run_id)

    assert summary.status is RunStatus.COMPLETED
    assert summary.total_cases == 2
    # Target cleanup is the worker's responsibility (its finally block);
    # the pipeline itself never closes the caller's target.
    assert target.closed is False
    await target.close()
    assert target.closed is True

    async with clean_db.session() as session:
        results = await EvaluationRepository().list_case_results(session, run_id)
        by_prompt = {r.input.get("prompt"): r for r in results}
        assert by_prompt["question one"].actual_output == "app-answer for case 1"
        assert by_prompt["question one"].status is CaseStatus.EXECUTED
        assert by_prompt["question one"].metrics["application"] == "fake-app"
        assert by_prompt["question one"].metrics["sources_count"] == 3

    # The prompts reached the application; they never appear in Evalyx logs.
    assert target.prompts == ["question one", "question two"]
    serialized = str(logs)
    assert "question one" not in serialized
    assert "app-answer" not in serialized


async def test_application_invocation_failure_becomes_case_error(clean_db):
    """One failing invocation → one ERROR case; the run still completes."""
    _app_id, run_id, _dsv_id, _case_ids = await seed(
        clean_db, case_inputs=[{"prompt": "good"}, {"prompt": "boom"}]
    )
    async with clean_db.session() as session:
        run = await EvaluationRepository().get_run(session, run_id)
        await session.execute(
            sa.update(type(run))
            .where(type(run).id == run_id)
            .values(agent_model="application:fake-app")
        )

    target = FakeApplication(fail_on="boom")
    pipeline = EvaluationPipeline(
        provider=FakeProvider(),
        session_factory=clean_db.session_factory,
        application_target=target,
    )
    summary = await pipeline.execute_and_score_existing_run(run_id)

    assert summary.status is RunStatus.COMPLETED
    assert summary.error_cases == 1
    assert summary.executed_cases == 1

    # The failed case carries typed failure metadata (Phase 12); the
    # passing one has none. Quality (guardrail) results stay separate.
    async with clean_db.session() as session:
        results = await EvaluationRepository().list_case_results(session, run_id)
    errored = next(r for r in results if r.status is CaseStatus.ERROR)
    ok = next(r for r in results if r.status is CaseStatus.EXECUTED)
    assert errored.metrics["failure"]["category"] == "application_http_error"
    assert errored.metrics["failure"]["http_status"] == 500
    assert errored.metrics["failure"]["retryable"] is False
    assert "failure" not in (ok.metrics or {})


# -- gated live MLGPT tests ---------------------------------------------------------

requires_mlgpt = pytest.mark.skipif(
    os.environ.get("EVALYX_RUN_MLGPT_INTEGRATION_TESTS") != "1",
    reason="live MLGPT integration disabled (set EVALYX_RUN_MLGPT_INTEGRATION_TESTS=1)",
)


@requires_mlgpt
async def test_live_mlgpt_health(settings):
    def _check() -> httpx.Response:
        return httpx.get(f"{settings.mlgpt_base_url}/health", timeout=10.0)

    response = await anyio.to_thread.run_sync(_check)
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


@requires_mlgpt
async def test_live_mlgpt_invocation_shape_and_latency(settings):
    """One harmless prompt against the real MLGPT RAG pipeline."""
    target = HttpApplicationTarget(settings.mlgpt_base_url, application_name="mlgpt")
    try:
        response = await target.invoke(
            "What is machine learning, in one short sentence?"
        )
    finally:
        await target.close()
    assert response.status_code == 200
    assert response.content.strip()
    assert response.latency_ms > 0
    assert response.metadata["application"] == "mlgpt"
