"""Evaluation engine tests: FakeProvider + real PostgreSQL (localhost:5433).

No network calls — the LLM provider is always a deterministic fake.
"""

import asyncio
import uuid
from collections.abc import AsyncIterator

import pytest
from sqlalchemy import text

from evalyx.db.models import CaseStatus, RunStatus
from evalyx.db.repositories import (
    ApplicationRepository,
    DatasetRepository,
    EvaluationRepository,
)
from evalyx.db.session import DatabaseManager
from evalyx.evaluation import EvaluationRunner, RunnerError
from evalyx.llm.base import LLMProvider, LLMResponse, TokenUsage
from evalyx.llm.errors import LLMRateLimitError, LLMTimeoutError
from evalyx.db.session import DatabaseManager

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
)


class FakeProvider:
    """Deterministic fake LLMProvider; no network, records every call."""

    def __init__(self, handler=None) -> None:
        self.calls: list[dict] = []
        self._handler = handler

    async def complete(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
        system: str | None = None,
    ) -> LLMResponse:
        self.calls.append(
            {
                "prompt": prompt,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "system": system,
            }
        )
        if self._handler is not None:
            outcome = self._handler(prompt)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        return LLMResponse(
            content=f"echo: {prompt}",
            model=model,
            latency_ms=7,
            usage=TokenUsage(prompt_tokens=3, completion_tokens=5, total_tokens=8),
            finish_reason="stop",
        )

    async def close(self) -> None:
        pass


@pytest.fixture
async def clean_db(db_manager: DatabaseManager) -> AsyncIterator[DatabaseManager]:
    async with db_manager.engine.begin() as conn:
        await conn.execute(
            text(f"TRUNCATE {', '.join(DOMAIN_TABLES)} RESTART IDENTITY CASCADE")
        )
    yield db_manager


async def seed(
    db_manager: DatabaseManager,
    *,
    case_inputs: list,
    run_version: int = 1,
    extra_version_cases: list | None = None,
    agent_model: str = "agent-model:free",
    judge_model: str = "judge-model:free",
    snapshot: dict | None = None,
    with_expected: bool = False,
):
    """Create application + dataset (+ versions) + a pending run."""
    async with db_manager.session() as session:
        apps, datasets, evaluations = (
            ApplicationRepository(),
            DatasetRepository(),
            EvaluationRepository(),
        )
        app = await apps.create(session, name=f"app-{uuid.uuid4().hex[:8]}")
        dataset = await datasets.create(session, name=f"dataset-{uuid.uuid4().hex[:8]}")
        version = await datasets.create_version(
            session, dataset_id=dataset.id, version=run_version
        )
        case_ids = []
        for index, case_input in enumerate(case_inputs):
            case = await datasets.add_test_case(
                session,
                dataset_version_id=version.id,
                name=f"case-{index}",
                input=case_input,
                expected_output={"marker": "never-in-prompt"} if with_expected else None,
            )
            case_ids.append(case.id)
        if extra_version_cases is not None:
            v2 = await datasets.create_version(
                session, dataset_id=dataset.id, version=run_version + 1
            )
            for index, case_input in enumerate(extra_version_cases):
                await datasets.add_test_case(
                    session, dataset_version_id=v2.id, name=f"v2-case-{index}", input=case_input
                )
        run = await evaluations.create_run(
            session,
            application_id=app.id,
            dataset_version_id=version.id,
            agent_model=agent_model,
            judge_model=judge_model,
            configuration_snapshot=snapshot or {},
        )
    return app.id, run.id, version.id, case_ids


def response_for(prompt: str) -> LLMResponse:
    return LLMResponse(
        content=f"reply:{prompt}",
        model="agent-model:free",
        latency_ms=42,
        usage=TokenUsage(prompt_tokens=2, completion_tokens=3, total_tokens=5),
        finish_reason="stop",
    )


async def get_run_status(db_manager: DatabaseManager, run_id: uuid.UUID) -> RunStatus:
    async with db_manager.session() as session:
        run = await EvaluationRepository().get_run(session, run_id)
        assert run is not None
        return run.status


async def get_results(db_manager: DatabaseManager, run_id: uuid.UUID):
    async with db_manager.session() as session:
        return await EvaluationRepository().list_case_results(session, run_id)


async def test_basic_execution_completes_run_with_three_results(clean_db):
    _, run_id, _, case_ids = await seed(
        clean_db, case_inputs=["case one", "case two", "case three"]
    )
    provider = FakeProvider()
    runner = EvaluationRunner(provider, clean_db.session_factory)

    summary = await runner.execute_run(run_id)

    assert summary.total_cases == 3
    assert summary.executed_cases == 3
    assert summary.error_cases == 0
    assert summary.status is RunStatus.COMPLETED
    assert summary.duration_ms >= 0
    assert len(provider.calls) == 3
    assert await get_run_status(clean_db, run_id) is RunStatus.COMPLETED
    results = await get_results(clean_db, run_id)
    assert {r.test_case_id for r in results} == set(case_ids)


async def test_run_convenience_creates_and_executes(clean_db):
    app_id, _, version_id, case_ids = await seed(clean_db, case_inputs=["only case"])
    provider = FakeProvider()
    runner = EvaluationRunner(provider, clean_db.session_factory)

    # A fresh run created through the public run() API (new run, same pinned
    # dataset version as the seeded pending one — only the new run executes).
    summary = await runner.run(
        application_id=app_id,
        dataset_version_id=version_id,
        agent_model="agent-model:free",
        judge_model="judge-model:free",
    )

    assert summary.total_cases == 1
    assert summary.executed_cases == 1
    results = await get_results(clean_db, summary.run_id)
    assert {r.test_case_id for r in results} == set(case_ids)


async def test_input_mapping_reaches_the_provider(clean_db):
    _, run_id, _, _ = await seed(
        clean_db,
        case_inputs=[
            "plain string input",
            {"prompt": "structured prompt"},
            {"b": 2, "a": 1},
            {"question": "no prompt key here"},
        ],
    )
    provider = FakeProvider()
    runner = EvaluationRunner(provider, clean_db.session_factory)
    await runner.execute_run(run_id)

    prompts = [call["prompt"] for call in provider.calls]
    assert prompts[0] == "plain string input"
    assert prompts[1] == "structured prompt"
    assert prompts[2] == '{"a": 1, "b": 2}'
    assert prompts[3] == '{"question": "no prompt key here"}'


async def test_context_included_and_expected_output_excluded(clean_db):
    async with clean_db.session() as session:
        datasets = DatasetRepository()
        dataset = await datasets.create(session, name=f"ctx-{uuid.uuid4().hex[:8]}")
        version = await datasets.create_version(session, dataset_id=dataset.id, version=1)
        await datasets.add_test_case(
            session,
            dataset_version_id=version.id,
            name="with-context",
            input="Answer me.",
            context={"persona": "support-agent"},
            expected_output={"must_contain": "SECRET-EXPECTED-ANSWER"},
        )
        version_id = version.id

    provider = FakeProvider()
    runner = EvaluationRunner(provider, clean_db.session_factory)
    async with clean_db.session() as session:
        app = await ApplicationRepository().create(
            session, name=f"ctx-app-{uuid.uuid4().hex[:8]}"
        )
        run = await EvaluationRepository().create_run(
            session,
            application_id=app.id,
            dataset_version_id=version_id,
            agent_model="agent-model:free",
        )
    await runner.execute_run(run.id)

    prompt = provider.calls[0]["prompt"]
    assert "Context:" in prompt and "support-agent" in prompt
    assert "SECRET-EXPECTED-ANSWER" not in prompt


async def test_agent_model_used_and_judge_model_never_called(clean_db):
    _, run_id, _, _ = await seed(
        clean_db,
        case_inputs=["a", "b"],
        agent_model="nvidia/nemotron-3-ultra-550b-a55b:free",
        judge_model="minimax/minimax-m3:free",
    )
    provider = FakeProvider()
    runner = EvaluationRunner(provider, clean_db.session_factory)
    await runner.execute_run(run_id)

    assert len(provider.calls) == 2
    assert all(
        call["model"] == "nvidia/nemotron-3-ultra-550b-a55b:free"
        for call in provider.calls
    )
    assert not any("minimax" in str(call) for call in provider.calls)


async def test_configuration_snapshot_drives_execution_parameters(clean_db):
    _, run_id, _, _ = await seed(
        clean_db,
        case_inputs=["one"],
        snapshot={"temperature": 0.7, "max_tokens": 99, "system": "Be terse."},
    )
    provider = FakeProvider()
    runner = EvaluationRunner(provider, clean_db.session_factory)
    await runner.execute_run(run_id)

    call = provider.calls[0]
    assert call["temperature"] == 0.7
    assert call["max_tokens"] == 99
    assert call["system"] == "Be terse."


async def test_invalid_snapshot_parameters_raise_runner_error(clean_db):
    _, run_id, _, _ = await seed(
        clean_db, case_inputs=["one"], snapshot={"temperature": "hot"}
    )
    provider = FakeProvider()
    runner = EvaluationRunner(provider, clean_db.session_factory)

    with pytest.raises(RunnerError, match="temperature"):
        await runner.execute_run(run_id)
    assert provider.calls == []


async def test_latency_and_usage_persisted(clean_db):
    _, run_id, _, _ = await seed(clean_db, case_inputs=["case"])
    provider = FakeProvider()
    runner = EvaluationRunner(provider, clean_db.session_factory)
    await runner.execute_run(run_id)

    results = await get_results(clean_db, run_id)
    assert results[0].latency_ms == 7
    assert results[0].metrics["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 5,
        "total_tokens": 8,
    }
    assert results[0].metrics["finish_reason"] == "stop"
    assert results[0].metrics["model"] == "agent-model:free"


async def test_missing_usage_is_not_invented(clean_db):
    _, run_id, _, _ = await seed(clean_db, case_inputs=["case"])
    provider = FakeProvider(
        handler=lambda prompt: LLMResponse(
            content="ok", model="agent-model:free", latency_ms=11
        )
    )
    runner = EvaluationRunner(provider, clean_db.session_factory)
    await runner.execute_run(run_id)

    results = await get_results(clean_db, run_id)
    assert "usage" not in results[0].metrics


async def test_partial_failure_continues_and_completes(clean_db):
    _, run_id, _, _ = await seed(clean_db, case_inputs=["ok-1", "boom", "ok-2"])

    def handler(prompt: str):
        if "boom" in prompt:
            return LLMTimeoutError("provider timed out")
        return response_for(prompt)

    provider = FakeProvider(handler=handler)
    runner = EvaluationRunner(provider, clean_db.session_factory)
    summary = await runner.execute_run(run_id)

    assert summary.status is RunStatus.COMPLETED
    assert summary.total_cases == 3
    assert summary.executed_cases == 2
    assert summary.error_cases == 1
    assert len(provider.calls) == 3  # the run continued after the failure

    results = await get_results(clean_db, run_id)
    assert len(results) == 3
    errored = next(r for r in results if r.status is CaseStatus.ERROR)
    assert "LLMTimeoutError" in errored.error
    assert errored.metrics["provider_error"] == "LLMTimeoutError"
    assert errored.metrics["provider_error_retryable"] is True
    assert errored.actual_output is None
    assert sum(1 for r in results if r.status is CaseStatus.EXECUTED) == 2
    assert all(r.input is not None for r in results)  # snapshots persisted


async def test_all_cases_erroring_still_completes(clean_db):
    _, run_id, _, _ = await seed(clean_db, case_inputs=["a", "b"])
    provider = FakeProvider(handler=lambda prompt: LLMRateLimitError("rate limited"))
    runner = EvaluationRunner(provider, clean_db.session_factory)
    summary = await runner.execute_run(run_id)

    assert summary.status is RunStatus.COMPLETED
    assert summary.executed_cases == 0
    assert summary.error_cases == 2
    assert await get_run_status(clean_db, run_id) is RunStatus.COMPLETED


async def test_empty_dataset_completes_with_zero_results(clean_db):
    _, run_id, _, _ = await seed(clean_db, case_inputs=[])
    provider = FakeProvider()
    runner = EvaluationRunner(provider, clean_db.session_factory)
    summary = await runner.execute_run(run_id)

    assert summary.total_cases == 0
    assert summary.status is RunStatus.COMPLETED
    assert provider.calls == []
    assert await get_results(clean_db, run_id) == []


async def test_dataset_version_is_pinned_not_latest(clean_db):
    _, run_id, _, v1_case_ids = await seed(
        clean_db,
        case_inputs=["v1-a", "v1-b"],
        extra_version_cases=["v2-a", "v2-b", "v2-c"],
    )
    provider = FakeProvider()
    runner = EvaluationRunner(provider, clean_db.session_factory)
    summary = await runner.execute_run(run_id)

    assert summary.total_cases == 2  # v1 only, even though v2 exists
    assert len(provider.calls) == 2
    results = await get_results(clean_db, run_id)
    assert {r.test_case_id for r in results} == set(v1_case_ids)


async def test_duplicate_execution_does_not_duplicate_results(clean_db):
    _, run_id, _, case_ids = await seed(clean_db, case_inputs=["a", "b", "c"])
    provider = FakeProvider()
    runner = EvaluationRunner(provider, clean_db.session_factory)

    await runner.execute_run(run_id)
    with pytest.raises(RunnerError, match="already completed"):
        await runner.execute_run(run_id)

    results = await get_results(clean_db, run_id)
    assert len(results) == 3  # never duplicated
    assert len(provider.calls) == 3


async def test_resume_skips_already_resulted_cases(clean_db):
    _, run_id, _, case_ids = await seed(clean_db, case_inputs=["a", "b", "c"])
    provider = FakeProvider()
    runner = EvaluationRunner(provider, clean_db.session_factory)

    # Simulate a partially executed run: one case already has a result.
    async with clean_db.session() as session:
        await EvaluationRepository().add_case_result(
            session,
            evaluation_run_id=run_id,
            test_case_id=case_ids[0],
            input={"pre": "existing"},
            status=CaseStatus.EXECUTED,
        )

    summary = await runner.execute_run(run_id)

    assert summary.total_cases == 3
    assert len(provider.calls) == 2  # only the two missing cases executed
    results = await get_results(clean_db, run_id)
    assert len(results) == 3  # no duplicates for the pre-existing case


async def test_cancellation_marks_run_cancelled_and_reraises(clean_db):
    _, run_id, _, _ = await seed(clean_db, case_inputs=["slow", "second"])

    class SlowProvider(FakeProvider):
        async def complete(self, prompt, **kwargs):
            if "slow" in prompt:
                await asyncio.sleep(3600)
            return await super().complete(prompt, **kwargs)

    runner = EvaluationRunner(SlowProvider(), clean_db.session_factory)
    task = asyncio.create_task(runner.execute_run(run_id))
    await asyncio.sleep(0.2)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert await get_run_status(clean_db, run_id) is RunStatus.CANCELLED





