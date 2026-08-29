"""Guardrail + scoring integration tests against real PostgreSQL (localhost:5433).

The LLM provider is always a fake — no network calls.
"""

import uuid

import pytest

from evalyx.db.models import CaseStatus, GuardrailStatus
from evalyx.db.repositories import (
    ApplicationRepository,
    DatasetRepository,
    EvaluationRepository,
)
from evalyx.db.session import DatabaseManager
from evalyx.evaluation.scoring import ScoringEngine
from evalyx.guardrails import GuardrailContext, GuardrailHarness
from evalyx.guardrails.policy import default_guardrail_policy
from evalyx.llm.base import LLMResponse, TokenUsage
from evalyx.llm.errors import LLMTimeoutError

pytestmark = pytest.mark.integration

JUDGE_MODEL = "minimax/minimax-m3:free"
AGENT_MODEL = "agent-model:free"


class JudgeAwareProvider:
    """Fake provider: agent calls return plain text; judge calls return JSON."""

    def __init__(self, judge_json: str | None = None, judge_error=None) -> None:
        self.judge_json = judge_json or (
            '{"passed": true, "score": 0.95, "reason": "meets criterion"}'
        )
        self.judge_error = judge_error
        self.judge_calls = 0

    async def complete(self, prompt, *, model, temperature=0.2, max_tokens=512, system=None):
        if model == JUDGE_MODEL:
            self.judge_calls += 1
            if self.judge_error is not None:
                raise self.judge_error
            return LLMResponse(
                content=self.judge_json,
                model=model,
                latency_ms=9,
                usage=TokenUsage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
                finish_reason="stop",
            )
        return LLMResponse(
            content=f"response for: {prompt}",
            model=model,
            latency_ms=11,
            usage=TokenUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
            finish_reason="stop",
        )

    async def close(self) -> None:
        pass


async def seed_case(
    db_manager: DatabaseManager,
    *,
    actual_output: str,
) -> tuple[uuid.UUID, uuid.UUID, uuid.UUID, uuid.UUID]:
    """Create app + dataset + case, run the case; return (app, run, version, case_result)."""
    async with db_manager.session() as session:
        apps = ApplicationRepository()
        datasets = DatasetRepository()
        evals = EvaluationRepository()
        app = await apps.create(session, name=f"app-{uuid.uuid4().hex[:8]}")
        dataset = await datasets.create(session, name=f"ds-{uuid.uuid4().hex[:8]}")
        version = await datasets.create_version(session, dataset_id=dataset.id, version=1)
        case = await datasets.add_test_case(
            session,
            dataset_version_id=version.id,
            name="case-1",
            input="Do the task please?",
            expected_output={"expected": True},
        )
        run = await evals.create_run(
            session,
            application_id=app.id,
            dataset_version_id=version.id,
            agent_model=AGENT_MODEL,
            judge_model=JUDGE_MODEL,
        )
        case_result = await evals.add_case_result(
            session,
            evaluation_run_id=run.id,
            test_case_id=case.id,
            input=case.input,
            expected_output=case.expected_output,
            actual_output=actual_output,
            status=CaseStatus.EXECUTED,
            latency_ms=10,
        )
        return app.id, run.id, version.id, case_result.id


def harness(db_manager: DatabaseManager, provider) -> GuardrailHarness:
    return GuardrailHarness(
        context=GuardrailContext(provider=provider, judge_model=JUDGE_MODEL),
        session_factory=db_manager.session_factory,
        policy=default_guardrail_policy(),
    )


async def guardrail_rows(db_manager: DatabaseManager, case_result_id: uuid.UUID):
    async with db_manager.session() as session:
        return await EvaluationRepository().list_guardrail_results(session, case_result_id)


async def test_harness_persists_all_five_verdicts(clean_db):
    _, _, _, case_id = await seed_case(clean_db, actual_output="The order ships in 3 days.")
    provider = JudgeAwareProvider()
    verdicts = await harness(clean_db, provider).evaluate_case(case_id)

    assert {v.name for v in verdicts} == {
        "pii",
        "prompt_injection",
        "instruction_following",
        "hallucination",
        "safety",
    }
    rows = await guardrail_rows(clean_db, case_id)
    assert len(rows) == 5
    # Deterministic checks ran (cheap, no LLM); judges used the provider.
    assert provider.judge_calls == 3  # instruction, hallucination, safety


async def test_harness_idempotent_no_duplicate_rows(clean_db):
    _, _, _, case_id = await seed_case(clean_db, actual_output="Fine, normal answer.")
    provider = JudgeAwareProvider()

    await harness(clean_db, provider).evaluate_case(case_id)
    first = await guardrail_rows(clean_db, case_id)
    await harness(clean_db, provider).evaluate_case(case_id)  # repeat
    second = await guardrail_rows(clean_db, case_id)

    assert len(second) == len(first) == 5
    assert sorted(r.name for r in second) == sorted(r.name for r in first)
    assert provider.judge_calls == 3  # judges NOT re-invoked on repeat


async def test_judge_timeout_does_not_erase_deterministic_results(clean_db):
    # Model output leaks an email -> PII fails deterministically.
    _, _, _, case_id = await seed_case(
        clean_db, actual_output="Email alice@example.com for support."
    )
    provider = JudgeAwareProvider(judge_error=LLMTimeoutError("judge timed out"))

    await harness(clean_db, provider).evaluate_case(case_id)

    rows = await guardrail_rows(clean_db, case_id)
    assert len(rows) == 5  # all five recorded
    by_name = {r.name: r for r in rows}

    assert by_name["pii"].status is GuardrailStatus.FAILED
    # Judge rows could not execute -> error status, distinct from failure.
    assert by_name["safety"].status is GuardrailStatus.ERROR
    assert by_name["instruction_following"].status is GuardrailStatus.ERROR
    assert "LLMTimeoutError" in by_name["safety"].metadata_["execution_error"]
    # The email value must never be persisted anywhere.
    persisted = str(by_name["pii"].metadata_) + str(by_name["pii"].reason)
    assert "alice@example.com" not in persisted


async def test_scoring_marks_case_failed_when_critical_pii_fails(clean_db):
    _, run_id, _, case_id = await seed_case(
        clean_db, actual_output="Call 555-123-4567 for help."
    )
    provider = JudgeAwareProvider()
    await harness(clean_db, provider).evaluate_case(case_id)

    counts = await ScoringEngine(clean_db.session_factory).score_run(run_id)
    assert counts["failed"] == 1
    assert counts["passed"] == 0

    async with clean_db.session() as session:
        case = await EvaluationRepository().get_case_result(session, case_id)
        assert case.status is CaseStatus.FAILED


async def test_scoring_all_pass_gives_passed(clean_db):
    _, run_id, _, case_id = await seed_case(
        clean_db, actual_output="A perfectly normal support answer."
    )
    provider = JudgeAwareProvider()
    await harness(clean_db, provider).evaluate_case(case_id)

    counts = await ScoringEngine(clean_db.session_factory).score_run(run_id)
    assert counts["passed"] == 1
    assert counts["failed"] == 0

    async with clean_db.session() as session:
        case = await EvaluationRepository().get_case_result(session, case_id)
        assert case.status is CaseStatus.PASSED


async def test_scoring_repeat_is_idempotent(clean_db):
    _, run_id, _, case_id = await seed_case(
        clean_db, actual_output="Another normal answer here."
    )
    provider = JudgeAwareProvider()
    await harness(clean_db, provider).evaluate_case(case_id)

    engine = ScoringEngine(clean_db.session_factory)
    await engine.score_run(run_id)
    await engine.score_run(run_id)  # repeat — must not duplicate rows

    rows = await guardrail_rows(clean_db, case_id)
    assert len(rows) == 5


async def test_pipeline_reports_evaluation_error_cases(clean_db):
    """A judge timeout leaves the case executed; the summary counts it."""
    _, run_id, _, _ = await seed_case(
        clean_db, actual_output="A perfectly normal support answer."
    )
    provider = JudgeAwareProvider(judge_error=LLMTimeoutError("judge timed out"))
    from evalyx.evaluation.pipeline import EvaluationPipeline

    pipeline = EvaluationPipeline(provider=provider, session_factory=clean_db.session_factory)
    summary = await pipeline.score_existing_run(run_id)

    assert summary.executed_cases == 1  # stays executed, not failed
    assert summary.evaluation_error_cases == 1
    assert summary.failed_cases == 0
    assert summary.passed_cases == 0


async def test_pipeline_end_to_end_summary_counts(clean_db):
    """3 cases: clean pass, PII fail, execution error -> passed=1, failed=1, error=1."""
    async with clean_db.session() as session:
        apps = ApplicationRepository()
        datasets = DatasetRepository()
        evals = EvaluationRepository()
        app = await apps.create(session, name=f"pipe-app-{uuid.uuid4().hex[:8]}")
        dataset = await datasets.create(session, name=f"pipe-ds-{uuid.uuid4().hex[:8]}")
        version = await datasets.create_version(session, dataset_id=dataset.id, version=1)
        c1 = await datasets.add_test_case(
            session, dataset_version_id=version.id, name="clean",
            input="Answer this question.", expected_output={"expected": True},
        )
        c2 = await datasets.add_test_case(
            session, dataset_version_id=version.id, name="pii",
            input="Answer this question.", expected_output={"expected": True},
        )
        c3 = await datasets.add_test_case(
            session, dataset_version_id=version.id, name="erroring",
            input="Answer this question.", expected_output={"expected": True},
        )
        run = await evals.create_run(
            session,
            application_id=app.id,
            dataset_version_id=version.id,
            agent_model=AGENT_MODEL,
            judge_model=JUDGE_MODEL,
        )
        await evals.add_case_result(
            session, evaluation_run_id=run.id, test_case_id=c1.id,
            input=c1.input, expected_output=c1.expected_output,
            actual_output="Perfectly fine answer.", status=CaseStatus.EXECUTED,
        )
        await evals.add_case_result(
            session, evaluation_run_id=run.id, test_case_id=c2.id,
            input=c2.input, expected_output=c2.expected_output,
            actual_output="Email contact me at bob@example.com.", status=CaseStatus.EXECUTED,
        )
        await evals.add_case_result(
            session, evaluation_run_id=run.id, test_case_id=c3.id,
            input=c3.input, expected_output=c3.expected_output,
            actual_output=None, status=CaseStatus.ERROR,
            error="LLMTimeoutError: timeout",
        )
        run_id = run.id

    provider = JudgeAwareProvider()
    from evalyx.evaluation.pipeline import EvaluationPipeline

    pipeline = EvaluationPipeline(provider=provider, session_factory=clean_db.session_factory)
    summary = await pipeline.score_existing_run(run_id)

    assert summary.total_cases == 3
    assert summary.passed_cases == 1
    assert summary.failed_cases == 1
    assert summary.error_cases == 1
    assert summary.executed_cases == 0

    async with clean_db.session() as session:
        results = await EvaluationRepository().list_case_results(session, run_id)
        statuses = sorted(r.status.value for r in results)
        assert statuses == ["error", "failed", "passed"]
        errored = next(r for r in results if r.status is CaseStatus.ERROR)
        # The execution-error case receives no guardrail rows.
        assert (
            await EvaluationRepository().list_guardrail_results(session, errored.id)
            == []
        )