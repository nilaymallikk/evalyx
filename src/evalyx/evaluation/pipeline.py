"""End-to-end evaluation orchestration.

Composes the Phase 5 execution runner, the guardrail harness, and the
scoring engine into one flow:

    EvaluationRunner  →  execution (case statuses = executed/error)
    GuardrailHarness  →  guardrail verdict rows for executed cases
    ScoringEngine     →  executed → passed/failed (errors stay error)

The pipeline is provider-independent; the caller supplies the provider.
"""

import uuid

from sqlalchemy.ext.asyncio import async_sessionmaker

from evalyx.db.models import CaseStatus, GuardrailStatus
from evalyx.db.repositories import EvaluationRepository
from evalyx.evaluation.runner import EvaluationRunner, EvaluationSummary, RunnerError
from evalyx.evaluation.scoring import ScoringEngine
from evalyx.guardrails.base import GuardrailContext
from evalyx.guardrails.harness import GuardrailHarness
from evalyx.guardrails.policy import GuardrailPolicy, default_guardrail_policy
from evalyx.llm.base import LLMProvider


class EvaluationPipeline:
    """Runs an evaluation end-to-end (execution + guardrails + scoring)."""

    def __init__(
        self,
        *,
        provider: LLMProvider,
        session_factory: async_sessionmaker,
        policy: GuardrailPolicy | None = None,
    ) -> None:
        self._provider = provider
        self._session_factory = session_factory
        self._policy = policy or default_guardrail_policy()
        self._runner = EvaluationRunner(provider, session_factory)
        self._evaluations = EvaluationRepository()

    async def run_and_score(
        self,
        *,
        application_id: uuid.UUID,
        dataset_version_id: uuid.UUID,
        agent_model: str,
        judge_model: str | None = None,
        application_version_id: uuid.UUID | None = None,
        configuration_snapshot: dict | None = None,
    ) -> EvaluationSummary:
        """Create, execute, and score an evaluation run."""
        execution = await self._runner.run(
            application_id=application_id,
            dataset_version_id=dataset_version_id,
            agent_model=agent_model,
            judge_model=judge_model,
            application_version_id=application_version_id,
            configuration_snapshot=configuration_snapshot,
        )
        return await self.score_existing_run(execution.run_id)

    async def score_existing_run(self, run_id: uuid.UUID) -> EvaluationSummary:
        """Run guardrails + scoring over an already-executed run."""
        if self._needs_judge_model:
            judge_model = await self._load_judge_model(run_id)
        else:
            judge_model = "unused-deterministic-only"
        context = GuardrailContext(
            provider=self._provider,
            judge_model=judge_model,
        )
        harness = GuardrailHarness(
            context=context,
            session_factory=self._session_factory,
            policy=self._policy,
        )
        await harness.evaluate_run(run_id)

        scoring = ScoringEngine(self._session_factory, policy=self._policy)
        counts = await scoring.score_run(run_id)

        run_status = await self._load_run_status(run_id)
        if run_status is None:
            raise RunnerError(f"Evaluation run {run_id} does not exist.")

        return EvaluationSummary(
            run_id=run_id,
            status=run_status,
            total_cases=sum(counts.values()),
            executed_cases=counts["executed"],
            error_cases=counts["error"],
            failed_cases=counts["failed"],
            passed_cases=counts["passed"],
            evaluation_error_cases=await self._count_guardrail_errors(run_id),
        )

    async def _count_guardrail_errors(self, run_id: uuid.UUID) -> int:
        """Count scored cases where at least one guardrail could not execute.

        These cases stay ``executed`` (evaluation incomplete, not a model
        failure); this count makes them visible in the summary.
        """
        async with self._session_factory() as session:
            case_results = await self._evaluations.list_case_results(session, run_id)
            error_cases = 0
            for case in case_results:
                if case.status is not CaseStatus.EXECUTED:
                    continue
                rows = await self._evaluations.list_guardrail_results(session, case.id)
                if any(row.status is GuardrailStatus.ERROR for row in rows):
                    error_cases += 1
        return error_cases

    @property
    def _needs_judge_model(self) -> bool:
        judge_names = {"instruction_following", "hallucination", "safety"}
        return bool(self._policy.enabled_guardrails & judge_names)

    async def _load_judge_model(self, run_id: uuid.UUID) -> str:
        async with self._session_factory() as session:
            run = await self._evaluations.get_run(session, run_id)
        if run is None:
            raise RunnerError(f"Evaluation run {run_id} does not exist.")
        if not run.judge_model:
            raise RunnerError(
                f"Evaluation run {run_id} has no judge_model; cannot run "
                "semantic guardrail evaluation."
            )
        return run.judge_model

    async def _load_run_status(self, run_id: uuid.UUID):
        async with self._session_factory() as session:
            run = await self._evaluations.get_run(session, run_id)
        return run.status if run is not None else None