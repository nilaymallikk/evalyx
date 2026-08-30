"""Guardrail harness: execute configured guardrails and persist verdicts.

- deterministic execution order (deterministic first, then judge checks)
- per-guardrail failure isolation: one failing guardrail never prevents the
  others from running
- idempotent persistence: a guardrail name already recorded for a case is
  skipped, and the database enforces uniqueness per (case, name)
- the harness owns persistence; the guardrails themselves never touch the DB
"""

import uuid

import structlog
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from evalyx.core.metrics import metrics
from evalyx.db.models import (
    CaseStatus,
    EvaluationCaseResult,
    GuardrailResult,
    GuardrailStatus,
)
from evalyx.db.repositories import EvaluationRepository
from evalyx.guardrails.base import (
    Guardrail,
    GuardrailContext,
    GuardrailVerdict,
)
from evalyx.guardrails.errors import GuardrailExecutionError
from evalyx.guardrails.hallucination import HallucinationJudge
from evalyx.guardrails.injection import PromptInjectionGuardrail
from evalyx.guardrails.instruction import InstructionFollowingJudge
from evalyx.guardrails.pii import PIIGuardrail
from evalyx.guardrails.policy import (
    DETERMINISTIC_NAMES,
    GuardrailPolicy,
    default_guardrail_policy,
)
from evalyx.guardrails.safety import SafetyJudge

logger = structlog.get_logger(__name__)


class GuardrailHarness:
    """Executes the configured guardrail set for evaluation cases."""

    def __init__(
        self,
        *,
        context: GuardrailContext,
        session_factory: async_sessionmaker[AsyncSession],
        policy: GuardrailPolicy | None = None,
    ) -> None:
        self._context = context
        self._session_factory = session_factory
        self._policy = policy or default_guardrail_policy()
        self._evaluations = EvaluationRepository()

    def guardrails(self) -> list[Guardrail]:
        """The enabled guardrails in deterministic execution order."""
        by_name: dict[str, Guardrail] = {
            PIIGuardrail.name: PIIGuardrail(),
            PromptInjectionGuardrail.name: PromptInjectionGuardrail(),
            InstructionFollowingJudge.name: InstructionFollowingJudge(),
            HallucinationJudge.name: HallucinationJudge(),
            SafetyJudge.name: SafetyJudge(),
        }
        enabled = [g for name, g in by_name.items() if self._policy.is_enabled(name)]
        enabled.sort(key=lambda g: (g.name not in DETERMINISTIC_NAMES, g.name))
        return enabled

    async def evaluate_case(self, case_result_id: uuid.UUID) -> list[GuardrailVerdict]:
        """Run guardrails for one case result and persist the verdicts.

        Returns every verdict (existing rows plus newly created ones).
        Idempotent: guardrail names already recorded for the case are not
        re-run.
        """
        case_result = await self._load_case_result(case_result_id)
        existing = {
            row.name: row for row in await self._load_existing_rows(case_result_id)
        }
        verdicts: list[GuardrailVerdict] = []

        for guardrail in self.guardrails():
            if guardrail.name in existing:
                verdicts.append(_verdict_from_row(guardrail.name, existing[guardrail.name]))
                continue

            verdict = await self._execute_one(guardrail, case_result)
            verdicts.append(verdict)
            await self._persist(case_result_id, verdict)

            if verdict.is_error:
                logger.warning(
                    "guardrail_execution_error",
                    case_result_id=str(case_result_id),
                    guardrail=guardrail.name,
                    error=verdict.execution_error,
                )
        return verdicts

    async def evaluate_run(self, run_id: uuid.UUID) -> dict[str, list[GuardrailVerdict]]:
        """Run guardrails for every EXECUTED case result of a run.

        Cases already scored (passed/failed) and execution-error cases are
        skipped. Returns {case_result_id: verdicts}.
        """
        async with self._session_factory() as session:
            case_results = await self._evaluations.list_case_results(session, run_id)

        outcomes: dict[str, list[GuardrailVerdict]] = {}
        for case in case_results:
            if case.status is not CaseStatus.EXECUTED:
                continue
            outcomes[str(case.id)] = await self.evaluate_case(case.id)
        return outcomes

    async def _load_case_result(self, case_result_id: uuid.UUID) -> EvaluationCaseResult:
        async with self._session_factory() as session:
            case_result = await self._evaluations.get_case_result(session, case_result_id)
        if case_result is None:
            raise ValueError(f"Evaluation case result {case_result_id} does not exist.")
        return case_result

    async def _load_existing_rows(self, case_result_id: uuid.UUID):
        async with self._session_factory() as session:
            return await self._evaluations.list_guardrail_results(session, case_result_id)

    async def _execute_one(
        self, guardrail: Guardrail, case_result: EvaluationCaseResult
    ) -> GuardrailVerdict:
        try:
            verdict = await guardrail.evaluate(case_result, context=self._context)
        except GuardrailExecutionError as exc:
            verdict = GuardrailVerdict(
                name=guardrail.name,
                type=guardrail.type,
                passed=False,
                score=None,
                reason="Guardrail could not execute.",
                metadata={"execution_error": str(exc)},
                execution_error=str(exc),
            )
        except Exception as exc:  # defensive: never let one guardrail kill the set  # noqa: BLE001
            verdict = GuardrailVerdict(
                name=guardrail.name,
                type=guardrail.type,
                passed=False,
                score=None,
                reason="Guardrail execution failed unexpectedly.",
                metadata={"execution_error": f"{type(exc).__name__}: {exc}"},
                execution_error=f"{type(exc).__name__}: {exc}",
            )
        # Aggregate guardrail outcome metric. Labels are bounded: guardrail
        # names come from the configured harness (a small, controlled set)
        # and status is the GuardrailStatus enum. Never PII values, prompt
        # or output content.
        metrics.increment(
            "guardrail_evaluations_total",
            {"name": guardrail.name, "status": _verdict_status(verdict).value},
        )
        return verdict

    async def _persist(self, case_result_id: uuid.UUID, verdict: GuardrailVerdict) -> None:
        async with self._session_factory() as session:
            await self._evaluations.add_guardrail_result(
                session,
                evaluation_case_result_id=case_result_id,
                name=verdict.name,
                type=verdict.type,
                passed=verdict.passed,
                status=_verdict_status(verdict),
                score=verdict.score,
                reason=verdict.reason,
                metadata=verdict.metadata,
            )


def _verdict_status(verdict: GuardrailVerdict) -> GuardrailStatus:
    if verdict.is_error:
        return GuardrailStatus.ERROR
    return GuardrailStatus.PASSED if verdict.passed else GuardrailStatus.FAILED


def _verdict_from_row(name: str, row: GuardrailResult) -> GuardrailVerdict:
    return GuardrailVerdict(
        name=name,
        type=row.type or "unknown",
        passed=row.passed,
        score=row.score,
        reason=row.reason,
        metadata=row.metadata_,
        execution_error=row.metadata_.get("execution_error"),
    )