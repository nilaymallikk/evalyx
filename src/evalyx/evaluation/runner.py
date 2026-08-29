"""Core evaluation engine.

Executes a run's pinned dataset version through an injected
:class:`evalyx.llm.base.LLMProvider` and persists structured results. The
runner depends only on the provider abstraction — it never imports concrete
providers (OpenRouter/Ollama) and never constructs provider clients.

Semantics (Phase 5 — execution only):

- **Result statuses are execution-honest.** A case that produced a provider
  response gets :attr:`CaseStatus.EXECUTED` — never ``passed``, because
  generation success does not imply correctness. Provider failures become
  :attr:`CaseStatus.ERROR` with safe diagnostics. Phase 6 scoring will
  transition EXECUTED cases to PASSED/FAILED.
- **Run lifecycle:** pending → running → completed. Even when every case
  errors, the run completes (errors are per-case records). ``failed`` is
  reserved for catastrophic runner-level failures (e.g. persistence errors).
  Cancellation marks the run cancelled and re-raises.
- **Empty dataset versions** complete successfully with zero results.
- **Sequential execution** — free models are rate-limited; no uncontrolled
  concurrency.
- **Idempotency:** test cases that already have a result in the run are
  skipped (resume semantics); the database enforces one result per
  (run, test case). Re-executing a terminal run raises :class:`RunnerError`.
- **Reproducibility:** the runner executes the run's pinned
  ``dataset_version_id``, ``agent_model``, and configuration snapshot —
  never "latest" versions or models.
- **Provider ownership:** the caller owns the provider; the runner uses it
  and never closes it.

Supported configuration snapshot keys (small, explicit set):
``temperature`` (number), ``max_tokens`` (positive int), ``system`` (str).
Recognized keys with wrong types raise :class:`RunnerError`; unknown keys
are ignored (they remain part of the recorded snapshot).
"""

import asyncio
import time
import uuid
from dataclasses import dataclass

import structlog
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from evalyx.db.models import CaseStatus, RunStatus, TestCase
from evalyx.db.repositories import DatasetRepository, EvaluationRepository
from evalyx.evaluation.prompts import build_prompt
from evalyx.llm.base import LLMProvider
from evalyx.llm.errors import LLMProviderError

logger = structlog.get_logger(__name__)

_DEFAULT_TEMPERATURE = 0.2
_DEFAULT_MAX_TOKENS = 512
_TERMINAL_STATUSES = frozenset(
    {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}
)


class RunnerError(Exception):
    """Raised for runner-level failures (bad run state, invalid config)."""


class EvaluationSummary(BaseModel):
    """Structured outcome of one evaluation run execution.

    Counts are execution-oriented unless scoring ran: ``passed_cases`` and
    ``evaluation_error_cases`` are filled by the Phase 6 scoring pipeline;
    the bare execution runner leaves them at 0 (cases are ``executed``).
    """

    run_id: uuid.UUID
    status: RunStatus
    total_cases: int
    #: Cases that produced a provider response but are not yet scored.
    executed_cases: int
    #: Cases that failed at the provider level (execution errors).
    error_cases: int
    #: Cases that failed evaluation (critical guardrail failure).
    failed_cases: int = 0
    #: Cases whose evaluation succeeded (all critical guardrails passed).
    passed_cases: int = 0
    #: Cases where a guardrail could not execute (scoring incomplete).
    evaluation_error_cases: int = 0
    duration_ms: int = 0


@dataclass(frozen=True)
class ExecutionParams:
    """The explicit set of per-run execution parameters."""

    temperature: float
    max_tokens: int
    system: str | None


def _execution_params(snapshot: dict) -> ExecutionParams:
    temperature = snapshot.get("temperature", _DEFAULT_TEMPERATURE)
    max_tokens = snapshot.get("max_tokens", _DEFAULT_MAX_TOKENS)
    system = snapshot.get("system")

    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise RunnerError(
            f"configuration_snapshot.temperature must be a number, got {temperature!r}"
        )
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise RunnerError(
            f"configuration_snapshot.max_tokens must be a positive integer, got {max_tokens!r}"
        )
    if system is not None and not isinstance(system, str):
        raise RunnerError(
            f"configuration_snapshot.system must be a string or null, got {system!r}"
        )

    return ExecutionParams(float(temperature), max_tokens, system)


@dataclass(frozen=True)
class _RunContext:
    """Values pinned at run-creation time, extracted before sessions close."""

    run_id: uuid.UUID
    dataset_version_id: uuid.UUID
    agent_model: str
    params: ExecutionParams


class EvaluationRunner:
    """Executes evaluation runs against an injected LLM provider.

    The runner manages its own short-lived sessions (one per operation) via
    the injected session factory, matching the repository pattern; sessions
    are never shared across concurrent work.
    """

    def __init__(
        self,
        provider: LLMProvider,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self._provider = provider
        self._session_factory = session_factory
        self._datasets = DatasetRepository()
        self._evaluations = EvaluationRepository()

    async def run(
        self,
        *,
        application_id: uuid.UUID,
        dataset_version_id: uuid.UUID,
        agent_model: str,
        judge_model: str | None = None,
        application_version_id: uuid.UUID | None = None,
        configuration_snapshot: dict | None = None,
    ) -> EvaluationSummary:
        """Create a new evaluation run and execute it to completion."""
        async with self._session_factory() as session:
            run = await self._evaluations.create_run(
                session,
                application_id=application_id,
                dataset_version_id=dataset_version_id,
                agent_model=agent_model,
                judge_model=judge_model,
                application_version_id=application_version_id,
                configuration_snapshot=configuration_snapshot or {},
            )
        return await self.execute_run(run.id)

    async def execute_run(self, run_id: uuid.UUID) -> EvaluationSummary:
        """Execute a pending (or partially executed) run."""
        context = await self._load_run_context(run_id)
        cases = await self._load_test_cases(context.dataset_version_id)
        already_done = await self._result_case_ids(run_id)
        pending = [case for case in cases if case.id not in already_done]

        await self._transition(run_id, RunStatus.RUNNING)
        logger.info(
            "evaluation_run_started",
            run_id=str(run_id),
            model=context.agent_model,
            total_cases=len(cases),
            already_resulted=len(already_done),
        )

        started = time.monotonic()
        executed = 0
        errors = 0
        try:
            for test_case in pending:
                case_status = await self._execute_case(context, test_case)
                if case_status is CaseStatus.EXECUTED:
                    executed += 1
                else:
                    errors += 1
        except asyncio.CancelledError:
            await self._best_effort_mark(run_id, RunStatus.CANCELLED)
            logger.warning("evaluation_run_cancelled", run_id=str(run_id))
            raise
        except Exception as exc:
            await self._best_effort_mark(run_id, RunStatus.FAILED)
            logger.error("evaluation_run_failed", run_id=str(run_id), error=str(exc))
            raise RunnerError(f"Evaluation run {run_id} failed: {exc}") from exc

        duration_ms = int((time.monotonic() - started) * 1000)
        await self._transition(run_id, RunStatus.COMPLETED)
        logger.info(
            "evaluation_run_completed",
            run_id=str(run_id),
            executed_cases=executed,
            error_cases=errors,
            duration_ms=duration_ms,
        )
        return EvaluationSummary(
            run_id=run_id,
            status=RunStatus.COMPLETED,
            total_cases=len(cases),
            executed_cases=executed,
            error_cases=errors,
            failed_cases=0,
            duration_ms=duration_ms,
        )

    # -- case execution ------------------------------------------------------

    async def _execute_case(
        self, context: _RunContext, test_case: TestCase
    ) -> CaseStatus:
        """Execute one case; provider failures become per-case error records."""
        logger.info(
            "evaluation_case_started",
            run_id=str(context.run_id),
            test_case_id=str(test_case.id),
        )

        prompt = build_prompt(test_case)
        status = CaseStatus.EXECUTED
        actual_output: str | None = None
        latency_ms: int | None = None
        error: str | None = None
        metrics: dict = {"model": context.agent_model}

        try:
            response = await self._provider.complete(
                prompt,
                model=context.agent_model,
                temperature=context.params.temperature,
                max_tokens=context.params.max_tokens,
                system=context.params.system,
            )
        except LLMProviderError as exc:
            status = CaseStatus.ERROR
            error = f"{type(exc).__name__}: {exc}"
            metrics["provider_error"] = type(exc).__name__
            metrics["provider_error_retryable"] = exc.retryable
            logger.warning(
                "evaluation_case_errored",
                run_id=str(context.run_id),
                test_case_id=str(test_case.id),
                provider_error=type(exc).__name__,
            )
        except Exception as exc:  # provider contract violation must not kill the run
            status = CaseStatus.ERROR
            error = f"UnexpectedProviderError: {exc}"
            metrics["provider_error"] = "UnexpectedProviderError"
            logger.warning(
                "evaluation_case_errored",
                run_id=str(context.run_id),
                test_case_id=str(test_case.id),
                provider_error="UnexpectedProviderError",
            )
        else:
            actual_output = response.content
            latency_ms = response.latency_ms
            if response.usage is not None:
                metrics["usage"] = response.usage.model_dump()
            if response.model:
                metrics["model"] = response.model
            if response.finish_reason:
                metrics["finish_reason"] = response.finish_reason
            logger.info(
                "evaluation_case_completed",
                run_id=str(context.run_id),
                test_case_id=str(test_case.id),
                latency_ms=response.latency_ms,
                status=status.value,
            )

        async with self._session_factory() as session:
            await self._evaluations.add_case_result(
                session,
                evaluation_run_id=context.run_id,
                test_case_id=test_case.id,
                input=test_case.input,  # snapshot of what was evaluated
                expected_output=test_case.expected_output,
                actual_output=actual_output,
                status=status,
                latency_ms=latency_ms,
                error=error,
                metrics=metrics,
            )
        return status

    # -- persistence helpers ---------------------------------------------------

    async def _load_run_context(self, run_id: uuid.UUID) -> _RunContext:
        async with self._session_factory() as session:
            run = await self._evaluations.get_run(session, run_id)
            if run is None:
                raise RunnerError(f"Evaluation run {run_id} does not exist.")
            if run.status in _TERMINAL_STATUSES:
                raise RunnerError(
                    f"Evaluation run {run_id} is already {run.status.value}; "
                    "create a new run instead of re-executing it."
                )
            # Pin execution to what the run recorded; extract values within
            # the session so detached attributes are safe afterwards.
            return _RunContext(
                run_id=run.id,
                dataset_version_id=run.dataset_version_id,
                agent_model=run.agent_model,
                params=_execution_params(run.configuration_snapshot),
            )

    async def _load_test_cases(self, dataset_version_id: uuid.UUID) -> list[TestCase]:
        """Load the pinned dataset version's cases (never another version)."""
        async with self._session_factory() as session:
            version = await self._datasets.get_version_by_id(session, dataset_version_id)
            if version is None:
                raise RunnerError(f"Dataset version {dataset_version_id} does not exist.")
            return await self._datasets.list_test_cases(session, dataset_version_id)

    async def _result_case_ids(self, run_id: uuid.UUID) -> set[uuid.UUID]:
        async with self._session_factory() as session:
            results = await self._evaluations.list_case_results(session, run_id)
            return {result.test_case_id for result in results}

    async def _transition(self, run_id: uuid.UUID, status: RunStatus) -> None:
        async with self._session_factory() as session:
            run = await self._evaluations.get_run(session, run_id)
            if run is not None:
                await self._evaluations.update_status(session, run, status)

    async def _best_effort_mark(self, run_id: uuid.UUID, status: RunStatus) -> None:
        """Attempt a terminal status transition; never mask the original error."""
        try:
            await self._transition(run_id, status)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "evaluation_run_status_update_failed",
                run_id=str(run_id),
                attempted_status=status.value,
                error=str(exc),
            )



