"""Async execution core for the worker: the Celery → asyncio boundary.

The Celery task is a thin synchronous bridge. Every piece of business logic
lives in the existing Phase 5/6 components (``EvaluationPipeline`` and
below); this module owns only orchestration concerns:

- create a ``DatabaseManager`` per task execution (sessions are never shared
  between tasks, and no module-level session exists)
- validate run eligibility (idempotent duplicate-delivery handling)
- construct the LLM provider through the provider factory — the worker owns
  the provider it creates and always closes it, including on failure
- invoke the existing ``EvaluationPipeline``
- return a small JSON-serializable result

Eligibility semantics (PostgreSQL ``EvaluationRun.status`` is authoritative;
Celery task state is operational metadata only):

- run does not exist            → :class:`PermanentEvaluationError`
- ``failed`` / ``cancelled``    → skipped (never blindly re-executed)
- ``completed``                 → idempotent re-score only (guardrails and
                                  scoring are idempotent; nothing re-runs)
- ``pending`` / ``running``     → executed. Phase 5 resume semantics skip
                                  cases that already have a result (unique
                                  ``(run, test_case)`` constraint), so a
                                  worker crash mid-run resumes without
                                  duplicating completed cases.

A ``running`` status is treated as an interrupted execution — with late acks,
Celery redelivers a message after worker loss, exactly matching that state.
Duplicate delivery while a twin task is still executing is not a normal
Celery outcome; the Phase 5 unique constraints and idempotent guardrail
persistence are the safety net, so no distributed locking is used.

Retry classification: only job-level infrastructure failures (PostgreSQL
connectivity, network errors) are retryable — with a bounded retry policy
owned by the task. Provider-level HTTP failures are Phase 4's concern and
case-level provider errors are already isolated per-case by Phase 5; they
never escalate to task retries.
"""

import uuid
from collections.abc import Callable
from contextlib import suppress

import structlog
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import async_sessionmaker

from evalyx.core.config import Settings
from evalyx.db.models import RunStatus
from evalyx.db.repositories import EvaluationRepository
from evalyx.db.session import DatabaseManager
from evalyx.evaluation.pipeline import EvaluationPipeline
from evalyx.evaluation.runner import EvaluationSummary
from evalyx.llm.base import LLMProvider
from evalyx.llm.factory import create_provider

logger = structlog.get_logger(__name__)

#: Job-level infrastructure failures worth a bounded task retry. Everything
#: else (bad run state, configuration errors, per-case provider errors) is
#: permanent and must not be retried.
RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    OperationalError,
    ConnectionError,
    OSError,
)


class PermanentEvaluationError(Exception):
    """The task cannot and should not execute (bad run id, missing run)."""


def is_retryable_infrastructure_error(exc: BaseException) -> bool:
    """True when an exception (or its cause chain) is retryable infra failure.

    ``RunnerError`` wraps catastrophic persistence errors with ``raise ... from``,
    so the cause chain is inspected — a PostgreSQL outage inside the pipeline
    must still classify as retryable.
    """
    current: BaseException | None = exc
    for _ in range(4):  # bounded chain walk; no infinite loops on self-cause
        if current is None:
            return False
        if isinstance(current, RETRYABLE_EXCEPTIONS):
            return True
        current = current.__cause__
    return False


def decide_action(status: RunStatus) -> str:
    """Map the authoritative run status to a worker action.

    One of ``"execute"``, ``"rescore"`` (completed: guardrails/scoring are
    idempotent, execution never re-runs), or ``"skip"`` (terminal failure —
    a failed/cancelled run is not blindly re-executed).
    """
    if status in (RunStatus.FAILED, RunStatus.CANCELLED):
        return "skip"
    if status is RunStatus.COMPLETED:
        return "rescore"
    return "execute"


def _default_pipeline(
    provider: LLMProvider, session_factory: async_sessionmaker
) -> EvaluationPipeline:
    return EvaluationPipeline(provider=provider, session_factory=session_factory)


def _summary_result(summary: EvaluationSummary, *, action: str) -> dict:
    """Small JSON-serializable task result; details stay in PostgreSQL."""
    return {
        "run_id": str(summary.run_id),
        "status": summary.status.value,
        "action": action,
        "total_cases": summary.total_cases,
        "executed_cases": summary.executed_cases,
        "error_cases": summary.error_cases,
        "passed_cases": summary.passed_cases,
        "failed_cases": summary.failed_cases,
        "evaluation_error_cases": summary.evaluation_error_cases,
    }


def _skipped_result(run_id: uuid.UUID, status: RunStatus) -> dict:
    return {
        "run_id": str(run_id),
        "status": status.value,
        "action": "skipped",
        "total_cases": 0,
        "executed_cases": 0,
        "error_cases": 0,
        "passed_cases": 0,
        "failed_cases": 0,
        "evaluation_error_cases": 0,
    }


async def _load_run_status(db: DatabaseManager, run_id: uuid.UUID) -> RunStatus | None:
    async with db.session() as session:
        run = await EvaluationRepository().get_run(session, run_id)
    return run.status if run is not None else None


async def execute_evaluation(
    run_id: uuid.UUID,
    settings: Settings,
    *,
    db_manager_factory: Callable[[Settings], DatabaseManager] | None = None,
    provider_factory: Callable[[Settings], LLMProvider] | None = None,
    pipeline_factory: Callable[[LLMProvider, async_sessionmaker], EvaluationPipeline]
    | None = None,
) -> dict:
    """Execute one evaluation run end-to-end and return a small result dict.

    The factories are injectable (call-time defaults) so tests can substitute
    fakes without a live database or provider.
    """
    db_manager_factory = db_manager_factory or DatabaseManager
    provider_factory = provider_factory or create_provider
    pipeline_factory = pipeline_factory or _default_pipeline

    db = db_manager_factory(settings)
    provider: LLMProvider | None = None
    try:
        status = await _load_run_status(db, run_id)
        if status is None:
            raise PermanentEvaluationError(f"Evaluation run {run_id} does not exist.")
        action = decide_action(status)
        if action == "skip":
            logger.info(
                "worker_run_skipped", run_id=str(run_id), run_status=status.value
            )
            return _skipped_result(run_id, status)

        provider = provider_factory(settings)
        pipeline = pipeline_factory(provider, db.session_factory)

        if action == "rescore":
            summary = await pipeline.score_existing_run(run_id)
            return _summary_result(summary, action="rescored")

        try:
            summary = await pipeline.execute_and_score_existing_run(run_id)
        except Exception as exc:
            # Race guard: another execution may have finished the run between
            # the eligibility check and execution. A now-completed run is
            # simply re-scored; anything else is a genuine failure.
            status_now = await _load_run_status(db, run_id)
            if status_now is RunStatus.COMPLETED:
                logger.warning(
                    "worker_run_completed_elsewhere",
                    run_id=str(run_id),
                    error=type(exc).__name__,
                )
                summary = await pipeline.score_existing_run(run_id)
                return _summary_result(summary, action="rescored")
            raise

        logger.info(
            "worker_run_executed",
            run_id=str(run_id),
            run_status=summary.status.value,
            passed_cases=summary.passed_cases,
            failed_cases=summary.failed_cases,
            error_cases=summary.error_cases,
        )
        return _summary_result(summary, action="executed")
    finally:
        # The worker owns the provider and the database engine it created:
        # both are closed even when the evaluation fails or is interrupted.
        if provider is not None:
            with suppress(Exception):
                await provider.close()
        with suppress(Exception):
            await db.dispose()


async def mark_run_failed_if_active(run_id: uuid.UUID, settings: Settings) -> None:
    """Best-effort: fail an active run after a permanent task failure.

    A run left ``pending``/``running`` while its task permanently failed must
    not look healthy. Terminal runs are left untouched (their state is
    authoritative). Never raises — the original task failure must not be
    masked.
    """
    db = DatabaseManager(settings)
    try:
        async with db.session() as session:
            run = await EvaluationRepository().get_run(session, run_id)
            if run is not None and run.status in (RunStatus.PENDING, RunStatus.RUNNING):
                await EvaluationRepository().update_status(session, run, RunStatus.FAILED)
                logger.warning("worker_run_marked_failed", run_id=str(run_id))
    except Exception as exc:  # noqa: BLE001 — best effort by design
        logger.error(
            "worker_run_mark_failed_error", run_id=str(run_id), error=type(exc).__name__
        )
    finally:
        with suppress(Exception):
            await db.dispose()
