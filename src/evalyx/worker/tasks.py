"""Celery task definitions — thin orchestration only.

Business logic lives in :mod:`evalyx.evaluation` and :mod:`evalyx.guardrails`
(Phases 5/6); a task receives a lightweight identifier, bridges into the async
pipeline through a single controlled event loop (``asyncio.run`` — one loop
per task invocation, never per LLM call), and classifies failures:

- retryable infrastructure failure → bounded Celery retry (exponential
  backoff, capped; ``WORKER_MAX_RETRIES`` from settings)
- soft time limit exceeded         → run marked failed; no retry (a job that
  outlived its time budget will not succeed by re-queuing)
- anything else                    → permanent task failure; the run is
  marked failed (best effort) so PostgreSQL never shows a healthy run
  behind a dead job

Task arguments contain identifiers only — never secrets, sessions, ORM
objects, or provider instances. The worker loads its configuration securely
from the environment.
"""

import asyncio
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import structlog
from billiard.exceptions import SoftTimeLimitExceeded

from evalyx.core.config import Settings, get_settings
from evalyx.worker.celery_app import celery_app
from evalyx.worker.execution import (
    PermanentEvaluationError,
    execute_evaluation,
    is_retryable_infrastructure_error,
    mark_run_failed_if_active,
)

logger = structlog.get_logger(__name__)


def _run_coroutine_sync(coroutine):
    """Bridge into asyncio with exactly one controlled event loop per call.

    Under a prefork Celery worker the task thread has no running loop, so
    ``asyncio.run`` is used directly. If a loop is already running in this
    thread (eager mode / async callers), the coroutine executes on an
    isolated loop in a dedicated thread instead — never on the caller's loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coroutine).result()


def _coerce_run_id(run_id: str | uuid.UUID) -> uuid.UUID:
    """Accept a run id as string or UUID; anything else is permanent."""
    if isinstance(run_id, uuid.UUID):
        return run_id
    return uuid.UUID(str(run_id))


def _retry_countdown(settings: Settings, retries: int) -> float:
    """Exponential backoff, capped to avoid retry storms."""
    delay = settings.worker_retry_backoff_seconds * (2**retries)
    return min(delay, settings.worker_retry_max_backoff_seconds)


@celery_app.task(bind=True, name="evalyx.worker.run_evaluation")
def run_evaluation(self, run_id: str | uuid.UUID) -> dict:
    """Execute one evaluation run in the background.

    Enqueue with ``run_evaluation.delay(str(run_id))``. The result is a small
    JSON dict; the PostgreSQL ``EvaluationRun`` remains the source of detailed
    results.
    """
    settings: Settings = get_settings()
    try:
        run_uuid = _coerce_run_id(run_id)
    except ValueError as exc:
        raise PermanentEvaluationError(f"Invalid run_id {run_id!r}: {exc}") from exc

    logger.info(
        "task_received",
        run_id=str(run_uuid),
        task_id=self.request.id,
        attempt=self.request.retries + 1,
    )
    started = time.monotonic()
    try:
        result = _run_coroutine_sync(execute_evaluation(run_uuid, settings))
    except SoftTimeLimitExceeded:
        logger.error(
            "task_time_limit_exceeded",
            run_id=str(run_uuid),
            task_id=self.request.id,
            soft_time_limit=settings.worker_soft_time_limit_seconds,
        )
        _best_effort_mark_failed(run_uuid)
        raise
    except Exception as exc:
        if is_retryable_infrastructure_error(exc):
            countdown = _retry_countdown(settings, self.request.retries)
            logger.warning(
                "task_retrying",
                run_id=str(run_uuid),
                task_id=self.request.id,
                attempt=self.request.retries + 1,
                max_retries=settings.worker_max_retries,
                error=type(exc).__name__,
                countdown=countdown,
            )
            raise self.retry(
                exc=exc,
                countdown=countdown,
                max_retries=settings.worker_max_retries,
            ) from exc
        logger.error(
            "task_failed",
            run_id=str(run_uuid),
            task_id=self.request.id,
            error=type(exc).__name__,
        )
        _best_effort_mark_failed(run_uuid)
        raise
    logger.info(
        "task_completed",
        run_id=str(run_uuid),
        task_id=self.request.id,
        status=result.get("status"),
        action=result.get("action"),
        duration_ms=int((time.monotonic() - started) * 1000),
    )
    return result


def _best_effort_mark_failed(run_id: uuid.UUID) -> None:
    """Mark an active run failed on permanent task failure; never masks the
    original error."""
    try:
        _run_coroutine_sync(mark_run_failed_if_active(run_id, get_settings()))
    except Exception:  # noqa: BLE001 — best effort by design
        logger.error("task_failure_state_update_failed", run_id=str(run_id))
