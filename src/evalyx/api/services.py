"""Evaluation submission service: HTTP boundary → PostgreSQL + Celery.

The API never evaluates anything itself. This service performs exactly the
orchestration the HTTP layer needs:

1. validate referenced entities (application, optional application version,
   dataset version)
2. create the ``EvaluationRun`` (status ``pending``) with the configuration
   snapshot persisted for reproducibility
3. commit (PostgreSQL owns run state)
4. enqueue the existing Celery task ``run_evaluation`` with the run id —
   the only task submission path; no retry/execution logic lives here
5. return the queued-run information

Transaction/enqueue strategy (no distributed transaction exists between
PostgreSQL and Redis — none is pretended):

- run persisted **and committed first**, then the task is enqueued
- enqueue happens outside the database session, in a worker thread (the
  broker round-trip is blocking I/O and must not stall the event loop)
- enqueue failure → the run is transitioned to the existing terminal
  ``failed`` status (best effort) and :class:`EvaluationSubmissionError`
  is raised → HTTP 503. The client sees the truth: nothing is queued.
- accepted consistency window: if the process dies *after* a successful
  enqueue but before the response, the worker still executes the queued
  job — an extra completed run, never a lost one. The reverse (enqueue
  fails, mark-failed also fails, run stays ``pending``) is logged loudly;
  the run remains resubmittable and no false success was returned.

Idempotency: submission is deliberately **not** idempotent in this phase.
Retrying the HTTP request creates a new run. Documented semantics: treat a
2xx response as success, and rely on run ids rather than client-side
deduplication. A distributed idempotency-key mechanism is explicitly
deferred (it would require a schema change).
"""

import uuid

import anyio
import structlog
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from evalyx.api.errors import EvaluationSubmissionError
from evalyx.api.schemas.evaluations import EvaluationCreate
from evalyx.db.models import ApplicationVersion, EvaluationRun, RunStatus
from evalyx.db.repositories import (
    ApplicationRepository,
    DatasetRepository,
    EvaluationRepository,
    NotFoundError,
)

logger = structlog.get_logger(__name__)


def default_enqueue(run_id: uuid.UUID) -> str | None:
    """Submit the existing worker task; returns the Celery task id.

    Imported lazily so importing the API layer does not pull Celery into
    every process (and so tests can monkeypatch the task's ``delay``).
    """
    from evalyx.worker.tasks import run_evaluation

    async_result = run_evaluation.delay(str(run_id))
    return async_result.id


class EvaluationService:
    """Creates evaluation runs and submits them to the background worker."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        enqueue=None,
    ) -> None:
        self._session_factory = session_factory
        self._enqueue = enqueue or default_enqueue
        self._applications = ApplicationRepository()
        self._datasets = DatasetRepository()
        self._evaluations = EvaluationRepository()

    async def submit(self, request: EvaluationCreate) -> tuple[EvaluationRun, str | None]:
        """Validate, persist, and enqueue one evaluation run.

        Returns ``(run, task_id)``. Raises :class:`NotFoundError` for missing
        references and :class:`EvaluationSubmissionError` when enqueueing
        fails (the run is marked ``failed`` before that).
        """
        async with self._session_factory() as session:
            run = await self._create_run(session, request)

        task_id: str | None
        try:
            # Blocking broker round-trip → off the event loop.
            task_id = await anyio.to_thread.run_sync(self._enqueue, run.id)
        except Exception as exc:
            logger.error(
                "evaluation_enqueue_failed",
                run_id=str(run.id),
                error=type(exc).__name__,
            )
            await self._mark_failed(run.id)
            raise EvaluationSubmissionError(
                str(run.id),
                "The evaluation run was created but could not be queued. "
                "The run was marked failed; retry submission with a new request.",
            ) from exc
        return run, task_id

    async def _create_run(self, session: AsyncSession, request: EvaluationCreate) -> EvaluationRun:
        """Validate references and persist the pending run (commits)."""
        application = await self._applications.get(session, request.application_id)
        if application is None:
            raise NotFoundError(f"Application {request.application_id} does not exist.")

        if request.application_version_id is not None:
            version = await session.get(ApplicationVersion, request.application_version_id)
            if version is None or version.application_id != request.application_id:
                raise NotFoundError(
                    f"Application version {request.application_version_id} does not "
                    f"exist for application {request.application_id}."
                )

        dataset_version = await self._datasets.get_version_by_id(
            session, request.dataset_version_id
        )
        if dataset_version is None:
            raise NotFoundError(
                f"Dataset version {request.dataset_version_id} does not exist."
            )

        return await self._evaluations.create_run(
            session,
            application_id=request.application_id,
            application_version_id=request.application_version_id,
            dataset_version_id=request.dataset_version_id,
            agent_model=request.agent_model,
            judge_model=request.judge_model,
            configuration_snapshot=request.configuration_snapshot,
        )

    async def _mark_failed(self, run_id: uuid.UUID) -> None:
        """Best-effort transition of a stranded run to the terminal failure."""
        try:
            async with self._session_factory() as session:
                run = await self._evaluations.get_run(session, run_id)
                if run is not None:
                    await self._evaluations.update_status(session, run, RunStatus.FAILED)
        except Exception:  # noqa: BLE001 — best effort by design
            logger.error("evaluation_failure_state_update_failed", run_id=str(run_id))
