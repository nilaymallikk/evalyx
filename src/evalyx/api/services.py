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
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from evalyx.api.errors import EvaluationSubmissionError, EvaluationValidationError
from evalyx.api.schemas.evaluations import EvaluationCreate
from evalyx.core.config import Settings
from evalyx.db.models import ApplicationVersion, EvaluationRun, RunStatus, TestCase
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
        settings: Settings | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._enqueue = enqueue or default_enqueue
        self._settings = settings
        self._applications = ApplicationRepository()
        self._datasets = DatasetRepository()
        self._evaluations = EvaluationRepository()

    async def submit(
        self, request: EvaluationCreate, *, organization_id: uuid.UUID
    ) -> tuple[EvaluationRun, str | None]:
        """Validate, persist, and enqueue one evaluation run.

        ``organization_id`` is the authenticated tenant (from the verified
        Clerk session, never from request data). Every referenced resource
        must belong to it. Returns ``(run, task_id)``. Raises
        :class:`NotFoundError` for missing/foreign references and
        :class:`EvaluationSubmissionError` when enqueueing fails (the run is
        marked ``failed`` before that).
        """
        async with self._session_factory() as session:
            run = await self._create_run(session, request, organization_id=organization_id)

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
        # Request → run → task correlation event. Safe fields only (ids and
        # model name); the request_id is merged automatically from the
        # correlation context. Prompts/outputs/secrets are never logged.
        logger.info(
            "evaluation_submitted",
            run_id=str(run.id),
            task_id=task_id,
            application_id=str(run.application_id),
            dataset_version_id=str(run.dataset_version_id),
            agent_model=run.agent_model,
        )
        return run, task_id

    async def _create_run(
        self,
        session: AsyncSession,
        request: EvaluationCreate,
        *,
        organization_id: uuid.UUID,
    ) -> EvaluationRun:
        """Validate references and persist the pending run (commits).

        All references are tenant-checked: a foreign application or dataset
        version is indistinguishable from a missing one (404).
        """
        application = await self._applications.get_in_organization(
            session, request.application_id, organization_id=organization_id
        )
        if application is None:
            raise NotFoundError(f"Application {request.application_id} does not exist.")

        if request.application_version_id is not None:
            version = await session.get(ApplicationVersion, request.application_version_id)
            if version is None or version.application_id != request.application_id:
                raise NotFoundError(
                    f"Application version {request.application_version_id} does not "
                    f"exist for application {request.application_id}."
                )

        # Phase 15: generic (connection_type="http") applications must have
        # a resolvable, connection-carrying version — otherwise the run
        # would fail case-by-case at execution time. Rejected up front with
        # the uniform 404 (never leaking another tenant's state).
        if application.connection_type == "http":
            if request.application_version_id is not None:
                if version is None or not isinstance(version.connection, dict):
                    raise NotFoundError(
                        f"Application version {request.application_version_id} "
                        "has no connection configuration."
                    )
            elif (
                await self._applications.latest_version_with_connection(
                    session, application.id
                )
                is None
            ):
                raise NotFoundError(
                    f"Application {application.id} has no version with a "
                    "connection configuration."
                )

        dataset_version = await self._datasets.get_version_in_organization(
            session, request.dataset_version_id, organization_id=organization_id
        )
        if dataset_version is None:
            raise NotFoundError(
                f"Dataset version {request.dataset_version_id} does not exist."
            )

        # Phase 17 production bound: one organization must not trivially
        # consume all worker capacity with a huge dataset version. Bounded
        # 422 (counts only, never payloads); Phase 18 adds quotas.
        max_cases = (
            self._settings.max_cases_per_evaluation
            if self._settings is not None
            else 500
        )
        case_count = int(
            await session.scalar(
                select(func.count())
                .select_from(TestCase)
                .where(TestCase.dataset_version_id == dataset_version.id)
            )
            or 0
        )
        if case_count > max_cases:
            raise EvaluationValidationError(
                f"Dataset version has {case_count} cases, exceeding the "
                f"limit of {max_cases} cases per evaluation."
            )

        return await self._evaluations.create_run(
            session,
            organization_id=organization_id,
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
