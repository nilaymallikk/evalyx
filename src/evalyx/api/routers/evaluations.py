"""Evaluation submission and inspection endpoints.

The API never executes evaluations: submission persists a run and enqueues
the existing Celery task; inspection reads persisted PostgreSQL state.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from evalyx.api.dependencies import (
    get_evaluation_service,
    get_regression_service,
    get_session,
    pagination_params,
)
from evalyx.api.schemas.common import Page
from evalyx.api.schemas.evaluations import (
    CaseStatusCounts,
    EvaluationCaseResultResponse,
    EvaluationCreate,
    EvaluationRunSummary,
    EvaluationSubmissionResponse,
    GuardrailResultResponse,
    RunReliabilityReport,
    failure_from_metrics,
)
from evalyx.api.services import EvaluationService
from evalyx.db.models import CaseStatus, EvaluationCaseResult, EvaluationRun, RunStatus
from evalyx.db.repositories import EvaluationRepository, NotFoundError
from evalyx.evaluation.regression.comparison import sanitize_configuration
from evalyx.evaluation.regression.models import RegressionReport
from evalyx.evaluation.regression.service import RegressionService

router = APIRouter(prefix="/evaluations", tags=["evaluations"])


def _repository() -> EvaluationRepository:
    return EvaluationRepository()


async def _get_run_or_404(session: AsyncSession, run_id: uuid.UUID) -> EvaluationRun:
    run = await _repository().get_run(session, run_id)
    if run is None:
        raise NotFoundError(f"Evaluation run {run_id} does not exist.")
    return run


def _to_summary(
    run: EvaluationRun, counts: CaseStatusCounts | None
) -> EvaluationRunSummary:
    """Column-only mapping (never touches lazy relationships).

    ``configuration_snapshot`` is re-sanitized on read — defense in depth so
    the API never surfaces secret-looking keys even if a legacy snapshot
    somehow contains one (writes are already sanitized at the boundary).
    """
    return EvaluationRunSummary(
        id=run.id,
        status=run.status,
        application_id=run.application_id,
        application_version_id=run.application_version_id,
        dataset_version_id=run.dataset_version_id,
        agent_model=run.agent_model,
        judge_model=run.judge_model,
        configuration_snapshot=sanitize_configuration(run.configuration_snapshot),
        started_at=run.started_at,
        completed_at=run.completed_at,
        created_at=run.created_at,
        counts=counts,
    )


def _counts_from(counts_by_status: dict[CaseStatus, int]) -> CaseStatusCounts:
    return CaseStatusCounts(
        total=sum(counts_by_status.values()),
        passed=counts_by_status.get(CaseStatus.PASSED, 0),
        failed=counts_by_status.get(CaseStatus.FAILED, 0),
        error=counts_by_status.get(CaseStatus.ERROR, 0),
        executed=counts_by_status.get(CaseStatus.EXECUTED, 0),
    )


def _guardrail_response(guardrail) -> GuardrailResultResponse:
    """Column-only mapping (``metadata_`` column → ``metadata`` field)."""
    return GuardrailResultResponse(
        id=guardrail.id,
        evaluation_case_result_id=guardrail.evaluation_case_result_id,
        name=guardrail.name,
        type=guardrail.type,
        status=guardrail.status.value,
        passed=guardrail.passed,
        score=guardrail.score,
        reason=guardrail.reason,
        metadata=guardrail.metadata_,
        created_at=guardrail.created_at,
    )


def _to_case_result(
    case: EvaluationCaseResult, guardrails: list[GuardrailResultResponse]
) -> EvaluationCaseResultResponse:
    """Column-only mapping; guardrails come from the batched run-level query."""
    return EvaluationCaseResultResponse(
        id=case.id,
        test_case_id=case.test_case_id,
        input=case.input,
        expected_output=case.expected_output,
        actual_output=case.actual_output,
        status=case.status,
        latency_ms=case.latency_ms,
        error=case.error,
        metrics=case.metrics,
        failure=failure_from_metrics(case.metrics),
        guardrail_results=guardrails,
        created_at=case.created_at,
    )


@router.post(
    "",
    response_model=EvaluationSubmissionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a background evaluation",
    description=(
        "Validate references, persist the run (status `pending`), and enqueue "
        "the existing `run_evaluation` Celery task. Execution happens in the "
        "worker — never inside the HTTP request. Poll `status_url` for state; "
        "PostgreSQL is authoritative. Not idempotent: a retried request "
        "creates a new run. If enqueueing fails, the run is marked `failed` "
        "and 503 is returned — the client is never told a job was queued "
        "when it was not."
    ),
    responses={
        202: {"description": "Evaluation accepted and queued."},
        404: {"description": "Referenced application/application version/dataset version not found."},
        422: {"description": "Invalid request body."},
        503: {"description": "Run persisted but the queue rejected the job."},
    },
)
async def submit_evaluation(
    payload: EvaluationCreate,
    service: Annotated[EvaluationService, Depends(get_evaluation_service)],
) -> EvaluationSubmissionResponse:
    run, task_id = await service.submit(payload)
    return EvaluationSubmissionResponse(
        run_id=run.id,
        status=run.status,
        task_id=task_id,
        status_url=f"/api/v1/evaluations/{run.id}",
    )


@router.get(
    "/{run_id}",
    response_model=EvaluationRunSummary,
    summary="Retrieve an evaluation run (summary)",
    description=(
        "Persisted run state, metadata, and case counts — no case outputs "
        "(use the results endpoint). Celery/Redis task state is never "
        "treated as authoritative."
    ),
    responses={404: {"description": "Run not found."}},
)
async def get_evaluation(
    run_id: uuid.UUID,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> EvaluationRunSummary:
    run = await _get_run_or_404(session, run_id)
    counts = await _repository().count_case_results_by_status(session, run_id)
    return _to_summary(run, _counts_from(counts))


@router.get(
    "",
    response_model=Page[EvaluationRunSummary],
    summary="List evaluation runs",
    description="Summaries only, newest first (created_at descending). Optionally filter by application.",
)
async def list_evaluations(
    session: Annotated[AsyncSession, Depends(get_session)],
    pagination: Annotated[tuple[int, int], Depends(pagination_params)],
    application_id: uuid.UUID | None = None,
) -> Page[EvaluationRunSummary]:
    limit, offset = pagination
    repository = _repository()
    runs = await repository.list_runs(
        session, limit=limit, offset=offset, application_id=application_id
    )
    items = [_to_summary(run, None) for run in runs]
    # Total via one cheap count query with the same filter.
    count_query = select(func.count()).select_from(EvaluationRun)
    if application_id is not None:
        count_query = count_query.where(EvaluationRun.application_id == application_id)
    total = int((await session.execute(count_query)).scalar_one())
    return Page[EvaluationRunSummary](items=items, total=total, limit=limit, offset=offset)


@router.get(
    "/{run_id}/results",
    response_model=Page[EvaluationCaseResultResponse],
    summary="Retrieve case-level results",
    description=(
        "Paginated case results with nested guardrail outcomes. Ordered by "
        "creation time ascending (stable page order). Loaded with batched "
        "queries, never one query per case."
    ),
    responses={404: {"description": "Run not found."}},
)
async def get_evaluation_results(
    run_id: uuid.UUID,
    session: Annotated[AsyncSession, Depends(get_session)],
    pagination: Annotated[tuple[int, int], Depends(pagination_params)],
) -> Page[EvaluationCaseResultResponse]:
    limit, offset = pagination
    repository = _repository()
    await _get_run_or_404(session, run_id)

    counts = await repository.count_case_results_by_status(session, run_id)
    cases = await repository.list_case_results(session, run_id, limit=limit, offset=offset)
    guardrails = await repository.list_guardrail_results_for_run(session, run_id)
    by_case: dict[uuid.UUID, list[GuardrailResultResponse]] = {}
    for guardrail in guardrails:
        by_case.setdefault(guardrail.evaluation_case_result_id, []).append(
            _guardrail_response(guardrail)
        )

    items = [
        _to_case_result(case, by_case.get(case.id, [])) for case in cases
    ]
    return Page[EvaluationCaseResultResponse](
        items=items, total=sum(counts.values()), limit=limit, offset=offset
    )


@router.get(
    "/{run_id}/guardrails",
    response_model=Page[GuardrailResultResponse],
    summary="Retrieve guardrail results",
    description=(
        "Flat, paginated list of every guardrail check in the run. Metadata "
        "contains categories/counts only — the platform never persists raw "
        "PII matches, API keys, or provider payloads."
    ),
    responses={404: {"description": "Run not found."}},
)
async def get_evaluation_guardrails(
    run_id: uuid.UUID,
    session: Annotated[AsyncSession, Depends(get_session)],
    pagination: Annotated[tuple[int, int], Depends(pagination_params)],
) -> Page[GuardrailResultResponse]:
    limit, offset = pagination
    repository = _repository()
    await _get_run_or_404(session, run_id)
    guardrails = await repository.list_guardrail_results_for_run(
        session, run_id, limit=limit, offset=offset
    )
    total = await repository.count_guardrail_results_for_run(session, run_id)
    return Page[GuardrailResultResponse](
        items=[_guardrail_response(g) for g in guardrails],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{run_id}/regressions",
    response_model=Page[RegressionReport],
    summary="List regression comparisons involving a run",
    description=(
        "Every persisted comparison where the run is the baseline or the "
        "current side, newest first. Delegates entirely to RegressionService."
    ),
    responses={404: {"description": "Run not found."}},
)
async def list_regressions_for_run(
    run_id: uuid.UUID,
    session: Annotated[AsyncSession, Depends(get_session)],
    service: Annotated[RegressionService, Depends(get_regression_service)],
) -> Page[RegressionReport]:
    await _get_run_or_404(session, run_id)
    reports = await service.list_for_run(run_id)
    return Page[RegressionReport](
        items=reports, total=len(reports), limit=len(reports), offset=0
    )


@router.get(
    "/{run_id}/status",
    response_model=RunStatus,
    include_in_schema=False,
    summary="Retrieve only the run status",
    description="Convenience alias returning the plain run status value.",
    responses={404: {"description": "Run not found."}},
)
async def get_evaluation_status(
    run_id: uuid.UUID,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> RunStatus:
    run = await _get_run_or_404(session, run_id)
    return run.status


@router.get(
    "/{run_id}/reliability",
    response_model=RunReliabilityReport,
    summary="Summarize execution reliability for one run",
    description=(
        "Deterministic breakdown of the run's execution failures by "
        "category (Phase 12 failure analysis). Answers 'did the application "
        "fail to answer, and why' — quality failures live in the case "
        "results and guardrail endpoints. Categories come from the "
        "classifier at execution time; rows persisted before Phase 12 count "
        "under `unknown` only when classified data exists."
    ),
    responses={404: {"description": "Run not found."}},
)
async def get_run_reliability(
    run_id: uuid.UUID,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> RunReliabilityReport:
    await _get_run_or_404(session, run_id)
    cases = await _repository().list_case_results(session, run_id)

    failures = [failure_from_metrics(case.metrics) for case in cases]
    breakdown: dict[str, int] = {}
    retryable = 0
    for failure in failures:
        if failure is not None:
            breakdown[failure.category] = breakdown.get(failure.category, 0) + 1
            retryable += 1 if failure.retryable else 0

    total = len(cases)
    errored = sum(1 for case in cases if case.status is CaseStatus.ERROR)
    return RunReliabilityReport(
        total_cases=total,
        errored_cases=errored,
        classified_failures=sum(breakdown.values()),
        unclassified_execution_failures=errored - sum(breakdown.values()),
        retryable_failures=retryable,
        failure_breakdown=dict(
            sorted(breakdown.items(), key=lambda item: (-item[1], item[0]))
        ),
    )
