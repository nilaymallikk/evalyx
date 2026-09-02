"""Repository for evaluation runs, case results, and guardrail results."""

import uuid
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from evalyx.db.models import (
    CaseStatus,
    EvaluationCaseResult,
    EvaluationRun,
    GuardrailResult,
    GuardrailStatus,
    RunStatus,
)
from evalyx.db.repositories.errors import NotFoundError


def _utc_now() -> datetime:
    return datetime.now(UTC)


class EvaluationRepository:
    """Async data access for evaluation runs and their results."""

    async def create_run(
        self,
        session: AsyncSession,
        *,
        organization_id: uuid.UUID,
        application_id: uuid.UUID,
        dataset_version_id: uuid.UUID,
        agent_model: str,
        judge_model: str | None = None,
        application_version_id: uuid.UUID | None = None,
        configuration_snapshot: dict | None = None,
    ) -> EvaluationRun:
        run = EvaluationRun(
            organization_id=organization_id,
            application_id=application_id,
            application_version_id=application_version_id,
            dataset_version_id=dataset_version_id,
            agent_model=agent_model,
            judge_model=judge_model,
            configuration_snapshot=configuration_snapshot or {},
            status=RunStatus.PENDING,
        )
        session.add(run)
        await session.commit()
        await session.refresh(run)
        return run

    async def get_run(
        self,
        session: AsyncSession,
        run_id: uuid.UUID,
    ) -> EvaluationRun | None:
        return await session.get(EvaluationRun, run_id)

    async def get_run_in_organization(
        self,
        session: AsyncSession,
        run_id: uuid.UUID,
        *,
        organization_id: uuid.UUID,
    ) -> EvaluationRun | None:
        """Tenant-scoped fetch: other tenants' runs read as missing."""
        result = await session.scalars(
            select(EvaluationRun).filter_by(id=run_id, organization_id=organization_id)
        )
        return result.first()

    async def update_status(
        self,
        session: AsyncSession,
        run: EvaluationRun,
        status: RunStatus,
    ) -> EvaluationRun:
        """Transition a run's status, maintaining lifecycle timestamps."""
        run.status = status
        if status is RunStatus.RUNNING and run.started_at is None:
            run.started_at = _utc_now()
        elif status in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED):
            run.completed_at = _utc_now()
        session.add(run)
        await session.commit()
        await session.refresh(run)
        return run

    async def add_case_result(
        self,
        session: AsyncSession,
        *,
        evaluation_run_id: uuid.UUID,
        test_case_id: uuid.UUID,
        input: dict,
        status: CaseStatus,
        expected_output: dict | None = None,
        actual_output: str | None = None,
        latency_ms: int | None = None,
        error: str | None = None,
        metrics: dict | None = None,
    ) -> EvaluationCaseResult:
        run = await session.get(EvaluationRun, evaluation_run_id)
        if run is None:
            raise NotFoundError(f"Evaluation run {evaluation_run_id} does not exist.")

        case_result = EvaluationCaseResult(
            evaluation_run_id=evaluation_run_id,
            test_case_id=test_case_id,
            input=input,
            expected_output=expected_output,
            actual_output=actual_output,
            status=status,
            latency_ms=latency_ms,
            error=error,
            metrics=metrics,
        )
        session.add(case_result)
        await session.commit()
        await session.refresh(case_result)
        return case_result

    async def get_case_result(
        self,
        session: AsyncSession,
        case_result_id: uuid.UUID,
    ) -> EvaluationCaseResult | None:
        return await session.get(EvaluationCaseResult, case_result_id)

    async def list_runs(
        self,
        session: AsyncSession,
        *,
        limit: int | None = None,
        offset: int = 0,
        application_id: uuid.UUID | None = None,
        organization_id: uuid.UUID | None = None,
    ) -> list[EvaluationRun]:
        """List runs newest first, optionally filtered by application.

        Tenant-aware: ``organization_id`` scopes the listing to one
        organization; the API always supplies it (internal callers such as
        the worker may omit it for a specific run they already verified).

        ``limit``/``offset`` back API pagination; ``None`` (repository
        default) preserves the unpaginated behavior for internal callers.
        Ordering: ``created_at`` descending, ``id`` descending as a stable
        tiebreaker.
        """
        query = select(EvaluationRun).order_by(
            EvaluationRun.created_at.desc(), EvaluationRun.id.desc()
        )
        if application_id is not None:
            query = query.where(EvaluationRun.application_id == application_id)
        if organization_id is not None:
            query = query.where(EvaluationRun.organization_id == organization_id)
        if offset:
            query = query.offset(offset)
        if limit is not None:
            query = query.limit(limit)
        result = await session.execute(query)
        return list(result.scalars().all())

    async def count_case_results_by_status(
        self,
        session: AsyncSession,
        run_id: uuid.UUID,
    ) -> dict[CaseStatus, int]:
        """Case-result counts grouped by status in one query."""
        result = await session.execute(
            select(EvaluationCaseResult.status, func.count())
            .where(EvaluationCaseResult.evaluation_run_id == run_id)
            .group_by(EvaluationCaseResult.status)
        )
        return {status: count for status, count in result.all()}

    async def list_case_results(
        self,
        session: AsyncSession,
        run_id: uuid.UUID,
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[EvaluationCaseResult]:
        """Case results for a run, oldest first (stable page order).

        ``limit``/``offset`` back API pagination; ``None`` (repository
        default) preserves the unpaginated behavior for internal callers.
        """
        query = (
            select(EvaluationCaseResult)
            .where(EvaluationCaseResult.evaluation_run_id == run_id)
            .order_by(EvaluationCaseResult.created_at, EvaluationCaseResult.id)
        )
        if offset:
            query = query.offset(offset)
        if limit is not None:
            query = query.limit(limit)
        result = await session.execute(query)
        return list(result.scalars().all())

    async def add_guardrail_result(
        self,
        session: AsyncSession,
        *,
        evaluation_case_result_id: uuid.UUID,
        name: str,
        passed: bool,
        type: str | None = None,
        score: float | None = None,
        reason: str | None = None,
        metadata: dict | None = None,
        status: GuardrailStatus | None = None,
    ) -> GuardrailResult:
        case_result = await session.get(EvaluationCaseResult, evaluation_case_result_id)
        if case_result is None:
            raise NotFoundError(
                f"Evaluation case result {evaluation_case_result_id} does not exist."
            )

        if status is None:
            # Backward-compatible default: derive from the legacy bool.
            status = GuardrailStatus.PASSED if passed else GuardrailStatus.FAILED

        guardrail_result = GuardrailResult(
            evaluation_case_result_id=evaluation_case_result_id,
            name=name,
            type=type,
            status=status,
            passed=passed,
            score=score,
            reason=reason,
            metadata_=metadata or {},
        )
        session.add(guardrail_result)
        await session.commit()
        await session.refresh(guardrail_result)
        return guardrail_result

    async def update_case_result_status(
        self,
        session: AsyncSession,
        case_result: EvaluationCaseResult,
        status: CaseStatus,
    ) -> EvaluationCaseResult:
        """Transition a case result's status (used by the scoring engine)."""
        case_result.status = status
        session.add(case_result)
        await session.commit()
        await session.refresh(case_result)
        return case_result

    async def list_guardrail_results(
        self,
        session: AsyncSession,
        case_result_id: uuid.UUID,
    ) -> list[GuardrailResult]:
        result = await session.execute(
            select(GuardrailResult)
            .where(GuardrailResult.evaluation_case_result_id == case_result_id)
            .order_by(GuardrailResult.created_at)
        )
        return list(result.scalars().all())

    async def list_guardrail_results_for_run(
        self,
        session: AsyncSession,
        run_id: uuid.UUID,
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[GuardrailResult]:
        """All guardrail results for a run in one query (batched loading).

        Ordered deterministically by creation time; used by the regression
        service and the API to avoid per-case queries. ``limit``/``offset``
        back API pagination; ``None`` (default) preserves unpaginated
        behavior for internal callers.
        """
        query = (
            select(GuardrailResult)
            .join(
                EvaluationCaseResult,
                GuardrailResult.evaluation_case_result_id == EvaluationCaseResult.id,
            )
            .where(EvaluationCaseResult.evaluation_run_id == run_id)
            .order_by(GuardrailResult.created_at, GuardrailResult.id)
        )
        if offset:
            query = query.offset(offset)
        if limit is not None:
            query = query.limit(limit)
        result = await session.execute(query)
        return list(result.scalars().all())

    async def count_guardrail_results_for_run(
        self,
        session: AsyncSession,
        run_id: uuid.UUID,
    ) -> int:
        """Total guardrail-result count for a run (pagination support)."""
        result = await session.execute(
            select(func.count())
            .select_from(GuardrailResult)
            .join(
                EvaluationCaseResult,
                GuardrailResult.evaluation_case_result_id == EvaluationCaseResult.id,
            )
            .where(EvaluationCaseResult.evaluation_run_id == run_id)
        )
        return int(result.scalar_one())
