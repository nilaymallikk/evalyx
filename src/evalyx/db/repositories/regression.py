"""Repository for regression comparison artifacts."""

import uuid

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from evalyx.db.models import ComparisonResult, RegressionComparison


class RegressionRepository:
    """Async data access for persisted regression comparisons."""

    async def create(
        self,
        session: AsyncSession,
        *,
        baseline_run_id: uuid.UUID,
        current_run_id: uuid.UUID,
        result: ComparisonResult,
        regression_detected: bool,
        comparison_version: str,
        policy_fingerprint: str,
        thresholds: dict,
        summary: dict,
    ) -> RegressionComparison:
        comparison = RegressionComparison(
            baseline_run_id=baseline_run_id,
            current_run_id=current_run_id,
            result=result,
            regression_detected=regression_detected,
            comparison_version=comparison_version,
            policy_fingerprint=policy_fingerprint,
            thresholds=thresholds,
            summary=summary,
        )
        session.add(comparison)
        await session.commit()
        await session.refresh(comparison)
        return comparison

    async def get(
        self,
        session: AsyncSession,
        comparison_id: uuid.UUID,
    ) -> RegressionComparison | None:
        return await session.get(RegressionComparison, comparison_id)

    async def get_by_pair_and_policy(
        self,
        session: AsyncSession,
        *,
        baseline_run_id: uuid.UUID,
        current_run_id: uuid.UUID,
        policy_fingerprint: str,
    ) -> RegressionComparison | None:
        """The existing artifact for this run pair + threshold policy, if any.

        Backing for deterministic idempotency: repeated comparisons of the
        same pair under the same policy return the original artifact.
        """
        result = await session.execute(
            select(RegressionComparison).where(
                RegressionComparison.baseline_run_id == baseline_run_id,
                RegressionComparison.current_run_id == current_run_id,
                RegressionComparison.policy_fingerprint == policy_fingerprint,
            )
        )
        return result.scalars().first()

    async def list_for_run(
        self,
        session: AsyncSession,
        run_id: uuid.UUID,
    ) -> list[RegressionComparison]:
        """Comparisons where the run is the baseline OR the current side."""
        result = await session.execute(
            select(RegressionComparison)
            .where(
                or_(
                    RegressionComparison.baseline_run_id == run_id,
                    RegressionComparison.current_run_id == run_id,
                )
            )
            .order_by(RegressionComparison.created_at.desc())
        )
        return list(result.scalars().all())
