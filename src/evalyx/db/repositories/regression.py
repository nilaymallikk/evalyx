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
        organization_id: uuid.UUID,
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
            organization_id=organization_id,
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

    async def get_in_organization(
        self,
        session: AsyncSession,
        comparison_id: uuid.UUID,
        *,
        organization_id: uuid.UUID,
    ) -> RegressionComparison | None:
        """Tenant-scoped fetch: other tenants' comparisons read as missing."""
        result = await session.scalars(
            select(RegressionComparison).filter_by(
                id=comparison_id, organization_id=organization_id
            )
        )
        return result.first()

    async def get_by_pair_and_policy(
        self,
        session: AsyncSession,
        *,
        baseline_run_id: uuid.UUID,
        current_run_id: uuid.UUID,
        policy_fingerprint: str,
        organization_id: uuid.UUID,
    ) -> RegressionComparison | None:
        """The existing artifact for this run pair + threshold policy, if any.

        Backing for deterministic idempotency: repeated comparisons of the
        same pair under the same policy return the original artifact.
        """
        result = await session.scalars(
            select(RegressionComparison).filter_by(
                baseline_run_id=baseline_run_id,
                current_run_id=current_run_id,
                policy_fingerprint=policy_fingerprint,
                organization_id=organization_id,
            )
        )
        return result.first()

    async def list_for_run(
        self,
        session: AsyncSession,
        run_id: uuid.UUID,
        *,
        organization_id: uuid.UUID,
    ) -> list[RegressionComparison]:
        """Comparisons of one tenant where the run is either side.

        The organization filter is applied to the artifact itself (both runs
        of a tenant-owned comparison belong to that tenant), so a tenant can
        never list another tenant's comparisons even by guessing run ids.
        """
        result = await session.scalars(
            select(RegressionComparison)
            .filter_by(organization_id=organization_id)
            .where(
                or_(
                    RegressionComparison.baseline_run_id == run_id,
                    RegressionComparison.current_run_id == run_id,
                )
            )
            .order_by(RegressionComparison.created_at.desc())
        )
        return list(result.all())
