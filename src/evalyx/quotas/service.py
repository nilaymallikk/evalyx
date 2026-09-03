"""Organization quota enforcement (Phase 18).

Reusable server-side admission for organization-scoped resource usage:
applications, datasets, dataset cases, evaluation submissions (daily and
concurrent), and connection tests. Independent from billing/subscriptions
(there are none).

Race safety: every admission check runs inside the caller's session while
holding a ``SELECT ... FOR UPDATE`` row lock on the organization's row, and
the caller commits the admitted mutation in the same transaction. Two
concurrent admissions for one organization therefore serialize: the second
sees the first's committed (or still-locked) state. Counts are always
recomputed from authoritative PostgreSQL rows — there are no counters to
drift or get stuck.

Evaluation capacity release: the concurrency quota counts runs in a live
state (``pending``/``running``) created within the staleness horizon.
Every terminal transition (completed/failed/cancelled) releases capacity
automatically, so worker failure or duplicate delivery cannot wedge the
quota — a dead worker's run either gets redelivered (and resumes) or is
marked failed, and ancient stuck runs age out of the window. No separate
release step exists to forget.

Denials: the quota service writes a ``quota.exceeded`` audit row and
commits it immediately (via :func:`record_denial_and_commit`) before
raising :class:`QuotaExceededError` — the caller's session holds no other
pending changes at admission time, so committing just the denial is
correct and the denial is never lost to the subsequent rollback.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import structlog
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from evalyx.api.errors import QuotaExceededError
from evalyx.core.config import Settings
from evalyx.core.metrics import metrics
from evalyx.db.models import (
    Application,
    Dataset,
    EvaluationRun,
    Organization,
    OrganizationQuotaOverrides,
    RunStatus,
    TestCase,
)
from evalyx.security.audit import (
    APPLICATION_CONNECTION_TEST,
    record_audit_event,
    record_denial_and_commit,
)

logger = structlog.get_logger(__name__)

#: Run states holding evaluation capacity.
_ACTIVE_RUN_STATUSES: tuple[RunStatus, RunStatus] = (
    RunStatus.PENDING,
    RunStatus.RUNNING,
)

_audit_disabled_warned: bool = False


@dataclass(frozen=True)
class EffectiveLimits:
    """Merged quota dimensions for one organization."""

    max_applications: int
    max_datasets: int
    max_cases_per_dataset_version: int
    max_evaluations_per_day: int
    max_connection_tests_per_day: int
    max_concurrent_evaluations: int


class QuotaService:
    """Race-safe quota admission backed by PostgreSQL row locking."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        settings: Settings,
    ) -> None:
        self._session_factory = session_factory
        self._settings = settings

    @property
    def enabled(self) -> bool:
        return self._settings.quota_enabled

    # -- limits -------------------------------------------------------------

    def _merge_limits(
        self, overrides: OrganizationQuotaOverrides | None
    ) -> EffectiveLimits:
        """Merge one override row (or ``None``) over global defaults.

        Pure function of settings + row — unit-tested directly.
        """
        settings = self._settings
        get = lambda name, default: (
            getattr(overrides, name)
            if overrides is not None and getattr(overrides, name) is not None
            else default
        )
        return EffectiveLimits(
            max_applications=get("max_applications", settings.quota_max_applications),
            max_datasets=get("max_datasets", settings.quota_max_datasets),
            max_cases_per_dataset_version=get(
                "max_cases_per_dataset_version",
                settings.quota_max_cases_per_dataset_version,
            ),
            max_evaluations_per_day=get(
                "max_evaluations_per_day", settings.quota_max_evaluations_per_day
            ),
            max_connection_tests_per_day=get(
                "max_connection_tests_per_day",
                settings.quota_max_connection_tests_per_day,
            ),
            max_concurrent_evaluations=get(
                "max_concurrent_evaluations",
                settings.quota_max_concurrent_evaluations,
            ),
        )

    async def effective_limits(
        self, session: AsyncSession, organization_id: uuid.UUID
    ) -> EffectiveLimits:
        """Merge the per-organization override row over global defaults."""
        overrides = await session.scalar(
            select(OrganizationQuotaOverrides).filter_by(
                organization_id=organization_id
            )
        )
        return self._merge_limits(overrides)

    async def _lock_organization(
        self, session: AsyncSession, organization_id: uuid.UUID
    ) -> None:
        """Serialize admissions for one organization (row-level lock).

        The lock is held until the caller's transaction commits or rolls
        back, which is exactly the race-safety mechanism: concurrent
        admissions queue here and then re-read committed counts.
        """
        await session.execute(
            select(Organization).filter_by(id=organization_id).with_for_update()
        )

    async def _deny(
        self,
        session: AsyncSession,
        *,
        organization_id: uuid.UUID,
        clerk_user_id: str,
        resource: str,
        message: str,
        details: dict | None = None,
    ) -> None:
        """Commit a denial audit row and raise (never returns)."""
        metrics.increment("quota_denied_total", {"resource": resource})
        logger.warning("quota_exceeded", resource=resource)
        await record_denial_and_commit(
            session,
            organization_id=organization_id,
            clerk_user_id=clerk_user_id,
            action="quota.exceeded",
            resource_type="quota",
            resource_id=resource,
            details=details,
        )
        raise QuotaExceededError(resource, message)

    # -- admission checks ----------------------------------------------------

    async def admit_application_create(
        self,
        session: AsyncSession,
        *,
        organization_id: uuid.UUID,
        clerk_user_id: str,
    ) -> EffectiveLimits:
        """Admit one application creation (caller commits the insert)."""
        if not self.enabled:
            return await self.effective_limits(session, organization_id)
        await self._lock_organization(session, organization_id)
        limits = await self.effective_limits(session, organization_id)
        used = await session.scalar(
            select(func.count())
            .select_from(Application)
            .where(Application.organization_id == organization_id)
        )
        used = int(used or 0)
        if used >= limits.max_applications:
            await self._deny(
                session,
                organization_id=organization_id,
                clerk_user_id=clerk_user_id,
                resource="applications",
                message=(
                    f"Organization application quota exceeded ({used}/"
                    f"{limits.max_applications})."
                ),
                details={"used": used, "limit": limits.max_applications},
            )
        return limits

    async def admit_dataset_create(
        self,
        session: AsyncSession,
        *,
        organization_id: uuid.UUID,
        clerk_user_id: str,
    ) -> EffectiveLimits:
        """Admit one dataset creation (caller commits the insert)."""
        if not self.enabled:
            return await self.effective_limits(session, organization_id)
        await self._lock_organization(session, organization_id)
        limits = await self.effective_limits(session, organization_id)
        used = await session.scalar(
            select(func.count())
            .select_from(Dataset)
            .where(Dataset.organization_id == organization_id)
        )
        used = int(used or 0)
        if used >= limits.max_datasets:
            await self._deny(
                session,
                organization_id=organization_id,
                clerk_user_id=clerk_user_id,
                resource="datasets",
                message=(
                    f"Organization dataset quota exceeded ({used}/"
                    f"{limits.max_datasets})."
                ),
                details={"used": used, "limit": limits.max_datasets},
            )
        return limits

    async def admit_case_add(
        self,
        session: AsyncSession,
        *,
        organization_id: uuid.UUID,
        clerk_user_id: str,
        dataset_version_id: uuid.UUID,
    ) -> EffectiveLimits:
        """Admit one test-case append (caller commits the insert)."""
        if not self.enabled:
            return await self.effective_limits(session, organization_id)
        await self._lock_organization(session, organization_id)
        limits = await self.effective_limits(session, organization_id)
        used = await session.scalar(
            select(func.count())
            .select_from(TestCase)
            .where(TestCase.dataset_version_id == dataset_version_id)
        )
        used = int(used or 0)
        if used >= limits.max_cases_per_dataset_version:
            await self._deny(
                session,
                organization_id=organization_id,
                clerk_user_id=clerk_user_id,
                resource="dataset_cases",
                message=(
                    f"Dataset version case quota exceeded ({used}/"
                    f"{limits.max_cases_per_dataset_version})."
                ),
                details={"used": used, "limit": limits.max_cases_per_dataset_version},
            )
        return limits

    async def admit_evaluation(
        self,
        session: AsyncSession,
        *,
        organization_id: uuid.UUID,
        clerk_user_id: str,
    ) -> EffectiveLimits:
        """Admit one evaluation submission (concurrency + daily budget).

        The caller creates the run in the same transaction. Capacity is
        released by terminal run transitions (completed/failed/cancelled)
        or by aging out of the staleness horizon — never by an explicit
        release call that could be forgotten.
        """
        if not self.enabled:
            return await self.effective_limits(session, organization_id)
        await self._lock_organization(session, organization_id)
        limits = await self.effective_limits(session, organization_id)
        active = await self._count_active_runs(session, organization_id)
        if active >= limits.max_concurrent_evaluations:
            await self._deny(
                session,
                organization_id=organization_id,
                clerk_user_id=clerk_user_id,
                resource="concurrent_evaluations",
                message=(
                    f"Organization concurrent evaluation quota exceeded "
                    f"({active}/{limits.max_concurrent_evaluations} running)."
                ),
                details={"used": active, "limit": limits.max_concurrent_evaluations},
            )
        day_start = datetime.now(UTC).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        submitted = await session.scalar(
            select(func.count())
            .select_from(EvaluationRun)
            .where(
                EvaluationRun.organization_id == organization_id,
                EvaluationRun.created_at >= day_start,
            )
        )
        submitted = int(submitted or 0)
        if submitted >= limits.max_evaluations_per_day:
            await self._deny(
                session,
                organization_id=organization_id,
                clerk_user_id=clerk_user_id,
                resource="evaluations_per_day",
                message=(
                    f"Organization daily evaluation quota exceeded ({submitted}/"
                    f"{limits.max_evaluations_per_day})."
                ),
                details={"used": submitted, "limit": limits.max_evaluations_per_day},
            )
        return limits

    async def admit_connection_test(
        self,
        session: AsyncSession,
        *,
        organization_id: uuid.UUID,
        clerk_user_id: str,
    ) -> uuid.UUID | None:
        """Admit one connection test; return the admission audit event id.

        The admission row (``result="allowed"``, ``phase="started"``) is
        committed immediately so the row lock is never held across the
        outbound test call while concurrent admissions still count
        in-flight tests. The caller records the outcome with
        :meth:`record_connection_test_outcome`. Denials commit immediately
        before raising. Returns ``None`` when quotas or auditing are
        disabled (no row is written).
        """
        if not self.enabled:
            return None
        if not self._settings.audit_enabled:
            # The daily connection-test budget is counted from audit rows;
            # without the audit log there is nothing truthful to count, so
            # the check is skipped loudly rather than enforced against zero.
            global _audit_disabled_warned
            if not _audit_disabled_warned:
                _audit_disabled_warned = True
                logger.warning("quota_connection_tests_unaudited")
            return None
        await self._lock_organization(session, organization_id)
        limits = await self.effective_limits(session, organization_id)
        used = await self._count_connection_tests_today(session, organization_id)
        if used >= limits.max_connection_tests_per_day:
            await self._deny(
                session,
                organization_id=organization_id,
                clerk_user_id=clerk_user_id,
                resource="connection_tests_per_day",
                message=(
                    f"Organization daily connection-test quota exceeded "
                    f"({used}/{limits.max_connection_tests_per_day})."
                ),
                details={"used": used, "limit": limits.max_connection_tests_per_day},
            )
        event = await record_audit_event(
            session,
            organization_id=organization_id,
            clerk_user_id=clerk_user_id,
            action=APPLICATION_CONNECTION_TEST,
            resource_type="application",
            resource_id=None,
            result="allowed",
            details={"phase": "started"},
        )
        await session.commit()
        return event.id

    async def record_connection_test_outcome(
        self,
        session: AsyncSession,
        *,
        event_id: uuid.UUID | None,
        application_id: uuid.UUID,
        success: bool,
    ) -> None:
        """Stamp the outcome onto the admission row (caller commits).

        ``event_id`` is ``None`` when quotas/auditing are disabled (nothing
        to update). A missing row (deleted by retention mid-test) is
        ignored — the outcome is informational, the admission was counted.
        """
        if event_id is None or not self._settings.audit_enabled:
            return
        from evalyx.db.models.governance import AuditEvent

        event = await session.get(AuditEvent, event_id)
        if event is None:
            return
        event.resource_id = str(application_id)[:64]
        event.details = {**(event.details or {}), "success": success}

    # -- override management -------------------------------------------------

    async def get_overrides(
        self, session: AsyncSession, organization_id: uuid.UUID
    ) -> OrganizationQuotaOverrides | None:
        """The organization's override row, if any (``None`` = all defaults)."""
        return await session.scalar(
            select(OrganizationQuotaOverrides).filter_by(
                organization_id=organization_id
            )
        )

    async def set_overrides(
        self,
        session: AsyncSession,
        organization_id: uuid.UUID,
        **dimensions: int | None,
    ) -> OrganizationQuotaOverrides:
        """Create-or-replace the override row (operator/future admin use).

        Only known dimension names are accepted; values must be positive or
        ``None`` (clear back to the default). Commits in the caller's
        session.
        """
        allowed = {
            "max_applications",
            "max_datasets",
            "max_cases_per_dataset_version",
            "max_evaluations_per_day",
            "max_connection_tests_per_day",
            "max_concurrent_evaluations",
        }
        unknown = set(dimensions) - allowed
        if unknown:
            raise ValueError(f"Unknown quota dimensions: {sorted(unknown)}.")
        for name, value in dimensions.items():
            if value is not None and (not isinstance(value, int) or value < 1):
                raise ValueError(f"Quota override {name} must be a positive int or None.")
        overrides = await self.get_overrides(session, organization_id)
        if overrides is None:
            overrides = OrganizationQuotaOverrides(organization_id=organization_id)
            session.add(overrides)
        for name, value in dimensions.items():
            setattr(overrides, name, value)
        await session.commit()
        await session.refresh(overrides)
        return overrides

    # -- counting helpers -----------------------------------------------------

    async def _count_active_runs(
        self, session: AsyncSession, organization_id: uuid.UUID
    ) -> int:
        """Live-state runs inside the staleness horizon (holds capacity)."""
        cutoff = datetime.now(UTC) - timedelta(
            seconds=self._settings.quota_stale_run_seconds
        )
        count = await session.scalar(
            select(func.count())
            .select_from(EvaluationRun)
            .where(
                EvaluationRun.organization_id == organization_id,
                EvaluationRun.status.in_(_ACTIVE_RUN_STATUSES),
                EvaluationRun.created_at >= cutoff,
            )
        )
        return int(count or 0)

    async def _count_connection_tests_today(
        self, session: AsyncSession, organization_id: uuid.UUID
    ) -> int:
        """Admitted connection tests since UTC midnight (audit-sourced)."""
        from evalyx.db.models.governance import AuditEvent

        day_start = datetime.now(UTC).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        count = await session.scalar(
            select(func.count())
            .select_from(AuditEvent)
            .where(
                AuditEvent.organization_id == organization_id,
                AuditEvent.action == APPLICATION_CONNECTION_TEST,
                AuditEvent.result == "allowed",
                AuditEvent.created_at >= day_start,
            )
        )
        return int(count or 0)


__all__ = ["EffectiveLimits", "QuotaService"]
