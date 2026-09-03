"""Durable security/resource audit events + per-organization quota overrides.

``AuditEvent`` (Phase 18) is the tamper-evident-ish operational record of
security-relevant actions: who did what to which tenant resource, when, and
with what result. Writes happen in the same transaction as the action they
describe (so an audit row can never silently go missing for a committed
mutation); denial audits commit immediately before the denial is raised.
Retention is operator-managed (``AUDIT_RETENTION_DAYS`` + documented
cleanup); there is deliberately no API read surface in this phase.

``OrganizationQuotaOverrides`` holds optional per-organization quota
dimensions (``None`` = fall back to the global ``Settings`` defaults).
Quotas stay independent from billing/subscriptions (there are none).
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from evalyx.db.models.base import (
    Base,
    CreatedAtMixin,
    UUIDPrimaryKeyMixin,
)


class AuditEvent(Base, UUIDPrimaryKeyMixin, CreatedAtMixin):
    """One durable, tenant-scoped audit record.

    ``details`` carries small, safe, structured facts only (counts, names,
    non-sensitive configuration): never prompts, responses, credentials,
    tokens, or secrets — enforced by the recorder's sanitizer, with a
    database-level size discipline (callers keep details small).
    """

    __tablename__ = "audit_events"
    __table_args__ = (
        # Retention cleanup and quota-window counting filter on
        # (organization, recency) together.
        Index("ix_audit_events_org_created", "organization_id", "created_at"),
    )

    organization_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("organizations.id", ondelete="SET NULL"),
        index=True,
    )
    clerk_user_id: Mapped[str] = mapped_column(String(255), nullable=False)
    action: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    resource_type: Mapped[str | None] = mapped_column(String(64))
    resource_id: Mapped[str | None] = mapped_column(String(64))
    #: ``allowed`` | ``denied`` — whether the action was admitted.
    result: Mapped[str] = mapped_column(String(16), nullable=False)
    request_id: Mapped[str | None] = mapped_column(String(128))
    details: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)


class OrganizationQuotaOverrides(Base, UUIDPrimaryKeyMixin, CreatedAtMixin):
    """Optional per-organization quota dimensions.

    One row per organization at most; every column is nullable and ``None``
    means "use the global Settings default". The quota service merges the
    override row over settings at admission time.
    """

    __tablename__ = "organization_quota_overrides"
    __table_args__ = (
        UniqueConstraint("organization_id", name="uq_quota_override_organization"),
    )

    organization_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    max_applications: Mapped[int | None] = mapped_column(Integer)
    max_datasets: Mapped[int | None] = mapped_column(Integer)
    max_cases_per_dataset_version: Mapped[int | None] = mapped_column(Integer)
    max_evaluations_per_day: Mapped[int | None] = mapped_column(Integer)
    max_connection_tests_per_day: Mapped[int | None] = mapped_column(Integer)
    max_concurrent_evaluations: Mapped[int | None] = mapped_column(Integer)


__all__ = ["AuditEvent", "OrganizationQuotaOverrides"]
