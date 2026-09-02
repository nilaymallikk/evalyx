"""Organization (tenant) domain model (Phase 14 multi-tenancy).

Clerk owns identity, organizations, memberships, and roles; Evalyx keeps a
local :class:`Organization` row as the *tenant boundary for domain data*:

- ``clerk_organization_id`` is the external tenant identity (unique here —
  one Clerk organization maps to at most one Evalyx organization).
- All tenant-owned resources (applications, datasets, evaluation runs,
  regression comparisons) carry ``organization_id`` and are scoped by it at
  the repository boundary.

Evalyx deliberately does **not** duplicate Clerk's membership/role system:
roles come from the verified Clerk session token, not from this table.
"""

import uuid

from sqlalchemy import ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from evalyx.db.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class Organization(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """One tenant: a mirror of a Clerk organization for domain scoping."""

    __tablename__ = "organizations"
    __table_args__ = (
        # Exactly one Evalyx organization per Clerk organization.
        UniqueConstraint("clerk_organization_id", name="uq_organization_clerk_id"),
    )

    clerk_organization_id: Mapped[str] = mapped_column(
        String(255), index=True, nullable=False
    )
    #: Display metadata mirrored at bootstrap time (never authoritative).
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    created_by_clerk_user_id: Mapped[str | None] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text)


class OrganizationMembershipAudit(Base, UUIDPrimaryKeyMixin):
    """Audit trail of workspace bootstrap events (who created/mapped it).

    Membership truth lives in Clerk; this table records only Evalyx-side
    provisioning events (workspace bootstrap), not membership changes.
    """

    __tablename__ = "organization_membership_audit"

    organization_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    clerk_user_id: Mapped[str] = mapped_column(String(255), nullable=False)
    event: Mapped[str] = mapped_column(String(64), nullable=False)


__all__ = ["Organization", "OrganizationMembershipAudit"]
