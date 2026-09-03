"""audit events and per-organization quota overrides (Phase 18)

Revision ID: b82e41f9c3d7
Revises: a41f2c7d8e05
Create Date: 2026-09-04 00:00:00.000000

Adds production-hardening governance tables:

- ``audit_events`` — durable, tenant-scoped security/resource audit
  records (actor, organization, action, resource, result, request id,
  timestamp, safe details). ``organization_id`` is nullable with
  ``SET NULL`` so audit history survives organization deletion; action
  and (organization, created_at) are indexed for retention cleanup and
  quota-window counting.
- ``organization_quota_overrides`` — at most one row per organization
  with nullable per-dimension quota overrides (``None`` = global
  Settings default).

The migration is purely additive; downgrade drops both tables.
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "b82e41f9c3d7"
down_revision: str | None = "a41f2c7d8e05"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "audit_events",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "organization_id",
            sa.Uuid(),
            sa.ForeignKey("organizations.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("clerk_user_id", sa.String(length=255), nullable=False),
        sa.Column("action", sa.String(length=64), nullable=False),
        sa.Column("resource_type", sa.String(length=64), nullable=True),
        sa.Column("resource_id", sa.String(length=64), nullable=True),
        sa.Column("result", sa.String(length=16), nullable=False),
        sa.Column("request_id", sa.String(length=128), nullable=True),
        sa.Column(
            "details",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_audit_events_action", "audit_events", ["action"])
    op.create_index(
        "ix_audit_events_organization_id", "audit_events", ["organization_id"]
    )
    op.create_index(
        "ix_audit_events_org_created",
        "audit_events",
        ["organization_id", "created_at"],
    )

    op.create_table(
        "organization_quota_overrides",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "organization_id",
            sa.Uuid(),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("max_applications", sa.Integer(), nullable=True),
        sa.Column("max_datasets", sa.Integer(), nullable=True),
        sa.Column("max_cases_per_dataset_version", sa.Integer(), nullable=True),
        sa.Column("max_evaluations_per_day", sa.Integer(), nullable=True),
        sa.Column("max_connection_tests_per_day", sa.Integer(), nullable=True),
        sa.Column("max_concurrent_evaluations", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "organization_id", name="uq_quota_override_organization"
        ),
    )


def downgrade() -> None:
    op.drop_table("organization_quota_overrides")
    op.drop_index("ix_audit_events_org_created", table_name="audit_events")
    op.drop_index("ix_audit_events_organization_id", table_name="audit_events")
    op.drop_index("ix_audit_events_action", table_name="audit_events")
    op.drop_table("audit_events")
