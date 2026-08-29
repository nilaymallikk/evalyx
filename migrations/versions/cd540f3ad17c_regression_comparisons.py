"""regression comparisons

Revision ID: cd540f3ad17c
Revises: 05dd0b5763fa
Create Date: 2026-08-30 03:30:55.112595

Adds the Phase 8 regression-comparison artifact:

- a ``regression_comparison_result`` enum ('regression_detected',
  'no_regression', 'not_comparable') for the top-level decision;
  ``not_comparable`` covers comparisons without a meaningful denominator
  (e.g. zero evaluated cases) — distinct from a clean ``no_regression``.
- the ``regression_comparisons`` table: RESTRICT foreign keys to the
  baseline and current evaluation runs (referenced runs cannot be deleted
  while regression evidence depends on them), the persisted threshold
  snapshot (``thresholds`` JSONB) and full typed report (``summary``
  JSONB) for reproducibility, and idempotency via the unique
  (baseline_run_id, current_run_id, policy_fingerprint) constraint. A
  CHECK constraint forbids comparing a run with itself.
- justified indexes only: both run foreign keys and created_at.
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "cd540f3ad17c"
down_revision: str | None = "05dd0b5763fa"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

COMPARISON_RESULT = ("regression_detected", "no_regression", "not_comparable")


def upgrade() -> None:
    # sa.Enum creates the regression_comparison_result type implicitly with
    # the table (a manual CREATE TYPE here would collide with it).
    op.create_table(
        "regression_comparisons",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "baseline_run_id",
            sa.Uuid(),
            sa.ForeignKey("evaluation_runs.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "current_run_id",
            sa.Uuid(),
            sa.ForeignKey("evaluation_runs.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "result",
            sa.Enum(*COMPARISON_RESULT, name="regression_comparison_result"),
            nullable=False,
        ),
        sa.Column("regression_detected", sa.Boolean(), nullable=False),
        sa.Column("comparison_version", sa.String(length=16), nullable=False),
        sa.Column("policy_fingerprint", sa.String(length=64), nullable=False),
        sa.Column("thresholds", JSONB(), nullable=False),
        sa.Column("summary", JSONB(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "baseline_run_id <> current_run_id",
            name="ck_regression_comparison_distinct_runs",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "baseline_run_id",
            "current_run_id",
            "policy_fingerprint",
            name="uq_regression_comparison_pair_policy",
        ),
    )
    op.create_index(
        op.f("ix_regression_comparisons_baseline_run_id"),
        "regression_comparisons",
        ["baseline_run_id"],
    )
    op.create_index(
        op.f("ix_regression_comparisons_current_run_id"),
        "regression_comparisons",
        ["current_run_id"],
    )
    op.create_index(
        op.f("ix_regression_comparisons_created_at"),
        "regression_comparisons",
        ["created_at"],
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_regression_comparisons_created_at"), table_name="regression_comparisons"
    )
    op.drop_index(
        op.f("ix_regression_comparisons_current_run_id"), table_name="regression_comparisons"
    )
    op.drop_index(
        op.f("ix_regression_comparisons_baseline_run_id"), table_name="regression_comparisons"
    )
    op.drop_table("regression_comparisons")
    op.execute("DROP TYPE IF EXISTS regression_comparison_result")
