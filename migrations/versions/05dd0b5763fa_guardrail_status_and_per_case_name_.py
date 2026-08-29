"""guardrail status and per-case name uniqueness

Revision ID: 05dd0b5763fa
Revises: 88b63e747b0a
Create Date: 2026-08-29 22:23:21.182982

Adds:
- a guardrail_status enum ('passed', 'failed', 'error') so that a guardrail
  which could not execute (judge timeout, invalid judge output) is recorded
  distinctly from a real policy failure (passed=False). The existing bool
  `passed` column remains for backward compatibility; `status` is the source
  of truth. Existing rows are backfilled from `passed`.
- a unique constraint (evaluation_case_result_id, name) so a case never
  accumulates duplicate guardrail results during repeated scoring.
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "05dd0b5763fa"
down_revision: str | None = "88b63e747b0a"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

GUARDRAIL_STATUS = ("passed", "failed", "error")


def upgrade() -> None:
    op.execute(
        "CREATE TYPE guardrail_status AS ENUM "
        f"({', '.join(repr(v) for v in GUARDRAIL_STATUS)})"
    )
    op.add_column(
        "guardrail_results",
        sa.Column(
            "status",
            sa.Enum(*GUARDRAIL_STATUS, name="guardrail_status"),
            nullable=True,
        ),
    )
    op.execute(
        "UPDATE guardrail_results SET status = "
        "CASE WHEN passed THEN 'passed'::guardrail_status "
        "ELSE 'failed'::guardrail_status END WHERE status IS NULL"
    )
    op.alter_column("guardrail_results", "status", nullable=False)
    op.create_unique_constraint(
        "uq_guardrail_result_case_name",
        "guardrail_results",
        ["evaluation_case_result_id", "name"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_guardrail_result_case_name", "guardrail_results", type_="unique"
    )
    op.drop_column("guardrail_results", "status")
    op.execute("DROP TYPE IF EXISTS guardrail_status")
