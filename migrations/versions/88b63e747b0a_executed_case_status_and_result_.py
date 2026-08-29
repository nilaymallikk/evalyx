"""executed case status and result uniqueness

Revision ID: 88b63e747b0a
Revises: 8c5e623bc72a
Create Date: 2026-08-29 20:46:53.451772

Adds:
- the 'executed' value to the case_status enum (Phase 5 execution semantics:
  a case produced a provider response but no scoring criterion has been
  applied yet - never treated as semantic "passed").
- a unique constraint enforcing one result per (run, test case), so a run
  cannot accumulate duplicate case results.
"""
from collections.abc import Sequence

from alembic import op

revision: str = "88b63e747b0a"
down_revision: str | None = "8c5e623bc72a"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # PostgreSQL 12+ allows adding enum values inside a transaction as long
    # as the new value is not used in the same transaction.
    op.execute("ALTER TYPE case_status ADD VALUE IF NOT EXISTS 'executed'")
    op.create_unique_constraint(
        "uq_case_result_run_testcase",
        "evaluation_case_results",
        ["evaluation_run_id", "test_case_id"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_case_result_run_testcase", "evaluation_case_results", type_="unique"
    )
    # PostgreSQL cannot drop an enum value; the type must be recreated.
    # This fails if any row still uses status='executed' - such rows must be
    # migrated before downgrading.
    op.execute("ALTER TYPE case_status RENAME TO case_status_old")
    op.execute("CREATE TYPE case_status AS ENUM ('passed', 'failed', 'error')")
    op.execute(
        "ALTER TABLE evaluation_case_results ALTER COLUMN status TYPE case_status "
        "USING status::text::case_status"
    )
    op.execute("DROP TYPE case_status_old")
