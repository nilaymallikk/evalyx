"""generic application connections (Phase 15)

Revision ID: a41f2c7d8e05
Revises: 2d6765a1b770
Create Date: 2026-09-03 00:00:00.000000

Adds generic HTTP application connections:

- ``applications.connection_type`` — how Evalyx invokes the application
  (``"mlgpt"`` reference target backfilled for existing rows; ``"http"``
  for user-registered generic applications).
- ``applications.encrypted_secret`` / ``applications.secret_metadata`` —
  the AES-GCM credential envelope (plaintext never touches the database)
  and its non-secret metadata.
- ``application_versions.connection`` — the immutable, non-secret
  connection configuration (endpoint, method, request mapping, response
  extraction path, auth mode, timeouts) for ``connection_type="http"``
  applications.

The migration is additive; existing development data stays valid (existing
applications become ``connection_type="mlgpt"`` and keep working exactly
as before).
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "a41f2c7d8e05"
down_revision: str | None = "2d6765a1b770"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "applications",
        sa.Column(
            "connection_type",
            sa.String(length=32),
            server_default="mlgpt",
            nullable=False,
        ),
    )
    op.add_column(
        "applications",
        sa.Column("encrypted_secret", sa.Text(), nullable=True),
    )
    op.add_column(
        "applications",
        sa.Column("secret_metadata", postgresql.JSONB(), nullable=True),
    )
    op.add_column(
        "application_versions",
        sa.Column("connection", postgresql.JSONB(), nullable=True),
    )
    # Bounded lookups by connection type (e.g. listing generic applications).
    op.create_index(
        "ix_applications_connection_type",
        "applications",
        ["connection_type"],
    )


def downgrade() -> None:
    op.drop_index("ix_applications_connection_type", table_name="applications")
    op.drop_column("application_versions", "connection")
    op.drop_column("applications", "secret_metadata")
    op.drop_column("applications", "encrypted_secret")
    op.drop_column("applications", "connection_type")