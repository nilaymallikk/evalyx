"""Declarative base and shared model mixins for the Evalyx domain.

Conventions used across all domain models:
- UUID primary keys (``sqlalchemy.Uuid``), generated client-side.
- Timezone-aware timestamps stored as ``timestamptz`` (UTC in PostgreSQL).
- JSONB only for genuinely flexible data (snapshots, metrics, metadata);
  queryable/relational data uses normal columns.
"""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all Evalyx domain models."""


class UUIDPrimaryKeyMixin:
    """Consistent UUID primary key strategy for domain entities."""

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        default=uuid.uuid4,
    )


class TimestampMixin:
    """created_at / updated_at in UTC (timezone-aware)."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class CreatedAtMixin:
    """created_at only, for immutable result records."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
