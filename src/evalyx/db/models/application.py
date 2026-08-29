"""Application (AI application/agent under evaluation) domain models."""

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from evalyx.db.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from evalyx.db.models.evaluation import EvaluationRun


class Application(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """An AI application or agent that Evalyx evaluates."""

    __tablename__ = "applications"

    name: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)

    versions: Mapped[list["ApplicationVersion"]] = relationship(
        back_populates="application",
        cascade="all, delete-orphan",
    )
    runs: Mapped[list["EvaluationRun"]] = relationship(back_populates="application")


class ApplicationVersion(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """An immutable configuration snapshot of an application.

    Evaluation runs reference a specific application version so it is always
    known which application configuration produced a result. Versions are
    unique per application and are never overwritten (enforced by a database
    unique constraint; the repository raises on duplicates).
    """

    __tablename__ = "application_versions"
    __table_args__ = (
        UniqueConstraint("application_id", "version", name="uq_application_version"),
    )

    application_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("applications.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    #: Non-secret configuration metadata (e.g. prompt template id, params).
    configuration: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)

    application: Mapped["Application"] = relationship(back_populates="versions")
