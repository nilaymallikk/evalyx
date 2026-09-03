"""Application (AI application/agent under evaluation) domain models."""

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from evalyx.db.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from evalyx.db.models.evaluation import EvaluationRun
    from evalyx.db.models.organization import Organization


class Application(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """An AI application or agent that Evalyx evaluates.

    Tenant-owned: every row belongs to exactly one organization (the Clerk
    organization that created it). All queries are scoped by
    ``organization_id`` at the repository boundary — a tenant can never
    observe another tenant's applications.
    """

    __tablename__ = "applications"

    organization_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("organizations.id", ondelete="RESTRICT"),
        index=True,
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    #: How Evalyx invokes this application: ``"http"`` (generic user
    #: application driven by its version's connection configuration) or
    #: ``"mlgpt"`` (the Evalyx reference demo target; the historical default
    #: so pre-Phase-15 rows and the demo keep working unchanged).
    connection_type: Mapped[str] = mapped_column(
        String(32), default="mlgpt", server_default="mlgpt", nullable=False
    )
    #: Encrypted application credential (Phase 15). The plaintext never
    #: touches the database, logs, API responses, or task arguments — only
    #: this AES-GCM envelope (see ``evalyx.core.encryption``), decrypted
    #: solely at the execution boundary in the worker/test path.
    encrypted_secret: Mapped[str | None] = mapped_column(Text)
    #: Non-secret credential metadata (e.g. auth type, rotation timestamp).
    secret_metadata: Mapped[dict | None] = mapped_column(JSONB)

    organization: Mapped[Organization] = relationship()
    versions: Mapped[list[ApplicationVersion]] = relationship(
        back_populates="application",
        cascade="all, delete-orphan",
    )
    runs: Mapped[list[EvaluationRun]] = relationship(back_populates="application")


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
    #: Immutable non-secret connection configuration (Phase 15, generic
    #: ``connection_type="http"`` applications only): endpoint, method,
    #: request mapping, response extraction path, auth mode, timeouts.
    #: The credential itself is NOT here — it lives encrypted on the
    #: application row. ``None`` for reference/legacy applications.
    connection: Mapped[dict | None] = mapped_column(JSONB)

    application: Mapped[Application] = relationship(back_populates="versions")
