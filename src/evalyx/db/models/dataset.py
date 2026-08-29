"""Dataset domain models: Dataset, DatasetVersion, TestCase.

A Dataset is the logical collection; its DatasetVersions contain the
immutable test-case snapshots that evaluation runs actually execute against.
"""

import uuid

from sqlalchemy import ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from evalyx.db.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class Dataset(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """A logical collection of evaluation test cases."""

    __tablename__ = "datasets"

    name: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)

    versions: Mapped[list["DatasetVersion"]] = relationship(
        back_populates="dataset",
        cascade="all, delete-orphan",
    )


class DatasetVersion(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """An immutable snapshot of a dataset's test cases.

    Version numbers are integers unique per dataset (v1, v2, ...). Existing
    versions must never be silently changed: create a new version instead.
    The unique constraint below is the database-level protection; the
    repository raises :class:`DuplicateVersionError` on attempts to reuse a
    version number, and no update path is exposed for version content.
    """

    __tablename__ = "dataset_versions"
    __table_args__ = (
        UniqueConstraint("dataset_id", "version", name="uq_dataset_version"),
    )

    dataset_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)

    dataset: Mapped["Dataset"] = relationship(back_populates="versions")
    test_cases: Mapped[list["TestCase"]] = relationship(
        back_populates="dataset_version",
        cascade="all, delete-orphan",
    )


class TestCase(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """A single evaluation case belonging to one immutable dataset version."""

    __tablename__ = "test_cases"

    dataset_version_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("dataset_versions.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    #: Structured input payload for the application under test.
    input: Mapped[dict] = mapped_column(JSONB, nullable=False)
    #: Expected output/behavior used by future checks; kept flexible.
    expected_output: Mapped[dict | None] = mapped_column(JSONB)
    context: Mapped[dict | None] = mapped_column(JSONB)
    #: ``metadata`` is reserved by SQLAlchemy, hence the trailing underscore.
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict, nullable=False)

    dataset_version: Mapped["DatasetVersion"] = relationship(back_populates="test_cases")
