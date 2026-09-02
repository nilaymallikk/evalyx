"""Evaluation domain models: EvaluationRun, EvaluationCaseResult.

Reproducibility rule: an EvaluationRun persists the agent/judge models and a
JSONB configuration snapshot at run time, so results stay reconstructable
even if application configuration changes later. Case results additionally
snapshot the evaluated input and expected output.
"""

import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from evalyx.db.models.base import (
    Base,
    CreatedAtMixin,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)

if TYPE_CHECKING:
    from evalyx.db.models.application import Application
    from evalyx.db.models.dataset import DatasetVersion, TestCase
    from evalyx.db.models.guardrail import GuardrailResult


class RunStatus(enum.Enum):
    """Lifecycle of an evaluation run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CaseStatus(enum.Enum):
    """Outcome of a single test-case execution."""

    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    #: Execution succeeded but no scoring criterion has been applied yet.
    #: Phase 6 (guardrails/judge) transitions EXECUTED to PASSED/FAILED.
    EXECUTED = "executed"


def _enum_values(enum_cls: type[enum.Enum]) -> list[str]:
    return [member.value for member in enum_cls]


class EvaluationRun(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """The central execution entity: one evaluation of an application
    configuration against one immutable dataset version."""

    __tablename__ = "evaluation_runs"

    application_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("applications.id", ondelete="RESTRICT"),
        index=True,
        nullable=False,
    )
    #: Denormalized tenant column: the organization owning the application
    #: at run time (kept in sync transactionally on run creation; scoping
    #: by it avoids a join through applications).
    organization_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("organizations.id", ondelete="RESTRICT"),
        index=True,
        nullable=False,
    )
    application_version_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("application_versions.id", ondelete="RESTRICT"),
        index=True,
    )
    dataset_version_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("dataset_versions.id", ondelete="RESTRICT"),
        index=True,
        nullable=False,
    )

    #: Model identifiers recorded at run time (reproducibility snapshot).
    agent_model: Mapped[str] = mapped_column(String(255), nullable=False)
    judge_model: Mapped[str | None] = mapped_column(String(255))

    #: Full execution configuration at run time (temperature, max tokens,
    #: guardrail policy ids, ...). Must never contain secrets.
    configuration_snapshot: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)

    status: Mapped[RunStatus] = mapped_column(
        Enum(RunStatus, name="run_status", values_callable=_enum_values),
        default=RunStatus.PENDING,
        nullable=False,
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    application: Mapped[Application] = relationship(back_populates="runs")
    dataset_version: Mapped[DatasetVersion] = relationship()
    case_results: Mapped[list[EvaluationCaseResult]] = relationship(
        back_populates="evaluation_run",
        cascade="all, delete-orphan",
    )


class EvaluationCaseResult(Base, UUIDPrimaryKeyMixin, CreatedAtMixin):
    """The execution result of one test case within one evaluation run."""

    __tablename__ = "evaluation_case_results"

    evaluation_run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("evaluation_runs.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    test_case_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("test_cases.id", ondelete="RESTRICT"),
        index=True,
        nullable=False,
    )

    #: Snapshots of what was actually evaluated (reproducibility/debugging).
    input: Mapped[dict] = mapped_column(JSONB, nullable=False)
    expected_output: Mapped[dict | None] = mapped_column(JSONB)
    actual_output: Mapped[str | None] = mapped_column(Text)

    status: Mapped[CaseStatus] = mapped_column(
        Enum(CaseStatus, name="case_status", values_callable=_enum_values),
        nullable=False,
    )
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    error: Mapped[str | None] = mapped_column(Text)
    #: Flexible scoring/metrics (deterministic checks, judge output, ...).
    metrics: Mapped[dict | None] = mapped_column(JSONB)

    evaluation_run: Mapped[EvaluationRun] = relationship(back_populates="case_results")
    test_case: Mapped[TestCase] = relationship()
    guardrail_results: Mapped[list[GuardrailResult]] = relationship(
        back_populates="evaluation_case_result",
        cascade="all, delete-orphan",
    )
