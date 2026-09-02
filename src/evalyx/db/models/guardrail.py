"""Guardrail result domain model.

A first-class record of one deterministic or LLM-based guardrail check
performed on one evaluation case result. The guardrail *logic* is Phase 6;
this schema persists its outcomes.

``status`` (guardrail_status enum) is the source of truth:
- ``passed``  — the check ran and the response satisfied it.
- ``failed``  — the check ran and detected a policy violation.
- ``error``   — the check could not execute (judge timeout, invalid judge
  output). This is distinct from a real failure: a guardrail that cannot
  execute is not evidence the response violated the policy.

The legacy ``passed`` bool is kept for backward compatibility and is always
kept in sync (``passed == (status is PASSED)``).
"""

import enum
import uuid
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, Enum, Float, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from evalyx.db.models.base import Base, CreatedAtMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from evalyx.db.models.evaluation import EvaluationCaseResult


class GuardrailStatus(enum.Enum):
    """Outcome of a single guardrail check."""

    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"


def _enum_values(enum_cls: type[enum.Enum]) -> list[str]:
    return [member.value for member in enum_cls]


class GuardrailResult(Base, UUIDPrimaryKeyMixin, CreatedAtMixin):
    """Outcome of a single guardrail check for one evaluation case result.

    Multiple guardrail results may belong to one case result (PII, prompt
    injection, safety, hallucination, instruction following, ...). A result
    is unique per (case, guardrail name) — database-enforced.
    """

    __tablename__ = "guardrail_results"
    __table_args__ = (
        # One result per guardrail per case; repeated scoring must not
        # accumulate duplicates.
        UniqueConstraint(
            "evaluation_case_result_id",
            "name",
            name="uq_guardrail_result_case_name",
        ),
    )

    evaluation_case_result_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("evaluation_case_results.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    type: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[GuardrailStatus] = mapped_column(
        Enum(
            GuardrailStatus,
            name="guardrail_status",
            values_callable=_enum_values,
        ),
        nullable=False,
    )
    #: Source of truth is ``status``; this bool is kept in sync.
    passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    score: Mapped[float | None] = mapped_column(Float)
    reason: Mapped[str | None] = mapped_column(Text)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict, nullable=False)

    evaluation_case_result: Mapped[EvaluationCaseResult] = relationship(
        back_populates="guardrail_results"
    )
