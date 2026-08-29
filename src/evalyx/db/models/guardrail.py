"""Guardrail result domain model.

A first-class record of one deterministic or LLM-based guardrail check
performed on one evaluation case result. The guardrail *logic* itself is
Phase 6; this schema only persists its outcomes.
"""

import uuid

from sqlalchemy import Boolean, Float, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from evalyx.db.models.base import Base, CreatedAtMixin, UUIDPrimaryKeyMixin


class GuardrailResult(Base, UUIDPrimaryKeyMixin, CreatedAtMixin):
    """Outcome of a single guardrail check for one evaluation case result.

    Multiple guardrail results may belong to one case result (PII, prompt
    injection, safety, hallucination, instruction following, ...).
    """

    __tablename__ = "guardrail_results"

    evaluation_case_result_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("evaluation_case_results.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    type: Mapped[str | None] = mapped_column(String(64))
    passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    score: Mapped[float | None] = mapped_column(Float)
    reason: Mapped[str | None] = mapped_column(Text)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict, nullable=False)

    evaluation_case_result: Mapped["EvaluationCaseResult"] = relationship(
        back_populates="guardrail_results"
    )
