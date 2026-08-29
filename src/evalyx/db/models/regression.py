"""Regression comparison domain model (Phase 8).

A :class:`RegressionComparison` is an immutable analysis artifact answering
"did the current run regress against the baseline run?" It references two
*completed* :class:`EvaluationRun` rows and persists:

- the regression decision (``result`` / ``regression_detected``)
- the exact threshold policy used for the decision (``thresholds`` JSONB)
- the full typed report (``summary`` JSONB): metrics, deltas, threshold
  violations, case-level findings, guardrail comparisons, run context

Immutability rules:

- Baseline/current runs and their results are historical evidence; a
  comparison never modifies them. Both run foreign keys use ``RESTRICT`` so
  a referenced run cannot be deleted while regression evidence depends on
  it.
- The persisted threshold snapshot makes historical reports reproducible
  without relying on current configuration.

Idempotency: the unique ``(baseline_run_id, current_run_id,
policy_fingerprint)`` constraint means comparing the same pair with the same
threshold policy twice does not accumulate duplicate artifacts. Comparing
the same pair with a *different* threshold policy creates a distinct
artifact (each policy produces its own decision; both remain queryable).
"""

import enum
import uuid

from sqlalchemy import (
    CheckConstraint,
    Enum,
    ForeignKey,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from evalyx.db.models.base import Base, CreatedAtMixin, UUIDPrimaryKeyMixin


def _enum_values(enum_cls: type[enum.Enum]) -> list[str]:
    return [member.value for member in enum_cls]


class ComparisonResult(enum.Enum):
    """Top-level outcome of a regression comparison.

    ``NOT_COMPARABLE`` covers comparisons without a meaningful denominator
    (e.g. a run with zero evaluated cases) — explicitly distinct from
    ``NO_REGRESSION``, which means the comparison ran and found nothing.
    """

    REGRESSION_DETECTED = "regression_detected"
    NO_REGRESSION = "no_regression"
    NOT_COMPARABLE = "not_comparable"


class RegressionComparison(Base, UUIDPrimaryKeyMixin, CreatedAtMixin):
    """A persisted, reproducible regression analysis of two completed runs.

    The detailed evidence (metrics, per-case findings, guardrail
    comparisons) lives in the ``summary`` JSONB column with a stable,
    documented schema (:mod:`evalyx.evaluation.regression.models`). Case
    findings reference existing ``EvaluationCaseResult`` ids instead of
    duplicating prompts/outputs.
    """

    __tablename__ = "regression_comparisons"
    __table_args__ = (
        # Idempotency: one artifact per (run pair, threshold policy). The
        # fingerprint covers the threshold snapshot + comparison version.
        UniqueConstraint(
            "baseline_run_id",
            "current_run_id",
            "policy_fingerprint",
            name="uq_regression_comparison_pair_policy",
        ),
        # A run can never meaningfully regress against itself.
        CheckConstraint(
            "baseline_run_id <> current_run_id",
            name="ck_regression_comparison_distinct_runs",
        ),
    )

    baseline_run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("evaluation_runs.id", ondelete="RESTRICT"),
        index=True,
        nullable=False,
    )
    current_run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("evaluation_runs.id", ondelete="RESTRICT"),
        index=True,
        nullable=False,
    )

    #: Top-level decision (``regression_detected`` mirrors it as a bool for
    #: cheap filtering; ``result`` is the source of truth).
    result: Mapped[ComparisonResult] = mapped_column(
        Enum(
            ComparisonResult,
            name="regression_comparison_result",
            values_callable=_enum_values,
        ),
        nullable=False,
    )
    regression_detected: Mapped[bool] = mapped_column(nullable=False)

    #: Algorithm version of the comparison logic ("1" for the MVP).
    comparison_version: Mapped[str] = mapped_column(String(16), nullable=False)
    #: sha256 of the canonical threshold snapshot (idempotency key part).
    policy_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)

    #: Exact threshold snapshot used for the decision (reproducibility).
    thresholds: Mapped[dict] = mapped_column(JSONB, nullable=False)
    #: Full typed report (metrics, deltas, violations, findings, context).
    summary: Mapped[dict] = mapped_column(JSONB, nullable=False)
