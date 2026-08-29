"""Typed models for regression comparison (pure data, no I/O).

All rates are **percentages in 0–100** (e.g. ``pass_rate=95.0``). Rate
deltas are **percentage points** (pp): 95% → 88% is a delta of ``-7.0 pp``.
Latency deltas use an **absolute** millisecond value and a **relative
percent** change. Units are explicit in field names (``_pp`` /
``_percent``) and in :class:`ThresholdUnit` — no vague ``threshold=0.05``.

Boundary rule (deterministic, documented): a violation requires the delta
to be *strictly* worse than the threshold; a delta exactly equal to the
threshold is **not** a regression. Values are rounded to 6 decimal places
before comparison so floating-point noise cannot flip decisions.

Undefined values (zero denominators, missing observations) are ``None``,
never silently ``0``.
"""

import enum
import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from evalyx.db.models import CaseStatus, ComparisonResult, GuardrailStatus

#: Algorithm version of the comparison logic. Bump only when the decision
#: semantics change; persisted comparisons record it for reproducibility.
COMPARISON_VERSION = "1"

#: Decimal places kept for rates/deltas before threshold comparison.
PRECISION = 6


def round_metric(value: float) -> float:
    """Deterministic rounding for stored/compared metric values."""
    return round(value, PRECISION)


class RegressionThresholds(BaseModel):
    """Typed threshold policy for regression decisions.

    Evalyx *policy defaults* (chosen to be conservative and explainable —
    deliberately NOT claimed to be industry standards):

    - pass rate may not drop more than 2 percentage points
    - error rate may not rise more than 2 percentage points
    - a guardrail failure rate may not rise more than 2 percentage points
    - average latency may not rise more than 20% (``None`` disables the
      latency criterion entirely)
    """

    model_config = ConfigDict(extra="forbid")

    #: Maximum tolerated pass-rate DROP, percentage points (baseline - current).
    max_pass_rate_drop_pp: float = Field(default=2.0, ge=0)
    #: Maximum tolerated error-rate INCREASE (execution + evaluation errors),
    #: percentage points (current - baseline).
    max_error_rate_increase_pp: float = Field(default=2.0, ge=0)
    #: Maximum tolerated increase of any single guardrail's failure rate,
    #: percentage points.
    max_guardrail_failure_rate_increase_pp: float = Field(default=2.0, ge=0)
    #: Maximum tolerated relative increase of average latency in percent,
    #: or ``None`` to disable latency as a regression criterion.
    max_latency_increase_percent: float | None = Field(default=20.0, ge=0)

    @field_validator("max_latency_increase_percent")
    @classmethod
    def _latency_not_negative(cls, value: float | None) -> float | None:
        if value is not None and value < 0:
            raise ValueError("max_latency_increase_percent must be >= 0 or None.")
        return value


class ThresholdUnit(str, enum.Enum):
    """Explicit unit of a threshold violation's delta."""

    PERCENTAGE_POINTS = "percentage_points"
    PERCENT = "percent"


class ThresholdViolation(BaseModel):
    """One explainable threshold breach.

    ``delta`` is expressed in the worse-is-worse direction (a positive
    number is always bad): a pass-rate drop is reported as ``baseline -
    current``, increases are reported as ``current - baseline``.
    """

    model_config = ConfigDict(extra="forbid")

    #: ``pass_rate`` | ``error_rate`` | ``guardrail_failure_rate`` | ``latency``.
    metric: str
    #: Set only for per-guardrail violations.
    guardrail: str | None = None
    baseline: float | None = None
    current: float | None = None
    delta: float
    threshold: float
    unit: ThresholdUnit
    #: Human-readable, self-contained explanation of the breach.
    detail: str


class LatencyMetrics(BaseModel):
    """Average latency over the run's observed case executions."""

    model_config = ConfigDict(extra="forbid")

    #: ``None`` when no case recorded a latency (never silently 0).
    average_ms: float | None = None
    #: Number of latency observations the average is based on.
    observations: int = 0


class LatencyDelta(BaseModel):
    """Latency change current vs baseline."""

    model_config = ConfigDict(extra="forbid")

    absolute_ms: float | None = None
    #: Relative change vs baseline average, in percent. ``None`` when either
    #: average is undefined or the baseline average is exactly 0 (a relative
    #: change from zero is meaningless, never reported as a finite number).
    percent: float | None = None


class RunMetrics(BaseModel):
    """Metric snapshot for one side of a comparison.

    In a report these are computed over **matched cases** (present with
    results on both sides) so new/removed cases and missing results never
    distort the comparison; the full-run picture is available via
    ``matched_cases`` / ``new_cases`` / ``removed_cases`` /
    ``missing_case_results``.

    ``evaluated_cases`` counts cases with a *semantic* outcome
    (passed + failed). ``execution_error_cases`` are provider-level
    failures (status ``error``); ``evaluation_error_cases`` are cases whose
    evaluation could not complete (status ``executed`` in a completed run —
    guardrail error or missing guardrail execution). The two error kinds
    are kept distinct everywhere.
    """

    model_config = ConfigDict(extra="forbid")

    total_cases: int = 0
    evaluated_cases: int = 0
    passed_cases: int = 0
    failed_cases: int = 0
    execution_error_cases: int = 0
    evaluation_error_cases: int = 0

    #: All rates are percentages 0–100; ``None`` marks an undefined rate
    #: (zero denominator) — explicitly not 0%.
    pass_rate: float | None = None       # passed / evaluated
    failure_rate: float | None = None    # failed / evaluated
    execution_error_rate: float | None = None   # / total_cases
    evaluation_error_rate: float | None = None  # / total_cases
    #: Combined error rate (execution + evaluation errors) / total_cases.
    #: This is the rate the error-rate threshold applies to.
    error_rate: float | None = None
    latency: LatencyMetrics = Field(default_factory=LatencyMetrics)


class MetricDeltas(BaseModel):
    """Metric changes current vs baseline.

    Rate deltas are percentage points (current - baseline). Latency uses
    :class:`LatencyDelta`. ``None`` when either side is undefined.
    """

    model_config = ConfigDict(extra="forbid")

    pass_rate_pp: float | None = None
    failure_rate_pp: float | None = None
    execution_error_rate_pp: float | None = None
    evaluation_error_rate_pp: float | None = None
    error_rate_pp: float | None = None
    latency: LatencyDelta = Field(default_factory=LatencyDelta)


class GuardrailNameMetrics(BaseModel):
    """Aggregated outcome of one guardrail name across a run.

    ``failure_rate`` uses only cases where the guardrail produced a verdict
    (passed + failed). Rows that errored, or are missing for a case, are
    never treated as passes — they are reported via ``errors`` /
    ``missing`` and excluded from the denominator.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    total_evaluations: int = 0   # every row, including errors
    passed: int = 0
    failed: int = 0
    errors: int = 0
    #: Cases in the run with no row for this guardrail name.
    missing: int = 0
    failure_rate: float | None = None  # failed / (passed + failed), None if 0


class GuardrailComparison(BaseModel):
    """Per-guardrail comparison between baseline and current runs."""

    model_config = ConfigDict(extra="forbid")

    name: str
    baseline: GuardrailNameMetrics
    current: GuardrailNameMetrics
    #: Percentage-point change of the failure rate (current - baseline).
    failure_rate_delta_pp: float | None = None


class CaseTransition(str, enum.Enum):
    """Outcome transition of one matched case, baseline → current.

    Full matrix over {passed, failed, error, executed} — stable error
    family members (error/executed) map to ``STABLE_ERROR``; the exact
    statuses remain on the finding. Execution errors and evaluation
    degradation are never collapsed into semantic failure.
    """

    STABLE_PASS = "stable_pass"
    STABLE_FAILURE = "stable_failure"
    STABLE_ERROR = "stable_error"                 # error→error, executed→executed, cross error/executed
    NEWLY_FAILED = "newly_failed"                 # passed→failed (regression candidate)
    NEWLY_ERRORED = "newly_errored"               # passed→error | passed→executed
    FIXED = "fixed"                               # failed→passed (improvement)
    RECOVERED = "recovered"                       # error|executed → passed
    FAILURE_AFTER_ERROR = "failure_after_error"   # error|executed → failed
    ERROR_TRANSITION = "error_transition"         # failed→error|executed


def classify_case_transition(baseline: CaseStatus, current: CaseStatus) -> CaseTransition:
    """Deterministic case transition classification (pure function)."""
    if baseline is current:
        return CaseTransition.STABLE_PASS if baseline is CaseStatus.PASSED else (
            CaseTransition.STABLE_FAILURE
            if baseline is CaseStatus.FAILED
            else CaseTransition.STABLE_ERROR
        )
    if baseline is CaseStatus.PASSED:
        if current is CaseStatus.FAILED:
            return CaseTransition.NEWLY_FAILED
        return CaseTransition.NEWLY_ERRORED  # error or executed
    if baseline is CaseStatus.FAILED:
        if current is CaseStatus.PASSED:
            return CaseTransition.FIXED
        return CaseTransition.ERROR_TRANSITION  # error or executed
    # baseline is error or executed (non-semantic)
    if current is CaseStatus.PASSED:
        return CaseTransition.RECOVERED
    if current is CaseStatus.FAILED:
        return CaseTransition.FAILURE_AFTER_ERROR
    return CaseTransition.STABLE_ERROR  # error↔executed


class GuardrailTransition(str, enum.Enum):
    """Outcome transition of one guardrail on one matched case.

    Missing rows (``None``) are explicit states — absence of evidence is
    never counted as a pass.
    """

    STABLE = "stable"                             # pass→pass
    NEW_FAILURE = "new_failure"                   # pass→fail
    FIXED = "fixed"                               # fail→pass
    PERSISTENT_FAILURE = "persistent_failure"     # fail→fail
    RECOVERED_EVALUATION = "recovered_evaluation" # error→pass
    DEGRADED_EVALUATION = "degraded_evaluation"   # pass|fail → error
    PERSISTENT_ERROR = "persistent_error"         # error→error
    FAILURE_AFTER_ERROR = "failure_after_error"   # error→fail
    MISSING_BASELINE = "missing_baseline"         # absent in baseline, present in current
    MISSING_CURRENT = "missing_current"           # present in baseline, absent in current


def classify_guardrail_transition(
    baseline: GuardrailStatus | None,
    current: GuardrailStatus | None,
) -> GuardrailTransition:
    """Deterministic guardrail transition classification (pure function).

    Raises ``ValueError`` when both sides are missing — callers only ever
    classify guardrail names present on at least one side (the aggregation
    iterates the union of observed names).
    """
    if baseline is None and current is None:
        raise ValueError("Cannot classify a guardrail that is missing on both sides.")
    if baseline is None:
        return GuardrailTransition.MISSING_BASELINE
    if current is None:
        return GuardrailTransition.MISSING_CURRENT
    if baseline is GuardrailStatus.PASSED:
        if current is GuardrailStatus.PASSED:
            return GuardrailTransition.STABLE
        if current is GuardrailStatus.FAILED:
            return GuardrailTransition.NEW_FAILURE
        return GuardrailTransition.DEGRADED_EVALUATION
    if baseline is GuardrailStatus.FAILED:
        if current is GuardrailStatus.PASSED:
            return GuardrailTransition.FIXED
        if current is GuardrailStatus.FAILED:
            return GuardrailTransition.PERSISTENT_FAILURE
        return GuardrailTransition.DEGRADED_EVALUATION
    # baseline errored
    if current is GuardrailStatus.PASSED:
        return GuardrailTransition.RECOVERED_EVALUATION
    if current is GuardrailStatus.FAILED:
        return GuardrailTransition.FAILURE_AFTER_ERROR
    return GuardrailTransition.PERSISTENT_ERROR


class GuardrailResultSnapshot(BaseModel):
    """Minimal guardrail evidence used by the comparison (pure data)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    status: GuardrailStatus


class CaseResultSnapshot(BaseModel):
    """Minimal case evidence used by the comparison (pure data).

    ``identity`` is the stable matching key chosen by the service:
    ``str(test_case_id)`` when both runs share a dataset version, or the
    test-case ``name`` when comparing different versions of one dataset.
    """

    model_config = ConfigDict(extra="forbid")

    identity: str
    case_result_id: uuid.UUID
    test_case_id: uuid.UUID
    name: str
    status: CaseStatus
    latency_ms: int | None = None
    guardrails: list[GuardrailResultSnapshot] = Field(default_factory=list)

    @property
    def guardrail_failures(self) -> list[str]:
        """Names of guardrails that produced a FAILED verdict, sorted."""
        return sorted(g.name for g in self.guardrails if g.status is GuardrailStatus.FAILED)

    def guardrail_status(self, name: str) -> GuardrailStatus | None:
        """Status of one guardrail name, or ``None`` when missing."""
        for guardrail in self.guardrails:
            if guardrail.name == name:
                return guardrail.status
        return None


class MatchedCasePair(BaseModel):
    """One case matched across runs (either side may lack a result)."""

    model_config = ConfigDict(extra="forbid")

    identity: str
    name: str
    test_case_id: uuid.UUID | None
    baseline: CaseResultSnapshot | None = None
    current: CaseResultSnapshot | None = None


class ConfigurationChange(BaseModel):
    """One differing (non-secret) configuration leaf between the runs."""

    model_config = ConfigDict(extra="forbid")

    path: str
    baseline: Any = None
    current: Any = None


class ComparisonRunContext(BaseModel):
    """Per-run context preserved with a comparison (debugging metadata)."""

    model_config = ConfigDict(extra="forbid")

    run_id: uuid.UUID
    status: str
    agent_model: str
    judge_model: str | None = None
    application_id: uuid.UUID
    application_version_id: uuid.UUID | None = None
    dataset_version_id: uuid.UUID
    dataset_version: int | None = None
    dataset_name: str | None = None
    #: Sanitized snapshot (secret-looking keys removed recursively).
    configuration_snapshot: dict = Field(default_factory=dict)


class ComparisonContext(BaseModel):
    """Cross-run context: model/version/config changes.

    Context explains a regression (e.g. a model change); it never
    contributes to the regression decision itself.
    """

    model_config = ConfigDict(extra="forbid")

    baseline: ComparisonRunContext
    current: ComparisonRunContext
    agent_model_changed: bool = False
    judge_model_changed: bool = False
    application_version_changed: bool = False
    #: Leaf-level differences of the sanitized configuration snapshots.
    configuration_changes: list[ConfigurationChange] = Field(default_factory=list)


class CaseFinding(BaseModel):
    """Evidence for one case-level outcome change.

    References existing ``EvaluationCaseResult`` rows instead of
    duplicating prompts/outputs. Sorted deterministically by ``name``.
    """

    model_config = ConfigDict(extra="forbid")

    identity: str
    name: str
    test_case_id: uuid.UUID | None
    baseline_case_result_id: uuid.UUID
    current_case_result_id: uuid.UUID
    baseline_status: str
    current_status: str
    transition: CaseTransition
    baseline_guardrail_failures: list[str]
    current_guardrail_failures: list[str]
    #: Guardrails that went pass → fail (the actionable signal).
    new_guardrail_failures: list[str]
    #: Guardrails that went fail → pass.
    fixed_guardrail_failures: list[str]


class RegressionReport(BaseModel):
    """Typed, self-contained regression report.

    ``comparison_id`` / ``created_at`` are filled when the report is
    persisted (or reloaded from an existing artifact). Findings lists are
    deterministically ordered by case name; guardrail comparisons by name.
    """

    model_config = ConfigDict(extra="forbid")

    comparison_version: str = COMPARISON_VERSION
    comparison_id: uuid.UUID | None = None
    baseline_run_id: uuid.UUID
    current_run_id: uuid.UUID

    result: ComparisonResult
    regression_detected: bool
    #: Why the comparison had no meaningful denominator (NOT_COMPARABLE).
    not_comparable_reason: str | None = None

    baseline: RunMetrics
    current: RunMetrics
    deltas: MetricDeltas
    threshold_violations: list[ThresholdViolation] = Field(default_factory=list)

    matched_cases: int = 0
    newly_failed_cases: list[CaseFinding] = Field(default_factory=list)
    newly_errored_cases: list[CaseFinding] = Field(default_factory=list)
    fixed_cases: list[CaseFinding] = Field(default_factory=list)
    stable_failures: list[CaseFinding] = Field(default_factory=list)
    error_transition_cases: list[CaseFinding] = Field(default_factory=list)
    recovered_cases: list[CaseFinding] = Field(default_factory=list)

    #: Case identities present in the current dataset version but not the
    #: baseline's (no baseline outcome — never counted as regressions).
    new_cases: list[str] = Field(default_factory=list)
    #: Case identities present in the baseline but absent from current.
    removed_cases: list[str] = Field(default_factory=list)
    #: Dataset-version cases with no case result in a run ("baseline"/"current").
    missing_case_results: dict[str, list[str]] = Field(default_factory=dict)

    guardrail_comparisons: list[GuardrailComparison] = Field(default_factory=list)
    context: ComparisonContext

    created_at: datetime | None = None
