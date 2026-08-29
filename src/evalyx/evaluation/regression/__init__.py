"""Regression testing & baseline comparison (Phase 8).

Deterministic, threshold-based regression detection over persisted
evaluation runs — no LLM, no network, no Celery, no statistical
significance testing (explicit MVP scope; see the README).

Public API:

- :class:`RegressionService` — compare two completed runs, persist the
  artifact idempotently, reload reports
- :class:`RegressionThresholds` — typed threshold policy (percentage
  points for rates, percent for latency)
- :class:`RegressionReport` and friends — typed report structures
- :func:`comparison.compare` and helpers — pure comparison functions

Usage::

    service = RegressionService(db.session_factory)
    report = await service.compare_runs(baseline_run_id, current_run_id)
    report.regression_detected   # bool
    report.newly_failed_cases    # case-level evidence
"""

from evalyx.evaluation.regression.comparison import (
    ComparisonInput,
    calculate_guardrail_metrics,
    calculate_guardrail_name_metrics,
    calculate_latency_metrics,
    calculate_metric_deltas,
    calculate_run_metrics,
    compare,
    diff_configurations,
    evaluate_thresholds,
    pair_case_results,
    policy_fingerprint,
    sanitize_configuration,
)
from evalyx.evaluation.regression.models import (
    COMPARISON_VERSION,
    CaseFinding,
    CaseResultSnapshot,
    CaseTransition,
    ComparisonContext,
    ComparisonResult,
    ComparisonRunContext,
    ConfigurationChange,
    GuardrailComparison,
    GuardrailNameMetrics,
    GuardrailResultSnapshot,
    GuardrailTransition,
    LatencyDelta,
    LatencyMetrics,
    MatchedCasePair,
    MetricDeltas,
    RegressionReport,
    RegressionThresholds,
    RunMetrics,
    ThresholdUnit,
    ThresholdViolation,
    classify_case_transition,
    classify_guardrail_transition,
)
from evalyx.evaluation.regression.service import (
    RegressionError,
    RegressionService,
    RegressionValidationError,
    comparison_to_report,
)

__all__ = [
    "COMPARISON_VERSION",
    "CaseFinding",
    "CaseResultSnapshot",
    "CaseTransition",
    "ComparisonContext",
    "ComparisonInput",
    "ComparisonResult",
    "ComparisonRunContext",
    "ConfigurationChange",
    "GuardrailComparison",
    "GuardrailNameMetrics",
    "GuardrailResultSnapshot",
    "GuardrailTransition",
    "LatencyDelta",
    "LatencyMetrics",
    "MatchedCasePair",
    "MetricDeltas",
    "RegressionError",
    "RegressionReport",
    "RegressionService",
    "RegressionThresholds",
    "RegressionValidationError",
    "RunMetrics",
    "ThresholdUnit",
    "ThresholdViolation",
    "calculate_guardrail_metrics",
    "calculate_guardrail_name_metrics",
    "calculate_latency_metrics",
    "calculate_metric_deltas",
    "calculate_run_metrics",
    "classify_case_transition",
    "classify_guardrail_transition",
    "compare",
    "comparison_to_report",
    "diff_configurations",
    "evaluate_thresholds",
    "pair_case_results",
    "policy_fingerprint",
    "sanitize_configuration",
]
