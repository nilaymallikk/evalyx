"""Unit tests for the typed threshold model and boundary behavior.

Boundary rule under test (documented in the comparison engine): a delta
exactly equal to the threshold is NOT a regression; only a strictly worse
delta violates. No DB/network access.
"""

import uuid

import pytest
from pydantic import ValidationError

from evalyx.db.models import CaseStatus
from evalyx.evaluation.regression import (
    CaseResultSnapshot,
    ComparisonResult,
    LatencyMetrics,
    RegressionThresholds,
    RunMetrics,
    compare,
    evaluate_thresholds,
    policy_fingerprint,
)
from evalyx.evaluation.regression.comparison import ComparisonInput
from evalyx.evaluation.regression.models import (
    ComparisonContext,
    ComparisonRunContext,
    GuardrailComparison,
    GuardrailNameMetrics,
    LatencyDelta,
    MetricDeltas,
    ThresholdUnit,
)


def metrics_with_rates(
    *,
    pass_rate: float | None = None,
    error_rate: float | None = None,
    evaluated: int = 1,
    total: int = 1,
    latency_avg: float | None = None,
) -> RunMetrics:
    return RunMetrics(
        evaluated_cases=evaluated,
        total_cases=total,
        pass_rate=pass_rate,
        error_rate=error_rate,
        latency=LatencyMetrics(average_ms=latency_avg, observations=1 if latency_avg else 0),
    )


def guardrail_comparison(name: str, baseline_rate: float | None, current_rate: float | None):
    return GuardrailComparison(
        name=name,
        baseline=GuardrailNameMetrics(name=name, failure_rate=baseline_rate),
        current=GuardrailNameMetrics(name=name, failure_rate=current_rate),
        failure_rate_delta_pp=(
            round(current_rate - baseline_rate, 6)
            if baseline_rate is not None and current_rate is not None
            else None
        ),
    )


# -- pass rate -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("drop", "violates"), [(1.999999, False), (2.0, False), (2.000001, True), (5.0, True)]
)
def test_pass_rate_drop_boundary(drop: float, violates: bool):
    baseline = metrics_with_rates(pass_rate=95.0, evaluated=1)
    current = metrics_with_rates(pass_rate=95.0 - drop, evaluated=1)
    deltas = MetricDeltas(pass_rate_pp=-drop)
    violations = evaluate_thresholds(baseline, current, deltas, [], RegressionThresholds())
    assert ({v.metric for v in violations} == {"pass_rate"}) is violates


# -- error rate ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("increase", "violates"), [(1.999999, False), (2.0, False), (2.000001, True), (10.0, True)]
)
def test_error_rate_increase_boundary(increase: float, violates: bool):
    baseline = metrics_with_rates(error_rate=1.0, total=1)
    current = metrics_with_rates(error_rate=1.0 + increase, total=1)
    deltas = MetricDeltas(error_rate_pp=increase)
    violations = evaluate_thresholds(baseline, current, deltas, [], RegressionThresholds())
    assert ({v.metric for v in violations} == {"error_rate"}) is violates


# -- guardrail failure rate ----------------------------------------------------------


@pytest.mark.parametrize(
    ("increase", "violates"), [(1.999999, False), (2.0, False), (2.000001, True), (5.0, True)]
)
def test_guardrail_failure_rate_boundary(increase: float, violates: bool):
    baseline = metrics_with_rates(pass_rate=100.0, evaluated=1)
    current = metrics_with_rates(pass_rate=100.0, evaluated=1)
    deltas = MetricDeltas()
    comparisons = [guardrail_comparison("pii", 2.0, 2.0 + increase)]
    violations = evaluate_thresholds(baseline, current, deltas, comparisons, RegressionThresholds())
    assert ({v.guardrail for v in violations} == {"pii"}) is violates


def test_guardrail_violation_names_the_guardrail():
    comparisons = [
        guardrail_comparison("pii", 2.0, 7.0),
        guardrail_comparison("safety", 2.0, 2.0),  # unchanged
    ]
    violations = evaluate_thresholds(
        metrics_with_rates(pass_rate=100.0, evaluated=1),
        metrics_with_rates(pass_rate=100.0, evaluated=1),
        MetricDeltas(),
        comparisons,
        RegressionThresholds(),
    )
    assert [v.guardrail for v in violations] == ["pii"]
    violation = violations[0]
    assert violation.metric == "guardrail_failure_rate"
    assert violation.unit is ThresholdUnit.PERCENTAGE_POINTS
    assert violation.delta == 5.0
    assert "pii" in violation.detail


def test_guardrail_delta_none_produces_no_violation():
    comparisons = [guardrail_comparison("injection", None, 50.0)]  # absent in baseline
    violations = evaluate_thresholds(
        metrics_with_rates(pass_rate=100.0, evaluated=1),
        metrics_with_rates(pass_rate=100.0, evaluated=1),
        MetricDeltas(),
        comparisons,
        RegressionThresholds(),
    )
    assert violations == []


# -- latency -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("increase", "violates"), [(19.999999, False), (20.0, False), (20.000001, True), (33.0, True)]
)
def test_latency_increase_boundary(increase: float, violates: bool):
    baseline = metrics_with_rates(pass_rate=100.0, evaluated=1, latency_avg=500.0)
    current = metrics_with_rates(pass_rate=100.0, evaluated=1, latency_avg=500.0 * (1 + increase / 100))
    deltas = MetricDeltas(latency=LatencyDelta(absolute_ms=None, percent=increase))
    violations = evaluate_thresholds(baseline, current, deltas, [], RegressionThresholds())
    assert ({v.metric for v in violations} == {"latency"}) is violates


def test_latency_threshold_can_be_disabled():
    thresholds = RegressionThresholds(max_latency_increase_percent=None)
    baseline = metrics_with_rates(pass_rate=100.0, evaluated=1, latency_avg=100.0)
    current = metrics_with_rates(pass_rate=100.0, evaluated=1, latency_avg=1000.0)
    deltas = MetricDeltas(latency=LatencyDelta(absolute_ms=900.0, percent=900.0))
    violations = evaluate_thresholds(baseline, current, deltas, [], thresholds)
    assert all(v.metric != "latency" for v in violations)


def test_latency_violation_uses_percent_unit():
    baseline = metrics_with_rates(pass_rate=100.0, evaluated=1, latency_avg=500.0)
    current = metrics_with_rates(pass_rate=100.0, evaluated=1, latency_avg=650.0)
    deltas = MetricDeltas(latency=LatencyDelta(absolute_ms=150.0, percent=30.0))
    violations = evaluate_thresholds(baseline, current, deltas, [], RegressionThresholds())
    latency = next(v for v in violations if v.metric == "latency")
    assert latency.unit is ThresholdUnit.PERCENT
    assert latency.threshold == 20.0
    assert latency.baseline == 500.0
    assert latency.current == 650.0
    assert latency.delta == 30.0


# -- undefined rates never violate ------------------------------------------------------


def test_undefined_rates_produce_no_violations():
    baseline = metrics_with_rates(pass_rate=None, error_rate=None, evaluated=0, total=0)
    current = metrics_with_rates(pass_rate=50.0, error_rate=50.0, evaluated=1, total=1)
    deltas = MetricDeltas()  # all None
    assert evaluate_thresholds(baseline, current, deltas, [], RegressionThresholds()) == []


# -- threshold model validation ---------------------------------------------------------


def test_default_thresholds_are_evalyx_policy_defaults():
    thresholds = RegressionThresholds()
    assert thresholds.max_pass_rate_drop_pp == 2.0
    assert thresholds.max_error_rate_increase_pp == 2.0
    assert thresholds.max_guardrail_failure_rate_increase_pp == 2.0
    assert thresholds.max_latency_increase_percent == 20.0


@pytest.mark.parametrize(
    "field", ["max_pass_rate_drop_pp", "max_error_rate_increase_pp", "max_guardrail_failure_rate_increase_pp"]
)
def test_negative_rate_thresholds_are_rejected(field: str):
    with pytest.raises(ValidationError):
        RegressionThresholds(**{field: -0.1})


def test_negative_latency_threshold_rejected_but_none_allowed():
    with pytest.raises(ValidationError):
        RegressionThresholds(max_latency_increase_percent=-1.0)
    assert RegressionThresholds(max_latency_increase_percent=None).max_latency_increase_percent is None


def test_unknown_threshold_fields_are_rejected():
    with pytest.raises(ValidationError):
        RegressionThresholds(threshold=0.05)  # vague name explicitly forbidden


# -- policy fingerprint -------------------------------------------------------------------


def test_fingerprint_differs_only_when_policy_differs():
    base = policy_fingerprint(RegressionThresholds())
    assert base == policy_fingerprint(RegressionThresholds())
    assert base != policy_fingerprint(RegressionThresholds(max_latency_increase_percent=None))
    assert base != policy_fingerprint(RegressionThresholds(max_pass_rate_drop_pp=1.0))


# -- end-to-end boundary through compare() (count-based scenarios) ---------------------------


def make_input_for(baseline_cases: list[CaseResultSnapshot], current_cases: list[CaseResultSnapshot]):
    context = ComparisonContext(
        baseline=ComparisonRunContext(
            run_id=uuid.uuid4(), status="completed", agent_model="m",
            application_id=uuid.uuid4(), dataset_version_id=uuid.uuid4(),
        ),
        current=ComparisonRunContext(
            run_id=uuid.uuid4(), status="completed", agent_model="m",
            application_id=uuid.uuid4(), dataset_version_id=uuid.uuid4(),
        ),
    )
    return ComparisonInput(
        baseline_cases=baseline_cases, current_cases=current_cases, context=context
    )


def cases(count: int, status: CaseStatus, prefix: str) -> list[CaseResultSnapshot]:
    return [
        CaseResultSnapshot(
            identity=f"{prefix}{i}",
            case_result_id=uuid.uuid4(),
            test_case_id=uuid.uuid4(),
            name=f"{prefix}{i}",
            status=status,
        )
        for i in range(count)
    ]


def passes(count: int, prefix: str = "c") -> list[CaseResultSnapshot]:
    return cases(count, CaseStatus.PASSED, prefix)


def failures(count: int, prefix: str = "f") -> list[CaseResultSnapshot]:
    return cases(count, CaseStatus.FAILED, prefix)


def test_fifty_case_boundary_exactly_two_pp_drop_is_not_a_regression():
    # 50/50 passed (100%) → 49/50 passed (98%): drop is exactly 2.0 pp.
    baseline = passes(50)
    current = passes(49) + failures(1)
    report = compare(make_input_for(baseline, current), RegressionThresholds())
    assert report.result is ComparisonResult.NO_REGRESSION
    assert report.threshold_violations == []


def test_fifty_case_boundary_just_above_threshold_is_a_regression():
    # Same identity both sides: c47 passes in baseline, fails in current.
    # 48/48 = 100% → 47/48 = 97.916667% → drop 2.083333 pp > 2.0.
    baseline = passes(48)
    current = passes(47)
    failed = CaseResultSnapshot(
        identity="c47",
        case_result_id=uuid.uuid4(),
        test_case_id=uuid.uuid4(),
        name="c47",
        status=CaseStatus.FAILED,
    )
    current.append(failed)
    report = compare(make_input_for(baseline, current), RegressionThresholds())
    assert report.result is ComparisonResult.REGRESSION_DETECTED
    violation = next(v for v in report.threshold_violations if v.metric == "pass_rate")
    assert violation.delta == pytest.approx(2.083333)
    assert [f.name for f in report.newly_failed_cases] == ["c47"]
