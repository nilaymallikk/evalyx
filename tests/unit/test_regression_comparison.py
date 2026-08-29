"""Unit tests for the pure regression comparison engine.

No database, no network, no Celery, no Redis — the comparison layer is
deterministic by construction (spec: regression logic must be pure).
"""

import uuid

import pytest

from evalyx.db.models import CaseStatus, GuardrailStatus
from evalyx.evaluation.regression import (
    CaseResultSnapshot,
    CaseTransition,
    ComparisonResult,
    GuardrailResultSnapshot,
    GuardrailTransition,
    RegressionThresholds,
    RunMetrics,
    calculate_guardrail_metrics,
    calculate_guardrail_name_metrics,
    calculate_latency_metrics,
    calculate_metric_deltas,
    calculate_run_metrics,
    classify_case_transition,
    classify_guardrail_transition,
    compare,
    diff_configurations,
    pair_case_results,
    policy_fingerprint,
    sanitize_configuration,
)
from evalyx.evaluation.regression.comparison import ComparisonInput
from evalyx.evaluation.regression.models import (
    ComparisonContext,
    ComparisonRunContext,
    LatencyMetrics,
)


def snap(
    identity: str,
    status: CaseStatus,
    *,
    latency: int | None = None,
    guardrails: list[tuple[str, GuardrailStatus]] | None = None,
) -> CaseResultSnapshot:
    return CaseResultSnapshot(
        identity=identity,
        case_result_id=uuid.uuid4(),
        test_case_id=uuid.uuid4(),
        name=identity,
        status=status,
        latency_ms=latency,
        guardrails=[GuardrailResultSnapshot(name=n, status=s) for n, s in guardrails or []],
    )


def make_context(
    *,
    baseline_config: dict | None = None,
    current_config: dict | None = None,
    baseline_model: str = "agent-a",
    current_model: str = "agent-a",
) -> ComparisonContext:
    # configuration_changes mirrors the service: diffs of the sanitized
    # snapshots, computed before the pure comparison is invoked.
    return ComparisonContext(
        baseline=ComparisonRunContext(
            run_id=uuid.uuid4(),
            status="completed",
            agent_model=baseline_model,
            application_id=uuid.uuid4(),
            dataset_version_id=uuid.uuid4(),
            configuration_snapshot=baseline_config or {},
        ),
        current=ComparisonRunContext(
            run_id=uuid.uuid4(),
            status="completed",
            agent_model=current_model,
            application_id=uuid.uuid4(),
            dataset_version_id=uuid.uuid4(),
            configuration_snapshot=current_config or {},
        ),
        agent_model_changed=baseline_model != current_model,
        configuration_changes=diff_configurations(
            baseline_config or {}, current_config or {}
        ),
    )


def make_input(
    baseline_cases: list[CaseResultSnapshot],
    current_cases: list[CaseResultSnapshot],
    *,
    context: ComparisonContext | None = None,
    baseline_dataset: list[str] | None = None,
    current_dataset: list[str] | None = None,
) -> ComparisonInput:
    return ComparisonInput(
        baseline_cases=baseline_cases,
        current_cases=current_cases,
        context=context or make_context(),
        # dataset cases are identity -> name; in these tests identity == name
        baseline_dataset_cases={name: name for name in baseline_dataset or []},
        current_dataset_cases={name: name for name in current_dataset or []},
    )


# -- run metrics ----------------------------------------------------------------


def test_pass_and_failure_rate():
    cases = [snap(f"c{i}", CaseStatus.PASSED) for i in range(95)] + [
        snap(f"f{i}", CaseStatus.FAILED) for i in range(5)
    ]
    metrics = calculate_run_metrics(cases)
    assert metrics.total_cases == 100
    assert metrics.evaluated_cases == 100
    assert metrics.pass_rate == 95.0
    assert metrics.failure_rate == 5.0


def test_error_rates_are_distinct():
    cases = (
        [snap(f"p{i}", CaseStatus.PASSED) for i in range(60)]
        + [snap(f"f{i}", CaseStatus.FAILED) for i in range(20)]
        + [snap(f"e{i}", CaseStatus.ERROR) for i in range(15)]
        + [snap(f"x{i}", CaseStatus.EXECUTED) for i in range(5)]
    )
    metrics = calculate_run_metrics(cases)
    assert metrics.evaluated_cases == 80
    assert metrics.pass_rate == 75.0
    assert metrics.failure_rate == 25.0
    assert metrics.execution_error_rate == 15.0
    assert metrics.evaluation_error_rate == 5.0
    assert metrics.error_rate == 20.0  # combined: the threshold applies to this


def test_zero_cases_yield_none_rates_never_zero():
    metrics = calculate_run_metrics([])
    assert metrics.total_cases == 0
    assert metrics.pass_rate is None
    assert metrics.failure_rate is None
    assert metrics.execution_error_rate is None
    assert metrics.evaluation_error_rate is None
    assert metrics.error_rate is None
    assert metrics.latency.average_ms is None
    assert metrics.latency.observations == 0


def test_no_evaluated_cases_but_errors_present():
    cases = [snap("e1", CaseStatus.ERROR), snap("e2", CaseStatus.ERROR)]
    metrics = calculate_run_metrics(cases)
    assert metrics.evaluated_cases == 0
    assert metrics.pass_rate is None  # undefined, not 0%
    assert metrics.error_rate == 100.0  # denominator is total_cases


def test_rates_rounded_to_six_decimals():
    cases = [snap("p", CaseStatus.PASSED)] + [snap(f"f{i}", CaseStatus.FAILED) for i in range(2)]
    metrics = calculate_run_metrics(cases)
    assert metrics.pass_rate == round(100 / 3, 6)


# -- latency ----------------------------------------------------------------------


def test_latency_average_ignores_missing_values():
    cases = [snap("a", CaseStatus.PASSED, latency=100), snap("b", CaseStatus.FAILED, latency=200),
             snap("c", CaseStatus.ERROR)]  # no latency
    latency = calculate_latency_metrics(cases)
    assert latency.average_ms == 150.0
    assert latency.observations == 2


def test_latency_delta_percent_and_absolute():
    baseline = RunMetrics(latency=LatencyMetrics(average_ms=420.0, observations=3))
    current = RunMetrics(latency=LatencyMetrics(average_ms=510.0, observations=3))
    deltas = calculate_metric_deltas(baseline, current)
    assert deltas.latency.absolute_ms == 90.0
    assert deltas.latency.percent == round(90 / 420 * 100, 6)  # 21.428571


def test_latency_percent_undefined_when_baseline_is_zero():
    baseline = RunMetrics(latency=LatencyMetrics(average_ms=0.0, observations=1))
    current = RunMetrics(latency=LatencyMetrics(average_ms=100.0, observations=1))
    deltas = calculate_metric_deltas(baseline, current)
    assert deltas.latency.absolute_ms == 100.0
    assert deltas.latency.percent is None


def test_latency_delta_undefined_when_a_side_has_no_observations():
    baseline = RunMetrics(latency=LatencyMetrics(average_ms=None, observations=0))
    current = RunMetrics(latency=LatencyMetrics(average_ms=100.0, observations=1))
    deltas = calculate_metric_deltas(baseline, current)
    assert deltas.latency.absolute_ms is None
    assert deltas.latency.percent is None


def test_rate_deltas_are_percentage_points_current_minus_baseline():
    baseline = RunMetrics(pass_rate=95.0, failure_rate=3.0, error_rate=1.0,
                          execution_error_rate=1.0, evaluation_error_rate=0.0)
    current = RunMetrics(pass_rate=88.0, failure_rate=6.0, error_rate=6.0,
                         execution_error_rate=5.0, evaluation_error_rate=1.0)
    deltas = calculate_metric_deltas(baseline, current)
    assert deltas.pass_rate_pp == -7.0
    assert deltas.failure_rate_pp == 3.0
    assert deltas.error_rate_pp == 5.0
    assert deltas.execution_error_rate_pp == 4.0
    assert deltas.evaluation_error_rate_pp == 1.0


# -- guardrail metrics -------------------------------------------------------------


def test_guardrail_failure_rate_excludes_errors_and_missing():
    cases = [
        snap("a", CaseStatus.PASSED, guardrails=[("pii", GuardrailStatus.PASSED)]),
        snap("b", CaseStatus.FAILED, guardrails=[("pii", GuardrailStatus.FAILED)]),
        snap("c", CaseStatus.PASSED, guardrails=[("pii", GuardrailStatus.ERROR)]),
        snap("d", CaseStatus.PASSED),  # pii row entirely missing
    ]
    metrics = calculate_guardrail_name_metrics("pii", cases)
    assert metrics.total_evaluations == 3
    assert metrics.passed == 1
    assert metrics.failed == 1
    assert metrics.errors == 1
    assert metrics.missing == 1
    assert metrics.failure_rate == 50.0  # failed / (passed + failed)


def test_guardrail_failure_rate_none_without_verdicts():
    cases = [snap("a", CaseStatus.PASSED, guardrails=[("pii", GuardrailStatus.ERROR)])]
    assert calculate_guardrail_name_metrics("pii", cases).failure_rate is None


def test_guardrail_comparisons_cover_union_of_names_sorted():
    baseline = [snap("a", CaseStatus.PASSED, guardrails=[("pii", GuardrailStatus.PASSED),
                                                         ("safety", GuardrailStatus.PASSED)])]
    current = [snap("a", CaseStatus.FAILED, guardrails=[("pii", GuardrailStatus.FAILED),
                                                        ("injection", GuardrailStatus.FAILED)])]
    comparisons = calculate_guardrail_metrics(baseline, current)
    assert [c.name for c in comparisons] == ["injection", "pii", "safety"]
    by_name = {c.name: c for c in comparisons}
    assert by_name["pii"].failure_rate_delta_pp == 100.0
    assert by_name["injection"].baseline.failure_rate is None  # absent in baseline
    assert by_name["injection"].failure_rate_delta_pp is None
    assert by_name["safety"].current.failure_rate is None


# -- case transition matrix ----------------------------------------------------------


@pytest.mark.parametrize(
    ("baseline", "current", "expected"),
    [
        (CaseStatus.PASSED, CaseStatus.PASSED, CaseTransition.STABLE_PASS),
        (CaseStatus.PASSED, CaseStatus.FAILED, CaseTransition.NEWLY_FAILED),
        (CaseStatus.PASSED, CaseStatus.ERROR, CaseTransition.NEWLY_ERRORED),
        (CaseStatus.PASSED, CaseStatus.EXECUTED, CaseTransition.NEWLY_ERRORED),
        (CaseStatus.FAILED, CaseStatus.PASSED, CaseTransition.FIXED),
        (CaseStatus.FAILED, CaseStatus.FAILED, CaseTransition.STABLE_FAILURE),
        (CaseStatus.FAILED, CaseStatus.ERROR, CaseTransition.ERROR_TRANSITION),
        (CaseStatus.FAILED, CaseStatus.EXECUTED, CaseTransition.ERROR_TRANSITION),
        (CaseStatus.ERROR, CaseStatus.PASSED, CaseTransition.RECOVERED),
        (CaseStatus.ERROR, CaseStatus.FAILED, CaseTransition.FAILURE_AFTER_ERROR),
        (CaseStatus.ERROR, CaseStatus.ERROR, CaseTransition.STABLE_ERROR),
        (CaseStatus.ERROR, CaseStatus.EXECUTED, CaseTransition.STABLE_ERROR),
        (CaseStatus.EXECUTED, CaseStatus.PASSED, CaseTransition.RECOVERED),
        (CaseStatus.EXECUTED, CaseStatus.FAILED, CaseTransition.FAILURE_AFTER_ERROR),
        (CaseStatus.EXECUTED, CaseStatus.ERROR, CaseTransition.STABLE_ERROR),
        (CaseStatus.EXECUTED, CaseStatus.EXECUTED, CaseTransition.STABLE_ERROR),
    ],
)
def test_case_transition_matrix(baseline, current, expected):
    assert classify_case_transition(baseline, current) is expected


# -- guardrail transition matrix -------------------------------------------------------


@pytest.mark.parametrize(
    ("baseline", "current", "expected"),
    [
        (GuardrailStatus.PASSED, GuardrailStatus.PASSED, GuardrailTransition.STABLE),
        (GuardrailStatus.PASSED, GuardrailStatus.FAILED, GuardrailTransition.NEW_FAILURE),
        (GuardrailStatus.PASSED, GuardrailStatus.ERROR, GuardrailTransition.DEGRADED_EVALUATION),
        (GuardrailStatus.FAILED, GuardrailStatus.PASSED, GuardrailTransition.FIXED),
        (GuardrailStatus.FAILED, GuardrailStatus.FAILED, GuardrailTransition.PERSISTENT_FAILURE),
        (GuardrailStatus.FAILED, GuardrailStatus.ERROR, GuardrailTransition.DEGRADED_EVALUATION),
        (GuardrailStatus.ERROR, GuardrailStatus.PASSED, GuardrailTransition.RECOVERED_EVALUATION),
        (GuardrailStatus.ERROR, GuardrailStatus.FAILED, GuardrailTransition.FAILURE_AFTER_ERROR),
        (GuardrailStatus.ERROR, GuardrailStatus.ERROR, GuardrailTransition.PERSISTENT_ERROR),
        (None, GuardrailStatus.PASSED, GuardrailTransition.MISSING_BASELINE),
        (None, GuardrailStatus.FAILED, GuardrailTransition.MISSING_BASELINE),
        (GuardrailStatus.PASSED, None, GuardrailTransition.MISSING_CURRENT),
        (GuardrailStatus.FAILED, None, GuardrailTransition.MISSING_CURRENT),
    ],
)
def test_guardrail_transition_matrix(baseline, current, expected):
    assert classify_guardrail_transition(baseline, current) is expected


def test_guardrail_transition_both_missing_is_a_caller_error():
    with pytest.raises(ValueError):
        classify_guardrail_transition(None, None)


# -- case pairing ------------------------------------------------------------------------


def test_pairing_matches_by_identity_and_reports_new_removed():
    baseline = [snap(f"c{i}", CaseStatus.PASSED) for i in range(3)] + [snap("old", CaseStatus.PASSED)]
    current = [snap(f"c{i}", CaseStatus.PASSED) for i in range(3)] + [snap("fresh", CaseStatus.PASSED)]
    pairing = pair_case_results(baseline, current)
    assert [p.identity for p in pairing.pairs] == ["c0", "c1", "c2", "fresh", "old"]  # sorted
    assert pairing.new_cases == ["fresh"]
    assert pairing.removed_cases == ["old"]
    matched = [p for p in pairing.pairs if p.baseline and p.current]
    assert len(matched) == 3


def test_findings_carry_guardrail_evidence():
    baseline = [
        snap(
            "c1",
            CaseStatus.PASSED,
            guardrails=[("pii", GuardrailStatus.PASSED), ("safety", GuardrailStatus.PASSED)],
        )
    ]
    current = [
        snap(
            "c1",
            CaseStatus.FAILED,
            guardrails=[("pii", GuardrailStatus.FAILED), ("safety", GuardrailStatus.PASSED)],
        )
    ]
    report = compare(make_input(baseline, current), RegressionThresholds())
    assert len(report.newly_failed_cases) == 1
    finding = report.newly_failed_cases[0]
    assert finding.transition is CaseTransition.NEWLY_FAILED
    assert finding.new_guardrail_failures == ["pii"]
    assert finding.fixed_guardrail_failures == []
    assert finding.baseline_status == "passed"
    assert finding.current_status == "failed"


# -- top-level comparison ----------------------------------------------------------------


def ten_case_scenario():
    """Baseline: 7 pass / 2 fail / 1 error. Current: 5 pass / 2 fail / 2 errors.

    One case is newly failed (with a new PII guardrail failure), one newly
    errored, one fixed, one stable failure, one stable error.
    """
    baseline = (
        [snap(f"ok{i}", CaseStatus.PASSED, latency=450, guardrails=[
            ("pii", GuardrailStatus.PASSED), ("safety", GuardrailStatus.PASSED)]) for i in range(7)]
        + [snap("stable-fail", CaseStatus.FAILED, latency=450, guardrails=[
            ("pii", GuardrailStatus.PASSED), ("safety", GuardrailStatus.FAILED)]),
           snap("to-fix", CaseStatus.FAILED, latency=450, guardrails=[
               ("pii", GuardrailStatus.PASSED), ("safety", GuardrailStatus.FAILED)]),
           snap("err", CaseStatus.ERROR)]
    )
    current = (
        [snap(f"ok{i}", CaseStatus.PASSED, latency=600, guardrails=[
            ("pii", GuardrailStatus.PASSED), ("safety", GuardrailStatus.PASSED)]) for i in range(5)]
        + [snap("ok5", CaseStatus.FAILED, latency=600, guardrails=[
            ("pii", GuardrailStatus.FAILED), ("safety", GuardrailStatus.PASSED)]),
           snap("ok6", CaseStatus.ERROR),  # execution error: no guardrail rows
           snap("stable-fail", CaseStatus.FAILED, latency=600, guardrails=[
               ("pii", GuardrailStatus.PASSED), ("safety", GuardrailStatus.FAILED)]),
           snap("to-fix", CaseStatus.PASSED, latency=600, guardrails=[
               ("pii", GuardrailStatus.PASSED), ("safety", GuardrailStatus.PASSED)]),
           snap("err", CaseStatus.ERROR)]
    )
    return baseline, current


def test_compare_detects_regression_with_violations():
    baseline, current = ten_case_scenario()
    report = compare(make_input(baseline, current), RegressionThresholds())
    assert report.result is ComparisonResult.REGRESSION_DETECTED
    assert report.regression_detected is True
    assert report.baseline.pass_rate == pytest.approx(77.777778)
    assert report.current.pass_rate == pytest.approx(75.0)
    metrics = {v.metric for v in report.threshold_violations}
    assert "pass_rate" in metrics
    assert "error_rate" in metrics
    assert "guardrail_failure_rate" in metrics
    assert "latency" in metrics  # 450 → 600 ms is +33%
    assert report.matched_cases == 10
    assert [f.name for f in report.newly_failed_cases] == ["ok5"]
    assert [f.name for f in report.fixed_cases] == ["to-fix"]
    assert [f.name for f in report.stable_failures] == ["stable-fail"]
    assert [f.name for f in report.newly_errored_cases] == ["ok6"]
    assert report.newly_errored_cases[0].new_guardrail_failures == []  # error case, no rows
    # every violation is explainable in words
    assert all(v.detail for v in report.threshold_violations)
    assert all(v.delta > v.threshold for v in report.threshold_violations)


def test_compare_reports_newly_errored_separately_from_newly_failed():
    baseline = [snap("a", CaseStatus.PASSED), snap("b", CaseStatus.PASSED)]
    current = [snap("a", CaseStatus.ERROR), snap("b", CaseStatus.FAILED)]
    report = compare(make_input(baseline, current), RegressionThresholds())
    assert [f.name for f in report.newly_errored_cases] == ["a"]
    assert [f.name for f in report.newly_failed_cases] == ["b"]
    finding = report.newly_errored_cases[0]
    assert finding.baseline_status == "passed"
    assert finding.current_status == "error"
    assert finding.transition is CaseTransition.NEWLY_ERRORED


def test_compare_no_regression_within_thresholds():
    baseline = [snap(f"c{i}", CaseStatus.PASSED) for i in range(10)]
    current = [snap(f"c{i}", CaseStatus.PASSED) for i in range(10)]
    report = compare(make_input(baseline, current), RegressionThresholds())
    assert report.result is ComparisonResult.NO_REGRESSION
    assert report.regression_detected is False
    assert report.threshold_violations == []


def test_compare_not_comparable_without_evaluated_cases():
    baseline = [snap("e", CaseStatus.ERROR)]
    current = [snap("a", CaseStatus.PASSED)]
    report = compare(make_input(baseline, current), RegressionThresholds())
    assert report.result is ComparisonResult.NOT_COMPARABLE
    assert report.regression_detected is False
    assert report.not_comparable_reason is not None
    assert "evaluated case" in report.not_comparable_reason
    assert report.threshold_violations == []


def test_compare_is_deterministic():
    baseline, current = ten_case_scenario()
    thresholds = RegressionThresholds()
    input_data = make_input(baseline, current)
    first = compare(input_data, thresholds)
    second = compare(input_data, thresholds)
    # comparison_id/created_at are the only persistence-dependent fields
    assert first.model_dump(exclude={"comparison_id", "created_at"}) == second.model_dump(
        exclude={"comparison_id", "created_at"}
    )


def test_compare_findings_and_guardrails_are_deterministically_ordered():
    baseline = [
        snap(f"c{i:02d}", CaseStatus.PASSED, guardrails=[("z", GuardrailStatus.PASSED),
                                                         ("a", GuardrailStatus.PASSED)])
        for i in range(5)
    ]
    current = [
        snap(f"c{i:02d}", CaseStatus.FAILED, guardrails=[("z", GuardrailStatus.FAILED),
                                                         ("a", GuardrailStatus.FAILED)])
        for i in range(5)
    ]
    report = compare(make_input(baseline, current), RegressionThresholds())
    assert [f.name for f in report.newly_failed_cases] == sorted(
        f.name for f in report.newly_failed_cases
    )
    assert [c.name for c in report.guardrail_comparisons] == ["a", "z"]


def test_compare_reports_new_removed_and_missing_case_results():
    baseline = [snap("kept", CaseStatus.PASSED)]
    current = [snap("kept", CaseStatus.PASSED), snap("added", CaseStatus.PASSED)]
    report = compare(
        make_input(
            baseline,
            current,
            baseline_dataset=["kept", "dropped"],
            current_dataset=["kept", "added"],
        ),
        RegressionThresholds(),
    )
    assert report.new_cases == ["added"]
    # "dropped" never produced a baseline result, so it is missing data —
    # not a removed case (removed = baseline RESULT without current result).
    assert report.removed_cases == []
    assert report.missing_case_results == {"baseline": ["dropped"], "current": []}
    # a dataset case with no result at all is surfaced explicitly
    report2 = compare(
        make_input(
            baseline,
            current,
            baseline_dataset=["kept", "no-result-baseline"],
            current_dataset=["kept", "added", "no-result-current"],
        ),
        RegressionThresholds(),
    )
    assert report2.missing_case_results == {
        "baseline": ["no-result-baseline"],
        "current": ["no-result-current"],
    }


def test_new_cases_are_never_regressions():
    baseline = [snap("a", CaseStatus.PASSED)]
    current = [snap("a", CaseStatus.PASSED)] + [
        snap(f"new{i}", CaseStatus.FAILED) for i in range(5)
    ]
    report = compare(make_input(baseline, current), RegressionThresholds())
    assert report.new_cases == ["new0", "new1", "new2", "new3", "new4"]
    # Metrics use matched cases only: five failing cases with no baseline
    # outcome cannot manufacture a pass-rate regression (spec: new cases
    # are reported separately, never classified as regressions).
    assert report.current.pass_rate == 100.0
    assert report.baseline.pass_rate == 100.0
    assert report.result is ComparisonResult.NO_REGRESSION
    assert report.threshold_violations == []
    assert report.newly_failed_cases == []


def test_context_records_model_and_configuration_changes():
    context = make_context(
        baseline_config={"temperature": 0.2, "max_tokens": 512},
        current_config={"temperature": 0.7, "max_tokens": 512},
        current_model="agent-b",
    )
    baseline = [snap("a", CaseStatus.PASSED)]
    current = [snap("a", CaseStatus.PASSED)]
    report = compare(make_input(baseline, current, context=context), RegressionThresholds())
    assert report.context.agent_model_changed is True
    changes = {c.path: c for c in report.context.configuration_changes}
    assert set(changes) == {"temperature"}
    assert changes["temperature"].baseline == 0.2
    assert changes["temperature"].current == 0.7


# -- configuration sanitization ------------------------------------------------------------


def test_sanitize_configuration_strips_secret_looking_keys_recursively():
    snapshot = {
        "temperature": 0.2,
        "OPENROUTER_API_KEY": "should-never-appear",
        "max_tokens": 512,  # innocuous: must survive
        "nested": {
            "authorization": "Bearer x",
            "db_password": "hunter2",
            "auth_token": "t",
            "max_tokens": 512,
        },
        "list": [{"secret_value": 1, "name": "keep"}],
    }
    cleaned = sanitize_configuration(snapshot)
    assert cleaned["temperature"] == 0.2
    assert cleaned["max_tokens"] == 512
    assert "OPENROUTER_API_KEY" not in cleaned
    assert cleaned["nested"] == {"max_tokens": 512}
    assert cleaned["list"] == [{"name": "keep"}]


def test_diff_configurations_reports_leaf_changes_sorted():
    diff = diff_configurations(
        {"a": {"x": 1, "y": 2}, "b": 3, "api_key": "s"},
        {"a": {"x": 1, "y": 9}, "c": 4, "api_key": "s2"},
    )
    assert [c.path for c in diff] == ["a.y", "b", "c"]
    assert diff[0].baseline == 2 and diff[0].current == 9
    assert diff[1].current is None  # removed
    assert diff[2].baseline is None  # added
    assert not any("api_key" in c.path for c in diff)
    assert not any("s" == c.baseline or "s2" == c.current for c in diff)


# -- policy fingerprint -------------------------------------------------------------------


def test_policy_fingerprint_is_stable_and_policy_sensitive():
    thresholds = RegressionThresholds()
    assert policy_fingerprint(thresholds) == policy_fingerprint(RegressionThresholds())
    assert policy_fingerprint(thresholds) != policy_fingerprint(
        RegressionThresholds(max_pass_rate_drop_pp=3.0)
    )
    assert len(policy_fingerprint(thresholds)) == 64
