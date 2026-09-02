"""Pure regression comparison functions (no DB, no network, no LLM, no I/O).

Deterministic by construction: same inputs → identical outputs. All
functions accept the typed snapshots from :mod:`.models` and return typed
results. The service layer (:mod:`.service`) owns data loading and
persistence only.

Metric conventions (see :mod:`.models` for full units):

- rates are percentages 0–100; rate deltas are percentage points
- a violation requires ``delta > threshold`` strictly — equal is not a
  regression
- zero/missing denominators yield ``None`` (never 0)
"""

import hashlib
import json
import statistics

from pydantic import BaseModel, ConfigDict, Field

from evalyx.db.models import CaseStatus, ComparisonResult, GuardrailStatus
from evalyx.evaluation.regression.models import (
    COMPARISON_VERSION,
    CaseFinding,
    CaseResultSnapshot,
    CaseTransition,
    ComparisonContext,
    ConfigurationChange,
    GuardrailComparison,
    GuardrailNameMetrics,
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
    round_metric,
)

__all__ = [
    "ComparisonInput",
    "calculate_guardrail_metrics",
    "calculate_guardrail_name_metrics",
    "calculate_latency_metrics",
    "calculate_metric_deltas",
    "calculate_run_metrics",
    "compare",
    "diff_configurations",
    "evaluate_thresholds",
    "pair_case_results",
    "policy_fingerprint",
    "sanitize_configuration",
]


# -- policy identity ----------------------------------------------------------


def policy_fingerprint(thresholds: RegressionThresholds) -> str:
    """sha256 over the canonical (comparison version, thresholds) JSON.

    Deterministic across processes and runs (sorted keys); used as the
    idempotency key component persisted on the comparison row.
    """
    payload = json.dumps(
        {"comparison_version": COMPARISON_VERSION, "thresholds": thresholds.model_dump()},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# -- metrics -------------------------------------------------------------------


def calculate_latency_metrics(cases: list[CaseResultSnapshot]) -> LatencyMetrics:
    """Average latency over cases that recorded one (``None`` if none do)."""
    observations = [case.latency_ms for case in cases if case.latency_ms is not None]
    if not observations:
        return LatencyMetrics(average_ms=None, observations=0)
    return LatencyMetrics(
        average_ms=round_metric(statistics.fmean(observations)),
        observations=len(observations),
    )


def calculate_run_metrics(cases: list[CaseResultSnapshot]) -> RunMetrics:
    """Full metric snapshot for one run's case results (pure, total-safe)."""
    total = len(cases)
    passed = sum(1 for c in cases if c.status is CaseStatus.PASSED)
    failed = sum(1 for c in cases if c.status is CaseStatus.FAILED)
    execution_errors = sum(1 for c in cases if c.status is CaseStatus.ERROR)
    evaluation_errors = sum(1 for c in cases if c.status is CaseStatus.EXECUTED)
    evaluated = passed + failed
    error_cases = execution_errors + evaluation_errors

    def rate(numerator: int, denominator: int) -> float | None:
        return round_metric(numerator / denominator * 100) if denominator else None

    return RunMetrics(
        total_cases=total,
        evaluated_cases=evaluated,
        passed_cases=passed,
        failed_cases=failed,
        execution_error_cases=execution_errors,
        evaluation_error_cases=evaluation_errors,
        pass_rate=rate(passed, evaluated),
        failure_rate=rate(failed, evaluated),
        execution_error_rate=rate(execution_errors, total),
        evaluation_error_rate=rate(evaluation_errors, total),
        error_rate=rate(error_cases, total),
        latency=calculate_latency_metrics(cases),
    )


def calculate_guardrail_name_metrics(
    name: str,
    cases: list[CaseResultSnapshot],
) -> GuardrailNameMetrics:
    """Aggregate one guardrail name across a run's cases.

    Missing rows are counted as ``missing`` (never as passes). The failure
    rate's denominator counts only cases with an actual verdict.
    """
    passed = failed = errors = missing = 0
    for case in cases:
        status = case.guardrail_status(name)
        if status is None:
            missing += 1
        elif status is GuardrailStatus.PASSED:
            passed += 1
        elif status is GuardrailStatus.FAILED:
            failed += 1
        else:
            errors += 1
    verdicts = passed + failed
    return GuardrailNameMetrics(
        name=name,
        total_evaluations=passed + failed + errors,
        passed=passed,
        failed=failed,
        errors=errors,
        missing=missing,
        failure_rate=round_metric(failed / verdicts * 100) if verdicts else None,
    )


def calculate_guardrail_metrics(
    baseline_cases: list[CaseResultSnapshot],
    current_cases: list[CaseResultSnapshot],
) -> list[GuardrailComparison]:
    """Compare every observed guardrail name, sorted by name."""
    baseline_names = {g.name for case in baseline_cases for g in case.guardrails}
    current_names = {g.name for case in current_cases for g in case.guardrails}
    comparisons: list[GuardrailComparison] = []
    for name in sorted(baseline_names | current_names):
        baseline = calculate_guardrail_name_metrics(name, baseline_cases)
        current = calculate_guardrail_name_metrics(name, current_cases)
        delta = (
            round_metric(current.failure_rate - baseline.failure_rate)
            if baseline.failure_rate is not None and current.failure_rate is not None
            else None
        )
        comparisons.append(
            GuardrailComparison(
                name=name,
                baseline=baseline,
                current=current,
                failure_rate_delta_pp=delta,
            )
        )
    return comparisons


def calculate_metric_deltas(baseline: RunMetrics, current: RunMetrics) -> MetricDeltas:
    """Percentage-point deltas (current - baseline); ``None`` if undefined."""

    def pp(b: float | None, c: float | None) -> float | None:
        return round_metric(c - b) if b is not None and c is not None else None

    baseline_avg = baseline.latency.average_ms
    current_avg = current.latency.average_ms
    if baseline_avg is None or current_avg is None:
        latency = LatencyDelta(absolute_ms=None, percent=None)
    else:
        percent = (
            round_metric((current_avg - baseline_avg) / baseline_avg * 100)
            if baseline_avg > 0
            else None  # relative change from exactly 0 ms is undefined
        )
        latency = LatencyDelta(
            absolute_ms=round_metric(current_avg - baseline_avg),
            percent=percent,
        )
    return MetricDeltas(
        pass_rate_pp=pp(baseline.pass_rate, current.pass_rate),
        failure_rate_pp=pp(baseline.failure_rate, current.failure_rate),
        execution_error_rate_pp=pp(baseline.execution_error_rate, current.execution_error_rate),
        evaluation_error_rate_pp=pp(baseline.evaluation_error_rate, current.evaluation_error_rate),
        error_rate_pp=pp(baseline.error_rate, current.error_rate),
        latency=latency,
    )


# -- threshold evaluation -------------------------------------------------------


def evaluate_thresholds(
    baseline: RunMetrics,
    current: RunMetrics,
    deltas: MetricDeltas,
    guardrail_comparisons: list[GuardrailComparison],
    thresholds: RegressionThresholds,
) -> list[ThresholdViolation]:
    """All threshold breaches, deterministically ordered.

    Violation rule: ``delta > threshold`` strictly (equal → no regression).
    Guardrail violations are ordered by guardrail name; the fixed metric
    order is pass_rate, error_rate, latency.
    """
    violations: list[ThresholdViolation] = []

    if baseline.pass_rate is not None and current.pass_rate is not None:
        drop = round_metric(baseline.pass_rate - current.pass_rate)
        if drop > thresholds.max_pass_rate_drop_pp:
            violations.append(
                ThresholdViolation(
                    metric="pass_rate",
                    baseline=baseline.pass_rate,
                    current=current.pass_rate,
                    delta=drop,
                    threshold=thresholds.max_pass_rate_drop_pp,
                    unit=ThresholdUnit.PERCENTAGE_POINTS,
                    detail=(
                        f"Pass rate dropped {drop} pp "
                        f"({baseline.pass_rate}% → {current.pass_rate}%), exceeding the "
                        f"maximum tolerated drop of {thresholds.max_pass_rate_drop_pp} pp."
                    ),
                )
            )

    if baseline.error_rate is not None and current.error_rate is not None:
        increase = round_metric(current.error_rate - baseline.error_rate)
        if increase > thresholds.max_error_rate_increase_pp:
            violations.append(
                ThresholdViolation(
                    metric="error_rate",
                    baseline=baseline.error_rate,
                    current=current.error_rate,
                    delta=increase,
                    threshold=thresholds.max_error_rate_increase_pp,
                    unit=ThresholdUnit.PERCENTAGE_POINTS,
                    detail=(
                        f"Error rate (execution + evaluation errors) rose {increase} pp "
                        f"({baseline.error_rate}% → {current.error_rate}%), exceeding the "
                        f"maximum tolerated increase of {thresholds.max_error_rate_increase_pp} pp."
                    ),
                )
            )

    for comparison in guardrail_comparisons:
        delta = comparison.failure_rate_delta_pp
        if delta is None:
            continue
        if delta > thresholds.max_guardrail_failure_rate_increase_pp:
            violations.append(
                ThresholdViolation(
                    metric="guardrail_failure_rate",
                    guardrail=comparison.name,
                    baseline=comparison.baseline.failure_rate,
                    current=comparison.current.failure_rate,
                    delta=delta,
                    threshold=thresholds.max_guardrail_failure_rate_increase_pp,
                    unit=ThresholdUnit.PERCENTAGE_POINTS,
                    detail=(
                        f"Guardrail {comparison.name!r} failure rate rose {delta} pp "
                        f"({comparison.baseline.failure_rate}% → "
                        f"{comparison.current.failure_rate}%), exceeding the maximum "
                        f"tolerated increase of "
                        f"{thresholds.max_guardrail_failure_rate_increase_pp} pp."
                    ),
                )
            )

    latency_percent = deltas.latency.percent
    if (
        thresholds.max_latency_increase_percent is not None
        and latency_percent is not None
        and latency_percent > thresholds.max_latency_increase_percent
    ):
        violations.append(
            ThresholdViolation(
                metric="latency",
                baseline=baseline.latency.average_ms,
                current=current.latency.average_ms,
                delta=latency_percent,
                threshold=thresholds.max_latency_increase_percent,
                unit=ThresholdUnit.PERCENT,
                detail=(
                    f"Average latency rose {latency_percent}% "
                    f"({baseline.latency.average_ms} ms → "
                    f"{current.latency.average_ms} ms), exceeding the maximum tolerated "
                    f"increase of {thresholds.max_latency_increase_percent}%."
                ),
            )
        )

    return violations


# -- case pairing & findings -----------------------------------------------------


class _Pairing(BaseModel):
    """Result of identity-based case matching (pure data)."""

    model_config = ConfigDict(extra="forbid")

    pairs: list[MatchedCasePair] = Field(default_factory=list)
    new_cases: list[str] = Field(default_factory=list)
    removed_cases: list[str] = Field(default_factory=list)


def pair_case_results(
    baseline_cases: list[CaseResultSnapshot],
    current_cases: list[CaseResultSnapshot],
) -> _Pairing:
    """Match case results by their ``identity`` key (stable, not positional).

    Unmatched identities are reported as new (current only) or removed
    (baseline only) cases — never silently dropped or counted as
    regressions. Pairs are ordered by identity.
    """
    baseline_by_identity = {case.identity: case for case in baseline_cases}
    current_by_identity = {case.identity: case for case in current_cases}
    all_identities = sorted(set(baseline_by_identity) | set(current_by_identity))

    pairs: list[MatchedCasePair] = []
    for identity in all_identities:
        baseline = baseline_by_identity.get(identity)
        current = current_by_identity.get(identity)
        anchor = baseline if baseline is not None else current
        if anchor is None:  # pragma: no cover - impossible by construction
            continue
        pairs.append(
            MatchedCasePair(
                identity=identity,
                name=anchor.name,
                test_case_id=anchor.test_case_id,
                baseline=baseline,
                current=current,
            )
        )

    return _Pairing(
        pairs=pairs,
        new_cases=sorted(set(current_by_identity) - set(baseline_by_identity)),
        removed_cases=sorted(set(baseline_by_identity) - set(current_by_identity)),
    )


def _finding(pair: MatchedCasePair) -> CaseFinding:
    assert pair.baseline is not None and pair.current is not None  # for finders
    baseline_failures = pair.baseline.guardrail_failures
    current_failures = pair.current.guardrail_failures
    return CaseFinding(
        identity=pair.identity,
        name=pair.name,
        test_case_id=pair.test_case_id,
        baseline_case_result_id=pair.baseline.case_result_id,
        current_case_result_id=pair.current.case_result_id,
        baseline_status=pair.baseline.status.value,
        current_status=pair.current.status.value,
        transition=classify_case_transition(pair.baseline.status, pair.current.status),
        baseline_guardrail_failures=baseline_failures,
        current_guardrail_failures=current_failures,
        new_guardrail_failures=sorted(set(current_failures) - set(baseline_failures)),
        fixed_guardrail_failures=sorted(set(baseline_failures) - set(current_failures)),
        current_failure_category=pair.current.failure_category,
    )


def _findings_by_transition(
    pairs: list[MatchedCasePair],
) -> dict[CaseTransition, list[CaseFinding]]:
    """Findings for matched pairs with results on both sides, per transition."""
    findings: dict[CaseTransition, list[CaseFinding]] = {}
    for pair in pairs:
        if pair.baseline is None or pair.current is None:
            continue
        finding = _finding(pair)
        findings.setdefault(finding.transition, []).append(finding)
    # Deterministic ordering everywhere: by case name, then identity.
    for group in findings.values():
        group.sort(key=lambda f: (f.name, f.identity))
    return findings


# -- top-level comparison ---------------------------------------------------------


class ComparisonInput(BaseModel):
    """Everything the pure comparison needs (built by the service)."""

    model_config = ConfigDict(extra="forbid")

    baseline_cases: list[CaseResultSnapshot]
    current_cases: list[CaseResultSnapshot]
    context: ComparisonContext
    #: Dataset-version cases per side as ``identity -> name``; used to detect
    #: dataset cases that produced no case result in a run (missing data).
    baseline_dataset_cases: dict[str, str] = Field(default_factory=dict)
    current_dataset_cases: dict[str, str] = Field(default_factory=dict)


def _missing_case_results(
    input_data: ComparisonInput,
    baseline_cases: list[CaseResultSnapshot],
    current_cases: list[CaseResultSnapshot],
) -> dict[str, list[str]]:
    """Dataset-version cases that produced no case result in a run.

    Reported by human-readable case name (never an opaque id): absence of
    evidence is surfaced explicitly, not invented into a pass.
    """
    baseline_results = {case.identity for case in baseline_cases}
    current_results = {case.identity for case in current_cases}
    return {
        "baseline": sorted(
            input_data.baseline_dataset_cases[identity]
            for identity in set(input_data.baseline_dataset_cases) - baseline_results
        ),
        "current": sorted(
            input_data.current_dataset_cases[identity]
            for identity in set(input_data.current_dataset_cases) - current_results
        ),
    }


def compare(input_data: ComparisonInput, thresholds: RegressionThresholds) -> RegressionReport:
    """Run the full deterministic comparison and build the typed report.

    Metrics are computed over **matched cases** — identities present with
    results on both sides. New/removed cases and missing results are
    reported separately and never influence rates: new failing cases have
    no baseline outcome and must not manufacture a regression (see
    ``new_cases`` / ``removed_cases`` / ``missing_case_results``).

    Decision: ``REGRESSION_DETECTED`` iff at least one threshold violation
    exists; ``NOT_COMPARABLE`` when either side lacks evaluated cases (no
    meaningful denominator); else ``NO_REGRESSION``. Case findings,
    guardrail comparisons, and context are always preserved as evidence.
    """
    pairing = pair_case_results(input_data.baseline_cases, input_data.current_cases)
    matched_baseline = [p.baseline for p in pairing.pairs if p.baseline and p.current]
    matched_current = [p.current for p in pairing.pairs if p.baseline and p.current]

    baseline_metrics = calculate_run_metrics(matched_baseline)
    current_metrics = calculate_run_metrics(matched_current)
    deltas = calculate_metric_deltas(baseline_metrics, current_metrics)
    guardrail_comparisons = calculate_guardrail_metrics(matched_baseline, matched_current)

    not_comparable_reason: str | None = None
    if baseline_metrics.evaluated_cases == 0 or current_metrics.evaluated_cases == 0:
        not_comparable_reason = (
            "Comparison has no meaningful denominator: baseline has "
            f"{baseline_metrics.evaluated_cases} evaluated case(s) and current has "
            f"{current_metrics.evaluated_cases} evaluated case(s); at least one "
            "semantic outcome (passed/failed) per run is required."
        )
        violations: list[ThresholdViolation] = []
    else:
        violations = evaluate_thresholds(
            baseline_metrics, current_metrics, deltas, guardrail_comparisons, thresholds
        )

    if violations:
        result = ComparisonResult.REGRESSION_DETECTED
    elif not_comparable_reason is not None:
        result = ComparisonResult.NOT_COMPARABLE
    else:
        result = ComparisonResult.NO_REGRESSION

    findings = _findings_by_transition(pairing.pairs)

    return RegressionReport(
        comparison_version=COMPARISON_VERSION,
        baseline_run_id=input_data.context.baseline.run_id,
        current_run_id=input_data.context.current.run_id,
        result=result,
        regression_detected=result is ComparisonResult.REGRESSION_DETECTED,
        not_comparable_reason=not_comparable_reason,
        baseline=baseline_metrics,
        current=current_metrics,
        deltas=deltas,
        threshold_violations=violations,
        matched_cases=len(matched_baseline),
        newly_failed_cases=findings.get(CaseTransition.NEWLY_FAILED, []),
        newly_errored_cases=findings.get(CaseTransition.NEWLY_ERRORED, []),
        fixed_cases=findings.get(CaseTransition.FIXED, []),
        stable_failures=findings.get(CaseTransition.STABLE_FAILURE, []),
        error_transition_cases=findings.get(CaseTransition.ERROR_TRANSITION, []),
        recovered_cases=findings.get(CaseTransition.RECOVERED, []),
        new_cases=pairing.new_cases,
        removed_cases=pairing.removed_cases,
        missing_case_results=_missing_case_results(
            input_data, input_data.baseline_cases, input_data.current_cases
        ),
        guardrail_comparisons=guardrail_comparisons,
        context=input_data.context,
    )


# -- configuration sanitization ----------------------------------------------------

#: Substrings that mark a configuration key as secret; matching keys are
#: stripped from any configuration data preserved in a comparison.
_FORBIDDEN_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "secret",
    "password",
    "credential",
    "private_key",
    "access_token",
    "auth_token",
    "bearer",
)


def sanitize_configuration(snapshot: dict) -> dict:
    """Recursively drop secret-looking keys from a configuration snapshot.

    Never raises on odd shapes: non-dict/list values are passed through;
    dict keys are matched case-insensitively against forbidden substrings.
    """
    return _sanitize_value(snapshot)  # type: ignore[arg-type,return-value]


def _sanitize_value(value: object) -> object:
    if isinstance(value, dict):
        cleaned: dict = {}
        for key, item in value.items():
            if isinstance(key, str) and any(
                part in key.lower() for part in _FORBIDDEN_KEY_PARTS
            ):
                continue
            cleaned[key] = _sanitize_value(item)
        return cleaned
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    return value


def _flatten_leaves(value: object, path: str = "") -> dict[str, object]:
    """Flatten nested dicts/lists into ``path.to.leaf`` scalar entries."""
    leaves: dict[str, object] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            leaves.update(_flatten_leaves(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            child = f"{path}[{index}]"
            leaves.update(_flatten_leaves(item, child))
    else:
        leaves[path] = value
    return leaves


def diff_configurations(
    baseline: dict, current: dict
) -> list[ConfigurationChange]:
    """Leaf-level differences between two sanitized snapshots (sorted)."""
    baseline_leaves = _flatten_leaves(sanitize_configuration(baseline))
    current_leaves = _flatten_leaves(sanitize_configuration(current))
    changes: list[ConfigurationChange] = []
    for path in sorted(set(baseline_leaves) | set(current_leaves)):
        if path not in baseline_leaves:
            changes.append(ConfigurationChange(path=path, baseline=None, current=current_leaves[path]))
        elif path not in current_leaves:
            changes.append(ConfigurationChange(path=path, baseline=baseline_leaves[path], current=None))
        elif baseline_leaves[path] != current_leaves[path]:
            changes.append(
                ConfigurationChange(
                    path=path,
                    baseline=baseline_leaves[path],
                    current=current_leaves[path],
                )
            )
    return changes
