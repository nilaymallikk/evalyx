"""Unit tests for scoring semantics (pure apply_policy, no DB)."""

import uuid

from evalyx.db.models import (
    CaseStatus,
    EvaluationCaseResult,
    GuardrailResult,
    GuardrailStatus,
)
from evalyx.evaluation.scoring import apply_policy
from evalyx.guardrails.policy import GuardrailPolicy, default_guardrail_policy


def make_case(status: CaseStatus = CaseStatus.EXECUTED) -> EvaluationCaseResult:
    return EvaluationCaseResult(
        id=uuid.uuid4(),
        evaluation_run_id=uuid.uuid4(),
        test_case_id=uuid.uuid4(),
        input={},
        actual_output="some output",
        status=status,
    )


def row(case, name: str, status: GuardrailStatus) -> GuardrailResult:
    return GuardrailResult(
        evaluation_case_result_id=case.id,
        name=name,
        type="deterministic",
        status=status,
        passed=status is GuardrailStatus.PASSED,
        score=1.0 if status is GuardrailStatus.PASSED else 0.0,
    )


POLICY = default_guardrail_policy()


def test_execution_error_remains_error():
    case = make_case(status=CaseStatus.ERROR)
    assert apply_policy(case, [], POLICY) is CaseStatus.ERROR
    assert apply_policy(case, [row(case, "pii", GuardrailStatus.PASSED)], POLICY) is CaseStatus.ERROR


def test_all_guardrails_pass_gives_passed():
    case = make_case()
    rows = [
        row(case, "pii", GuardrailStatus.PASSED),
        row(case, "safety", GuardrailStatus.PASSED),
        row(case, "hallucination", GuardrailStatus.PASSED),
        row(case, "prompt_injection", GuardrailStatus.PASSED),
        row(case, "instruction_following", GuardrailStatus.PASSED),
    ]
    assert apply_policy(case, rows, POLICY) is CaseStatus.PASSED


def test_critical_pii_failure_gives_failed():
    case = make_case()
    rows = [
        row(case, "pii", GuardrailStatus.FAILED),
        row(case, "safety", GuardrailStatus.PASSED),
    ]
    assert apply_policy(case, rows, POLICY) is CaseStatus.FAILED


def test_critical_safety_failure_gives_failed():
    case = make_case()
    rows = [
        row(case, "pii", GuardrailStatus.PASSED),
        row(case, "safety", GuardrailStatus.FAILED),
    ]
    assert apply_policy(case, rows, POLICY) is CaseStatus.FAILED


def test_critical_hallucination_failure_gives_failed():
    case = make_case()
    rows = [
        row(case, "pii", GuardrailStatus.PASSED),
        row(case, "hallucination", GuardrailStatus.FAILED),
    ]
    assert apply_policy(case, rows, POLICY) is CaseStatus.FAILED


def test_non_critical_failures_still_pass():
    """Documented policy: non-critical failures are indicators, not case-failures."""
    case = make_case()
    rows = [
        row(case, "pii", GuardrailStatus.PASSED),
        row(case, "safety", GuardrailStatus.PASSED),
        row(case, "prompt_injection", GuardrailStatus.FAILED),
        row(case, "instruction_following", GuardrailStatus.FAILED),
    ]
    assert apply_policy(case, rows, POLICY) is CaseStatus.PASSED


def test_judge_execution_error_keeps_case_executed():
    case = make_case()
    rows = [
        row(case, "pii", GuardrailStatus.PASSED),
        row(case, "safety", GuardrailStatus.ERROR),
    ]
    assert apply_policy(case, rows, POLICY) is CaseStatus.EXECUTED


def test_execution_error_with_critical_failure_stays_error():
    """An execution error is never 'upgraded' to a semantic failure."""
    case = make_case(status=CaseStatus.ERROR)
    rows = [row(case, "pii", GuardrailStatus.FAILED)]
    assert apply_policy(case, rows, POLICY) is CaseStatus.ERROR


def test_empty_guardrail_rows_keeps_case_executed():
    case = make_case()
    assert apply_policy(case, [], POLICY) is CaseStatus.EXECUTED


def test_default_policy_has_expected_criticality():
    assert POLICY.is_critical("pii")
    assert POLICY.is_critical("safety")
    assert POLICY.is_critical("hallucination")
    assert not POLICY.is_critical("prompt_injection")
    assert not POLICY.is_critical("instruction_following")


def test_custom_policy_can_promote_non_critical_to_critical():
    policy = GuardrailPolicy(
        enabled_guardrails=frozenset({"pii", "prompt_injection"}),
        critical_guardrails=frozenset({"pii", "prompt_injection"}),
    )
    case = make_case()
    rows = [row(case, "prompt_injection", GuardrailStatus.FAILED)]
    assert apply_policy(case, rows, policy) is CaseStatus.FAILED