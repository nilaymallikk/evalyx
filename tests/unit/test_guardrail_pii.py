"""Unit tests for the deterministic PII guardrail (no network)."""

import uuid

import pytest

from evalyx.db.models import CaseStatus, EvaluationCaseResult
from evalyx.guardrails.pii import PIIGuardrail

_RUN_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")
_CASE_ID = uuid.UUID("00000000-0000-0000-0000-000000000002")


def make_result(actual_output: str | None) -> EvaluationCaseResult:
    return EvaluationCaseResult(
        evaluation_run_id=_RUN_ID,
        test_case_id=_CASE_ID,
        input={"prompt": "hello"},
        actual_output=actual_output,
        status=CaseStatus.EXECUTED,
    )


@pytest.mark.asyncio
async def test_clean_output_passes():
    verdict = await PIIGuardrail().evaluate(make_result("Your order ships on Friday."))
    assert verdict.passed is True
    assert verdict.score == 0.0


@pytest.mark.asyncio
async def test_email_detected():
    verdict = await PIIGuardrail().evaluate(
        make_result("Contact me at alice@example.com for help.")
    )
    assert verdict.passed is False
    assert "email" in verdict.reason


@pytest.mark.asyncio
async def test_phone_detected():
    verdict = await PIIGuardrail().evaluate(
        make_result("Call us at (555) 123-4567 anytime.")
    )
    assert verdict.passed is False
    assert "phone" in verdict.reason


@pytest.mark.asyncio
async def test_ssn_like_detected():
    verdict = await PIIGuardrail().evaluate(
        make_result("Your reference is 123-45-6789.")
    )
    assert verdict.passed is False
    assert "ssn" in verdict.reason


@pytest.mark.asyncio
async def test_multiple_categories_detected():
    verdict = await PIIGuardrail().evaluate(
        make_result("Email alice@example.com or call 555-123-4567.")
    )
    assert verdict.passed is False
    assert set(verdict.metadata["categories"]) == {"email", "phone"}
    assert verdict.metadata["count"] == 2


@pytest.mark.asyncio
async def test_no_raw_pii_in_metadata_or_reason():
    output = "Email alice@example.com, call (555)123-4567, SSN 123-45-6789."
    verdict = await PIIGuardrail().evaluate(make_result(output))
    persisted_txt = str(verdict.metadata) + (verdict.reason or "")
    for secret in ("alice@example.com", "555-123-4567", "123-45-6789"):
        assert secret not in persisted_txt


@pytest.mark.asyncio
async def test_no_output_raises_execution_error():
    from evalyx.guardrails.errors import GuardrailExecutionError

    with pytest.raises(GuardrailExecutionError):
        await PIIGuardrail().evaluate(make_result(None))