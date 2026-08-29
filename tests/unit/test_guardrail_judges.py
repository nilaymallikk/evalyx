"""Unit tests for the three LLM-judge guardrails (fake provider, no network)."""

import uuid

import pytest

from evalyx.db.models import CaseStatus, EvaluationCaseResult
from evalyx.guardrails import (
    GuardrailContext,
    GuardrailExecutionError,
    HallucinationJudge,
    InstructionFollowingJudge,
    SafetyJudge,
)


def make_result(
    *,
    actual_output="A coherent answer.",
    input=None,
    expected_output=None,
):
    return EvaluationCaseResult(
        evaluation_run_id=uuid.uuid4(),
        test_case_id=uuid.uuid4(),
        input=input if input is not None else {"prompt": "Do the task."},
        actual_output=actual_output,
        expected_output=expected_output,
        status=CaseStatus.EXECUTED,
    )


def judge_context(provider, judge_model="minimax/minimax-m3:free") -> GuardrailContext:
    return GuardrailContext(provider=provider, judge_model=judge_model)


def pass_json(call):
    return '{"passed": true, "score": 0.95, "reason": "meets criterion"}'


@pytest.mark.asyncio
async def test_instruction_following_uses_judge_model_and_input(fake_provider):
    provider = fake_provider(response_builder=pass_json)
    result = make_result(actual_output="Definitely follows.", input="Do X in Y format.")
    verdict = await InstructionFollowingJudge().evaluate(
        result, context=judge_context(provider)
    )
    assert verdict.passed is True
    call = provider.calls[0]
    assert call["model"] == "minimax/minimax-m3:free"
    assert "Do X in Y format." in call["prompt"]
    assert "Definitely follows." in call["prompt"]


@pytest.mark.asyncio
async def test_instruction_following_without_context_errors(fake_provider):
    with pytest.raises(GuardrailExecutionError):
        await InstructionFollowingJudge().evaluate(make_result())


@pytest.mark.asyncio
async def test_instruction_following_expected_output_included_only_if_present(fake_provider):
    provider = fake_provider(response_builder=pass_json)
    result = make_result(
        expected_output={"mentions": "reset link"},
        actual_output="Use the reset link.",
    )
    await InstructionFollowingJudge().evaluate(result, context=judge_context(provider))
    assert "reset link" in provider.calls[0]["prompt"]

    provider2 = fake_provider(response_builder=pass_json)
    await InstructionFollowingJudge().evaluate(
        make_result(actual_output="hi"), context=judge_context(provider2)
    )
    assert "<reference_expectations>" not in provider2.calls[0]["prompt"]


@pytest.mark.asyncio
async def test_hallucination_requires_reference_material(fake_provider):
    provider = fake_provider(response_builder=pass_json)
    # No expected_output -> cannot evaluate groundedness -> execution error.
    with pytest.raises(GuardrailExecutionError, match="reference"):
        await HallucinationJudge().evaluate(
            make_result(actual_output="A wild unsupported claim."),
            context=judge_context(provider),
        )


@pytest.mark.asyncio
async def test_hallucination_with_reference_uses_provider(fake_provider):
    provider = fake_provider(
        response_builder=lambda call: '{"passed": false, "score": 0.3, "reason": "unsupported"}'
    )
    result = make_result(
        actual_output="Your order ships in 3 days.",
        expected_output={"shipping_window_days": 10},
    )
    verdict = await HallucinationJudge().evaluate(result, context=judge_context(provider))
    assert verdict.passed is False
    assert verdict.score == 0.3
    assert "shipping_window_days" in provider.calls[0]["prompt"]


@pytest.mark.asyncio
async def test_safety_judge_evaluates_output_only(fake_provider):
    provider = fake_provider(response_builder=pass_json)
    result = make_result(actual_output="This is a normal polite response.")
    verdict = await SafetyJudge().evaluate(result, context=judge_context(provider))
    assert verdict.passed is True
    call = provider.calls[0]
    # Safety evaluates the output only — no original request in the prompt.
    assert "<original_request>" not in call["prompt"]
    assert "This is a normal polite response." in call["prompt"]


@pytest.mark.asyncio
async def test_safety_judge_without_context_errors(fake_provider):
    with pytest.raises(GuardrailExecutionError):
        await SafetyJudge().evaluate(make_result())


@pytest.mark.asyncio
async def test_failed_verdict_does_not_become_pass(fake_provider):
    provider = fake_provider(
        response_builder=lambda call: '{"passed": false, "score": 0.1, "reason": "nope"}'
    )
    verdict = await SafetyJudge().evaluate(
        make_result(actual_output="You should harm them."),
        context=judge_context(provider),
    )
    assert verdict.passed is False