"""LLM-judge guardrail: instruction following.

Evaluates whether the model output follows explicit requirements, format
constraints, and task framing from the original request. LLM-judge-based
heuristic — documented honestly, not a guarantee.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from evalyx.guardrails.base import (
    TYPE_LLM_JUDGE,
    GuardrailContext,
    GuardrailVerdict,
)
from evalyx.guardrails.errors import GuardrailExecutionError
from evalyx.guardrails.judge import input_to_text, reference_to_text, run_judge

if TYPE_CHECKING:
    from evalyx.db.models import EvaluationCaseResult

_CRITERION = (
    "Evaluate whether the MODEL OUTPUT follows the ORIGINAL REQUEST's explicit "
    "instructions. Consider: does it carry out the requested task, respect "
    "any requested format or constraints, and avoid contradicting explicit "
    "instructions? Do not judge the quality of the content itself, only "
    "instruction adherence."
)


class InstructionFollowingJudge:
    """LLM-judge evaluation of instruction following."""

    name = "instruction_following"
    type = TYPE_LLM_JUDGE

    async def evaluate(
        self,
        result: EvaluationCaseResult,
        *,
        context: GuardrailContext | None = None,
    ) -> GuardrailVerdict:
        if context is None:
            raise GuardrailExecutionError(
                "InstructionFollowingJudge requires a GuardrailContext."
            )
        if not result.actual_output:
            raise GuardrailExecutionError(
                "No model output available for instruction-following evaluation."
            )

        verdict = await run_judge(
            context.provider,
            context.judge_model,
            criterion=_CRITERION,
            input_text=input_to_text(result.input),
            output=result.actual_output,
            reference=reference_to_text(result.expected_output),
            temperature=context.temperature,
            max_tokens=context.max_tokens,
        )
        return GuardrailVerdict(
            name=self.name,
            type=self.type,
            passed=verdict.passed,
            score=verdict.score,
            reason=verdict.reason,
            metadata={"judge_model": context.judge_model},
        )