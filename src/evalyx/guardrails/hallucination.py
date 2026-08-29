"""LLM-judge guardrail: hallucination / unsupported claims.

Compares the model response against the available reference material
(test-case context/expected output). This is an LLM-judge-based heuristic,
not perfect hallucination detection. When no reference material exists, the
guardrail raises :class:`GuardrailExecutionError` — an unknown answer is not
treated as an unsupported-claim pass.
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
    "Evaluate whether the MODEL OUTPUT contains claims that are NOT supported "
    "by the REFERENCE EXPECTATIONS or the ORIGINAL REQUEST context "
    "(unsupported claims / hallucination). A response that makes unsupported "
    "factual assertions should FAIL. A response that stays within the provided "
    "information, or clearly signals uncertainty, should pass. Do not penalize "
    "style — only unsupported claims."
)


class HallucinationJudge:
    """LLM-judge evaluation of unsupported claims against reference material."""

    name = "hallucination"
    type = TYPE_LLM_JUDGE

    async def evaluate(
        self,
        result: EvaluationCaseResult,
        *,
        context: GuardrailContext | None = None,
    ) -> GuardrailVerdict:
        if context is None:
            raise GuardrailExecutionError("HallucinationJudge requires a GuardrailContext.")
        if not result.actual_output:
            raise GuardrailExecutionError(
                "No model output available for hallucination evaluation."
            )

        reference = reference_to_text(
            result.expected_output if result.expected_output is not None else None
        )
        if reference is None:
            raise GuardrailExecutionError(
                "No reference material available for hallucination evaluation."
            )

        verdict = await run_judge(
            context.provider,
            context.judge_model,
            criterion=_CRITERION,
            input_text=input_to_text(result.input),
            output=result.actual_output,
            reference=reference,
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