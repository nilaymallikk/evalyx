"""Provider-neutral guardrail abstractions.

Evalyx guardrails depend only on:
- :class:`evalyx.llm.base.LLMProvider` (for LLM-judge checks)
- the Phase 3 domain models (``EvaluationCaseResult``)

They never import concrete providers (OpenRouter/Ollama).
"""

from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, Field

from evalyx.db.models import EvaluationCaseResult
from evalyx.llm.base import LLMProvider

#: Guardrail type taxonomy (kept deliberately small).
TYPE_DETERMINISTIC = "deterministic"
TYPE_LLM_JUDGE = "llm_judge"


class GuardrailVerdict(BaseModel):
    """Structured outcome of one guardrail check (not yet persisted)."""

    name: str
    type: str
    passed: bool
    score: float | None = None
    reason: str | None = None
    #: Non-sensitive diagnostic metadata (categories/pattern names/products).
    #: Never contains raw PII, full malicious prompts, API keys, or secrets.
    metadata: dict[str, Any] = Field(default_factory=dict)
    #: When set, the guardrail could not execute; ``passed`` carries no
    #: meaning. Distinct from a policy failure (passed=False).
    execution_error: str | None = None

    @property
    def is_error(self) -> bool:
        return self.execution_error is not None


@dataclass(frozen=True)
class GuardrailContext:
    """Shared inputs for semantic (LLM-judge) guardrails."""

    provider: LLMProvider
    judge_model: str
    temperature: float = 0.0
    max_tokens: int = 400


class Guardrail(Protocol):
    """Interface every Evalyx guardrail implements."""

    name: str
    type: str

    async def evaluate(
        self,
        result: EvaluationCaseResult,
        *,
        context: GuardrailContext | None = None,
    ) -> GuardrailVerdict:
        """Evaluate one case result and return a structured verdict.

        May raise :class:`GuardrailExecutionError` when the check cannot
        produce a verdict.
        """
        ...