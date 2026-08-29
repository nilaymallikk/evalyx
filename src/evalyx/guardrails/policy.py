"""Guardrail policy: which guardrails run and which failures are critical.

The default MVP policy runs all five guardrails (deterministic first, then
judge checks) so the result keeps complete diagnostic information — no
short-circuiting.

Criticality:
- ``pii``, ``safety``, ``hallucination`` are critical — a failure fails the
  case.
- ``prompt_injection`` and ``instruction_following`` are non-critical
  diagnostic indicators — their failures are persisted as guardrail
  failures but do not individually flip the case.

The policy is config-driven; a stricter policy may promote the
non-critical names to critical.
"""

from dataclasses import dataclass, field

GUARDRAIL_PII = "pii"
GUARDRAIL_PROMPT_INJECTION = "prompt_injection"
GUARDRAIL_INSTRUCTION_FOLLOWING = "instruction_following"
GUARDRAIL_HALLUCINATION = "hallucination"
GUARDRAIL_SAFETY = "safety"

DEFAULT_GUARDRAILS = (
    GUARDRAIL_PII,
    GUARDRAIL_PROMPT_INJECTION,
    GUARDRAIL_INSTRUCTION_FOLLOWING,
    GUARDRAIL_HALLUCINATION,
    GUARDRAIL_SAFETY,
)

#: Deterministic checks always run before judge checks (cheap, no LLM calls).
DETERMINISTIC_NAMES = frozenset({GUARDRAIL_PII, GUARDRAIL_PROMPT_INJECTION})


@dataclass(frozen=True)
class GuardrailPolicy:
    """Selection and criticality for guardrails in a scoring run."""

    enabled_guardrails: frozenset[str] = field(
        default_factory=lambda: frozenset(DEFAULT_GUARDRAILS)
    )
    critical_guardrails: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {GUARDRAIL_PII, GUARDRAIL_SAFETY, GUARDRAIL_HALLUCINATION}
        )
    )

    def is_enabled(self, name: str) -> bool:
        return name in self.enabled_guardrails

    def is_critical(self, name: str) -> bool:
        return name in self.critical_guardrails


def default_guardrail_policy() -> GuardrailPolicy:
    return GuardrailPolicy()