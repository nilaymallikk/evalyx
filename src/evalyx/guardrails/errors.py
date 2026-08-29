"""Guardrail-level errors.

A ``GuardrailExecutionError`` means a guardrail could not produce a verdict
(provider failure, malformed judge output, missing reference material). It
is NEVER treated as a policy violation: a guardrail that cannot execute is
not evidence the response violated the policy.
"""


class GuardrailExecutionError(Exception):
    """A guardrail could not execute and has no verdict."""