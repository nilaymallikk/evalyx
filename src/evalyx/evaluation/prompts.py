"""Deterministic input-to-prompt mapping for evaluation cases.

Phase 5 rules (intentionally simple and explicit):

- ``TestCase.input`` as a plain string is used as the prompt verbatim.
- A dict input containing a string ``"prompt"`` key uses that string as the
  core request; all other keys in the input are ignored (they may carry case
  metadata for future scoring).
- Any other JSON value is serialized deterministically (sorted keys,
  compact separators).
- ``TestCase.context``, when a non-empty dict, is rendered as a clearly
  labeled JSON block ("Context:") before the request.
- ``TestCase.expected_output`` is NEVER sent to the model; it is reserved
  for future scoring (Phase 6).
"""

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evalyx.db.models import TestCase


def build_prompt(test_case: TestCase) -> str:
    """Build the user prompt sent to the provider for one test case."""
    core = _core_request(test_case.input)
    context = test_case.context
    if context:
        return f"Context:\n{_to_json(context)}\n\nRequest:\n{core}"
    return core


def _core_request(test_case_input: object) -> str:
    if isinstance(test_case_input, str):
        return test_case_input
    if isinstance(test_case_input, dict) and isinstance(test_case_input.get("prompt"), str):
        return test_case_input["prompt"]
    return _to_json(test_case_input)


def _to_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(", ", ": "))
