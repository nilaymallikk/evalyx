"""Unit tests for the deterministic input-to-prompt mapping (no DB, no I/O)."""

from evalyx.db.models import TestCase as TestCaseModel
from evalyx.evaluation.prompts import build_prompt


def make_case(**overrides) -> TestCaseModel:
    defaults = {
        "dataset_version_id": None,
        "name": "case",
        "input": "What is 2 + 2?",
    }
    fields = {**defaults, **overrides}
    return TestCaseModel(**fields)


def test_string_input_used_verbatim():
    assert build_prompt(make_case(input="What is 2 + 2?")) == "What is 2 + 2?"


def test_dict_input_with_prompt_key_uses_prompt_string():
    case = make_case(input={"prompt": "Summarize this", "case_type": "summary"})
    prompt = build_prompt(case)
    assert prompt == "Summarize this"
    assert "case_type" not in prompt


def test_structured_input_without_prompt_key_is_serialized_deterministically():
    case = make_case(input={"b": 2, "a": 1})
    prompt = build_prompt(case)
    assert prompt == '{"a": 1, "b": 2}'


def test_context_is_included_clearly_before_request():
    case = make_case(
        input="Answer the request.",
        context={"persona": "support-agent"},
    )
    prompt = build_prompt(case)
    assert prompt.startswith("Context:\n")
    assert '"persona": "support-agent"' in prompt
    assert "Request:\nAnswer the request." in prompt


def test_empty_context_is_omitted():
    case = make_case(input="hi", context={})
    assert build_prompt(case) == "hi"


def test_expected_output_is_never_sent_to_the_model():
    case = make_case(
        input="What is the refund policy?",
        expected_output={"must_contain": "30 days"},
    )
    prompt = build_prompt(case)
    assert "30 days" not in prompt
    assert "must_contain" not in prompt
