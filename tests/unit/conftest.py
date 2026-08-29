"""Shared unit-test fixtures for the guardrail/judge tests (no network)."""

import json

import pytest

from evalyx.llm.base import LLMResponse, TokenUsage


class FakeProvider:
    """Deterministic fake LLMProvider; no network; records every call."""

    def __init__(self, response_builder=None, error=None) -> None:
        self.calls: list[dict] = []
        #: Optional callable(json_content) -> text to return as judge output.
        self._response_builder = response_builder
        self._error = error

    async def complete(self, prompt, *, model, temperature=0.0, max_tokens=512, system=None):
        self.calls.append(
            {
                "prompt": prompt,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "system": system,
            }
        )
        if self._error is not None:
            raise self._error
        content = (
            self._response_builder(self.calls[-1]) if self._response_builder else "{}"
        )
        return LLMResponse(
            content=content,
            model=model,
            latency_ms=5,
            usage=TokenUsage(prompt_tokens=4, completion_tokens=3, total_tokens=7),
            finish_reason="stop",
        )

    async def close(self) -> None:
        pass


@pytest.fixture
def fake_provider():
    return FakeProvider


def json_response(data: dict) -> str:
    """Render a dict as the JSON text the judge model would return."""
    return json.dumps(data)