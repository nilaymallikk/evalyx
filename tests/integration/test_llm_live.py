"""Optional live OpenRouter integration test.

Double-gated — requires BOTH:
  - EVALYX_RUN_INTEGRATION_TESTS=1   (conftest integration gate)
  - EVALYX_RUN_LLM_INTEGRATION_TESTS=1  (this module's gate)
plus a configured OPENROUTER_API_KEY. Skipped otherwise; never part of the
default test run.
"""

import os

import pytest

from evalyx.core.config import Settings
from evalyx.llm.factory import create_provider

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("EVALYX_RUN_LLM_INTEGRATION_TESTS") != "1",
        reason="live LLM test; set EVALYX_RUN_LLM_INTEGRATION_TESTS=1 to run",
    ),
]


async def test_openrouter_live_completion():
    settings = Settings()
    if not settings.openrouter_api_key.get_secret_value().strip():
        pytest.skip("OPENROUTER_API_KEY is not configured")

    provider = create_provider(settings)
    try:
        response = await provider.complete(
            "Reply with exactly the word: OK",
            model=settings.evalyx_agent_model,
            temperature=0.0,
            max_tokens=16,
        )
    finally:
        await provider.close()

    assert isinstance(response.content, str) and response.content
    assert response.model == settings.evalyx_agent_model
    assert response.latency_ms >= 0
    assert response.metadata["provider"] == "openrouter"
