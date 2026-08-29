"""Unit tests for provider selection via the factory."""

import pytest
from pydantic import SecretStr

from evalyx.core.config import Settings
from evalyx.llm.errors import LLMConfigurationError
from evalyx.llm.factory import create_provider
from evalyx.llm.ollama import OllamaProvider
from evalyx.llm.openrouter import OpenRouterProvider


def make_settings(provider: str) -> Settings:
    return Settings(
        _env_file=None,
        evalyx_secret_key="unit-test-secret",
        openrouter_api_key=SecretStr("factory-test-key"),
        llm_provider=provider,  # type: ignore[arg-type]
    )


def test_openrouter_provider_is_selected():
    provider = create_provider(make_settings("openrouter"))
    assert isinstance(provider, OpenRouterProvider)


def test_ollama_provider_is_selected():
    provider = create_provider(make_settings("ollama"))
    assert isinstance(provider, OllamaProvider)


def test_explicit_provider_argument_overrides_settings():
    provider = create_provider(make_settings("openrouter"), provider="ollama")
    assert isinstance(provider, OllamaProvider)


def test_unknown_provider_raises_configuration_error():
    with pytest.raises(LLMConfigurationError, match="Unknown LLM provider"):
        create_provider(make_settings("openrouter"), provider="azure_openai")  # type: ignore[arg-type]


def test_default_configuration_is_openrouter():
    settings = make_settings("openrouter")
    assert settings.llm_provider == "openrouter"
    # Free-model defaults remain intact (guard against paid-model drift).
    assert settings.evalyx_agent_model.endswith(":free")
    assert settings.evalyx_judge_model.endswith(":free")
