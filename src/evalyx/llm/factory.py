"""Provider selection: configuration -> provider name -> LLMProvider instance.

Centralizes provider construction so the rest of Evalyx never branches on
provider names. Unknown providers raise a clear configuration error; there
is no silent fallback. (Azure OpenAI is a future provider and intentionally
absent.)
"""

from typing import Literal

from evalyx.core.config import Settings
from evalyx.llm.base import LLMProvider, RetryPolicy
from evalyx.llm.errors import LLMConfigurationError
from evalyx.llm.ollama import OllamaProvider
from evalyx.llm.openrouter import OpenRouterProvider

ProviderName = Literal["openrouter", "ollama"]


def create_provider(
    settings: Settings,
    *,
    provider: ProviderName | None = None,
    retry_policy: RetryPolicy | None = None,
) -> LLMProvider:
    """Create the configured LLM provider from application settings."""
    name = provider or settings.llm_provider
    if retry_policy is None:
        retry_policy = RetryPolicy()

    if name == "openrouter":
        return OpenRouterProvider(
            settings.openrouter_api_key,
            retry_policy=retry_policy,
        )
    if name == "ollama":
        return OllamaProvider(
            base_url=settings.ollama_base_url,
            retry_policy=retry_policy,
        )

    raise LLMConfigurationError(
        f"Unknown LLM provider {name!r}. Supported providers: openrouter, ollama."
    )
