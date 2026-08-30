"""OpenRouter LLM provider (async httpx).

The request shape mirrors the proven structure of the root-level
``test_openrouter.py`` connectivity script, restructured into:
configuration (constructor) / transport (injected or owned AsyncClient) /
request construction / response parsing / error mapping.

Only the free OpenRouter models configured through Evalyx settings are used
by default; the provider itself accepts any OpenRouter model ID.
"""

import httpx
from pydantic import SecretStr

from evalyx.llm.base import (
    DEFAULT_TIMEOUT,
    LLMResponse,
    RetryPolicy,
    TokenUsage,
    send_with_retries,
)
from evalyx.llm.errors import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMRequestError,
    LLMResponseError,
    LLMServerError,
)

DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterProvider:
    """Async OpenRouter chat-completion provider."""

    def __init__(
        self,
        api_key: SecretStr,
        *,
        base_url: str = DEFAULT_OPENROUTER_BASE_URL,
        timeout: httpx.Timeout = DEFAULT_TIMEOUT,
        retry_policy: RetryPolicy = RetryPolicy(),
        client: httpx.AsyncClient | None = None,
    ) -> None:
        api_key_value = api_key.get_secret_value()
        if not api_key_value or not api_key_value.strip():
            raise LLMConfigurationError(
                "OPENROUTER_API_KEY is not configured. Set it in the environment or .env."
            )
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._retry_policy = retry_policy
        # The client may be injected (tests); otherwise the provider owns it.
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = client is None

    def __repr__(self) -> str:  # pragma: no cover - trivial
        # Never include the API key in the repr.
        return f"OpenRouterProvider(base_url={self._base_url!r})"

    def _auth_headers(self) -> dict[str, str]:
        """Construct request headers. The API key is only touched here."""
        return {
            "Authorization": f"Bearer {self._api_key.get_secret_value()}",
            "Content-Type": "application/json",
            "X-Title": "Evalyx",
        }

    async def complete(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
        system: str | None = None,
    ) -> LLMResponse:
        messages: list[dict[str, str]] = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        async def send() -> httpx.Response:
            return await self._client.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                headers=self._auth_headers(),
            )

        response, latency_ms = await send_with_retries(
            send, self._retry_policy, provider="openrouter", model=model
        )
        return self._interpret(response, requested_model=model, latency_ms=latency_ms)

    def _interpret(
        self, response: httpx.Response, *, requested_model: str, latency_ms: int
    ) -> LLMResponse:
        """Map HTTP status to typed errors and parse a successful payload."""
        if response.status_code >= 400:
            raise _error_for_status(response)

        try:
            data = response.json()
        except ValueError as exc:
            raise LLMResponseError("OpenRouter returned invalid JSON.") from exc

        return parse_chat_completion_response(
            data, requested_model=requested_model, latency_ms=latency_ms
        )

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()


def _error_for_status(response: httpx.Response) -> LLMProviderError:
    """Map an HTTP error response to a typed provider exception.

    Messages include only safe diagnostic information (status code plus a
    short body excerpt); the API key can never appear here.
    """
    status = response.status_code
    excerpt = response.text[:200] if response.text else ""

    if status in (401, 403):
        return LLMAuthenticationError(
            f"OpenRouter rejected the credentials (HTTP {status})."
        )
    if status == 429:
        retry_after: float | None = None
        raw = response.headers.get("Retry-After")
        if raw is not None:
            try:
                retry_after = float(raw)
            except ValueError:
                retry_after = None
        return LLMRateLimitError(
            "OpenRouter rate limit reached (HTTP 429).",
            retry_after_seconds=retry_after,
        )
    if status == 404:
        return LLMRequestError(
            "OpenRouter endpoint or model not found (HTTP 404).", response=response
        )
    if 400 <= status < 500:
        return LLMRequestError(
            f"OpenRouter rejected the request (HTTP {status}): {excerpt}",
            response=response,
        )
    if status >= 500:
        return LLMServerError(
            f"OpenRouter server error (HTTP {status}).", status_code=status
        )
    return LLMProviderError(f"Unexpected OpenRouter response (HTTP {status}).")


def parse_chat_completion_response(
    data: object,
    *,
    requested_model: str,
    latency_ms: int,
) -> LLMResponse:
    """Validate and convert an OpenRouter chat-completion payload.

    Never assumes ``choices[0].message.content`` exists; raises
    :class:`LLMResponseError` with safe diagnostics when malformed. Token
    usage is captured only when the provider supplies it.
    """
    if not isinstance(data, dict):
        raise LLMResponseError("OpenRouter response is not a JSON object.")

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMResponseError(
            f"OpenRouter response has no choices (keys={sorted(data.keys())})."
        )

    first = choices[0]
    if not isinstance(first, dict):
        raise LLMResponseError("OpenRouter response choice is not an object.")

    message = first.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise LLMResponseError(
            "OpenRouter response is missing message content "
            f"(choice keys={sorted(first.keys())})."
        )

    usage = None
    usage_data = data.get("usage")
    if isinstance(usage_data, dict):
        usage = _parse_usage(usage_data)

    finish_reason = first.get("finish_reason")
    if not isinstance(finish_reason, str):
        finish_reason = None

    model = data.get("model")
    return LLMResponse(
        content=content,
        model=model if isinstance(model, str) else requested_model,
        latency_ms=latency_ms,
        usage=usage,
        finish_reason=finish_reason,
        metadata={
            "provider": "openrouter",
            "response_id": data.get("id"),
            "provider_name": data.get("provider"),
        },
    )


def _parse_usage(usage_data: dict) -> TokenUsage:
    """Extract token usage; derive the total only when it is omitted."""
    prompt_tokens = usage_data.get("prompt_tokens")
    completion_tokens = usage_data.get("completion_tokens")
    total_tokens = usage_data.get("total_tokens")
    if not isinstance(prompt_tokens, int):
        prompt_tokens = None
    if not isinstance(completion_tokens, int):
        completion_tokens = None
    if not isinstance(total_tokens, int):
        if prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens
        else:
            total_tokens = None
    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )

