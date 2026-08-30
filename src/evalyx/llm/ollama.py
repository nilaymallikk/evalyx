"""Ollama LLM provider (optional, local).

Implements the same :class:`evalyx.llm.base.LLMProvider` interface against
Ollama's native ``/api/chat`` endpoint (``stream: false``). Ollama is never
the default provider and is not required by the normal test suite; when it
is unreachable the provider raises a clear :class:`LLMConnectionError`.
"""

import httpx

from evalyx.llm.base import (
    DEFAULT_TIMEOUT,
    LLMResponse,
    RetryPolicy,
    TokenUsage,
    send_with_retries,
)
from evalyx.llm.errors import (
    LLMProviderError,
    LLMRateLimitError,
    LLMRequestError,
    LLMResponseError,
    LLMServerError,
)

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


class OllamaProvider:
    """Async Ollama chat provider (local, optional)."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        timeout: httpx.Timeout = DEFAULT_TIMEOUT,
        retry_policy: RetryPolicy = RetryPolicy(max_retries=1),
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._retry_policy = retry_policy
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = client is None

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"OllamaProvider(base_url={self._base_url!r})"

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
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        async def send() -> httpx.Response:
            return await self._client.post(
                f"{self._base_url}/api/chat",
                json=payload,
            )

        response, latency_ms = await send_with_retries(
            send, self._retry_policy, provider="ollama", model=model
        )

        if response.status_code >= 400:
            raise _error_for_status(response)

        try:
            data = response.json()
        except ValueError as exc:
            raise LLMResponseError("Ollama returned invalid JSON.") from exc

        return parse_ollama_chat_response(data, requested_model=model, latency_ms=latency_ms)

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()


def _error_for_status(response: httpx.Response) -> LLMProviderError:
    status = response.status_code
    excerpt = response.text[:200] if response.text else ""
    if status == 429:
        return LLMRateLimitError("Ollama rate limited the request (HTTP 429).")
    if status == 404:
        return LLMRequestError(
            f"Ollama model not found or endpoint missing (HTTP 404): {excerpt}",
            response=response,
        )
    if 400 <= status < 500:
        return LLMRequestError(
            f"Ollama rejected the request (HTTP {status}): {excerpt}",
            response=response,
        )
    if status >= 500:
        return LLMServerError(f"Ollama server error (HTTP {status}).", status_code=status)
    return LLMProviderError(f"Unexpected Ollama response (HTTP {status}).")


def parse_ollama_chat_response(
    data: object,
    *,
    requested_model: str,
    latency_ms: int,
) -> LLMResponse:
    """Validate and convert an Ollama ``/api/chat`` (non-streaming) payload."""
    if not isinstance(data, dict):
        raise LLMResponseError("Ollama response is not a JSON object.")

    message = data.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise LLMResponseError(
            f"Ollama response is missing message content (keys={sorted(data.keys())})."
        )

    usage = None
    prompt_tokens = data.get("prompt_eval_count")
    completion_tokens = data.get("eval_count")
    if isinstance(prompt_tokens, int) or isinstance(completion_tokens, int):
        total = None
        if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
            total = prompt_tokens + completion_tokens
        usage = TokenUsage(
            prompt_tokens=prompt_tokens if isinstance(prompt_tokens, int) else None,
            completion_tokens=(
                completion_tokens if isinstance(completion_tokens, int) else None
            ),
            total_tokens=total,
        )

    finish_reason = data.get("done_reason")
    if not isinstance(finish_reason, str):
        finish_reason = None

    model = data.get("model")
    return LLMResponse(
        content=content,
        model=model if isinstance(model, str) else requested_model,
        latency_ms=latency_ms,
        usage=usage,
        finish_reason=finish_reason,
        metadata={"provider": "ollama"},
    )
