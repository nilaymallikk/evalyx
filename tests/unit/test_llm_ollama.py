"""Unit tests for OllamaProvider using mocked HTTP transport (no server)."""

import httpx
import pytest

from evalyx.llm.base import DEFAULT_TIMEOUT, RetryPolicy
from evalyx.llm.errors import (
    LLMConnectionError,
    LLMRequestError,
    LLMResponseError,
)
from evalyx.llm.ollama import OllamaProvider, parse_ollama_chat_response

SUCCESS_BODY = {
    "model": "llama3.2:latest",
    "message": {"role": "assistant", "content": "Local model reply."},
    "done_reason": "stop",
    "prompt_eval_count": 9,
    "eval_count": 12,
}


def make_provider(handler, **kwargs) -> tuple[OllamaProvider, httpx.AsyncClient]:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=DEFAULT_TIMEOUT)
    provider = OllamaProvider(
        retry_policy=RetryPolicy(max_retries=kwargs.pop("max_retries", 1), backoff_seconds=0),
        client=client,
        **kwargs,
    )
    return provider, client


async def test_success_response_is_parsed():
    payloads: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(request.read())
        return httpx.Response(200, json=SUCCESS_BODY)

    provider, client = make_provider(handler)
    try:
        response = await provider.complete("hi", model="llama3.2:latest", temperature=0.5)
    finally:
        await client.aclose()

    assert response.content == "Local model reply."
    assert response.model == "llama3.2:latest"
    assert response.usage is not None
    assert response.usage.prompt_tokens == 9
    assert response.usage.completion_tokens == 12
    assert response.usage.total_tokens == 21
    assert response.finish_reason == "stop"
    assert response.metadata["provider"] == "ollama"

    import json

    body = json.loads(payloads[0])
    assert body["stream"] is False
    assert body["options"]["temperature"] == 0.5
    assert body["options"]["num_predict"] == 512


async def test_usage_fields_missing_yields_none():
    body = {"message": {"content": "hi"}}
    response = parse_ollama_chat_response(body, requested_model="m", latency_ms=1)
    assert response.usage is None


async def test_malformed_payloads_raise_response_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": True})

    provider, client = make_provider(handler, max_retries=2)
    try:
        with pytest.raises(LLMResponseError):
            await provider.complete("hi", model="llama3.2:latest")
    finally:
        await client.aclose()


async def test_model_not_found_raises_request_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text='{"error":"model not found"}')

    provider, client = make_provider(handler, max_retries=0)
    try:
        with pytest.raises(LLMRequestError):
            await provider.complete("hi", model="missing-model")
    finally:
        await client.aclose()


async def test_unreachable_ollama_raises_connection_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    provider, client = make_provider(handler, max_retries=0)
    try:
        with pytest.raises(LLMConnectionError, match="Could not reach LLM provider"):
            await provider.complete("hi", model="llama3.2:latest")
    finally:
        await client.aclose()
