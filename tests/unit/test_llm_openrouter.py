"""Unit tests for OpenRouterProvider using mocked HTTP transport.

No network access is performed. API keys used here are dummies and are
asserted never to leak into exceptions or reprs.
"""

import json

import httpx
import pytest
from pydantic import SecretStr

from evalyx.llm.base import DEFAULT_TIMEOUT, RetryPolicy
from evalyx.llm.errors import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMRequestError,
    LLMResponseError,
    LLMServerError,
    LLMTimeoutError,
)
from evalyx.llm.openrouter import (
    OpenRouterProvider,
    parse_chat_completion_response,
)

API_KEY = SecretStr("test-key-not-a-real-secret")
SUCCESS_BODY = {
    "id": "resp-123",
    "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
    "choices": [
        {
            "message": {"role": "assistant", "content": "Hello from the model."},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    "provider": "Nvidia",
}


def make_provider(handler, **kwargs) -> tuple[OpenRouterProvider, httpx.AsyncClient]:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=DEFAULT_TIMEOUT)
    provider = OpenRouterProvider(
        API_KEY,
        retry_policy=RetryPolicy(max_retries=kwargs.pop("max_retries", 2), backoff_seconds=0),
        client=client,
        **kwargs,
    )
    return provider, client


async def test_success_response_is_parsed_completely():
    seen_headers: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.update(request.headers)
        return httpx.Response(200, json=SUCCESS_BODY)

    provider, client = make_provider(handler)
    try:
        response = await provider.complete(
            "Say hello", model="nvidia/nemotron-3-ultra-550b-a55b:free"
        )
    finally:
        await client.aclose()

    assert response.content == "Hello from the model."
    assert response.model == "nvidia/nemotron-3-ultra-550b-a55b:free"
    assert response.latency_ms >= 0
    assert response.usage is not None
    assert response.usage.prompt_tokens == 11
    assert response.usage.completion_tokens == 7
    assert response.usage.total_tokens == 18
    assert response.finish_reason == "stop"
    assert response.metadata["provider"] == "openrouter"
    assert response.metadata["response_id"] == "resp-123"
    # Authorization header is constructed at request time from the SecretStr.
    assert seen_headers["authorization"] == f"Bearer {API_KEY.get_secret_value()}"


async def test_system_prompt_and_options_are_sent():
    payloads: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(request.read())
        return httpx.Response(200, json=SUCCESS_BODY)

    provider, client = make_provider(handler)
    try:
        await provider.complete(
            "Evaluate this",
            model="minimax/minimax-m3:free",
            temperature=0.1,
            max_tokens=64,
            system="You are a strict judge.",
        )
    finally:
        await client.aclose()

    body = json.loads(payloads[0])
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert body["temperature"] == 0.1
    assert body["max_tokens"] == 64


async def test_usage_missing_yields_none_not_invented():
    body = {"choices": [{"message": {"content": "hi"}}]}
    response = parse_chat_completion_response(body, requested_model="m", latency_ms=1)
    assert response.usage is None
    assert response.finish_reason is None


async def test_total_tokens_derived_when_provider_omits_it():
    body = {
        "choices": [{"message": {"content": "hi"}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4},
    }
    response = parse_chat_completion_response(body, requested_model="m", latency_ms=1)
    assert response.usage is not None
    assert response.usage.total_tokens == 7


async def test_401_raises_authentication_error_without_retry():
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(401, json={"error": "invalid key"})

    provider, client = make_provider(handler, max_retries=3)
    try:
        with pytest.raises(LLMAuthenticationError):
            await provider.complete("hi", model="some-model")
    finally:
        await client.aclose()
    assert len(attempts) == 1  # authentication failures are never retried


async def test_400_raises_request_error_without_retry():
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(400, json={"error": "bad request"})

    provider, client = make_provider(handler, max_retries=3)
    try:
        with pytest.raises(LLMRequestError):
            await provider.complete("hi", model="some-model")
    finally:
        await client.aclose()
    assert len(attempts) == 1


async def test_404_raises_request_error_without_retry():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "no such model"})

    provider, client = make_provider(handler, max_retries=3)
    try:
        with pytest.raises(LLMRequestError):
            await provider.complete("hi", model="nope")
    finally:
        await client.aclose()


async def test_429_retries_then_raises_rate_limit_error():
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(429, json={"error": "rate limited"})

    provider, client = make_provider(handler, max_retries=2)
    try:
        with pytest.raises(LLMRateLimitError):
            await provider.complete("hi", model="some-model")
    finally:
        await client.aclose()
    assert len(attempts) == 3  # 1 initial + 2 bounded retries


async def test_429_recovers_on_later_attempt():
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) == 1:
            return httpx.Response(429, headers={"Retry-After": "0"}, json={})
        return httpx.Response(200, json=SUCCESS_BODY)

    provider, client = make_provider(handler, max_retries=2)
    try:
        response = await provider.complete("hi", model="some-model")
    finally:
        await client.aclose()
    assert len(attempts) == 2
    assert response.content == "Hello from the model."


async def test_500_retries_then_raises_server_error():
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(500, json={"error": "boom"})

    provider, client = make_provider(handler, max_retries=2)
    try:
        with pytest.raises(LLMServerError):
            await provider.complete("hi", model="some-model")
    finally:
        await client.aclose()
    assert len(attempts) == 3


async def test_503_recovers_on_later_attempt():
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) <= 2:
            return httpx.Response(503, json={})
        return httpx.Response(200, json=SUCCESS_BODY)

    provider, client = make_provider(handler, max_retries=3)
    try:
        response = await provider.complete("hi", model="some-model")
    finally:
        await client.aclose()
    assert len(attempts) == 3
    assert response.latency_ms >= 0


async def test_timeout_retries_then_raises_timeout_error():
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        raise httpx.ConnectTimeout("timed out")

    provider, client = make_provider(handler, max_retries=2)
    try:
        with pytest.raises(LLMTimeoutError):
            await provider.complete("hi", model="some-model")
    finally:
        await client.aclose()
    assert len(attempts) == 3


async def test_connection_error_retries_then_raises_connection_error():
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        raise httpx.ConnectError("connection refused")

    provider, client = make_provider(handler, max_retries=1)
    try:
        with pytest.raises(LLMConnectionError):
            await provider.complete("hi", model="some-model")
    finally:
        await client.aclose()
    assert len(attempts) == 2


@pytest.mark.parametrize(
    "body",
    [
        {"no_choices": True},  # missing choices
        {"choices": []},  # empty choices
        {"choices": ["not-an-object"]},  # choice not an object
        {"choices": [{"finish_reason": "stop"}]},  # missing message
        {"choices": [{"message": {}}]},  # missing content
        {"choices": [{"message": {"content": 42}}]},  # content not a string
        "just-a-string",  # not an object
    ],
)
async def test_malformed_payloads_raise_response_error_without_retry(body):
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(200, json=body)

    provider, client = make_provider(handler, max_retries=3)
    try:
        with pytest.raises(LLMResponseError):
            await provider.complete("hi", model="some-model")
    finally:
        await client.aclose()
    assert len(attempts) == 1  # malformed payloads are not retried


async def test_invalid_json_raises_response_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>not json</html>")

    provider, client = make_provider(handler, max_retries=2)
    try:
        with pytest.raises(LLMResponseError):
            await provider.complete("hi", model="some-model")
    finally:
        await client.aclose()


async def test_missing_api_key_raises_configuration_error():
    with pytest.raises(LLMConfigurationError):
        OpenRouterProvider(SecretStr(""))
    with pytest.raises(LLMConfigurationError):
        OpenRouterProvider(SecretStr("   "))


async def test_api_key_never_leaks_into_exceptions_or_repr():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "denied"})

    provider, client = make_provider(handler, max_retries=0)
    try:
        with pytest.raises(LLMAuthenticationError) as exc_info:
            await provider.complete("hi", model="some-model")
    finally:
        await client.aclose()

    assert API_KEY.get_secret_value() not in str(exc_info.value)
    assert API_KEY.get_secret_value() not in repr(exc_info.value)
    assert API_KEY.get_secret_value() not in repr(provider)



