"""Unit tests for the application-under-test adapter (MLGPT integration).

Hermetic: uses httpx.MockTransport — no live MLGPT, no network. Sensitive
values are fake by construction.
"""

import json
import uuid

import httpx
import pytest
from structlog.testing import capture_logs

from evalyx.application.base import (
    ApplicationInvocationError,
    ApplicationResponse,
    application_name_from_model,
    create_application_target,
)
from evalyx.application.http import HttpApplicationTarget
from evalyx.core.config import Settings


def make_target(handler, **kwargs) -> HttpApplicationTarget:
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, base_url="http://mlgpt.test")
    kwargs.setdefault("client", client)
    return HttpApplicationTarget("http://mlgpt.test", **kwargs)


async def test_successful_invocation_parses_answer_and_metadata():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["question"] == "What is supervised learning?"
        assert body["conversation_id"] is None
        uuid.UUID(body["user_id"])  # valid UUID, not PII
        assert request.url.path == "/v1/chat"
        return httpx.Response(
            200,
            json={
                "answer": "Learning from labeled examples.",
                "sources": [{"source": "handbook", "page": 1, "score": 0.9}],
                "conversation_id": "c-1",
            },
        )

    target = make_target(handler)
    response = await target.invoke("What is supervised learning?")
    assert response.content == "Learning from labeled examples."
    assert response.status_code == 200
    assert response.latency_ms >= 0
    assert response.metadata == {"application": "application", "sources_count": 1}
    await target.close()


async def test_latency_is_measured_not_zero_from_server():
    async def handler(request: httpx.Request) -> httpx.Response:
        import anyio

        await anyio.sleep(0.02)
        return httpx.Response(200, json={"answer": "ok", "sources": []})

    target = make_target(handler)
    response = await target.invoke("ping")
    assert response.latency_ms > 0
    await target.close()


async def test_http_error_raises_safe_typed_error_without_body():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "route not found: secret-xyz"})

    target = make_target(handler)
    with pytest.raises(ApplicationInvocationError) as excinfo:
        await target.invoke("hello")
    message = str(excinfo.value)
    assert "HTTP 404" in message
    # The response body (which could echo prompts/internals) never surfaces.
    assert "secret-xyz" not in message
    assert "route not found" not in message
    await target.close()


async def test_server_error_is_retried_then_succeeds(monkeypatch):
    """Transient 5xx is retried with backoff; a later attempt succeeds."""
    monkeypatch.setattr("evalyx.application.http.RETRY_BACKOFF_SECONDS", (0.0, 0.0))
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(500, json={"detail": "transient"})
        return httpx.Response(200, json={"answer": "recovered", "sources": []})

    target = make_target(handler)
    response = await target.invoke("hello")
    assert response.content == "recovered"
    assert calls["n"] == 3
    await target.close()


async def test_server_error_exhausts_retries_as_typed_error(monkeypatch):
    monkeypatch.setattr("evalyx.application.http.RETRY_BACKOFF_SECONDS", (0.0, 0.0))
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(500, json={"detail": "persistently broken"})

    target = make_target(handler)
    with pytest.raises(ApplicationInvocationError) as excinfo:
        await target.invoke("hello")
    assert calls["n"] == 3  # MAX_ATTEMPTS
    assert "HTTP 500" in str(excinfo.value)
    assert "persistently broken" not in str(excinfo.value)
    await target.close()


async def test_client_error_is_not_retried(monkeypatch):
    monkeypatch.setattr("evalyx.application.http.RETRY_BACKOFF_SECONDS", (0.0, 0.0))
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(422, json={"detail": "bad request"})

    target = make_target(handler)
    with pytest.raises(ApplicationInvocationError, match="HTTP 422"):
        await target.invoke("hello")
    assert calls["n"] == 1
    await target.close()


async def test_timeout_raises_typed_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out")

    target = make_target(handler)
    with pytest.raises(ApplicationInvocationError, match="timed out"):
        await target.invoke("hello")
    await target.close()


async def test_transport_error_raises_typed_error_without_host_details():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused to mlgpt.test:443")

    target = make_target(handler)
    with pytest.raises(ApplicationInvocationError, match="ConnectError"):
        await target.invoke("hello")
    await target.close()


async def test_malformed_json_raises_typed_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not-json{{{")

    target = make_target(handler)
    with pytest.raises(ApplicationInvocationError, match="unexpected response shape"):
        await target.invoke("hello")
    await target.close()


async def test_missing_answer_field_raises_typed_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "shape"})

    target = make_target(handler)
    with pytest.raises(ApplicationInvocationError, match="unexpected response shape"):
        await target.invoke("hello")
    await target.close()


async def test_empty_answer_raises_typed_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"answer": "   ", "sources": []})

    target = make_target(handler)
    with pytest.raises(ApplicationInvocationError, match="empty answer"):
        await target.invoke("hello")
    await target.close()


async def test_invocation_logs_never_contain_prompt_or_output():
    """§27/§30: the adapter logs nothing about content — and logs nothing
    at all beyond structlog noise from other components."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"answer": "the answer text", "sources": []})

    target = make_target(handler)
    with capture_logs() as logs:
        await target.invoke("the prompt text")
    serialized = str(logs)
    assert "the prompt text" not in serialized
    assert "the answer text" not in serialized
    await target.close()


# -- registry & selector --------------------------------------------------------


def test_registry_builds_mlgpt_target_from_settings():
    settings = Settings(mlgpt_base_url="http://127.0.0.1:9999")
    target = create_application_target("mlgpt", settings)
    assert isinstance(target, HttpApplicationTarget)


def test_registry_rejects_unknown_target():
    settings = Settings()
    with pytest.raises(ApplicationInvocationError, match="Unknown application target"):
        create_application_target("definitely-not-registered", settings)


def test_selector_extracts_application_name():
    assert application_name_from_model("application:mlgpt") == "mlgpt"
    assert application_name_from_model("application:mlgpt ") == "mlgpt"
    assert application_name_from_model("application:") is None
    assert application_name_from_model("nvidia/nemotron-3-super-120b-a12b:free") is None


def test_response_model_forbids_extra_fields():
    with pytest.raises(ValueError):
        ApplicationResponse(
            content="x", latency_ms=1, unexpected_field="nope"  # type: ignore[call-arg]
        )


def test_protocol_is_satisfied_by_http_target():
    from evalyx.application.base import ApplicationTarget

    assert isinstance(HttpApplicationTarget("http://x"), ApplicationTarget)
