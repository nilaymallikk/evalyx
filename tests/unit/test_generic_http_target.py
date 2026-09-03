"""Hermetic tests for the generic HTTP application target (Phase 15 Step 3).

Uses ``httpx.MockTransport`` — no network. The SSRF resolution check is
monkeypatched with a deterministic local validator so both allow and block
behavior is testable without DNS.
"""

import json
from urllib.parse import urlparse

import httpx
import pytest

import evalyx.application.generic_http as gh
from evalyx.application.base import ApplicationInvocationError
from evalyx.application.connection import ConnectionConfig
from evalyx.application.generic_http import HTTPApplicationTarget
from evalyx.application.ssrf import SSRFViolationError, assert_static_url_allowed
from evalyx.core.metrics import metrics

PUBLIC = "https://93.184.216.34/v1/chat"
SECRET = "sk-live-" + "not-a-real-credential"


@pytest.fixture(autouse=True)
def _fast_and_clean(monkeypatch):
    """No retry backoff sleeps; isolated metric registry per test."""
    monkeypatch.setattr(gh, "RETRY_BACKOFF_SECONDS", (0.0, 0.0, 0.0))
    metrics.reset()
    yield
    metrics.reset()


def _allowing_resolver(*, blocked_hosts: set[str] | None = None):
    """Deterministic stand-in for the DNS-based SSRF check."""
    blocked = blocked_hosts or set()

    async def resolve(url: str) -> None:
        assert_static_url_allowed(url)
        host = (urlparse(url).hostname or "").lower()
        if host in blocked:
            raise SSRFViolationError("blocked in test")

    return resolve


@pytest.fixture
def mock_client():
    """Build a target with a scripted MockTransport handler."""

    def _make(connection: ConnectionConfig, handler, **kwargs) -> HTTPApplicationTarget:
        resolver = kwargs.pop("resolver", _allowing_resolver())
        monkeypatch_target_resolver(resolver)
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            trust_env=False,
            follow_redirects=False,
        )
        return HTTPApplicationTarget(connection, client=client, **kwargs)

    return _make


def monkeypatch_target_resolver(resolver):
    if not hasattr(gh, "_original_resolver"):
        gh._original_resolver = gh.assert_url_resolves_public  # type: ignore[attr-defined]
    gh.assert_url_resolves_public = resolver  # type: ignore[method-assign]


@pytest.fixture(autouse=True)
def _restore_resolver():
    yield
    original = getattr(gh, "_original_resolver", None)
    if original is not None:
        gh.assert_url_resolves_public = original  # type: ignore[method-assign]


def _json_response(payload: dict, status_code: int = 200) -> httpx.Response:
    return httpx.Response(status_code, json=payload)


# -- success, mapping, and extraction ------------------------------------------


async def test_success_with_field_mapping(mock_client):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers.get("authorization")
        return _json_response({"answer": "supervised learning is..."})

    connection = ConnectionConfig(
        endpoint=PUBLIC, request={"input_field": "question"}
    )
    target = mock_client(connection, handler)
    response = await target.invoke("What is supervised learning?")
    assert response.content == "supervised learning is..."
    assert response.status_code == 200
    assert seen["body"] == {"question": "What is supervised learning?"}
    assert seen["auth"] is None  # no auth configured, no header sent
    await target.close()


async def test_nested_extraction_openai_style(mock_client):
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(
            {"choices": [{"message": {"content": "hello from app"}}]}
        )

    connection = ConnectionConfig(endpoint=PUBLIC, response_path="choices.0.message.content")
    target = mock_client(connection, handler)
    assert (await target.invoke("hi")).content == "hello from app"
    await target.close()


async def test_bearer_secret_sent_and_never_logged_or_raised(mock_client, caplog):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.headers.get("authorization") != f"Bearer {SECRET}":
            return httpx.Response(401)
        return _json_response({"answer": "ok"})

    connection = ConnectionConfig(endpoint=PUBLIC, auth={"type": "bearer"})
    target = mock_client(connection, handler, secret=SECRET)
    response = await target.invoke("hi")
    assert response.content == "ok"
    assert caplog.text == "" or SECRET not in caplog.text
    await target.close()


async def test_api_key_header_sent(mock_client):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["key"] = request.headers.get("x-api-key")
        return _json_response({"answer": "ok"})

    connection = ConnectionConfig(
        endpoint=PUBLIC, auth={"type": "api_key", "header_name": "X-API-Key"}
    )
    target = mock_client(connection, handler, secret=SECRET)
    await target.invoke("hi")
    assert seen["key"] == SECRET
    await target.close()


async def test_missing_secret_fails_fast_without_request(mock_client):
    connection = ConnectionConfig(endpoint=PUBLIC, auth={"type": "bearer"})
    with pytest.raises(ApplicationInvocationError) as excinfo:
        HTTPApplicationTarget(connection, client=httpx.AsyncClient(transport=httpx.MockTransport(lambda r: _json_response({}))))
    assert "credential" in str(excinfo.value)
    assert SECRET not in str(excinfo.value)


# -- failure classification and retry behavior ---------------------------------


@pytest.mark.parametrize(
    "status_code,category",
    [(401, "authentication"), (403, "authentication"), (429, "rate_limited")],
)
async def test_client_error_categories_not_retried(mock_client, status_code, category):
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(status_code)

    target = mock_client(ConnectionConfig(endpoint=PUBLIC), handler)
    with pytest.raises(ApplicationInvocationError) as excinfo:
        await target.invoke("hi")
    assert excinfo.value.category == category
    assert excinfo.value.attempts == 1  # never retried
    assert len(calls) == 1
    await target.close()


async def test_500_not_retried(mock_client):
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(500)

    target = mock_client(ConnectionConfig(endpoint=PUBLIC), handler)
    with pytest.raises(ApplicationInvocationError) as excinfo:
        await target.invoke("hi")
    assert excinfo.value.category == "application_http_error"
    assert len(calls) == 1  # 500 is not in the transient retry set
    await target.close()


async def test_502_then_success_retries(mock_client):
    responses = [httpx.Response(502), _json_response({"answer": "recovered"})]
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return responses.pop(0)

    target = mock_client(ConnectionConfig(endpoint=PUBLIC), handler)
    response = await target.invoke("hi")
    assert response.content == "recovered"
    assert len(calls) == 2
    await target.close()


async def test_retry_exhaustion_records_attempts(mock_client):
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(503)

    target = mock_client(ConnectionConfig(endpoint=PUBLIC), handler)
    with pytest.raises(ApplicationInvocationError) as excinfo:
        await target.invoke("hi")
    assert excinfo.value.category == "application_http_error"
    assert excinfo.value.attempts == 3  # default max_attempts
    assert len(calls) == 3
    await target.close()


async def test_timeout_is_transient_and_classified(mock_client):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out")

    target = mock_client(ConnectionConfig(endpoint=PUBLIC), handler)
    with pytest.raises(ApplicationInvocationError) as excinfo:
        await target.invoke("hi")
    assert excinfo.value.category == "timeout"
    assert excinfo.value.attempts == 3
    await target.close()


async def test_connection_error_classified(mock_client):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    target = mock_client(ConnectionConfig(endpoint=PUBLIC), handler)
    with pytest.raises(ApplicationInvocationError) as excinfo:
        await target.invoke("hi")
    assert excinfo.value.category == "connection_error"
    await target.close()


async def test_invalid_json_malformed_response(mock_client):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<not-json>")

    target = mock_client(ConnectionConfig(endpoint=PUBLIC), handler)
    with pytest.raises(ApplicationInvocationError) as excinfo:
        await target.invoke("hi")
    assert excinfo.value.category == "malformed_response"
    assert "<not-json>" not in str(excinfo.value)
    await target.close()


async def test_missing_answer_field_application_response_invalid(mock_client):
    target = mock_client(
        ConnectionConfig(endpoint=PUBLIC),
        lambda r: _json_response({"unexpected": "shape"}),
    )
    with pytest.raises(ApplicationInvocationError) as excinfo:
        await target.invoke("hi")
    assert excinfo.value.category == "application_response_invalid"
    await target.close()


async def test_error_messages_never_contain_url_or_secret(mock_client):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401)

    target = mock_client(
        ConnectionConfig(endpoint=PUBLIC, auth={"type": "bearer"}),
        handler,
        secret=SECRET,
    )
    with pytest.raises(ApplicationInvocationError) as excinfo:
        await target.invoke("hi")
    message = str(excinfo.value)
    assert PUBLIC not in message
    assert SECRET not in message
    await target.close()


# -- limits and SSRF at transport level ------------------------------------------


async def test_oversized_response_rejected(mock_client):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=b'{"answer":"' + b"x" * (gh.MAX_RESPONSE_BYTES + 1) + b'"}'
        )

    target = mock_client(ConnectionConfig(endpoint=PUBLIC), handler)
    with pytest.raises(ApplicationInvocationError) as excinfo:
        await target.invoke("hi")
    assert "size limit" in str(excinfo.value)
    await target.close()


async def test_redirect_to_public_followed(mock_client):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/redirect"):
            return httpx.Response(302, headers={"Location": f"{PUBLIC}/final"})
        return _json_response({"answer": "after redirect"})

    connection = ConnectionConfig(endpoint=PUBLIC + "/redirect")
    target = mock_client(connection, handler)
    assert (await target.invoke("hi")).content == "after redirect"
    await target.close()


async def test_redirect_to_private_blocked(mock_client):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            302, headers={"Location": "http://169.254.169.254/latest/meta-data"}
        )

    target = mock_client(ConnectionConfig(endpoint=PUBLIC), handler)
    with pytest.raises(ApplicationInvocationError) as excinfo:
        await target.invoke("hi")
    assert "SSRF" in str(excinfo.value)
    assert excinfo.value.category == "unknown"  # non-retryable
    await target.close()


async def test_redirect_rebinding_blocked(mock_client):
    """A redirect whose destination resolves privately is blocked."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"Location": "https://93.184.216.34/x"})

    async def rebinding_resolver(url: str) -> None:
        raise SSRFViolationError("resolved to private address")

    target = mock_client(
        ConnectionConfig(endpoint=PUBLIC), handler, resolver=rebinding_resolver
    )
    with pytest.raises(ApplicationInvocationError) as excinfo:
        await target.invoke("hi")
    assert "SSRF" in str(excinfo.value)
    await target.close()


async def test_redirect_limit_enforced(mock_client):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"Location": f"{PUBLIC}/loop"})

    connection = ConnectionConfig(endpoint=PUBLIC + "/loop")
    target = mock_client(connection, handler)
    with pytest.raises(ApplicationInvocationError) as excinfo:
        await target.invoke("hi")
    assert "redirect" in str(excinfo.value)
    await target.close()


async def test_ssrf_blocked_initial_destination(mock_client):
    blocked = _allowing_resolver(blocked_hosts={"93.184.216.34"})
    target = mock_client(
        ConnectionConfig(endpoint=PUBLIC),
        lambda r: _json_response({"answer": "never"}),
        resolver=blocked,
    )
    with pytest.raises(ApplicationInvocationError) as excinfo:
        await target.invoke("hi")
    assert "SSRF" in str(excinfo.value)
    await target.close()


# -- observability ----------------------------------------------------------------


async def test_metrics_recorded_with_bounded_labels(mock_client):
    target = mock_client(
        ConnectionConfig(endpoint=PUBLIC), lambda r: _json_response({"answer": "ok"})
    )
    await target.invoke("hi")
    snapshot = metrics.snapshot()
    assert "application_requests_total" in snapshot
    labels = snapshot["application_requests_total"][0]["labels"]
    assert labels["target"] == "http"
    assert labels["outcome"] == "success"
    assert "application_id" not in labels  # cardinality policy holds
    await target.close()