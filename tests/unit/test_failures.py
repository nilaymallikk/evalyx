"""Unit tests for Phase 12 failure classification (deterministic, hermetic).

Sensitive values in these tests are fake by construction; no network.
"""

import pytest
from httpx import Response

from evalyx.application.base import ApplicationInvocationError
from evalyx.evaluation.failures import (
    ExecutionFailure,
    FailureCategory,
    classify_failure,
)
from evalyx.llm.errors import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMRequestError,
    LLMResponseError,
    LLMServerError,
    LLMTimeoutError,
)

# -- provider-error classification ---------------------------------------------------


def test_timeout_is_retryable():
    failure = classify_failure(LLMTimeoutError("t"))
    assert failure.category is FailureCategory.TIMEOUT
    assert failure.retryable is True
    assert failure.http_status is None


def test_connection_error_is_retryable():
    failure = classify_failure(LLMConnectionError("c"))
    assert failure.category is FailureCategory.CONNECTION_ERROR
    assert failure.retryable is True


@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_server_errors_map_to_provider_unavailable_with_status(status):
    failure = classify_failure(LLMServerError("s", status_code=status))
    assert failure.category is FailureCategory.PROVIDER_UNAVAILABLE
    assert failure.retryable is True
    assert failure.http_status == status


def test_rate_limit_is_retryable():
    failure = classify_failure(LLMRateLimitError("Rate limit reached."))
    assert failure.category is FailureCategory.RATE_LIMITED
    assert failure.retryable is True
    assert failure.http_status == 429


def test_daily_quota_exhaustion_is_not_retryable():
    """The Phase 11 incident shape: 429 whose message signals the daily
    free-model quota. Retrying cannot succeed → NOT retryable."""
    failure = classify_failure(
        LLMRateLimitError("Rate limit exceeded: free-models-per-day.")
    )
    assert failure.category is FailureCategory.QUOTA_EXHAUSTED
    assert failure.retryable is False
    assert failure.http_status == 429


def test_authentication_is_not_retryable():
    failure = classify_failure(LLMAuthenticationError("a"))
    assert failure.category is FailureCategory.AUTHENTICATION
    assert failure.retryable is False


def test_malformed_response_is_not_retryable():
    failure = classify_failure(LLMResponseError("bad payload"))
    assert failure.category is FailureCategory.MALFORMED_RESPONSE
    assert failure.retryable is False


def test_request_error_is_unknown_with_status():
    request = object()
    failure = classify_failure(
        LLMRequestError("rejected", response=Response(400, request=request))  # type: ignore[arg-type]
    )
    assert failure.category is FailureCategory.UNKNOWN
    assert failure.retryable is False
    assert failure.http_status == 400


def test_base_provider_error_is_unknown():
    from evalyx.llm.errors import LLMProviderError

    failure = classify_failure(LLMProviderError("opaque"))
    assert failure.category is FailureCategory.UNKNOWN


# -- application-error classification -------------------------------------------------


def test_application_http_500_is_application_http_error_with_attempts():
    failure = classify_failure(
        ApplicationInvocationError("Application 'mlgpt' returned HTTP 500.", attempts=3)
    )
    assert failure.category is FailureCategory.APPLICATION_HTTP_ERROR
    assert failure.http_status == 500
    assert failure.attempts == 3
    assert failure.retryable is False  # transport already exhausted its retries


def test_application_429_is_rate_limited_and_retryable():
    failure = classify_failure(
        ApplicationInvocationError("Application 'mlgpt' returned HTTP 429.")
    )
    assert failure.category is FailureCategory.RATE_LIMITED
    assert failure.retryable is True
    assert failure.http_status == 429


def test_application_4xx_is_response_invalid_and_not_retryable():
    failure = classify_failure(
        ApplicationInvocationError("Application 'mlgpt' returned HTTP 422.")
    )
    assert failure.category is FailureCategory.APPLICATION_RESPONSE_INVALID
    assert failure.retryable is False


def test_application_unreachable_is_connection_error():
    failure = classify_failure(
        ApplicationInvocationError("Could not reach application 'mlgpt': ConnectError.")
    )
    assert failure.category is FailureCategory.CONNECTION_ERROR
    assert failure.retryable is True


def test_application_timeout_is_timeout():
    failure = classify_failure(
        ApplicationInvocationError("Application 'mlgpt' timed out after 60 s.")
    )
    assert failure.category is FailureCategory.TIMEOUT
    assert failure.retryable is True


def test_application_attempts_default_to_none():
    failure = classify_failure(
        ApplicationInvocationError("Application 'x' returned HTTP 500.")
    )
    assert failure.attempts is None


# -- sanitization & boundaries ---------------------------------------------------------


@pytest.mark.parametrize(
    "exc",
    [
        ApplicationInvocationError("secret-abc-42 leaked in body"),
        LLMRateLimitError("quota hint: sk-live-should-never-appear"),
    ],
)
def test_reasons_never_contain_exception_message(exc):
    """Reason strings are classifier-authored templates, never bodies —
    a body could echo prompts, headers, or credentials."""
    failure = classify_failure(exc)
    assert "secret-abc-42" not in failure.reason
    assert "sk-live-should-never-appear" not in failure.reason


def test_arbitrary_exception_becomes_unknown_without_details():
    class WeirdError(Exception):
        pass

    failure = classify_failure(WeirdError("internal detail /home/user/secret"))
    assert failure.category is FailureCategory.UNKNOWN
    assert failure.retryable is False
    assert "internal detail" not in failure.reason
    assert failure.reason == "WeirdError"


def test_execution_failure_metric_dict_is_json_safe():
    data = ExecutionFailure(
        category=FailureCategory.TIMEOUT, reason="r", retryable=True
    ).metric_dict()
    assert data["category"] == "timeout"
    assert set(data) == {"category", "reason", "retryable", "http_status", "attempts"}
