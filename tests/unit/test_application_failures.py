"""Typed failure-category integration with the Phase 12 taxonomy (Phase 15)."""

import pytest

from evalyx.application.base import ApplicationInvocationError
from evalyx.evaluation.failures import (
    FailureCategory,
    classify_application_error,
    classify_failure,
)


@pytest.mark.parametrize(
    "category,expected",
    [
        ("authentication", FailureCategory.AUTHENTICATION),
        ("timeout", FailureCategory.TIMEOUT),
        ("connection_error", FailureCategory.CONNECTION_ERROR),
        ("rate_limited", FailureCategory.RATE_LIMITED),
        ("application_http_error", FailureCategory.APPLICATION_HTTP_ERROR),
        ("application_response_invalid", FailureCategory.APPLICATION_RESPONSE_INVALID),
        ("malformed_response", FailureCategory.MALFORMED_RESPONSE),
        ("unknown", FailureCategory.UNKNOWN),
    ],
)
def test_typed_category_preferred(category, expected):
    exc = ApplicationInvocationError(
        f"Application 'app' returned HTTP 500. {category}", category=category
    )
    failure = classify_application_error(exc, attempts=3)
    assert failure.category is expected
    assert failure.attempts == 3


def test_authentication_is_not_retryable():
    exc = ApplicationInvocationError("HTTP 401.", category="authentication")
    failure = classify_failure(exc)
    assert failure.category is FailureCategory.AUTHENTICATION
    assert failure.retryable is False
    assert failure.http_status == 401


def test_malformed_response_reason_is_templated():
    exc = ApplicationInvocationError("bad json happened", category="malformed_response")
    failure = classify_failure(exc)
    assert failure.category is FailureCategory.MALFORMED_RESPONSE
    assert "bad json happened" not in failure.reason  # never echo messages
    assert failure.reason.startswith("Application under test")


def test_invalid_category_falls_back_to_message_parsing():
    exc = ApplicationInvocationError("Application 'app' returned HTTP 503.")
    failure = classify_failure(exc)
    assert failure.category is FailureCategory.APPLICATION_HTTP_ERROR


def test_legacy_mlgpt_error_unchanged():
    """The MLGPT target sets no category; legacy parsing must keep working."""
    exc = ApplicationInvocationError("Application 'mlgpt' returned HTTP 429.")
    failure = classify_failure(exc)
    assert failure.category is FailureCategory.RATE_LIMITED
    assert failure.retryable is True


def test_timeout_message_parsing_unchanged():
    exc = ApplicationInvocationError("Application 'mlgpt' timed out after 60 s.")
    failure = classify_failure(exc)
    assert failure.category is FailureCategory.TIMEOUT