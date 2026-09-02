"""Failure analysis: typed, deterministic classification of execution failures.

Phase 12 answers one question: *when a case errors, why?* The distinction
that matters everywhere downstream is:

- **Quality failure** — the application answered, but the answer was bad
  (guardrails/scoring; unchanged).
- **Execution failure** — no usable answer was produced. That is what this
  module classifies.

Classification is deterministic on exception types and HTTP status codes —
never an LLM, never body sniffing. When the evidence is ambiguous the result
is :attr:`FailureCategory.UNKNOWN` (no invented root causes).

Security: every produced reason string is safe by construction — the inputs
are exception types and status codes only, never response bodies, headers,
or credentials.
"""

import enum
import uuid

from pydantic import BaseModel

from evalyx.application.base import ApplicationInvocationError
from evalyx.llm.errors import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMProviderError,
    LLMRateLimitError,
    LLMRequestError,
    LLMResponseError,
    LLMServerError,
    LLMTimeoutError,
)


class FailureCategory(str, enum.Enum):
    """Execution-failure taxonomy. Kept small: only distinctions someone
    can act on (retry now, wait for quota, fix config, fix the app)."""

    PROVIDER_UNAVAILABLE = "provider_unavailable"
    RATE_LIMITED = "rate_limited"
    QUOTA_EXHAUSTED = "quota_exhausted"
    TIMEOUT = "timeout"
    CONNECTION_ERROR = "connection_error"
    MALFORMED_RESPONSE = "malformed_response"
    APPLICATION_HTTP_ERROR = "application_http_error"
    APPLICATION_RESPONSE_INVALID = "application_response_invalid"
    AUTHENTICATION = "authentication"
    UNKNOWN = "unknown"


#: OpenRouter signals daily free-tier exhaustion in the error message; the
#: safe (secret-free) marker is the rate-limit code name, not the body text.
_QUOTA_MARKER = "free-models-per-day"


def is_quota_exhaustion(rate_limit_message: str) -> bool:
    """True when a rate-limit error is actually daily quota exhaustion."""
    return _QUOTA_MARKER in rate_limit_message


def _retryable_for(category: FailureCategory) -> bool:
    """Conservative default (free models): only wait-and-retry categories
    are retryable. Quota exhaustion is NOT — retrying burns nothing but
    confirms the same 429."""
    return category in (
        FailureCategory.PROVIDER_UNAVAILABLE,
        FailureCategory.TIMEOUT,
        FailureCategory.CONNECTION_ERROR,
        FailureCategory.RATE_LIMITED,
    )


class ExecutionFailure(BaseModel):
    """Typed execution-failure record for one errored case.

    Stored inside the case result's ``metrics["failure"]`` (JSONB — no
    migration) and surfaced verbatim through the API.
    """

    category: FailureCategory
    reason: str
    retryable: bool
    http_status: int | None = None
    attempts: int | None = None

    def metric_dict(self) -> dict:
        data = self.model_dump(mode="json")
        return data

    @classmethod
    def unknown(cls, error: Exception) -> ExecutionFailure:
        return cls(
            category=FailureCategory.UNKNOWN,
            reason=f"{type(error).__name__}",
            retryable=False,
        )


def classify_provider_error(exc: LLMProviderError) -> ExecutionFailure:
    """Classify a typed provider exception deterministically."""
    if isinstance(exc, LLMServerError):
        return ExecutionFailure(
            category=FailureCategory.PROVIDER_UNAVAILABLE,
            reason=f"Provider returned HTTP {exc.status_code}.",
            retryable=True,
            http_status=exc.status_code,
        )
    if isinstance(exc, LLMRateLimitError):
        if is_quota_exhaustion(str(exc)):
            return ExecutionFailure(
                category=FailureCategory.QUOTA_EXHAUSTED,
                reason="Provider daily free-model quota exhausted (HTTP 429).",
                retryable=False,
                http_status=429,
            )
        retryable = exc.retryable
        return ExecutionFailure(
            category=FailureCategory.RATE_LIMITED,
            reason="Provider rate limit reached (HTTP 429).",
            retryable=retryable,
            http_status=429,
        )
    if isinstance(exc, LLMTimeoutError):
        return ExecutionFailure(
            category=FailureCategory.TIMEOUT,
            reason="Provider did not respond within the timeout.",
            retryable=True,
        )
    if isinstance(exc, LLMConnectionError):
        return ExecutionFailure(
            category=FailureCategory.CONNECTION_ERROR,
            reason="Provider could not be reached.",
            retryable=True,
        )
    if isinstance(exc, LLMAuthenticationError):
        return ExecutionFailure(
            category=FailureCategory.AUTHENTICATION,
            reason="Provider rejected the credentials.",
            retryable=False,
        )
    if isinstance(exc, LLMResponseError):
        return ExecutionFailure(
            category=FailureCategory.MALFORMED_RESPONSE,
            reason="Provider response was malformed or unusable.",
            retryable=False,
        )
    if isinstance(exc, LLMRequestError):
        status = exc.response.status_code if exc.response is not None else None
        return ExecutionFailure(
            category=FailureCategory.UNKNOWN,
            reason=f"Provider rejected the request (HTTP {status})."
            if status is not None
            else "Provider rejected the request.",
            retryable=False,
            http_status=status,
        )
    return ExecutionFailure(
        category=FailureCategory.UNKNOWN,
        reason=type(exc).__name__,
        retryable=False,
    )


def classify_application_error(
    exc: ApplicationInvocationError, *, attempts: int | None = None
) -> ExecutionFailure:
    """Classify an application-boundary failure.

    Only what is observable at the application boundary is recorded. The
    exception's message is safe by contract (status code and error kind
    only — never bodies), and the classification derives from the status
    code embedded there.
    """
    message = str(exc)
    http_status = _extract_status(message)
    category = _application_category(message, http_status)
    return ExecutionFailure(
        category=category,
        reason=_application_reason(category, message),
        retryable=_retryable_for(category),
        http_status=http_status,
        attempts=exc.attempts if attempts is None else attempts,
    )


def _extract_status(message: str) -> int | None:
    """Pull the HTTP status out of an ApplicationInvocationError message
    (they always embed ``HTTP <code>`` when one exists)."""
    marker = "HTTP "
    index = message.rfind(marker)
    if index == -1:
        return None
    digits = ""
    for char in message[index + len(marker) :]:
        if char.isdigit():
            digits += char
        else:
            break
    return int(digits) if digits else None


def _application_category(message: str, status: int | None) -> FailureCategory:
    if status is None:
        if "timed out" in message:
            return FailureCategory.TIMEOUT
        return FailureCategory.CONNECTION_ERROR  # unreachable / transport
    if status == 429:
        return FailureCategory.RATE_LIMITED
    if status >= 500:
        return FailureCategory.APPLICATION_HTTP_ERROR
    if status >= 400:
        return FailureCategory.APPLICATION_RESPONSE_INVALID
    return FailureCategory.UNKNOWN


def _application_reason(category: FailureCategory, message: str) -> str:
    """Classifier-authored reason templates only — never the exception
    message (a message could in principle carry echoed prompt content)."""
    if category is FailureCategory.APPLICATION_HTTP_ERROR:
        return "Application under test returned a server error (HTTP 500-class)."
    if category is FailureCategory.APPLICATION_RESPONSE_INVALID:
        return "Application under test rejected the request (HTTP 4xx)."
    if category is FailureCategory.TIMEOUT:
        return "Application under test did not respond within the timeout."
    if category is FailureCategory.CONNECTION_ERROR:
        return "Application under test could not be reached."
    if category is FailureCategory.RATE_LIMITED:
        return "Application under test reported rate limiting (HTTP 429)."
    return "Application invocation failed for an unclear reason."


def classify_failure(
    exc: Exception, *, attempts: int | None = None
) -> ExecutionFailure:
    """Classify any execution failure from its typed exception.

    Deterministic dispatch on exception type (never body sniffing, never an
    LLM); ambiguous evidence yields :attr:`FailureCategory.UNKNOWN`.
    """
    if isinstance(exc, ApplicationInvocationError):
        return classify_application_error(exc, attempts=attempts)
    if isinstance(exc, LLMProviderError):
        return classify_provider_error(exc)
    return ExecutionFailure.unknown(exc)


__all__ = [
    "ExecutionFailure",
    "FailureCategory",
    "classify_application_error",
    "classify_failure",
    "classify_provider_error",
    "is_quota_exhaustion",
    "uuid",
]
