"""Provider-level exception hierarchy for LLM calls.

The hierarchy lets future phases (evaluation engine, workers) distinguish
retryable/transient failures from permanent configuration problems without
knowing provider internals. Exception messages never include API keys.
"""

from httpx import Response


class LLMProviderError(Exception):
    """Base class for all LLM provider errors."""

    retryable: bool = False


class LLMConfigurationError(LLMProviderError):
    """Missing/invalid provider configuration (e.g. no API key).

    Non-retryable: requires a configuration change, not another attempt.
    """


class LLMAuthenticationError(LLMProviderError):
    """The provider rejected the credentials (HTTP 401/403). Non-retryable."""


class LLMRequestError(LLMProviderError):
    """The request itself was rejected (HTTP 400/404/422, unknown model...).

    Non-retryable: the same request will fail again.
    """

    def __init__(self, message: str, *, response: Response | None = None) -> None:
        super().__init__(message)
        self.response = response


class LLMRateLimitError(LLMProviderError):
    """The provider signaled rate limiting (HTTP 429). Retryable."""

    retryable = True

    def __init__(self, message: str, *, retry_after_seconds: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class LLMTimeoutError(LLMProviderError):
    """The provider did not respond within the configured timeout. Retryable."""

    retryable = True


class LLMConnectionError(LLMProviderError):
    """The provider could not be reached (DNS/network/connection errors). Retryable."""

    retryable = True


class LLMServerError(LLMProviderError):
    """The provider failed server-side (HTTP 5xx). Retryable."""

    retryable = True

    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class LLMResponseError(LLMProviderError):
    """The provider responded but the payload was malformed/unusable.

    Non-retryable by default: an identical request would likely return an
    identical malformed payload.
    """
