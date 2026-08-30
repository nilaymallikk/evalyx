"""Provider-neutral LLM abstraction.

The rest of Evalyx depends on :class:`LLMProvider` (a typing.Protocol) and
:class:`LLMResponse` — never on OpenRouter/Ollama specifics. Concrete
providers live in sibling modules and are selected via
:mod:`evalyx.llm.factory`.

This module also contains the shared bounded-retry logic used by HTTP-based
providers: only plausibly transient failures (network errors, timeouts, 429,
selected 5xx) are retried, with exponential backoff capped by policy and
honoring ``Retry-After`` when the provider supplies one.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
import structlog
from pydantic import BaseModel, Field

from evalyx.llm.errors import (
    LLMConnectionError,
    LLMRateLimitError,
    LLMTimeoutError,
)

logger = structlog.get_logger(__name__)

#: Explicit HTTP timeouts; an LLM call must never hang indefinitely.
DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)

#: Status codes that are plausibly transient and safe to retry.
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


class TokenUsage(BaseModel):
    """Token usage reported by the provider, when available."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class LLMResponse(BaseModel):
    """Typed, provider-neutral result of one LLM completion."""

    content: str
    model: str
    #: Wall-clock duration of the provider call, measured with a monotonic clock.
    latency_ms: int
    usage: TokenUsage | None = None
    finish_reason: str | None = None
    #: Non-critical provider-specific information (never secrets, never raw HTTP).
    metadata: dict[str, Any] = Field(default_factory=dict)


class LLMProvider(Protocol):
    """Interface every Evalyx LLM provider implements."""

    async def complete(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
        system: str | None = None,
    ) -> LLMResponse:
        """Execute one completion and return a typed response."""
        ...

    async def close(self) -> None:
        """Release underlying HTTP resources."""
        ...


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded retry configuration for transient provider failures."""

    max_retries: int = 2
    backoff_seconds: float = 1.0
    max_backoff_seconds: float = 30.0


def _retry_delay(
    policy: RetryPolicy,
    attempt: int,
    retry_after_seconds: float | None = None,
) -> float:
    """Compute the backoff delay for a retry attempt (bounded, no storms)."""
    if retry_after_seconds is not None:
        return min(retry_after_seconds, policy.max_backoff_seconds)
    return min(policy.backoff_seconds * (2**attempt), policy.max_backoff_seconds)


async def send_with_retries(
    send: Callable[[], Awaitable[httpx.Response]],
    policy: RetryPolicy,
    *,
    provider: str | None = None,
    model: str | None = None,
) -> tuple[httpx.Response, int]:
    """Execute ``send`` with bounded retries for transient failures.

    Returns ``(response, latency_ms)`` where latency is measured with a
    monotonic clock around the final (successful or returned) attempt.

    Raises :class:`LLMTimeoutError` or :class:`LLMConnectionError` when the
    final attempt fails at the network level. All other outcomes are
    returned to the caller for status-code mapping and parsing.

    Each deferred retry emits a structured ``provider_retry_scheduled``
    event with safe fields only (provider, model, attempt, error type,
    bounded delay) — never the provider response body, prompt, or output.
    """
    last_latency_ms = 0
    for attempt in range(policy.max_retries + 1):
        start = time.monotonic()
        try:
            response = await send()
        except httpx.TimeoutException as exc:
            if attempt >= policy.max_retries:
                raise LLMTimeoutError(f"LLM request timed out after retries: {exc}") from exc
            _log_provider_retry(provider, model, attempt, policy, "timeout")
            await asyncio.sleep(_retry_delay(policy, attempt))
            continue
        except httpx.TransportError as exc:
            if attempt >= policy.max_retries:
                raise LLMConnectionError(f"Could not reach LLM provider: {exc}") from exc
            _log_provider_retry(provider, model, attempt, policy, type(exc).__name__)
            await asyncio.sleep(_retry_delay(policy, attempt))
            continue

        last_latency_ms = int((time.monotonic() - start) * 1000)

        if response.status_code in RETRYABLE_STATUS_CODES and attempt < policy.max_retries:
            retry_after: float | None = None
            if response.status_code == 429:
                raw = response.headers.get("Retry-After")
                if raw is not None:
                    try:
                        retry_after = float(raw)
                    except ValueError:
                        retry_after = None
            _log_provider_retry(
                provider, model, attempt, policy, f"http_{response.status_code}", retry_after
            )
            await asyncio.sleep(_retry_delay(policy, attempt, retry_after))
            continue

        return response, last_latency_ms

    # Unreachable: the loop either returns or raises.
    raise AssertionError("send_with_retries exhausted retries without outcome")


def _log_provider_retry(
    provider: str | None,
    model: str | None,
    attempt: int,
    policy: RetryPolicy,
    error_type: str,
    retry_after_seconds: float | None = None,
) -> None:
    """Emit a safe structured retry event (attempt zero-based)."""
    logger.warning(
        "provider_retry_scheduled",
        provider=provider,
        model=model,
        attempt=attempt + 1,
        max_attempts=policy.max_retries + 1,
        error_type=error_type,
        retry_after_seconds=retry_after_seconds,
    )


__all__ = [
    "DEFAULT_TIMEOUT",
    "LLMProvider",
    "LLMResponse",
    "RETRYABLE_STATUS_CODES",
    "RetryPolicy",
    "TokenUsage",
    "send_with_retries",
]
