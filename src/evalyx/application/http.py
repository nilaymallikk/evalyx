"""HTTP transport for application targets (MLGPT reference integration).

Drives an external AI application over HTTP — for Evalyx's reference demo,
MLGPT's FastAPI backend (``POST /v1/chat``). The transport is deliberately
contract-shaped but parameterized: base URL, path, and field names are
constructor arguments, so a future application with a slightly different
JSON contract needs a different *instance*, not new domain code.

Safety properties:

- The client sends only the prompt text and a fixed anonymous user id —
  never credentials.
- Errors are :class:`ApplicationInvocationError` with the HTTP status code
  and error kind only; response bodies are never embedded in exceptions or
  logs (a body could echo the prompt).
- Latency is measured with a monotonic clock around the whole invocation.
"""

import time
import uuid

import anyio
import httpx
import structlog

from evalyx.application.base import (
    ANONYMOUS_EVALUATION_USER_ID,
    ApplicationInvocationError,
    ApplicationResponse,
)

logger = structlog.get_logger(__name__)

#: MLGPT's chat path (documented in its README); the base URL is configured.
DEFAULT_CHAT_PATH = "/v1/chat"

#: Transient-failure retry policy for application invocations. Free-tier
#: model backends behind an application frequently rate-limit or return
#: transient 5xx (the reference MLGPT does not even log them); bounded
#: transport-level retries keep one hiccup from failing a case, mirroring
#: the Phase 4 provider retry philosophy.
MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = (1.0, 4.0)


class _TransientApplicationError(Exception):
    """Internal: a retryable failure wrapping the final typed error."""

    def __init__(self, error: ApplicationInvocationError) -> None:
        super().__init__(str(error))
        self.error = error


class HttpApplicationTarget:
    """Invoke an application's HTTP chat endpoint with one prompt."""

    def __init__(
        self,
        base_url: str,
        *,
        path: str = DEFAULT_CHAT_PATH,
        timeout_seconds: float = 60.0,
        user_id: str = ANONYMOUS_EVALUATION_USER_ID,
        question_field: str = "question",
        answer_field: str = "answer",
        client: httpx.AsyncClient | None = None,
        application_name: str = "application",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._path = path
        self._user_id = user_id
        self._question_field = question_field
        self._answer_field = answer_field
        self._application_name = application_name
        self._client = client or httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(connect=10.0, read=timeout_seconds, write=10.0, pool=10.0),
        )

    async def invoke(self, prompt: str) -> ApplicationResponse:
        """POST one prompt; return the normalized response.

        Each invocation uses a fresh conversation (no ``conversation_id``),
        so cases are independent — exactly what an evaluation dataset needs.
        Transient failures (HTTP 5xx, connect errors, read timeouts) are
        retried with bounded backoff; permanent failures (4xx, unexpected
        response shapes) are not.
        """
        payload = {
            self._question_field: prompt,
            "user_id": self._user_id or str(uuid.uuid4()),
            "conversation_id": None,
        }
        last_error: ApplicationInvocationError | None = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                return await self._invoke_once(payload)
            except _TransientApplicationError as exc:
                last_error = exc.error
                if attempt < MAX_ATTEMPTS - 1:
                    delay = RETRY_BACKOFF_SECONDS[min(attempt, len(RETRY_BACKOFF_SECONDS) - 1)]
                    logger.warning(
                        "application_invocation_retry",
                        application=self._application_name,
                        attempt=attempt + 1,
                        next_delay_seconds=delay,
                        reason=exc.error.__class__.__name__,
                    )
                    await anyio.sleep(delay)
        assert last_error is not None  # loop ran at least once
        raise last_error

    async def _invoke_once(self, payload: dict) -> ApplicationResponse:
        started = time.monotonic()
        try:
            response = await self._client.post(self._path, json=payload)
        except httpx.TimeoutException as exc:
            raise _TransientApplicationError(
                ApplicationInvocationError(
                    f"Application {self._application_name!r} timed out "
                    f"after {self._client.timeout.read} s."
                )
            ) from exc
        except httpx.TransportError as exc:
            raise _TransientApplicationError(
                ApplicationInvocationError(
                    f"Could not reach application {self._application_name!r}: "
                    f"{type(exc).__name__}."
                )
            ) from exc
        latency_ms = int((time.monotonic() - started) * 1000)

        if response.status_code >= 500:
            # Status code only — the body may echo the prompt or internals.
            raise _TransientApplicationError(
                ApplicationInvocationError(
                    f"Application {self._application_name!r} returned "
                    f"HTTP {response.status_code}."
                )
            )
        if response.status_code >= 400:
            raise ApplicationInvocationError(
                f"Application {self._application_name!r} returned "
                f"HTTP {response.status_code}."
            )

        try:
            body = response.json()
            content = body[self._answer_field]
        except (ValueError, KeyError, TypeError) as exc:
            raise ApplicationInvocationError(
                f"Application {self._application_name!r} returned an "
                "unexpected response shape."
            ) from exc
        if not isinstance(content, str) or not content.strip():
            raise ApplicationInvocationError(
                f"Application {self._application_name!r} returned an "
                "empty answer."
            )

        sources = body.get("sources")
        return ApplicationResponse(
            content=content,
            latency_ms=latency_ms,
            status_code=response.status_code,
            metadata={
                "application": self._application_name,
                # Bounded count only — never the raw source documents.
                "sources_count": len(sources) if isinstance(sources, list) else 0,
            },
        )

    async def close(self) -> None:
        """Release the underlying HTTP connection pool."""
        await self._client.aclose()
