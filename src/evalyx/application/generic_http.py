"""Generic HTTP application target (Phase 15).

Drives an arbitrary user-registered AI application over HTTP using its
immutable connection configuration — the production-quality counterpart to
the MLGPT-specific :class:`evalyx.application.http.HttpApplicationTarget`
(which stays unchanged for the reference demo). Implements the existing
:class:`evalyx.application.base.ApplicationTarget` protocol, so the
evaluation engine cannot tell the two apart — it only calls ``invoke``.

Safety properties (all bounded, all secret-free):

- **SSRF**: every destination (including every redirect hop) is validated
  by :mod:`evalyx.application.ssrf` before the request; proxies from the
  environment are ignored; redirects are followed manually, never by the
  HTTP client.
- **HTTP**: bounded connect/read timeouts, a hard response-size cap, and
  bounded retries with exponential backoff — retrying only transient
  failures (connect errors, timeouts, HTTP 502/503/504).
- **Secrets**: the credential is injected into one header per request and
  is never logged, never embedded in an exception, and never echoed in a
  response body. Error messages carry the status code and failure kind
  only.
- **Observability**: bounded-label metrics only (taxonomy categories,
  outcomes) — no URLs, no application ids, no payloads.
"""

import json
import time

import anyio
import httpx
import structlog

from evalyx.application.base import ApplicationInvocationError, ApplicationResponse
from evalyx.application.connection import (
    ConnectionConfig,
    ResponseExtractionError,
    build_request_body,
    extract_answer,
)
from evalyx.application.ssrf import (
    SSRFViolationError,
    assert_url_resolves_public,
    is_redirect,
)
from evalyx.core.metrics import metrics

logger = structlog.get_logger(__name__)

#: Hard cap on the response body read from an external application.
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
#: Maximum manual redirect hops (each re-validated against SSRF rules).
MAX_REDIRECTS = 3
#: Backoff schedule between transient-failure retries (seconds, capped).
RETRY_BACKOFF_SECONDS = (1.0, 2.0, 4.0)
#: Connect timeout is deliberately tighter than the read timeout.
CONNECT_TIMEOUT_SECONDS = 10.0

#: Retryable server statuses (transient infrastructure failures only).
_RETRYABLE_STATUSES = frozenset({502, 503, 504})

_METRIC_TARGET = "http"


class _TransientApplicationError(Exception):
    """Internal: a retryable failure wrapping the final typed error."""

    def __init__(self, error: ApplicationInvocationError) -> None:
        super().__init__(str(error))
        self.error = error


class HTTPApplicationTarget:
    """Invoke a user-registered HTTP application with one prompt."""

    def __init__(
        self,
        connection: ConnectionConfig,
        *,
        secret: str | None = None,
        application_name: str = "application",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if connection.auth.requires_secret and not secret:
            raise ApplicationInvocationError(
                f"Application {application_name!r} requires a credential but "
                "none is configured.",
                category="unknown",
            )
        self._connection = connection
        self._secret = secret
        self._application_name = application_name
        self._last_status: int | None = None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=CONNECT_TIMEOUT_SECONDS,
                read=connection.timeout_seconds,
                write=CONNECT_TIMEOUT_SECONDS,
                pool=CONNECT_TIMEOUT_SECONDS,
            ),
            # SSRF hardening: never route through ambient proxy config, and
            # never let httpx follow redirects (each hop is validated here).
            trust_env=False,
            follow_redirects=False,
        )

    async def invoke(self, prompt: str) -> ApplicationResponse:
        """Map the prompt, call the application, and extract the answer.

        Transient failures (connect errors, timeouts, HTTP 502/503/504) are
        retried with bounded backoff; everything else — including invalid
        authentication, malformed requests, and response extraction
        failures — fails immediately.
        """
        body = build_request_body(self._connection.request, prompt)
        headers = self._build_headers()
        last_error: ApplicationInvocationError | None = None
        attempts = 0
        for attempt in range(self._connection.max_attempts):
            attempts = attempt + 1
            try:
                return await self._invoke_once(
                    self._connection.method,
                    headers,
                    body if self._connection.method == "POST" else None,
                )
            except _TransientApplicationError as exc:
                last_error = exc.error
                if attempt < self._connection.max_attempts - 1:
                    delay = RETRY_BACKOFF_SECONDS[
                        min(attempt, len(RETRY_BACKOFF_SECONDS) - 1)
                    ]
                    logger.warning(
                        "application_invocation_retry",
                        application=self._application_name,
                        attempt=attempts,
                        next_delay_seconds=delay,
                        reason=exc.error.category or "transient",
                    )
                    metrics.increment(
                        "application_request_retries_total", {"target": _METRIC_TARGET}
                    )
                    await anyio.sleep(delay)
            except ApplicationInvocationError as exc:
                # Permanent failure (auth, 4xx, extraction, SSRF block, ...):
                # never retried, so exactly one attempt was made.
                exc.attempts = attempts
                raise self._record_error(exc) from exc
        assert last_error is not None  # the loop ran at least once
        last_error.attempts = attempts  # typed failure metadata (Phase 12)
        raise self._record_error(last_error) from last_error

    async def close(self) -> None:
        """Release the underlying HTTP connection pool."""
        await self._client.aclose()

    # -- transport internals --------------------------------------------------

    async def _invoke_once(
        self, method: str, headers: dict[str, str], body: dict | None
    ) -> ApplicationResponse:
        started = time.monotonic()
        content = await self._request_following_redirects(method, headers, body)
        latency_ms = int((time.monotonic() - started) * 1000)
        metrics.observe(
            "application_request_latency_ms", latency_ms, {"target": _METRIC_TARGET}
        )
        metrics.increment(
            "application_requests_total",
            {"target": _METRIC_TARGET, "outcome": "success", "category": "none"},
        )
        return ApplicationResponse(
            content=content,
            latency_ms=latency_ms,
            status_code=self._last_status,
            metadata={"application": self._application_name},
        )

    def _record_error(
        self, error: ApplicationInvocationError
    ) -> ApplicationInvocationError:
        """Count one errored invocation; category labels stay bounded."""
        metrics.increment(
            "application_requests_total",
            {
                "target": _METRIC_TARGET,
                "outcome": "error",
                "category": error.category or "unknown",
            },
        )
        return error

    def _build_headers(self) -> dict[str, str]:
        """Static headers plus the credential header (never logged)."""
        headers = dict(self._connection.headers)
        if self._connection.auth.type == "bearer":
            headers["Authorization"] = f"Bearer {self._secret}"
        elif self._connection.auth.type == "api_key":
            headers[self._connection.auth.header_name] = self._secret or ""
        return headers

    async def _request_following_redirects(
        self, method: str, headers: dict[str, str], body: dict | None
    ) -> str:
        """One validated request, following at most MAX_REDIRECTS hops.

        Every hop re-runs the SSRF destination check (DNS-rebinding
        mitigation); redirects are followed manually — the HTTP client
        itself never redirects.
        """
        url: str = self._connection.endpoint
        current_method, current_body = method, body
        for hop in range(MAX_REDIRECTS + 1):
            try:
                await assert_url_resolves_public(url)
            except SSRFViolationError:
                raise ApplicationInvocationError(
                    "Application endpoint was blocked by SSRF protection.",
                    category="unknown",
                ) from None
            response = await self._send(current_method, url, headers, current_body)
            if not is_redirect(response.status_code):
                try:
                    return await self._read_and_extract(response)
                finally:
                    await response.aclose()
            # Redirect: close this hop, validate the destination, follow.
            await response.aclose()
            location = response.headers.get("location")
            if not location or hop == MAX_REDIRECTS:
                raise ApplicationInvocationError(
                    f"Application {self._application_name!r} returned an "
                    "invalid redirect.",
                    category="application_response_invalid",
                )
            url = str(self._client.build_request("GET", url).url.join(location))
            if response.status_code in (301, 302, 303):
                # Historic-POST redirects degrade to GET (browser behavior);
                # 307/308 preserve method and body.
                current_method, current_body = "GET", None
        raise ApplicationInvocationError(
            f"Application {self._application_name!r} exceeded the redirect limit.",
            category="application_response_invalid",
        )

    async def _send(
        self, method: str, url: str, headers: dict[str, str], body: dict | None
    ) -> httpx.Response:
        """One HTTP attempt; transport failures become typed errors.

        Only connect errors, timeouts, and 502/503/504 are transient.
        The URL and headers are never included in error messages.
        """
        request = self._client.build_request(method, url, json=body, headers=headers)
        try:
            response = await self._client.send(request, stream=True)
        except httpx.TimeoutException as exc:
            raise _TransientApplicationError(
                ApplicationInvocationError(
                    f"Application {self._application_name!r} timed out after "
                    f"{self._connection.timeout_seconds} s.",
                    category="timeout",
                )
            ) from exc
        except httpx.TransportError as exc:
            raise _TransientApplicationError(
                ApplicationInvocationError(
                    f"Could not reach application {self._application_name!r}: "
                    f"{type(exc).__name__}.",
                    category="connection_error",
                )
            ) from exc
        self._last_status = response.status_code
        status = response.status_code
        if status in _RETRYABLE_STATUSES:
            raise _TransientApplicationError(
                ApplicationInvocationError(
                    f"Application {self._application_name!r} returned HTTP {status}.",
                    category="application_http_error",
                )
            )
        if status == 429:
            raise ApplicationInvocationError(
                f"Application {self._application_name!r} reported rate limiting "
                "(HTTP 429).",
                category="rate_limited",
            )
        if status in (401, 403):
            raise ApplicationInvocationError(
                f"Application {self._application_name!r} rejected the credentials "
                f"(HTTP {status}).",
                category="authentication",
            )
        if status >= 500:
            raise ApplicationInvocationError(
                f"Application {self._application_name!r} returned HTTP {status}.",
                category="application_http_error",
            )
        if status >= 400:
            raise ApplicationInvocationError(
                f"Application {self._application_name!r} rejected the request "
                f"(HTTP {status}).",
                category="application_response_invalid",
            )
        return response

    async def _read_and_extract(self, response: httpx.Response) -> str:
        """Size-capped body read, JSON parse, and answer extraction."""
        received = 0
        chunks: list[bytes] = []
        try:
            async for chunk in response.aiter_bytes():
                received += len(chunk)
                if received > MAX_RESPONSE_BYTES:
                    raise ApplicationInvocationError(
                        f"Application {self._application_name!r} response "
                        "exceeded the size limit.",
                        category="application_response_invalid",
                    )
                chunks.append(chunk)
        except httpx.TimeoutException as exc:
            raise _TransientApplicationError(
                ApplicationInvocationError(
                    f"Application {self._application_name!r} timed out after "
                    f"{self._connection.timeout_seconds} s.",
                    category="timeout",
                )
            ) from exc
        raw = b"".join(chunks)
        try:
            parsed = json.loads(raw)
        except ValueError:
            raise ApplicationInvocationError(
                f"Application {self._application_name!r} returned a malformed "
                "(non-JSON) response.",
                category="malformed_response",
            ) from None
        try:
            return extract_answer(parsed, self._connection.response_path)
        except ResponseExtractionError as exc:
            raise ApplicationInvocationError(
                f"Application {self._application_name!r}: {exc}",
                category="application_response_invalid",
            ) from exc