"""HTTP observability middleware: correlation, lifecycle logs, latency metrics.

Pure-ASGI middleware (no BaseHTTPMiddleware task overhead) that, for every
HTTP request:

1. resolves a **request id** — reuses a safe client-sent ``X-Request-ID``
   (bounded length, restricted charset) or generates a UUID4; the resolved id
   is always echoed back as the ``X-Request-ID`` response header
2. binds the id into the correlation context (:mod:`evalyx.core.context`) so
   every structured log event during the request carries ``request_id`` and
   the context is cleared afterwards (no cross-request leakage)
3. emits ``http_request_started`` (debug) and ``http_request_completed``
   (info) events with method, **route template** (never concrete dynamic
   paths), status code and ``duration_ms`` measured with a monotonic clock —
   logging happens even when the request raises
4. records bounded-label metrics: ``http_requests_total{method, route,
   status}`` and ``http_request_duration_ms{method, route}``
5. warns on requests exceeding the configured slow-request threshold

Privacy: request bodies, headers (Authorization/cookies), query strings,
prompts and model outputs are **never** logged — only explicitly selected
safe fields. Invalid client request ids are rejected without logging the
rejected value (only its length).
"""

import re
import time
import uuid
from typing import Final

import structlog

from evalyx.core.context import clear_correlation_context, set_request_id
from evalyx.core.metrics import metrics

logger = structlog.get_logger(__name__)

_REQUEST_ID_HEADER: Final[bytes] = b"x-request-id"
_REQUEST_ID_MAX_LENGTH: Final[int] = 128
# Allowed client-provided request ids: letters, digits, dot, underscore,
# hyphen. Anything else (spaces, control characters, oversized values) is
# replaced with a generated UUID4.
_SAFE_REQUEST_ID_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9._-]{1,128}$")

#: The versioned surface prefix. Routes registered under it are matched
#: against leaf paths relative to their sub-router on some FastAPI versions,
#: so the prefix is re-attached to produce the full route template.
API_V1_PREFIX: Final[str] = "/api/v1"

UNMATCHED_ROUTE: Final[str] = "/unmatched"
"""Bounded stand-in for requests that matched no route (e.g. 404s on
arbitrary paths) — concrete paths would make metric labels unbounded."""

SCOPE_REQUEST_ID_KEY: Final[str] = "evalyx_request_id"
"""ASGI scope key carrying the resolved request id.

Needed because the outer server-error middleware (which answers unhandled
exceptions) runs *after* this middleware's ``finally`` block has cleared the
correlation context — the scope dict still travels with the exception. The
handler reads this key (never a global) to stamp the response header."""


def resolve_request_id(raw: str | None) -> tuple[str, str]:
    """Return ``(request_id, origin)`` for a client-supplied header value.

    ``origin`` is ``"client"`` (valid header reused), ``"generated"`` (no
    header or an unsafe value replaced). Invalid values are never logged;
    a bounded warning records only the rejection reason and the raw length.
    """
    if raw:
        if len(raw) <= _REQUEST_ID_MAX_LENGTH and _SAFE_REQUEST_ID_RE.match(raw):
            return raw, "client"
        logger.warning(
            "request_id_rejected",
            reason="oversized" if len(raw) > _REQUEST_ID_MAX_LENGTH else "invalid_format",
            length=len(raw),
        )
    return str(uuid.uuid4()), "generated"


def route_template(scope: dict) -> str:
    """Best-effort route template for metric labels and logs.

    Uses the matched route object when the router recorded it (bounded
    templates like ``/api/v1/evaluations/{run_id}``); falls back to
    :data:`UNMATCHED_ROUTE` otherwise. Deliberately never falls back to the
    concrete request path, which would create unbounded label cardinality.
    """
    route = scope.get("route")
    path = getattr(route, "path", None)
    if not isinstance(path, str) or not path:
        return UNMATCHED_ROUTE
    request_path = scope.get("path", "")
    if (
        isinstance(request_path, str)
        and request_path.startswith(API_V1_PREFIX)
        and not path.startswith(API_V1_PREFIX)
    ):
        # FastAPI may report the leaf path relative to its sub-router;
        # re-attach the version prefix so labels show the full template.
        return API_V1_PREFIX + path
    return path


class ObservabilityMiddleware:
    """Request correlation + structured lifecycle logging + HTTP metrics."""

    def __init__(self, app, slow_request_threshold_ms: int = 1000) -> None:
        self.app = app
        self.slow_request_threshold_ms = slow_request_threshold_ms

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Isolate context: drop anything a previous request/task left behind.
        clear_correlation_context()
        raw = self._request_id_header(scope)
        request_id, origin = resolve_request_id(raw)
        set_request_id(request_id)
        scope[SCOPE_REQUEST_ID_KEY] = request_id

        method = scope.get("method", "?")
        started = time.monotonic()  # monotonic latency clock
        status_code: int | None = None

        async def send_wrapper(message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                headers = list(message.get("headers", []))
                headers.append((_REQUEST_ID_HEADER, request_id.encode("latin-1")))
                message = {**message, "headers": headers}
            await send(message)

        logger.debug(
            "http_request_started",
            request_id=request_id,
            origin=origin,
            method=method,
        )
        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            self._finish(scope, method, request_id, started, status_code)
            clear_correlation_context()

    @staticmethod
    def _request_id_header(scope: dict) -> str | None:
        """Extract the raw ``X-Request-ID`` header value, if present."""
        for key, value in scope.get("headers") or []:
            if key == _REQUEST_ID_HEADER:
                # Header bytes may contain arbitrary latin-1-decodable data;
                # resolve_request_id validates before anything is reused.
                return value.decode("latin-1")
        return None

    def _finish(
        self,
        scope: dict,
        method: str,
        request_id: str,
        started: float,
        status_code: int | None,
    ) -> None:
        """Log completion and record metrics — always, success or failure.

        An exception that escaped the app (answered by the outer server-error
        middleware) is reported as HTTP 500 for both the log event and the
        metric label.
        """
        duration_ms = (time.monotonic() - started) * 1000.0
        route = route_template(scope)
        effective_status = status_code if status_code is not None else 500
        logger.info(
            "http_request_completed",
            request_id=request_id,
            method=method,
            route=route,
            status_code=effective_status,
            duration_ms=round(duration_ms, 2),
        )
        metrics.increment(
            "http_requests_total",
            {"method": method, "route": route, "status": str(effective_status)},
        )
        metrics.observe(
            "http_request_duration_ms",
            duration_ms,
            {"method": method, "route": route},
        )
        if duration_ms > self.slow_request_threshold_ms:
            logger.warning(
                "http_request_slow",
                request_id=request_id,
                method=method,
                route=route,
                duration_ms=round(duration_ms, 2),
                threshold_ms=self.slow_request_threshold_ms,
            )
