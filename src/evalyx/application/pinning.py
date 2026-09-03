"""IP-pinning HTTP transport closing the SSRF DNS/TCP TOCTOU window (Phase 18).

Problem: validating a hostname's DNS answers and *then* connecting leaves a
window where a hostile DNS server (or a rebinding answer) can steer the
subsequent TCP connect to a different, private address.

This transport closes the window by resolving + validating **inside** the
request path and connecting only to a validated address:

- the original hostname is resolved via
  :func:`evalyx.application.ssrf.resolve_public_addresses` (every answer
  must be public, same as the request-time check);
- the request URL's host is rewritten to the first validated address while
  the original ``Host`` header (and TLS SNI via the ``sni_hostname``
  extension) is preserved, so virtual hosting and certificates keep working;
- literal-IP destinations skip resolution (already validated statically);
- non-HTTP(S) traffic is never issued by the connector and is rejected here
  defensively.

The per-hop ``assert_url_resolves_public`` in the connector stays in place
as defense in depth (it re-validates the hostname before each attempt); the
transport guarantees the connected address was validated in the same
request. Redirect hops each pass through the transport again, so every hop
is pinned independently.

Residual risk (documented honestly): a hostile DNS server can still choose
*which* public address is returned — it cannot steer to a private one.
"""

from __future__ import annotations

import ipaddress
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from evalyx.application.ssrf import (
    SSRFViolationError,
    assert_static_url_allowed,
    resolve_public_addresses,
)

#: Schemes this transport will issue (anything else is refused).
_PINNED_SCHEMES = frozenset({"http", "https"})
_DEFAULT_PORTS = {"http": 80, "https": 443}

Resolver = Callable[[str, int], Awaitable[list[str]]]
"""Resolve + validate ``(hostname, port)`` into validated address strings."""


def _is_literal_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


class PinningAsyncHTTPTransport(httpx.AsyncHTTPTransport):
    """httpx transport that connects only to SSRF-validated addresses."""

    def __init__(
        self,
        *args: Any,
        resolver: Resolver | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._resolver = resolver or resolve_public_addresses

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        url = request.url
        scheme = url.scheme.lower()
        if scheme not in _PINNED_SCHEMES or not url.host:
            raise SSRFViolationError("Application endpoint must use http or https.")
        # Cheap static boundary first (scheme/userinfo/fragment/port rules).
        assert_static_url_allowed(str(url))
        host = url.host
        port = url.port or _DEFAULT_PORTS[scheme]
        if _is_literal_ip(host):
            # Already validated statically above; connect as-is.
            return await super().handle_async_request(request)
        validated = await self._resolver(host, port)
        if not validated:
            raise SSRFViolationError(
                "Application endpoint hostname could not be resolved."
            )
        pinned = validated[0]
        host_header = host if port == _DEFAULT_PORTS[scheme] else f"{host}:{port}"
        headers = httpx.Headers(request.headers)
        headers["host"] = host_header
        extensions = dict(request.extensions)
        if scheme == "https":
            # Preserve certificate validation against the real hostname
            # while the TCP connection goes to the validated address.
            extensions["sni_hostname"] = host
        pinned_request = httpx.Request(
            request.method,
            url.copy_with(host=pinned, port=port),
            headers=headers,
            content=request.content,
            extensions=extensions,
        )
        return await super().handle_async_request(pinned_request)


__all__ = ["PinningAsyncHTTPTransport"]
