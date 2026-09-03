"""SSRF protection for user-registered application endpoints (Phase 15).

Users can register arbitrary HTTP endpoints, which makes Evalyx a potential
SSRF (server-side request forgery) service unless every outbound destination
is validated. This module is the single validation boundary:

- :func:`assert_static_url_allowed` — cheap, synchronous checks applied at
  *configuration time* (schema validation): scheme, credentials, fragment,
  and non-public literal hosts (``localhost``, private IPs, ...).
- :func:`assert_url_resolves_public` — the authoritative check, applied at
  *request time*: resolves the hostname via DNS and requires **every**
  resolved address to be a public, globally-routable IP.

Blocked destinations (defense in depth — several mechanisms overlap):

- non-HTTP(S) schemes, embedded userinfo, fragments
- literal hostnames: ``localhost`` and ``*.localhost`` subdomains
- resolved addresses that are loopback, RFC1918-private, link-local
  (which covers the cloud metadata service 169.254.169.254), multicast,
  reserved, unspecified (``0.0.0.0``/``::``), IPv4-mapped IPv6, or
  otherwise not globally routable

Known, honestly-documented limitation: the resolution check and the TCP
connect are two separate steps (a TOCTOU window exists between them).
Defenses that shrink the window: the check re-runs before **every**
transport attempt, redirects are followed manually with a full re-check of
each hop, and proxy environment variables are ignored (``trust_env=False``)
so an HTTP client cannot be redirected through a local proxy. Fully closing
the window would require pinning connections to validated IPs at the
transport layer (a custom httpcore transport) — deferred as future work.
"""

import asyncio
import ipaddress
import socket
from urllib.parse import urlparse

_ALLOWED_SCHEMES = frozenset({"http", "https"})
_DEFAULT_PORTS = {"http": 80, "https": 443}
FORBIDDEN_HEADER_NAMES = frozenset(
    {
        "host",
        "content-length",
        "connection",
        "transfer-encoding",
        "authorization",
        "expect",
        "te",
        "trailer",
        "upgrade",
        "keep-alive",
        "proxy-authorization",
        "proxy-connection",
    }
)


class SSRFViolationError(Exception):
    """A URL targets a non-public (or malformed) destination.

    Safe by construction: the message describes the *rule* violated — never
    the URL's credentials (there must not be any) or internal topology.
    """


def _assert_public_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> None:
    """Raise when ``ip`` is not a public, globally-routable address."""
    # Unwrap IPv4-mapped IPv6 addresses (::ffff:127.0.0.1) so they cannot
    # masquerade as public IPv6.
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    blocked = (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
        or not ip.is_global
    )
    if blocked:
        raise SSRFViolationError(
            "Application endpoint must resolve to a public IP address."
        )


def assert_static_url_allowed(url: str) -> None:
    """Cheap synchronous URL validation (configuration-time boundary).

    Does not resolve DNS — the authoritative per-request check is
    :func:`assert_url_resolves_public`. Raises :class:`SSRFViolationError`.
    """
    try:
        parsed = urlparse(url)
    except ValueError:
        raise SSRFViolationError("Application endpoint URL is malformed.") from None
    if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
        raise SSRFViolationError("Application endpoint must use http or https.")
    if parsed.username is not None or parsed.password is not None:
        raise SSRFViolationError("Application endpoint must not embed credentials.")
    if parsed.fragment:
        raise SSRFViolationError("Application endpoint must not include a fragment.")
    hostname = parsed.hostname
    if not hostname:
        raise SSRFViolationError("Application endpoint must include a hostname.")
    if parsed.port is not None and not (0 < parsed.port <= 65535):
        raise SSRFViolationError("Application endpoint port is out of range.")
    lowered = hostname.lower().rstrip(".")
    if lowered == "localhost" or lowered.endswith(".localhost"):
        raise SSRFViolationError("Application endpoint must not target localhost.")
    # Literal IPs are fully checkable right now; hostnames are checked at
    # request time via DNS resolution.
    try:
        _assert_public_ip(ipaddress.ip_address(lowered))
    except ValueError:
        return  # a hostname, not a literal IP — resolved later


async def assert_url_resolves_public(url: str) -> None:
    """Resolve ``url``'s hostname and require every address to be public.

    The authoritative SSRF check. Re-run before every transport attempt
    (including every redirect hop) to keep DNS-rebinding windows small.
    """
    assert_static_url_allowed(url)
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    port = parsed.port or _DEFAULT_PORTS.get(parsed.scheme.lower(), 80)
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except socket.gaierror:
        raise SSRFViolationError(
            "Application endpoint hostname could not be resolved."
        ) from None
    if not infos:
        raise SSRFViolationError(
            "Application endpoint hostname could not be resolved."
        )
    seen: set[str] = set()
    for info in infos:
        address = str(info[4][0])
        if address in seen:
            continue
        seen.add(address)
        _assert_public_ip(ipaddress.ip_address(address))


def is_redirect(status_code: int) -> bool:
    """True for the redirect statuses the connector follows manually."""
    return status_code in (301, 302, 303, 307, 308)


__all__ = [
    "FORBIDDEN_HEADER_NAMES",
    "SSRFViolationError",
    "assert_static_url_allowed",
    "assert_url_resolves_public",
    "is_redirect",
]