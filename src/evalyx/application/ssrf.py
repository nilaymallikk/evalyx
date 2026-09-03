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
so an HTTP client cannot be redirected through a local proxy. The window is
closed by :mod:`evalyx.application.pinning`, which resolves and validates
*inside* the HTTP transport and connects only to a validated address
(preserving Host/SNI) — the validated address is the connected address.
Residual risk: a hostile DNS server that answers differently per query can
still steer *which* public address is used, but every candidate address is
validated public before use, so steering cannot reach private targets.
"""

import asyncio
import ipaddress
import socket
import struct
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


def _parse_numeric_ipv4(host: str) -> ipaddress.IPv4Address | None:
    """Interpret obfuscated numeric IPv4 literals (defense in depth).

    Attackers write 127.0.0.1 as ``2130706433`` (decimal), ``0x7f000001``
    (hex), ``0177.0.0.1`` (octal quads) or ``127.1`` (short form) to dodge
    naive string blocklists. ``ipaddress`` rejects all of these, but the OS
    resolver (``inet_aton`` semantics) accepts them — so configuration-time
    validation must recognize them too. Returns the parsed address, or
    ``None`` when ``host`` is not a numeric form.
    """
    try:
        packed = socket.inet_aton(host)
    except OSError:
        packed = None
    if packed is not None:
        return ipaddress.IPv4Address(struct.unpack("!I", packed)[0])
    candidate = host.strip()
    try:
        if candidate.lower().startswith("0x"):
            value = int(candidate, 16)
        elif candidate.isdigit():
            value = int(candidate, 10)
        else:
            return None
    except ValueError:
        return None
    if 0 <= value <= 0xFFFFFFFF:
        return ipaddress.IPv4Address(value)
    return None


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
    try:
        port = parsed.port
    except ValueError:
        raise SSRFViolationError("Application endpoint port is out of range.") from None
    if port is not None and not (0 < port <= 65535):
        raise SSRFViolationError("Application endpoint port is out of range.")
    lowered = hostname.lower().rstrip(".")
    if lowered == "localhost" or lowered.endswith(".localhost"):
        raise SSRFViolationError("Application endpoint must not target localhost.")
    # Literal IPs are fully checkable right now; hostnames are checked at
    # request time via DNS resolution. Obfuscated numeric literals (decimal,
    # hex, octal, short forms) are recognized too — the OS resolver would
    # accept them even though ``ipaddress`` does not.
    try:
        _assert_public_ip(ipaddress.ip_address(lowered))
    except ValueError:
        pass
    else:
        return
    obfuscated = _parse_numeric_ipv4(lowered)
    if obfuscated is not None:
        _assert_public_ip(obfuscated)


async def resolve_public_addresses(hostname: str, port: int) -> list[str]:
    """Resolve ``hostname`` and require every address to be public.

    Returns the deduplicated validated address strings (first answer wins
    for connection purposes). Raises :class:`SSRFViolationError` when
    resolution fails or any address is non-public. Shared by the
    request-time check and the pinning transport so both validate the same
    way.
    """
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
    validated: list[str] = []
    seen: set[str] = set()
    for info in infos:
        address = str(info[4][0])
        if address in seen:
            continue
        seen.add(address)
        _assert_public_ip(ipaddress.ip_address(address))
        validated.append(address)
    return validated


async def assert_url_resolves_public(url: str) -> None:
    """Resolve ``url``'s hostname and require every address to be public.

    The authoritative SSRF check. Re-run before every transport attempt
    (including every redirect hop) to keep DNS-rebinding windows small.
    """
    assert_static_url_allowed(url)
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    port = parsed.port or _DEFAULT_PORTS.get(parsed.scheme.lower(), 80)
    await resolve_public_addresses(hostname, port)


def is_redirect(status_code: int) -> bool:
    """True for the redirect statuses the connector follows manually."""
    return status_code in (301, 302, 303, 307, 308)


__all__ = [
    "FORBIDDEN_HEADER_NAMES",
    "SSRFViolationError",
    "assert_static_url_allowed",
    "assert_url_resolves_public",
    "is_redirect",
    "resolve_public_addresses",
]