"""Phase 18 SSRF adversarial tests (hermetic, no real network).

Configuration-time boundary: obfuscated numeric literals, private hosts,
userinfo, fragments, schemes, ports. Request-time boundary: rebinding
resolvers (every answer validated). Pinning transport: the real
``PinningAsyncHTTPTransport`` with the socket layer stubbed at the parent
class, asserting URL/Host/SNI rewriting and refusal paths.
"""

from __future__ import annotations

import asyncio
import socket

import httpx
import pytest

from evalyx.application.pinning import PinningAsyncHTTPTransport
from evalyx.application.ssrf import (
    SSRFViolationError,
    _parse_numeric_ipv4,
    assert_static_url_allowed,
    assert_url_resolves_public,
    resolve_public_addresses,
)


class TestNumericLiteralDetection:
    @pytest.mark.parametrize(
        "host",
        [
            "2130706433",  # 127.0.0.1 decimal
            "0x7f000001",  # 127.0.0.1 hex
            "0x7F.0.0.1",
            "0177.0.0.1",  # octal quads
            "127.1",  # short form
            "127.0.1",
            "10.0.0.1",
            "192.168.1.1",
            "169.254.169.254",  # cloud metadata
            "0.0.0.0",
        ],
    )
    def test_private_numeric_forms_blocked_at_config_time(self, host):
        with pytest.raises(SSRFViolationError):
            assert_static_url_allowed(f"http://{host}/chat")

    def test_public_decimal_form_allowed_statically(self):
        # 93.184.216.34 as a bare decimal literal is public: static check
        # passes (request-time resolution remains authoritative).
        assert_static_url_allowed("http://1572391218/chat")

    def test_out_of_range_integer_is_hostname_path(self):
        assert _parse_numeric_ipv4("99999999999") is None

    def test_plain_hostnames_untouched(self):
        assert _parse_numeric_ipv4("example.com") is None
        assert _parse_numeric_ipv4("api.example.com") is None

    @pytest.mark.parametrize(
        "url",
        [
            "http://user:pass@example.com/",
            "https://example.com/#frag",
            "ftp://example.com/x",
            "file:///etc/passwd",
            "gopher://example.com/",
            "http://localhost:8000/",
            "http://localhost./",
            "http://x.localhost/",
            "http://LOCALHOST/",
            "http://127.0.0.1/",
            "http://[::1]/",
            "http://[::ffff:127.0.0.1]/",
            "http://[fe80::1]/",
            "http://example.com:99999/",
        ],
    )
    def test_static_boundary_rejects(self, url):
        with pytest.raises(SSRFViolationError):
            assert_static_url_allowed(url)


class _FakeLoop:
    """Stand-in event loop with scripted getaddrinfo answers."""

    def __init__(self, answers):
        self._answers = list(answers)
        self.calls = 0

    async def getaddrinfo(self, host, port, type=None):
        self.calls += 1
        answer = self._answers[min(self.calls - 1, len(self._answers) - 1)]
        if isinstance(answer, Exception):
            raise answer
        return [(2, 1, 6, "", (ip, port)) for ip in answer]


def _with_loop(monkeypatch, answers):
    from evalyx.application import ssrf

    loop = _FakeLoop(answers)
    monkeypatch.setattr(ssrf.asyncio, "get_running_loop", lambda: loop)
    return loop


class TestRequestTimeRebinding:
    def test_mixed_public_private_answers_rejected(self, monkeypatch):
        _with_loop(monkeypatch, [["93.184.216.34", "10.0.0.5"]])
        with pytest.raises(SSRFViolationError):
            asyncio.run(assert_url_resolves_public("http://example.com/chat"))

    def test_rebinding_across_attempts_each_checked(self, monkeypatch):
        """First attempt resolves public, second rebinds to private."""
        loop = _with_loop(monkeypatch, [["93.184.216.34"], ["127.0.0.1"]])
        asyncio.run(assert_url_resolves_public("http://example.com/a"))
        with pytest.raises(SSRFViolationError):
            asyncio.run(assert_url_resolves_public("http://example.com/b"))
        assert loop.calls == 2

    def test_all_public_answers_returned(self, monkeypatch):
        _with_loop(monkeypatch, [["93.184.216.34", "93.184.216.35"]])
        assert asyncio.run(resolve_public_addresses("example.com", 80)) == [
            "93.184.216.34",
            "93.184.216.35",
        ]

    def test_unresolvable_rejected(self, monkeypatch):
        _with_loop(monkeypatch, [socket.gaierror("nope")])
        with pytest.raises(SSRFViolationError):
            asyncio.run(assert_url_resolves_public("http://nonexistent.invalid/"))


class TestPinningTransport:
    """Real transport, socket layer stubbed at the parent class."""

    @staticmethod
    def _stub_socket(monkeypatch, seen):
        async def fake_handle(self, request):
            seen.append(request)
            return httpx.Response(200, json={"answer": "ok"})

        monkeypatch.setattr(
            httpx.AsyncHTTPTransport, "handle_async_request", fake_handle
        )

    @staticmethod
    async def _get(transport, url):
        async with httpx.AsyncClient(
            transport=transport, trust_env=False, follow_redirects=False
        ) as client:
            return await client.get(url)

    def test_http_pinned_to_validated_ip_with_host_preserved(self, monkeypatch):
        async def resolver(host, port):
            assert (host, port) == ("example.com", 80)
            return ["93.184.216.34", "93.184.216.35"]

        seen: list = []
        self._stub_socket(monkeypatch, seen)
        transport = PinningAsyncHTTPTransport(resolver=resolver)
        response = asyncio.run(self._get(transport, "http://example.com/chat"))
        assert response.status_code == 200
        (request,) = seen
        assert request.url.host == "93.184.216.34"
        assert request.headers["host"] == "example.com"
        assert "sni_hostname" not in request.extensions

    def test_https_pins_ip_but_keeps_sni(self, monkeypatch):
        async def resolver(host, port):
            assert (host, port) == ("example.com", 443)
            return ["93.184.216.34"]

        seen: list = []
        self._stub_socket(monkeypatch, seen)
        transport = PinningAsyncHTTPTransport(resolver=resolver)
        response = asyncio.run(self._get(transport, "https://example.com/chat"))
        assert response.status_code == 200
        (request,) = seen
        assert request.url.host == "93.184.216.34"
        assert request.url.scheme == "https"
        assert request.headers["host"] == "example.com"
        assert request.extensions["sni_hostname"] == "example.com"

    def test_non_default_port_in_host_header(self, monkeypatch):
        async def resolver(host, port):
            assert (host, port) == ("example.com", 8443)
            return ["93.184.216.34"]

        seen: list = []
        self._stub_socket(monkeypatch, seen)
        transport = PinningAsyncHTTPTransport(resolver=resolver)
        asyncio.run(self._get(transport, "https://example.com:8443/chat"))
        (request,) = seen
        assert request.url.host == "93.184.216.34"
        assert request.url.port == 8443
        assert request.headers["host"] == "example.com:8443"

    def test_literal_ip_skips_resolution(self, monkeypatch):
        async def resolver(host, port):  # pragma: no cover
            raise AssertionError("resolver must not run for literal IPs")

        seen: list = []
        self._stub_socket(monkeypatch, seen)
        transport = PinningAsyncHTTPTransport(resolver=resolver)
        asyncio.run(self._get(transport, "http://93.184.216.34/chat"))
        (request,) = seen
        assert request.url.host == "93.184.216.34"

    def test_private_literal_refused_before_resolution(self, monkeypatch):
        async def resolver(host, port):  # pragma: no cover
            raise AssertionError("resolver must not run for blocked URLs")

        seen: list = []
        self._stub_socket(monkeypatch, seen)
        transport = PinningAsyncHTTPTransport(resolver=resolver)
        with pytest.raises(SSRFViolationError):
            asyncio.run(self._get(transport, "http://127.0.0.1/chat"))
        with pytest.raises(SSRFViolationError):
            asyncio.run(self._get(transport, "ftp://example.com/x"))
        assert seen == []
