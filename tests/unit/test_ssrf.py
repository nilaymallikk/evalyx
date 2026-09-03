"""Hermetic SSRF validation tests (Phase 15 Step 11).

Static URL rules are tested directly; DNS-based rules are tested through a
monkeypatched ``getaddrinfo`` (no network).
"""

import asyncio
import socket

import pytest

from evalyx.application.ssrf import (
    SSRFViolationError,
    assert_static_url_allowed,
    assert_url_resolves_public,
    is_redirect,
)

PUBLIC_IP = "93.184.216.34"  # example.com's address; public/global


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost/v1/chat",
        "https://api.localhost/v1/chat",
        "http://127.0.0.1/v1/chat",
        "http://0.0.0.0/v1/chat",
        "http://10.0.0.5/v1/chat",
        "http://172.16.0.9/v1/chat",
        "http://192.168.1.10/v1/chat",
        "http://169.254.169.254/latest/meta-data",  # cloud metadata
        "http://[::1]/v1/chat",
        "http://[::ffff:127.0.0.1]/v1/chat",  # IPv4-mapped IPv6 loopback
        "http://[fe80::1]/v1/chat",  # link-local IPv6
        "http://100.64.0.1/v1/chat",  # CGNAT (shared address space)
        "ftp://93.184.216.34/v1/chat",  # non-HTTP scheme
        "http://user:pass@93.184.216.34/v1/chat",  # embedded credentials
        "https://93.184.216.34/v1/chat#fragment",
        "http:///v1/chat",  # missing host
    ],
)
def test_static_url_blocked(url: str):
    with pytest.raises(SSRFViolationError):
        assert_static_url_allowed(url)


@pytest.mark.parametrize(
    "url",
    [
        f"https://{PUBLIC_IP}/v1/chat",
        f"https://{PUBLIC_IP}:8443/v1/chat",
    ],
)
def test_static_url_public_accepted(url: str):
    assert_static_url_allowed(url)


def _fake_getaddrinfo(monkeypatch, address: str):
    """Patch socket resolution to always answer with ``address``."""

    def fake_getaddrinfo(host, port, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)


def test_resolution_public_passes(monkeypatch):
    _fake_getaddrinfo(monkeypatch, PUBLIC_IP)
    asyncio.run(assert_url_resolves_public("https://api.example.com/v1/chat"))


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.1.2.3",
        "169.254.169.254",
        "0.0.0.0",
    ],
)
def test_dns_rebinding_to_private_blocked(monkeypatch, address: str):
    # The hostname is innocent; the *resolved* address is not (rebinding).
    _fake_getaddrinfo(monkeypatch, address)
    with pytest.raises(SSRFViolationError):
        asyncio.run(assert_url_resolves_public("https://api.example.com/v1/chat"))


def test_unresolvable_host_rejected(monkeypatch):
    def fake_getaddrinfo(host, port, *args, **kwargs):
        raise socket.gaierror(-2, "Name or service not known")

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    with pytest.raises(SSRFViolationError, match="resolved"):
        asyncio.run(assert_url_resolves_public("https://missing.example.com/v1"))


def test_ipv4_mapped_ipv6_loopback_blocked_at_resolution(monkeypatch):
    _fake_getaddrinfo(monkeypatch, "::ffff:127.0.0.1")
    with pytest.raises(SSRFViolationError):
        asyncio.run(assert_url_resolves_public("https://api.example.com/v1"))


def test_is_redirect_statuses():
    assert is_redirect(301) and is_redirect(308)
    assert not is_redirect(200) and not is_redirect(404)


def test_literal_non_global_ipv6_blocked():
    with pytest.raises(SSRFViolationError):
        assert_static_url_allowed("http://[fd00::1]/v1/chat")  # ULA private