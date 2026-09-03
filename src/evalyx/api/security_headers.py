"""HTTP security headers for the API boundary (Phase 17).

The API serves no browser UI, so the header set is minimal and safe:

- X-Content-Type-Options: nosniff
- X-Frame-Options: DENY (no embedding)
- Referrer-Policy: no-referrer
- Permissions-Policy: locked down
- HSTS is set by the reverse proxy (TLS termination), not here.

CORS stays disabled unless CORS_ALLOWED_ORIGINS names explicit origins —
never "*".
"""

from __future__ import annotations

_SECURITY_HEADERS: dict[str, str] = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}


class SecurityHeadersMiddleware:
    """Stamp safe security headers on every HTTP response."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_wrapper(message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                existing = {k.lower() for k, _ in headers}
                for name, value in _SECURITY_HEADERS.items():
                    if name.lower().encode("latin-1") not in existing:
                        headers.append(
                            (name.lower().encode("latin-1"), value.encode("latin-1"))
                        )
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_wrapper)
