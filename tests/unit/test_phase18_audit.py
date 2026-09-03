"""Phase 18 audit-recorder tests (hermetic).

Sanitization is the security-critical property: prompts, responses,
credentials, tokens, and other secrets must never land in audit rows even
if a caller passes them by mistake. Recording behavior (append-without-
commit vs deny-and-commit) is verified against a stub session.
"""

from __future__ import annotations

import uuid

import pytest

from evalyx.security.audit import (
    APPLICATION_CREATE,
    record_audit_event,
    record_denial_and_commit,
    sanitize_details,
)


class TestSanitizeDetails:
    @pytest.mark.parametrize(
        "key",
        [
            "secret",
            "api_key",
            "client_secret",
            "token",
            "id_token",
            "password",
            "db_password",
            "authorization",
            "bearer",
            "clerk_secret_key",
            "encryption_key",
            "credential",
            "api_credential",
            "prompt",
            "system_prompt",
            "response",
            "model_response",
            "input",
            "user_input",
            "output",
            "expected_output",
            "payload",
            "request_body",
        ],
    )
    def test_secret_shaped_keys_dropped(self, key):
        cleaned = sanitize_details({key: "super-secret-value", "name": "ok"})
        assert key not in cleaned
        assert "super-secret-value" not in str(cleaned)
        assert cleaned["name"] == "ok"

    def test_case_insensitive(self):
        cleaned = sanitize_details({"API_KEY": "x", "Secret": "y"})
        assert cleaned == {}

    def test_nested_secret_keys_dropped(self):
        cleaned = sanitize_details({"config": {"token": "abc", "temperature": 0.2}})
        assert cleaned == {"config": {"temperature": 0.2}}

    def test_long_strings_truncated(self):
        cleaned = sanitize_details({"name": "n" * 2000})
        assert len(cleaned["name"]) == 500

    def test_key_count_bounded(self):
        cleaned = sanitize_details({f"k{i}": i for i in range(100)})
        assert len(cleaned) == 20

    def test_non_string_values_stringified_safely(self):
        cleaned = sanitize_details({"count": 3, "ok": True, "nothing": None})
        assert cleaned == {"count": 3, "ok": True, "nothing": None}

    def test_empty_or_none(self):
        assert sanitize_details(None) == {}
        assert sanitize_details({}) == {}

    def test_lists_bounded(self):
        cleaned = sanitize_details({"ids": list(range(50))})
        assert cleaned["ids"] == list(range(10))


class _StubSession:
    """Minimal async session: records adds, tracks commits."""

    def __init__(self) -> None:
        self.added: list = []
        self.commits = 0

    def add(self, row) -> None:
        self.added.append(row)

    async def commit(self) -> None:
        self.commits += 1


class TestRecording:
    async def test_allowed_event_appended_without_commit(self):
        session = _StubSession()
        event = await record_audit_event(
            session,  # type: ignore[arg-type]
            organization_id=uuid.uuid4(),
            clerk_user_id="user_123",
            action=APPLICATION_CREATE,
            resource_type="application",
            resource_id=uuid.uuid4(),
            details={"name": "demo", "secret": "must-vanish"},
        )
        assert len(session.added) == 1
        assert session.commits == 0  # caller commits with its own transaction
        assert event.result == "allowed"
        assert event.details == {"name": "demo"}
        assert event.request_id is None or isinstance(event.request_id, str)

    async def test_denial_commits_immediately(self):
        session = _StubSession()
        event = await record_denial_and_commit(
            session,  # type: ignore[arg-type]
            organization_id=None,
            clerk_user_id="user_123",
            action="auth.organization_required",
        )
        assert event.result == "denied"
        assert session.commits == 1

    async def test_resource_id_truncated(self):
        session = _StubSession()
        event = await record_audit_event(
            session,  # type: ignore[arg-type]
            organization_id=None,
            clerk_user_id="u",
            action=APPLICATION_CREATE,
            resource_id="r" * 200,
        )
        assert event.resource_id is not None
        assert len(event.resource_id) == 64
