"""Durable security/resource audit recording (Phase 18).

:func:`record_audit_event` appends one :class:`AuditEvent` row to the
caller's session (no commit — the caller commits as part of its own
transaction, so a committed mutation always has its audit row and a rolled
back mutation never leaves a stray one). Denial paths use
:func:`record_denial_and_commit`, which commits the denial row immediately
before the caller raises (the operation is aborted, so committing just the
audit row is correct).

Safety rules:

- ``details`` are sanitized: keys containing secret-shaped substrings are
  dropped, over-long strings truncated, non-JSON values stringified with a
  size cap, and the whole payload bounded. Prompts, responses, tokens,
  credentials, and secrets must never reach this function — and if they do
  by mistake, the sanitizer drops the recognizable shapes.
- ``action`` comes from the fixed :class:`AuditAction` set (bounded labels
  for metrics/queries); ``result`` is ``"allowed"`` or ``"denied"``.
- Recording never raises for sanitization reasons; only genuine database
  errors propagate (and callers treat audit failure like any other
  persistence failure inside their transaction).
"""

from __future__ import annotations

import uuid
from typing import Any, Final

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from evalyx.core.context import get_request_id
from evalyx.core.metrics import metrics
from evalyx.db.models.governance import AuditEvent

logger = structlog.get_logger(__name__)

#: Bounded audit action vocabulary (new actions require a conscious addition
#: here — never free-form caller strings).
APPLICATION_CREATE: Final[str] = "application.create"
APPLICATION_UPDATE: Final[str] = "application.update"
APPLICATION_DELETE: Final[str] = "application.delete"
APPLICATION_VERSION_CREATE: Final[str] = "application.version.create"
APPLICATION_SECRET_ROTATE: Final[str] = "application.secret.rotate"
APPLICATION_CONNECTION_TEST: Final[str] = "application.connection.test"
DATASET_CREATE: Final[str] = "dataset.create"
DATASET_VERSION_CREATE: Final[str] = "dataset.version.create"
DATASET_CASE_ADD: Final[str] = "dataset.case.add"
EVALUATION_SUBMIT: Final[str] = "evaluation.submit"
AUTH_ORGANIZATION_REQUIRED: Final[str] = "auth.organization_required"
AUTH_INSUFFICIENT_ROLE: Final[str] = "auth.insufficient_role"
QUOTA_EXCEEDED: Final[str] = "quota.exceeded"

ALLOWED: Final[str] = "allowed"
DENIED: Final[str] = "denied"

#: Detail keys carrying these substrings are dropped, not stored.
_FORBIDDEN_DETAIL_SUBSTRINGS: Final[tuple[str, ...]] = (
    "secret",
    "token",
    "password",
    "api_key",
    "apikey",
    "authorization",
    "encryption_key",
    "clerk",
    "credential",
    "bearer",
    "prompt",
    "response",
    "input",
    "output",
    "payload",
    "body",
)

_MAX_DETAIL_STRING: Final[int] = 500
_MAX_DETAIL_KEYS: Final[int] = 20


def sanitize_details(details: dict[str, Any] | None) -> dict[str, Any]:
    """Return a bounded, secret-free copy of ``details``.

    Drops secret-shaped keys, truncates long strings, stringifies exotic
    values with a cap, and keeps at most ``_MAX_DETAIL_KEYS`` entries.
    Pure function — unit-tested directly.
    """
    if not details:
        return {}
    clean: dict[str, Any] = {}
    for key, value in details.items():
        if len(clean) >= _MAX_DETAIL_KEYS:
            break
        if not isinstance(key, str):
            continue
        lowered = key.lower()
        if any(part in lowered for part in _FORBIDDEN_DETAIL_SUBSTRINGS):
            continue
        clean[key] = _sanitize_value(value)
    return clean


def _sanitize_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:_MAX_DETAIL_STRING]
    if isinstance(value, (list, tuple)):
        return [_sanitize_value(item) for item in value[:10]]
    if isinstance(value, dict):
        return {
            str(key)[:64]: _sanitize_value(item)
            for key, item in list(value.items())[:10]
            if not any(part in str(key).lower() for part in _FORBIDDEN_DETAIL_SUBSTRINGS)
        }
    return str(value)[:_MAX_DETAIL_STRING]


async def record_audit_event(
    session: AsyncSession,
    *,
    organization_id: uuid.UUID | None,
    clerk_user_id: str,
    action: str,
    resource_type: str | None = None,
    resource_id: uuid.UUID | str | None = None,
    result: str = ALLOWED,
    details: dict[str, Any] | None = None,
    request_id: str | None = None,
) -> AuditEvent:
    """Append one audit row to ``session`` (caller commits).

    ``resource_id`` accepts UUIDs (stored as strings) or short bounded
    strings; overly long values are truncated.
    """
    resource_id_value: str | None = None
    if resource_id is not None:
        resource_id_value = str(resource_id)[:64]
    event = AuditEvent(
        organization_id=organization_id,
        clerk_user_id=clerk_user_id[:255],
        action=action,
        resource_type=resource_type,
        resource_id=resource_id_value,
        result=result,
        request_id=(request_id or get_request_id()),
        details=sanitize_details(details),
    )
    session.add(event)
    metrics.increment("audit_events_total", {"action": action, "result": result})
    logger.info(
        "audit_event",
        action=action,
        result=result,
        resource_type=resource_type,
        resource_id=resource_id_value,
    )
    return event


async def record_denial_and_commit(
    session: AsyncSession,
    *,
    organization_id: uuid.UUID | None,
    clerk_user_id: str,
    action: str,
    resource_type: str | None = None,
    resource_id: uuid.UUID | str | None = None,
    details: dict[str, Any] | None = None,
) -> AuditEvent:
    """Record a denial row and commit immediately (before raising).

    Used on paths that abort with an exception (quota/authorization
    denials): the surrounding session would otherwise roll the audit row
    back. The session must contain no other pending changes at this point
    (callers record denials before mutating anything).
    """
    event = await record_audit_event(
        session,
        organization_id=organization_id,
        clerk_user_id=clerk_user_id,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        result=DENIED,
        details=details,
    )
    await session.commit()
    return event


__all__ = [
    "ALLOWED",
    "APPLICATION_CONNECTION_TEST",
    "APPLICATION_CREATE",
    "APPLICATION_DELETE",
    "APPLICATION_SECRET_ROTATE",
    "APPLICATION_UPDATE",
    "APPLICATION_VERSION_CREATE",
    "AUTH_INSUFFICIENT_ROLE",
    "AUTH_ORGANIZATION_REQUIRED",
    "DATASET_CASE_ADD",
    "DATASET_CREATE",
    "DATASET_VERSION_CREATE",
    "DENIED",
    "EVALUATION_SUBMIT",
    "QUOTA_EXCEEDED",
    "record_audit_event",
    "record_denial_and_commit",
    "sanitize_details",
]
