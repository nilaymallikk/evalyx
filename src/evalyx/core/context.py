"""Correlation context for cross-cutting observability.

Typed helpers over :mod:`structlog.contextvars` (ContextVar-based, never a
mutable global). Bound values are merged into every structured log event by
the ``merge_contextvars`` processor that is already first in the logging
chain (see :mod:`evalyx.core.logging`).

Three correlation identifiers are supported:

- ``request_id`` — one HTTP API request (set by the observability middleware)
- ``run_id``     — one evaluation run (set by the Celery task that executes it)
- ``task_id``    — one Celery task invocation (reused from Celery's request
  context, never generated)

Correlation ids may appear in **logs** but must never be used as **metric
labels** (unbounded cardinality) — :mod:`evalyx.core.metrics` enforces this.

Every helper is safe to call from async handlers and worker threads: each
binding lives in its own ContextVar, so concurrent requests/tasks never see
each other's context (explicitly covered by tests).
"""

import structlog.contextvars

_REQUEST_ID = "request_id"
_RUN_ID = "run_id"
_TASK_ID = "task_id"

_CORRELATION_KEYS = (_REQUEST_ID, _RUN_ID, _TASK_ID)


def _bind(key: str, value: str) -> None:
    structlog.contextvars.bind_contextvars(**{key: value})


def _get(key: str) -> str | None:
    """Return the bound value for ``key`` in the current context, if any."""
    value = structlog.contextvars.get_contextvars().get(key)
    if value is None:
        return None
    assert isinstance(value, str)  # only these helpers bind these keys
    return value


def _unbind(*keys: str) -> None:
    structlog.contextvars.unbind_contextvars(*keys)


def set_request_id(value: str) -> None:
    """Bind the correlation id of the current HTTP request."""
    _bind(_REQUEST_ID, value)


def get_request_id() -> str | None:
    """Return the current request id, or ``None`` outside a request."""
    return _get(_REQUEST_ID)


def set_run_id(value: str) -> None:
    """Bind the id of the evaluation run being executed."""
    _bind(_RUN_ID, value)


def get_run_id() -> str | None:
    """Return the current evaluation run id, if bound."""
    return _get(_RUN_ID)


def set_task_id(value: str) -> None:
    """Bind the id of the Celery task invocation being executed."""
    _bind(_TASK_ID, value)


def get_task_id() -> str | None:
    """Return the current Celery task id, if bound."""
    return _get(_TASK_ID)


def clear_correlation_context() -> None:
    """Unbind all correlation ids.

    Called when a request/task ends so context never leaks into whatever
    work reuses the thread or connection afterwards.
    """
    _unbind(*_CORRELATION_KEYS)
