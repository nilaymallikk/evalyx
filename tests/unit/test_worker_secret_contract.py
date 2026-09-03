"""Secrets never travel through the Celery task boundary (Phase 15 Step 15).

The worker task accepts identifiers only — never credentials, never ORM
objects, never provider instances. These tests pin that contract so a
future change cannot silently start shipping secrets into task arguments.
"""

import inspect
import uuid

from evalyx.api.services import default_enqueue
from evalyx.worker.tasks import run_evaluation


def test_run_evaluation_accepts_only_run_id():
    """The task body signature is (run_id) — nothing else."""
    body = run_evaluation.__wrapped__
    parameters = list(inspect.signature(body).parameters)
    assert parameters == ["run_id"]


def test_enqueue_passes_only_the_run_id(monkeypatch):
    """``default_enqueue`` must serialize exactly one identifier to Celery."""
    captured: list[tuple] = []

    class _FakeAsyncResult:
        id = "task-000"

    class _FakeTask:
        @staticmethod
        def delay(*args, **kwargs):
            captured.append((args, kwargs))
            return _FakeAsyncResult()

    monkeypatch.setattr("evalyx.worker.tasks.run_evaluation", _FakeTask)
    run_id = uuid.uuid4()
    task_id = default_enqueue(run_id)
    assert task_id == "task-000"
    assert len(captured) == 1
    args, kwargs = captured[0]
    assert args == (str(run_id),)  # stringified identifier only
    assert kwargs == {}


def test_task_argument_is_a_bare_string_run_id():
    """Task arguments are stringified identifiers — never secret values."""
    from evalyx.worker.tasks import _coerce_run_id

    assert _coerce_run_id("123e4567-e89b-12d3-a456-426614174000") == uuid.UUID(
        "123e4567-e89b-12d3-a456-426614174000"
    )
    assert _coerce_run_id(uuid.uuid4()) is not None