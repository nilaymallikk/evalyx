"""Evalyx background worker package (Phase 7).

Thin Celery orchestration around the existing evaluation pipeline:

- :mod:`evalyx.worker.celery_app` — the Celery application (Redis broker/backend)
- :mod:`evalyx.worker.tasks` — task definitions (lightweight identifiers only)
- :mod:`evalyx.worker.execution` — the async execution core (Celery → asyncio
  boundary, run eligibility, provider lifecycle)

Business logic stays in :mod:`evalyx.evaluation` and :mod:`evalyx.guardrails`;
the worker never duplicates it. PostgreSQL remains the source of truth for
evaluation state; Redis/Celery only deliver jobs and track operational state.

Start a worker with::

    celery -A evalyx.worker.celery_app worker --loglevel=INFO
"""
