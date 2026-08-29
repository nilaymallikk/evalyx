"""Celery application for Evalyx background evaluation workers.

Redis (``REDIS_URL``) is both broker and result backend: it delivers jobs and
stores *operational* task state (PENDING / STARTED / RETRY / SUCCESS /
FAILURE). It is never the source of truth for evaluation state — PostgreSQL
owns ``EvaluationRun`` status, case results, and guardrail results. If Redis
disappears, persisted evaluation history remains intact.

The app is importable without starting a worker. Start one with:

    celery -A evalyx.worker.celery_app worker --loglevel=INFO

Reliability choices (goal: "do not lose evaluation jobs silently"):

- ``task_acks_late`` — a message is acknowledged only after the task
  finishes, so a worker crash redelivers the job instead of dropping it.
- ``task_reject_on_worker_lost`` — if the worker process is killed mid-task,
  the message is put back on the queue rather than acknowledged.
- ``worker_prefetch_multiplier = 1`` — fair dispatch; with late acks this
  prevents one worker from hoarding messages it may never finish.
- ``worker_concurrency`` from settings (default 2) — deliberately
  conservative: free OpenRouter models are rate-limited, and per-run case
  execution is sequential by design (Phase 5).
- ``visibility_timeout`` above — must exceed the hard time limit so Redis
  does not redeliver a message while its task is still executing.
- Bounded time limits from settings: evaluations may legitimately take
  time, but a stuck task must not run forever (see ``tasks.py`` for the
  soft-limit behavior).
"""

from celery import Celery
from celery.signals import setup_logging

from evalyx.core.config import Settings, get_settings
from evalyx.core.logging import configure_logging


def create_celery_app(settings: Settings | None = None) -> Celery:
    """Create the Evalyx Celery application from application settings."""
    settings = settings or get_settings()
    app = Celery("evalyx", include=["evalyx.worker.tasks"])
    app.conf.update(
        # Transports: Redis from existing configuration.
        broker_url=settings.celery_broker_url,
        result_backend=settings.celery_result_backend,
        # Serialization: JSON only — tasks receive lightweight identifiers,
        # never ORM objects, providers, or secrets.
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        # Reliability: do not lose evaluation jobs silently.
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        worker_prefetch_multiplier=1,
        worker_concurrency=settings.worker_concurrency,
        broker_connection_retry_on_startup=True,
        broker_transport_options={
            "visibility_timeout": settings.worker_visibility_timeout_seconds,
        },
        result_backend_transport_options={
            "visibility_timeout": settings.worker_visibility_timeout_seconds,
        },
        # Task state is operational metadata, not evaluation state.
        task_track_started=True,
        result_expires=settings.worker_result_ttl_seconds,
        # Time limits: explicit, generous, bounded.
        task_soft_time_limit=settings.worker_soft_time_limit_seconds,
        task_time_limit=settings.worker_hard_time_limit_seconds,
        task_default_queue="evalyx",
    )
    return app


@setup_logging.connect
def _configure_worker_logging(**_: object) -> None:
    """Use Evalyx structured logging in workers.

    Prevents Celery from hijacking the root logger with its own format.
    """
    configure_logging(get_settings())


#: Module-level app instance for `celery -A evalyx.worker.celery_app worker`.
celery_app = create_celery_app()
