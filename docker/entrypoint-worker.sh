#!/bin/sh
# Production Celery worker entrypoint (Phase 17).
#
# Same image + same environment configuration as the API; connects to Redis
# (broker) and PostgreSQL (state). Task args carry run ids only — never
# application secrets (see worker/tasks.py). Concurrency/acks/timeouts come
# from Settings so operators tune via WORKER_* env vars.
#
# Graceful shutdown: SIGTERM -> Celery warm shutdown (no new tasks, current
# task finishes within the soft/hard time limits, pools close).
set -eu

# Allow one-off inspection commands (e.g. `docker run <image> sh`).
if [ "$#" -gt 0 ]; then
    exec "$@"
fi

CONCURRENCY="${WORKER_CONCURRENCY:-2}"
LOGLEVEL="${WORKER_LOGLEVEL:-INFO}"

exec celery -A evalyx.worker.celery_app worker \
  --loglevel="$LOGLEVEL" \
  --concurrency="$CONCURRENCY" \
  --queues=evalyx
