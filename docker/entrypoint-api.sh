#!/bin/sh
# Production API entrypoint (Phase 17).
#
# Environment-driven, production-grade uvicorn:
# - host/port/workers from API_HOST / API_PORT / API_WORKERS
#   (backed by Settings api_host/api_port/api_workers)
# - NO --reload, ever
# - graceful shutdown is cooperative: SIGTERM stops the listener, in-flight
#   requests drain, then the lifespan handler closes DB/Redis pools.
set -eu

# Allow one-off inspection commands (e.g. `docker run <image> sh`).
if [ "$#" -gt 0 ]; then
    exec "$@"
fi

HOST="${API_HOST:-${EVALYX_API_HOST:-0.0.0.0}}"
PORT="${API_PORT:-${EVALYX_API_PORT:-8000}}"
WORKERS="${API_WORKERS:-1}"

exec python -m uvicorn evalyx.api.app:app \
  --host "$HOST" \
  --port "$PORT" \
  --workers "$WORKERS" \
  --proxy-headers \
  --forwarded-allow-ips='*' \
  --timeout-graceful-shutdown 30
