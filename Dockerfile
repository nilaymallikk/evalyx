# Evalyx production image (Phase 17).
#
# Multi-stage, deterministic, non-root, secret-free:
# - builder resolves dependencies from uv.lock (reproducible)
# - runtime contains only the app package + production deps
# - no .env, no credentials, no VCS metadata, no dev dependencies
# - runs as non-root user `evalyx`
#
# Build:   docker build -t evalyx:prod .
# Inspect: docker run --rm evalyx:prod python -c "import evalyx"

FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Dependency manifests first (better layer caching).
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Application source.
COPY src ./src
COPY migrations ./migrations
COPY alembic.ini ./
COPY README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

FROM python:3.14-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PATH=/app/.venv/bin:$PATH \
    APP_ENV=production

WORKDIR /app

# Non-root user first so all copied files can be owned correctly.
RUN groupadd --system --gid 10001 evalyx \
    && useradd --system --uid 10001 --gid evalyx --no-log-init \
        --home /app --shell /usr/sbin/nologin evalyx

COPY --from=builder --chown=evalyx:evalyx /app/.venv /app/.venv
COPY --from=builder --chown=evalyx:evalyx /app/src /app/src
COPY --from=builder --chown=evalyx:evalyx /app/migrations /app/migrations
COPY --chown=evalyx:evalyx alembic.ini ./
COPY --chown=evalyx:evalyx docker/entrypoint-api.sh /app/entrypoint-api.sh
COPY --chown=evalyx:evalyx docker/entrypoint-worker.sh /app/entrypoint-worker.sh
RUN chmod 0755 /app/entrypoint-api.sh /app/entrypoint-worker.sh

USER evalyx

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health').read()"

# Default: API. The worker service overrides the command (see
# docker-compose.production.yml).
ENTRYPOINT ["/app/entrypoint-api.sh"]
