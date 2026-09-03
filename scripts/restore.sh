#!/bin/sh
# PostgreSQL restore for Evalyx production (Phase 17).
#
# Usage:
#   POSTGRES_USER=evalyx POSTGRES_DB=evalyx ./scripts/restore.sh <dump-file>
#
# Procedure (also in docs/deployment.md):
#   1. stop api + worker (keep postgres running)
#   2. verify checksum (sha256sum -c <dump>.sha256)
#   3. restore with pg_restore into a scratch database first when possible
#   4. run migrations (alembic upgrade head) against the restored database
#   5. start the application and verify data (record counts, latest run)
#
# DANGER: this overwrites the target database. Take a fresh backup first.
set -eu

DUMP="${1:?usage: restore.sh <dump-file>}"
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.production.yml}"

echo "verify checksum first: sha256sum -c $DUMP.sha256"
docker compose -f "$COMPOSE_FILE" exec -T postgres \
  pg_restore -U "${POSTGRES_USER:?set POSTGRES_USER}" \
             -d "${POSTGRES_DB:-evalyx}" --clean --if-exists < "$DUMP"
echo "restore complete; next: run migrations, then verify (see docs/deployment.md)."
