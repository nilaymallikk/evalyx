#!/bin/sh
# PostgreSQL logical backup for Evalyx production (Phase 17).
#
# Usage:
#   POSTGRES_USER=evalyx POSTGRES_DB=evalyx ./scripts/backup.sh [output-dir]
# For a compose deployment:
#   docker compose -f docker-compose.production.yml exec -T postgres \
#     pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc > backup.dump
#
# This script runs pg_dump inside the postgres service and stores a
# timestamped custom-format dump plus SHA-256 checksum in the output dir.
# Operator responsibility: copy the dump + checksum off-host (encrypted
# storage), rotate per the retention policy in docs/deployment.md.
set -eu

OUT_DIR="${1:-./backups}"
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.production.yml}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
DEST="$OUT_DIR/evalyx-$STAMP.dump"

mkdir -p "$OUT_DIR"
docker compose -f "$COMPOSE_FILE" exec -T postgres \
  pg_dump -U "${POSTGRES_USER:?set POSTGRES_USER}" \
           -d "${POSTGRES_DB:-evalyx}" -Fc > "$DEST"
sha256sum "$DEST" > "$DEST.sha256"
echo "backup written: $DEST"
