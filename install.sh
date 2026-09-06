#!/usr/bin/env bash
# Evalyx one-line installer (local-first: no accounts, no login).
#
#   curl -fsSL https://raw.githubusercontent.com/nilaymallikk/evalyx/main/install.sh | bash
#
# What it does:
#   1. checks Docker, git, and Python 3
#   2. installs `uv` if missing
#   3. clones Evalyx to $EVALYX_DIR (default: ~/evalyx) or updates it
#   4. installs dependencies, creates .env with generated secrets
#   5. starts PostgreSQL + Redis, applies migrations
#
# Safe to re-run: it never overwrites an existing .env or database.
# Afterward, open two terminals:
#   uv run python main.py                                            # API
#   uv run celery -A evalyx.worker.celery_app worker --loglevel=INFO # worker
set -euo pipefail

REPO_URL="${EVALYX_REPO:-https://github.com/nilaymallikk/evalyx.git}"
EVALYX_DIR="${EVALYX_DIR:-$HOME/evalyx}"

say() { printf '%s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

# -- prerequisites ------------------------------------------------------------

command -v docker >/dev/null 2>&1 || die "Docker is required (https://docs.docker.com/get-docker/)"
docker info >/dev/null 2>&1 || die "Docker daemon is not running — start Docker first."
command -v git >/dev/null 2>&1 || die "git is required."
command -v python3 >/dev/null 2>&1 || die "Python 3 is required."

if ! command -v uv >/dev/null 2>&1; then
    say "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || die "uv installation failed."

# -- source -------------------------------------------------------------------

if [ -d "$EVALYX_DIR/.git" ]; then
    say "Updating existing checkout at $EVALYX_DIR..."
    git -C "$EVALYX_DIR" pull --ff-only || say "(keeping local changes; pull skipped)"
else
    say "Cloning Evalyx to $EVALYX_DIR..."
    git clone "$REPO_URL" "$EVALYX_DIR"
fi
cd "$EVALYX_DIR"

# -- dependencies + environment -----------------------------------------------

say "Installing dependencies..."
uv sync

if [ ! -f .env ]; then
    say "Creating .env..."
    cp .env.example .env
fi
if grep -qE '^EVALYX_SECRET_KEY=$' .env; then
    SECRET="$(uv run python -c 'import secrets; print(secrets.token_urlsafe(32))')"
    # Replace only the empty value, never an existing secret.
    sed -i "s|^EVALYX_SECRET_KEY=$|EVALYX_SECRET_KEY=$SECRET|" .env
    say "Generated EVALYX_SECRET_KEY."
fi
if ! grep -qE '^OPENROUTER_API_KEY=.+' .env; then
    say "NOTE: set OPENROUTER_API_KEY in .env to run LLM evaluations."
fi

# -- infrastructure + database -------------------------------------------------

say "Starting PostgreSQL + Redis..."
docker compose up -d

say "Waiting for PostgreSQL..."
for _ in $(seq 1 30); do
    if docker compose exec -T postgres pg_isready -U evalyx >/dev/null 2>&1; then
        break
    fi
    sleep 2
done
docker compose exec -T postgres pg_isready -U evalyx >/dev/null 2>&1 \
    || die "PostgreSQL did not become ready."

say "Applying migrations..."
uv run alembic upgrade head

say ""
say "Done. Evalyx is installed at $EVALYX_DIR"
say "  Terminal 1:  uv run python main.py"
say "  Terminal 2:  uv run celery -A evalyx.worker.celery_app worker --loglevel=INFO"
say "  Then:        evalyx whoami"
