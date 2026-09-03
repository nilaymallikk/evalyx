"""CLI-local configuration: API URL, organization preference, output prefs.

Precedence (documented in the README): CLI flag > environment variable
(``EVALYX_API_URL`` / ``EVALYX_ORG``) > config file > default.

The config file is non-secret by design (``~/.config/evalyx/config.toml``):
no tokens, no Clerk secrets, no application credentials ever live here.
Authentication credentials go to the keyring-backed store (``auth.py``) or,
when no keyring backend exists, to ``~/.config/evalyx/credentials.json``
with ``0600`` permissions.
"""

import os
import stat
import tomllib
from dataclasses import dataclass
from pathlib import Path

#: Default API URL: the local development server.
DEFAULT_API_URL = "http://127.0.0.1:8000"

CONFIG_DIR_ENV = "EVALYX_CONFIG_DIR"


def config_dir() -> Path:
    """The CLI configuration directory (overridable for tests)."""
    override = os.environ.get(CONFIG_DIR_ENV)
    if override:
        return Path(override)
    return Path(
        os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
    ) / "evalyx"


def config_file() -> Path:
    return config_dir() / "config.toml"


@dataclass(frozen=True)
class Config:
    """Resolved CLI configuration (non-secret)."""

    api_url: str = DEFAULT_API_URL
    org: str | None = None
    quiet: bool = False
    verbose: bool = False
    timeout: float = 30.0
    poll_interval: float = 2.0
    poll_timeout: float = 3600.0


def load_config(
    *,
    api_url: str | None = None,
    org: str | None = None,
    quiet: bool = False,
    verbose: bool = False,
    timeout: float | None = None,
) -> Config:
    """Resolve configuration: CLI flag > environment > config file > default."""
    file_values: dict = {}
    path = config_file()
    if path.is_file():
        try:
            file_values = tomllib.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 — a broken config file falls back to defaults
            file_values = {}

    resolved_api_url = (
        api_url
        or os.environ.get("EVALYX_API_URL")
        or file_values.get("api_url")
        or DEFAULT_API_URL
    )
    resolved_org = (
        org or os.environ.get("EVALYX_ORG") or file_values.get("org") or None
    )
    resolved_timeout = (
        timeout
        or float(os.environ.get("EVALYX_TIMEOUT", "") or 0)
        or float(file_values.get("timeout", 0) or 0)
        or 30.0
    )
    return Config(
        api_url=str(resolved_api_url).rstrip("/"),
        org=resolved_org,
        quiet=quiet,
        verbose=verbose,
        timeout=resolved_timeout,
        poll_interval=float(file_values.get("poll_interval", 2.0)),
        poll_timeout=float(file_values.get("poll_timeout", 3600.0)),
    )


def save_config(api_url: str | None = None, org: str | None = None) -> None:
    """Persist chosen non-secret preferences as a minimal TOML file."""
    directory = config_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = config_file()
    lines = ["# Evalyx CLI configuration (non-secret).", ""]
    if api_url:
        lines.append(f'api_url = "{api_url}"')
    if org:
        lines.append(f'org = "{org}"')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)  # 0600 — same caution as credentials
