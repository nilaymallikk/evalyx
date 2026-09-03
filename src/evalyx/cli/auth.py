"""Local credential storage for the CLI (Phase 16).

The CLI stores exactly one credential: the bearer token it attaches to
Evalyx API requests (a Clerk session token in production, the dev-mode
organization header value locally). Storage strategy, most secure first:

1. OS keyring (via ``keyring``) when a backend is available — the token
   never touches disk in plaintext.
2. Fallback: ``~/.config/evalyx/credentials.json`` with ``0600`` permissions.

The store deliberately refuses to hold server-side secrets: Clerk secret
keys, Evalyx encryption keys, and application API keys have no API here.
"""

import json
import os
import stat
import sys

from evalyx.cli.config import config_dir

_SERVICE = "evalyx-cli"
_ACCOUNT = "api-token"
_KEYRING_KEY = f"{_SERVICE}:{_ACCOUNT}"

#: Keys permitted in the fallback file — nothing else may ever appear.
_ALLOWED_KEYS = frozenset({"token"})


def _fallback_file():
    return config_dir() / "credentials.json"


def _keyring_available() -> bool:
    try:
        import keyring

        backend = keyring.get_keyring()
        # The "fail" backend raises on every operation; "null" silently drops.
        name = type(backend).__module__ + "." + type(backend).__name__
        return "fail" not in name.lower() and "null" not in name.lower()
    except Exception:  # noqa: BLE001 — any keyring trouble → fallback file
        return False


def load_token() -> str | None:
    """The stored bearer token, or ``None`` when logged out."""
    if _keyring_available():
        try:
            import keyring

            value = keyring.get_password(_SERVICE, _ACCOUNT)
            return value or None
        except Exception:  # noqa: BLE001, S110 — fall through to the file store
            pass
    path = _fallback_file()
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — unreadable store counts as logged out
        return None
    token = data.get("token")
    return token if isinstance(token, str) and token else None


def save_token(token: str) -> str:
    """Store the bearer token; returns ``"keyring"`` or ``"file"``."""
    if _keyring_available():
        try:
            import keyring

            keyring.set_password(_SERVICE, _ACCOUNT, token)
            return "keyring"
        except Exception:  # noqa: BLE001, S110 — fall through to the file store
            pass
    path = _fallback_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"token": token}
    if path.exists():
        try:  # keep any future (whitelisted) keys that may join the file
            existing = json.loads(path.read_text(encoding="utf-8"))
            payload = {
                key: value
                for key, value in {**existing, **payload}.items()
                if key in _ALLOWED_KEYS
            }
        except Exception:  # noqa: BLE001, S110 — unreadable file is simply replaced
            pass
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)  # 0600
    return "file"


def clear_token() -> None:
    """Remove the stored credential from every storage location."""
    if _keyring_available():
        try:
            import keyring

            keyring.delete_password(_SERVICE, _ACCOUNT)
        except Exception:  # noqa: BLE001, S110 — nothing stored / backend refused
            pass
    try:
        _fallback_file().unlink()
    except FileNotFoundError:
        pass
    except Exception:  # noqa: BLE001 — best-effort cleanup
        print(
            "warning: could not remove the fallback credential file", file=sys.stderr
        )
