"""Output helpers: human tables + strict JSON mode.

JSON mode is contract-grade: only valid JSON reaches stdout — no banners,
progress, or ANSI sequences. Human mode uses ASCII-safe glyphs with text
fallbacks (``[OK]`` / ``[FAIL]``) when Unicode support is unavailable.
"""

import json
import os
import sys
from typing import Any


def supports_unicode() -> bool:
    """True when stdout can render the tick/cross glyphs (never relied on
    for critical information — text fallbacks exist everywhere)."""
    if os.environ.get("EVALYX_ASCII"):
        return False
    try:
        "✓✗".encode(sys.stdout.encoding or "ascii")
    except (LookupError, UnicodeEncodeError):
        return False
    return True


def mark(ok: bool) -> str:
    if not supports_unicode():
        return "[OK]" if ok else "[FAIL]"
    return "✓" if ok else "✗"


def arrow_up() -> str:
    return "↑" if supports_unicode() else "+"


def arrow_down() -> str:
    return "↓" if supports_unicode() else "-"


def emit_json(data: Any) -> None:
    """Print strict machine-readable JSON (CI contract)."""
    json.dump(data, sys.stdout, indent=2, sort_keys=False, default=str)
    sys.stdout.write("\n")


def info(message: str) -> None:
    """Human-only progress line (suppressed automatically in JSON mode by
    callers; never written to stdout so stdout stays parseable)."""
    print(message, file=sys.stderr)


def human_table(rows: list[list[str]], headers: list[str]) -> None:
    """Fixed-width table with deterministic column sizing."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(line)
    print("  ".join("─" * w for w in widths))
    for row in rows:
        print("  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))


def kv(pairs: list[tuple[str, Any]]) -> None:
    """Aligned key/value block for detail views."""
    width = max(len(k) for k, _ in pairs) if pairs else 0
    for key, value in pairs:
        print(f"{key.ljust(width)}  {value}")
