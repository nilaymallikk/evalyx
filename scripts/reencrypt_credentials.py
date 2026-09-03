#!/usr/bin/env python
"""Re-encrypt stored application credentials to the current key (Phase 18).

Operator rotation workflow:

    1. generate a new key and deploy it as EVALYX_ENCRYPTION_KEY, moving the
       old key to EVALYX_PREVIOUS_ENCRYPTION_KEYS (reads keep working);
    2. dry-run:  uv run python scripts/reencrypt_credentials.py --dry-run
    3. apply:    uv run python scripts/reencrypt_credentials.py --apply
    4. dry-run again (expect zero remaining), then remove the old key from
       EVALYX_PREVIOUS_ENCRYPTION_KEYS.

Output is counts and application ids only — never plaintext, ciphertext,
or key material.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from evalyx.core.config import get_settings
from evalyx.core.key_rotation import reencrypt_application_secrets
from evalyx.db.session import DatabaseManager


async def _run(apply: bool) -> int:
    settings = get_settings()
    manager = DatabaseManager(settings)
    try:
        report = await reencrypt_application_secrets(
            manager.session_factory, settings, dry_run=not apply
        )
    finally:
        await manager.dispose()
    print(f"mode={'apply' if apply else 'dry-run'}")
    print(f"examined={report.examined} already_current={report.already_current} "
          f"reencrypted={report.reencrypted} "
          f"skipped_no_secret={report.skipped_no_secret} "
          f"undecryptable={report.undecryptable}")
    for application_id in report.undecryptable_application_ids:
        print(f"undecryptable={application_id}")
    if report.undecryptable:
        return 2
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    try:
        return asyncio.run(_run(apply=args.apply))
    except Exception as exc:  # noqa: BLE001 — operator script: loud failure
        # Type name only: exception text can carry settings/connection
        # values that must never reach CI logs.
        print(f"re-encryption failed: {type(exc).__name__}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
