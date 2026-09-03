"""Operator credential re-encryption workflow (Phase 18 rotation).

Safe, idempotent migration of stored application credentials to the
current encryption key:

1. operator generates a new key and deploys it as ``EVALYX_ENCRYPTION_KEY``,
   moving the old key to ``EVALYX_PREVIOUS_ENCRYPTION_KEYS`` (decrypt still
   works everywhere — old rows remain readable during the whole window);
2. operator runs this workflow (dry-run first, then apply): every stored
   envelope already on the current key is skipped; every other decryptable
   envelope is re-encrypted to the current key;
3. operator verifies (dry-run reports zero remaining) and removes the old
   key from ``EVALYX_PREVIOUS_ENCRYPTION_KEYS``.

Properties:

- idempotent and resumable: re-running skips ``is_current_envelope`` rows,
  commits per row, and never double-encrypts;
- safe output: the summary carries counts and application ids only — never
  plaintext, ciphertext, or keys. Undecryptable rows are reported by id so
  the operator can investigate without secret exposure;
- atomicity per row (one UPDATE + commit each), so an interrupted run keeps
  every completed row and can simply be re-run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from evalyx.core.config import Settings
from evalyx.core.encryption import CIPHERTEXT_VERSION, EncryptionError, SecretEncryptor
from evalyx.db.models import Application

logger = structlog.get_logger(__name__)


@dataclass
class ReencryptionReport:
    """Counts-only rotation summary (safe to print and log)."""

    examined: int = 0
    already_current: int = 0
    reencrypted: int = 0
    skipped_no_secret: int = 0
    undecryptable: int = 0
    undecryptable_application_ids: list[str] = field(default_factory=list)
    dry_run: bool = True


async def reencrypt_application_secrets(
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
    *,
    dry_run: bool = True,
    batch_size: int = 100,
) -> ReencryptionReport:
    """Re-encrypt stored credentials to the current key (idempotent).

    With ``dry_run=True`` nothing is written; the report shows what an
    apply run would do. Rows already on the current key are never touched.
    """
    encryptor = SecretEncryptor.from_settings(settings)
    report = ReencryptionReport(dry_run=dry_run)
    offset = 0
    while True:
        async with session_factory() as session:
            rows = (
                await session.execute(
                    select(Application.id, Application.encrypted_secret)
                    .order_by(Application.created_at, Application.id)
                    .offset(offset)
                    .limit(batch_size)
                )
            ).all()
            if not rows:
                break
            for application_id, envelope in rows:
                report.examined += 1
                if not envelope:
                    report.skipped_no_secret += 1
                    continue
                if encryptor.is_current_envelope(envelope):
                    report.already_current += 1
                    continue
                try:
                    plaintext = encryptor.decrypt(envelope)
                except EncryptionError:
                    report.undecryptable += 1
                    report.undecryptable_application_ids.append(str(application_id))
                    logger.error(
                        "reencryption_undecryptable",
                        application_id=str(application_id),
                    )
                    continue
                if dry_run:
                    report.reencrypted += 1
                    continue
                application = await session.get(Application, application_id)
                if application is None:  # deleted mid-run; next run converges
                    continue
                application.encrypted_secret = encryptor.encrypt(plaintext)
                metadata = dict(application.secret_metadata or {})
                metadata["key_version"] = CIPHERTEXT_VERSION
                metadata["key_id"] = encryptor.current_key_id
                application.secret_metadata = metadata
                session.add(application)
                await session.commit()
                report.reencrypted += 1
                logger.info(
                    "reencryption_reencrypted",
                    application_id=str(application_id),
                )
        offset += batch_size
    return report


async def count_pending_reencryption(
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
) -> int:
    """How many stored envelopes are not yet on the current key."""
    encryptor = SecretEncryptor.from_settings(settings)
    pending = 0
    offset = 0
    batch_size = 500
    while True:
        async with session_factory() as session:
            envelopes = (
                await session.execute(
                    select(Application.encrypted_secret)
                    .where(Application.encrypted_secret.is_not(None))
                    .order_by(Application.created_at, Application.id)
                    .offset(offset)
                    .limit(batch_size)
                )
            ).scalars().all()
            if not envelopes:
                break
            pending += sum(
                1
                for envelope in envelopes
                if envelope is not None and not encryptor.is_current_envelope(envelope)
            )
        offset += batch_size
    return pending


__all__ = [
    "ReencryptionReport",
    "count_pending_reencryption",
    "reencrypt_application_secrets",
]
