"""Phase 18 key-rotation integration tests (live PostgreSQL).

Full operator workflow: store under key A → deploy key B with A as history
(reads keep working) → dry-run → apply → verify → second run is a no-op.
Plaintext and keys never appear in reports, logs, or metadata.
"""

from __future__ import annotations

import uuid

import pytest

from evalyx.core.config import Settings
from evalyx.core.encryption import SecretEncryptor, generate_encryption_key
from evalyx.core.key_rotation import (
    count_pending_reencryption,
    reencrypt_application_secrets,
)
from evalyx.db.models import Application
from evalyx.db.session import DatabaseManager

pytestmark = pytest.mark.integration

_CREDENTIAL = "rotation-secret-" + "not-a-real-credential"


def _settings_for(key: str, previous: str = "") -> Settings:
    return Settings(
        _env_file=None,
        evalyx_secret_key="placeholder",
        auth_required=False,
        evalyx_encryption_key=key,
        evalyx_previous_encryption_keys=previous,
    )


async def _store_application(db: DatabaseManager, settings: Settings) -> uuid.UUID:
    async with db.session() as session:
        from evalyx.db.models import Organization

        org = Organization(
            clerk_organization_id=f"org_rotation_{uuid.uuid4().hex[:8]}",
            name="rotation org",
        )
        session.add(org)
        await session.commit()
        await session.refresh(org)
        encryptor = SecretEncryptor.from_settings(settings)
        app = Application(
            organization_id=org.id,
            name=f"rot-{uuid.uuid4().hex[:6]}",
            connection_type="http",
            encrypted_secret=encryptor.encrypt(_CREDENTIAL),
            secret_metadata={"key_version": "v2", "key_id": encryptor.current_key_id},
        )
        session.add(app)
        await session.commit()
        await session.refresh(app)
        return app.id


async def test_full_rotation_workflow(clean_db: DatabaseManager):
    key_a = generate_encryption_key()
    key_b = generate_encryption_key()
    settings_a = _settings_for(key_a)
    settings_b_with_history = _settings_for(key_b, previous=key_a)

    app_id = await _store_application(clean_db, settings_a)

    # Reads keep working through the rotation window (history key).
    async with clean_db.session() as session:
        row = await session.get(Application, app_id)
        assert row is not None and row.encrypted_secret is not None
        assert SecretEncryptor.from_settings(settings_b_with_history).decrypt(
            row.encrypted_secret
        ) == _CREDENTIAL

    # Dry-run changes nothing but reports the pending row.
    dry = await reencrypt_application_secrets(
        clean_db.session_factory, settings_b_with_history, dry_run=True
    )
    assert dry.examined >= 1
    assert dry.reencrypted == 1
    assert dry.undecryptable == 0
    assert await count_pending_reencryption(
        clean_db.session_factory, settings_b_with_history
    ) == 1

    # Apply re-encrypts; the report carries counts and ids only.
    applied = await reencrypt_application_secrets(
        clean_db.session_factory, settings_b_with_history, dry_run=False
    )
    assert applied.reencrypted == 1
    assert applied.undecryptable == 0
    assert _CREDENTIAL not in str(applied)
    assert key_a not in str(applied) and key_b not in str(applied)

    # New key reads the row; history can be dropped (fresh settings, no A).
    settings_b_alone = _settings_for(key_b)
    async with clean_db.session() as session:
        row = await session.get(Application, app_id)
        assert row is not None and row.encrypted_secret is not None
        assert SecretEncryptor.from_settings(settings_b_alone).decrypt(
            row.encrypted_secret
        ) == _CREDENTIAL
        assert row.secret_metadata is not None
        assert row.secret_metadata.get("key_version") == "v2"

    # Idempotent: a second run (and the pending counter) finds nothing to do.
    again = await reencrypt_application_secrets(
        clean_db.session_factory, settings_b_with_history, dry_run=False
    )
    assert again.reencrypted == 0
    assert again.already_current >= 1
    assert await count_pending_reencryption(
        clean_db.session_factory, settings_b_with_history
    ) == 0


async def test_undecryptable_rows_reported_by_id_only(
    clean_db: DatabaseManager,
):
    key = generate_encryption_key()
    settings = _settings_for(key)
    app_id = await _store_application(clean_db, settings)
    # Corrupt the envelope directly (simulates foreign-key ciphertext).
    async with clean_db.session() as session:
        row = await session.get(Application, app_id)
        assert row is not None
        row.encrypted_secret = "v2:deadbeef:AAAA:AAAA"
        await session.commit()

    report = await reencrypt_application_secrets(
        clean_db.session_factory, settings, dry_run=False
    )
    assert report.undecryptable == 1
    assert report.undecryptable_application_ids == [str(app_id)]
    assert report.reencrypted == 0
