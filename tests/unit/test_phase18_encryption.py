"""Phase 18 encryption keyring + rotation tests (hermetic).

Covers: v2 round-trip, legacy v1 readability through the ring, unknown key
ids, wrong keys, idempotency detection, settings-built keyrings, and the
rule that failures never carry key/ciphertext/plaintext material.
"""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from evalyx.core.encryption import (
    CIPHERTEXT_VERSION,
    LEGACY_CIPHERTEXT_VERSION,
    EncryptionError,
    SecretEncryptor,
    decode_encryption_key,
    generate_encryption_key,
    key_id_for_key,
)

SECRET = "sk-test-" + "rotation-credential"


def _encryptor_and_key():
    raw = generate_encryption_key()
    return SecretEncryptor(decode_encryption_key(raw)), raw


def test_v2_roundtrip():
    encryptor, _ = _encryptor_and_key()
    envelope = encryptor.encrypt(SECRET)
    assert envelope.startswith(CIPHERTEXT_VERSION + ":")
    assert len(envelope.split(":")) == 4
    assert encryptor.decrypt(envelope) == SECRET


def test_key_id_deterministic_and_short():
    key = decode_encryption_key(generate_encryption_key())
    assert key_id_for_key(key) == key_id_for_key(key)
    assert len(key_id_for_key(key)) == 8


def test_legacy_v1_still_decrypts():
    # Simulate a Phase 15 row: craft a v1 envelope with the legacy lib shape.
    import base64
    import os

    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    raw = os.urandom(32)
    aesgcm = AESGCM(raw)
    nonce = os.urandom(12)
    ciphertext = aesgcm.encrypt(nonce, SECRET.encode(), None)
    envelope = (
        f"{LEGACY_CIPHERTEXT_VERSION}:"
        f"{base64.urlsafe_b64encode(nonce).decode()}:"
        f"{base64.urlsafe_b64encode(ciphertext).decode()}"
    )
    ringed = SecretEncryptor(raw, previous_keys={})
    assert ringed.decrypt(envelope) == SECRET
    # A foreign key cannot read it (same generic message, no oracle).
    foreign, _ = _encryptor_and_key()
    with pytest.raises(EncryptionError, match="could not be decrypted"):
        foreign.decrypt(envelope)


def test_previous_key_in_ring_decrypts_old_envelopes():
    old_raw = generate_encryption_key()
    old = SecretEncryptor(decode_encryption_key(old_raw))
    old_envelope = old.encrypt(SECRET)

    new_raw = generate_encryption_key()
    new_key = decode_encryption_key(new_raw)
    old_key = decode_encryption_key(old_raw)
    ringed = SecretEncryptor(
        new_key, previous_keys={key_id_for_key(old_key): old_key}
    )
    # Old envelope readable through history...
    assert ringed.decrypt(old_envelope) == SECRET
    # ...new writes use the current key only.
    fresh = ringed.encrypt(SECRET)
    assert ringed.is_current_envelope(fresh) is True
    assert ringed.decrypt(fresh) == SECRET


def test_previous_keys_from_settings():
    old_raw = generate_encryption_key()
    new_raw = generate_encryption_key()
    old = SecretEncryptor(decode_encryption_key(old_raw))
    old_envelope = old.encrypt(SECRET)

    class _Settings:
        evalyx_encryption_key = SecretStr(new_raw)
        previous_encryption_key_values = (old_raw,)

    ringed = SecretEncryptor.from_settings(_Settings())
    assert ringed.decrypt(old_envelope) == SECRET
    # New writes use the current key.
    fresh = ringed.encrypt(SECRET)
    assert ringed.is_current_envelope(fresh) is True
    assert ringed.is_current_envelope(old_envelope) is False


def test_unknown_key_id_rejected_without_detail():
    encryptor, _ = _encryptor_and_key()
    with pytest.raises(EncryptionError, match="could not be decrypted"):
        encryptor.decrypt(f"{CIPHERTEXT_VERSION}:deadbeef:AAAA:AAAA")


def test_malformed_envelopes_rejected():
    encryptor, _ = _encryptor_and_key()
    for bad in ("", "v2:only", "v9:a:b:c", "v2:short:AAAA:AAAA", "not-an-envelope"):
        with pytest.raises(EncryptionError):
            encryptor.decrypt(bad)


def test_failures_carry_no_material():
    encryptor, raw = _encryptor_and_key()
    envelope = encryptor.encrypt(SECRET)
    for probe in (envelope[:-4] + "AAAA", f"{CIPHERTEXT_VERSION}:deadbeef:AAAA:AAAA"):
        try:
            encryptor.decrypt(probe)
        except EncryptionError as exc:
            message = str(exc)
            assert SECRET not in message
            assert envelope not in message
            assert raw not in message
        else:  # pragma: no cover — tampering must not verify
            raise AssertionError("tampered envelope verified")


def test_is_current_envelope_idempotent_marker():
    first, _ = _encryptor_and_key()
    second, _ = _encryptor_and_key()
    envelope = first.encrypt(SECRET)
    assert first.is_current_envelope(envelope) is True
    assert second.is_current_envelope(envelope) is False
    assert first.is_current_envelope("garbage") is False
