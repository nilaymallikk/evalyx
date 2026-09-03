"""Hermetic tests for the secret-encryption boundary (Phase 15 Step 20).

Covers round-tripping, nonce uniqueness, tamper detection, wrong-key
rejection, envelope versioning, and the rule that no key/ciphertext/
plaintext material ever appears in exception messages.
"""

import base64

import pytest

from evalyx.core.encryption import (
    CIPHERTEXT_VERSION,
    EncryptionError,
    SecretEncryptor,
    decode_encryption_key,
    generate_encryption_key,
)

SECRET = "sk-test-" + "not-a-real-credential"


@pytest.fixture
def encryptor() -> SecretEncryptor:
    return SecretEncryptor(decode_encryption_key(generate_encryption_key()))


def test_roundtrip(encryptor: SecretEncryptor):
    envelope = encryptor.encrypt(SECRET)
    assert encryptor.decrypt(envelope) == SECRET


def test_envelope_is_versioned(encryptor: SecretEncryptor):
    envelope = encryptor.encrypt(SECRET)
    assert envelope.startswith(CIPHERTEXT_VERSION + ":")
    assert len(envelope.split(":")) == 3


def test_plaintext_never_appears_in_envelope(encryptor: SecretEncryptor):
    envelope = encryptor.encrypt(SECRET)
    assert SECRET not in envelope
    assert base64.b64encode(SECRET.encode()).decode() not in envelope


def test_nonces_are_unique(encryptor: SecretEncryptor):
    a = encryptor.encrypt(SECRET)
    b = encryptor.encrypt(SECRET)
    assert a != b  # fresh random nonce per encryption


def test_tampered_ciphertext_rejected(encryptor: SecretEncryptor):
    envelope = encryptor.encrypt(SECRET)
    version, nonce, ciphertext = envelope.split(":")
    corrupted = list(ciphertext)
    corrupted[0] = "A" if corrupted[0] != "A" else "B"
    tampered = f"{version}:{nonce}:{''.join(corrupted)}"
    with pytest.raises(EncryptionError, match="could not be decrypted"):
        encryptor.decrypt(tampered)


def test_wrong_key_rejected():
    other = SecretEncryptor(decode_encryption_key(generate_encryption_key()))
    envelope = SecretEncryptor(decode_encryption_key(generate_encryption_key())).encrypt(
        SECRET
    )
    with pytest.raises(EncryptionError):
        other.decrypt(envelope)


def test_malformed_envelope_rejected(encryptor: SecretEncryptor):
    with pytest.raises(EncryptionError):
        encryptor.decrypt("v1:not-a-valid-envelope")
    with pytest.raises(EncryptionError):
        encryptor.decrypt("v9:aaaa:bbbb")  # unknown future version


def test_exception_messages_never_carry_material(encryptor: SecretEncryptor):
    envelope = encryptor.encrypt(SECRET)
    try:
        encryptor.decrypt(envelope[:-4] + "AAAA")
    except EncryptionError as exc:
        message = str(exc)
        assert SECRET not in message
        assert envelope not in message
        assert "v1" not in message


def test_decode_key_validates_length():
    with pytest.raises(ValueError, match="32 bytes"):
        decode_encryption_key(base64.urlsafe_b64encode(b"short").decode())
    with pytest.raises(ValueError):
        decode_encryption_key("!!!not-base64!!!")


def test_generate_key_roundtrips():
    key = decode_encryption_key(generate_encryption_key())
    assert len(key) == 32


def test_missing_key_raises_clear_error():
    class _Settings:
        evalyx_encryption_key = __import__("pydantic").SecretStr("")

    with pytest.raises(EncryptionError, match="EVALYX_ENCRYPTION_KEY"):
        SecretEncryptor.from_settings(_Settings())