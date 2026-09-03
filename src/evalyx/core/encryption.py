"""Authenticated secret encryption (Phase 15 security boundary).

The only module in Evalyx that performs cryptography. Application
credentials (API keys, bearer tokens) are encrypted with AES-256-GCM
(``cryptography`` — a well-established library; nothing is invented here)
before they touch the database, and decrypted only at the execution
boundary (the worker/test path that performs the outbound HTTP request).

Ciphertext envelope (versioned for future key rotation)::

    v1:<nonce_b64>:<ciphertext_b64>     AES-256-GCM, 12-byte nonce

``v1`` is the algorithm/version marker: a future format or key id becomes
``v2``/``v1.<key_id>:...`` without touching call sites. Full key rotation
(re-encrypting stored secrets) is *not* implemented in this phase — the
envelope format is the designed extension point (documented limitation).

Security rules enforced here:

- the key comes from environment configuration (``EVALYX_ENCRYPTION_KEY``,
  urlsafe base64 of exactly 32 bytes) and never appears in exceptions,
  logs, or ``repr``
- every encryption uses a fresh random nonce (no deterministic reuse)
- tampered/truncated/foreign-key ciphertexts raise :class:`EncryptionError`
  with a fixed, secret-free message
"""

import base64
import binascii
import os

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

#: AES-GCM standard nonce size for this envelope version.
NONCE_BYTES = 12
#: AES-256 key size.
KEY_BYTES = 32
#: Current ciphertext envelope version.
CIPHERTEXT_VERSION = "v1"


class EncryptionError(Exception):
    """A secret could not be encrypted or decrypted.

    Safe by construction: the message names the failure kind only — never
    the key, the ciphertext, or the plaintext (call sites pass fixed,
    secret-free literals).
    """


def decode_encryption_key(key_value: str) -> bytes:
    """Decode and validate a urlsafe base64-encoded 32-byte key.

    Raises :class:`ValueError` (with no key material in the message) when
    the value is not decodable or has the wrong length.
    """
    try:
        key = base64.urlsafe_b64decode(key_value.encode("ascii"))
    except (binascii.Error, ValueError, UnicodeEncodeError):
        raise ValueError("encryption key is not valid urlsafe base64.") from None
    if len(key) != KEY_BYTES:
        raise ValueError(
            f"encryption key must decode to exactly {KEY_BYTES} bytes."
        )
    return key


def generate_encryption_key() -> str:
    """Generate a fresh urlsafe base64-encoded 32-byte key (ops helper)."""
    return base64.urlsafe_b64encode(os.urandom(KEY_BYTES)).decode("ascii")


class SecretEncryptor:
    """Authenticated encryption for application credentials (AES-256-GCM)."""

    def __init__(self, key: bytes) -> None:
        if len(key) != KEY_BYTES:
            raise EncryptionError("encryption key must be 32 bytes.")
        self._aesgcm = AESGCM(key)

    @classmethod
    def from_settings(cls, settings) -> SecretEncryptor:
        """Build an encryptor from ``EVALYX_ENCRYPTION_KEY``.

        Raises :class:`EncryptionError` when the key is not configured —
        callers surface this as a configuration problem, never as a crypto
        failure carrying key material.
        """
        key_value = settings.evalyx_encryption_key.get_secret_value().strip()
        if key_value == "":
            raise EncryptionError(
                "EVALYX_ENCRYPTION_KEY is not configured; application "
                "secrets cannot be encrypted or decrypted."
            )
        try:
            return cls(decode_encryption_key(key_value))
        except ValueError:
            raise EncryptionError(
                "EVALYX_ENCRYPTION_KEY is malformed; cannot encrypt secrets."
            ) from None

    def encrypt(self, plaintext: str) -> str:
        """Encrypt one secret into the versioned envelope."""
        nonce = os.urandom(NONCE_BYTES)
        ciphertext = self._aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)
        return (
            f"{CIPHERTEXT_VERSION}:"
            f"{base64.urlsafe_b64encode(nonce).decode('ascii')}:"
            f"{base64.urlsafe_b64encode(ciphertext).decode('ascii')}"
        )

    def decrypt(self, envelope: str) -> str:
        """Decrypt one versioned envelope; tampering raises (no detail)."""
        parts = envelope.split(":")
        if len(parts) != 3 or parts[0] != CIPHERTEXT_VERSION:
            raise EncryptionError("ciphertext envelope is malformed.")
        try:
            nonce = base64.urlsafe_b64decode(parts[1].encode("ascii"))
            ciphertext = base64.urlsafe_b64decode(parts[2].encode("ascii"))
            plaintext = self._aesgcm.decrypt(nonce, ciphertext, None)
        except (binascii.Error, ValueError, InvalidTag, UnicodeEncodeError):
            raise EncryptionError("ciphertext could not be decrypted.") from None
        try:
            return plaintext.decode("utf-8")
        except UnicodeDecodeError:
            raise EncryptionError("ciphertext could not be decrypted.") from None


__all__ = [
    "CIPHERTEXT_VERSION",
    "EncryptionError",
    "SecretEncryptor",
    "decode_encryption_key",
    "generate_encryption_key",
]