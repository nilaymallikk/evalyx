"""Authenticated secret encryption (Phase 15 security boundary, Phase 18 rotation).

The only module in Evalyx that performs cryptography. Application
credentials (API keys, bearer tokens) are encrypted with AES-256-GCM
(``cryptography`` — a well-established library; nothing is invented here)
before they touch the database, and decrypted only at the execution
boundary (the worker/test path that performs the outbound HTTP request).

Ciphertext envelopes (versioned for key rotation)::

    v1:<nonce_b64>:<ciphertext_b64>          legacy (Phase 15), no key id
    v2:<key_id>:<nonce_b64>:<ciphertext_b64>  current; AES-256-GCM, 12-byte nonce

``key_id`` is the first 8 hex characters of SHA-256 over the raw key —
deterministic (no extra configuration to keep in sync) and non-sensitive
(a truncated hash reveals nothing usable about the key). The encryptor
holds a keyring: the current key encrypts, while current + previous keys
decrypt, so credentials stay readable throughout a rotation. Re-encrypting
stored secrets to the current key is an explicit operator workflow
(:mod:`evalyx.core.key_rotation`), never automatic.

Security rules enforced here:

- keys come from environment configuration (``EVALYX_ENCRYPTION_KEY`` plus
  optional ``EVALYX_PREVIOUS_ENCRYPTION_KEYS``) and never appear in
  exceptions, logs, or ``repr``
- every encryption uses a fresh random nonce (no deterministic reuse)
- tampered/truncated/foreign-key/unknown-key-id ciphertexts raise
  :class:`EncryptionError` with one fixed, secret-free message (no
  decryptability oracle between failure kinds)
"""

import base64
import binascii
import hashlib
import os

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

#: AES-GCM standard nonce size for this envelope version.
NONCE_BYTES = 12
#: AES-256 key size.
KEY_BYTES = 32
#: Envelope version emitted by :meth:`SecretEncryptor.encrypt`.
CIPHERTEXT_VERSION = "v2"
#: Envelope version read (but never written) for legacy Phase 15 rows.
LEGACY_CIPHERTEXT_VERSION = "v1"
#: Truncated SHA-256 hex length used as the key id.
KEY_ID_HEX_CHARS = 8


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


def key_id_for_key(key: bytes) -> str:
    """Deterministic non-sensitive id for a raw key (truncated SHA-256 hex)."""
    return hashlib.sha256(key).hexdigest()[:KEY_ID_HEX_CHARS]


class SecretEncryptor:
    """Authenticated encryption for application credentials (AES-256-GCM).

    Holds a keyring: ``key`` (the current key) encrypts; ``key`` plus
    ``previous_keys`` decrypt. Previous keys are decrypt-only — rotation
    narrows the ring back to one key once re-encryption completes.
    """

    def __init__(
        self,
        key: bytes,
        *,
        key_id: str | None = None,
        previous_keys: dict[str, bytes] | None = None,
    ) -> None:
        if len(key) != KEY_BYTES:
            raise EncryptionError("encryption key must be 32 bytes.")
        for previous in (previous_keys or {}).values():
            if len(previous) != KEY_BYTES:
                raise EncryptionError("encryption key must be 32 bytes.")
        self._current_key = key
        self._aesgcm = AESGCM(key)
        self._key_id = key_id or key_id_for_key(key)
        self._previous: dict[str, bytes] = dict(previous_keys or {})

    @classmethod
    def from_settings(cls, settings) -> SecretEncryptor:
        """Build a keyring encryptor from settings.

        Current key from ``EVALYX_ENCRYPTION_KEY``; decrypt-only history
        from ``EVALYX_PREVIOUS_ENCRYPTION_KEYS`` (validated comma-separated
        values by the Settings validators). Raises :class:`EncryptionError`
        when the current key is not configured — callers surface this as a
        configuration problem, never as a crypto failure carrying key
        material.
        """
        key_value = settings.evalyx_encryption_key.get_secret_value().strip()
        if key_value == "":
            raise EncryptionError(
                "EVALYX_ENCRYPTION_KEY is not configured; application "
                "secrets cannot be encrypted or decrypted."
            )
        try:
            key = decode_encryption_key(key_value)
        except ValueError:
            raise EncryptionError(
                "EVALYX_ENCRYPTION_KEY is malformed; cannot encrypt secrets."
            ) from None
        previous: dict[str, bytes] = {}
        for raw in settings.previous_encryption_key_values:
            try:
                previous_key = decode_encryption_key(raw)
            except ValueError:
                raise EncryptionError(
                    "EVALYX_PREVIOUS_ENCRYPTION_KEYS is malformed; cannot "
                    "decrypt secrets."
                ) from None
            previous[key_id_for_key(previous_key)] = previous_key
        previous.pop(key_id_for_key(key), None)  # current key wins its id
        return cls(key, previous_keys=previous)

    @property
    def current_key_id(self) -> str:
        """The key id new envelopes are written with (safe to persist)."""
        return self._key_id

    def encrypt(self, plaintext: str) -> str:
        """Encrypt one secret into the current (v2) envelope."""
        nonce = os.urandom(NONCE_BYTES)
        ciphertext = self._aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)
        return (
            f"{CIPHERTEXT_VERSION}:"
            f"{self._key_id}:"
            f"{base64.urlsafe_b64encode(nonce).decode('ascii')}:"
            f"{base64.urlsafe_b64encode(ciphertext).decode('ascii')}"
        )

    def decrypt(self, envelope: str) -> str:
        """Decrypt a v1 or v2 envelope; any failure raises (no detail)."""
        parts = envelope.split(":")
        candidates: list[bytes]
        nonce_b64: str
        ciphertext_b64: str
        if len(parts) == 3 and parts[0] == LEGACY_CIPHERTEXT_VERSION:
            # Legacy envelopes carry no key id: try the whole ring.
            candidates = [self._aesgcm_key(), *self._previous.values()]
            nonce_b64, ciphertext_b64 = parts[1], parts[2]
        elif (
            len(parts) == 4
            and parts[0] == CIPHERTEXT_VERSION
            and len(parts[1]) == KEY_ID_HEX_CHARS
        ):
            key = self._ring_lookup(parts[1])
            if key is None:
                raise EncryptionError("ciphertext could not be decrypted.")
            candidates = [key]
            nonce_b64, ciphertext_b64 = parts[2], parts[3]
        else:
            raise EncryptionError("ciphertext envelope is malformed.")
        last_error: Exception | None = None
        for candidate in candidates:
            try:
                nonce = base64.urlsafe_b64decode(nonce_b64.encode("ascii"))
                ciphertext = base64.urlsafe_b64decode(ciphertext_b64.encode("ascii"))
                plaintext = AESGCM(candidate).decrypt(nonce, ciphertext, None)
            except (binascii.Error, ValueError, InvalidTag, UnicodeEncodeError) as exc:
                last_error = exc
                continue
            try:
                return plaintext.decode("utf-8")
            except UnicodeDecodeError as exc:
                last_error = exc
                continue
        assert last_error is not None  # candidates is never empty
        raise EncryptionError("ciphertext could not be decrypted.")

    def is_current_envelope(self, envelope: str) -> bool:
        """True when ``envelope`` already uses the current key (v2 + id).

        The rotation workflow skips such rows, making re-encryption
        idempotent and resumable.
        """
        parts = envelope.split(":")
        return (
            len(parts) == 4
            and parts[0] == CIPHERTEXT_VERSION
            and parts[1] == self._key_id
        )

    def _aesgcm_key(self) -> bytes:
        """The raw current key (kept private; used for legacy decryption)."""
        # AESGCM does not expose the key; re-derive is impossible, so the
        # constructor keeps it.
        return self._current_key

    def _ring_lookup(self, key_id: str) -> bytes | None:
        if key_id == self._key_id:
            return self._current_key
        return self._previous.get(key_id)


__all__ = [
    "CIPHERTEXT_VERSION",
    "LEGACY_CIPHERTEXT_VERSION",
    "EncryptionError",
    "SecretEncryptor",
    "decode_encryption_key",
    "generate_encryption_key",
    "key_id_for_key",
]