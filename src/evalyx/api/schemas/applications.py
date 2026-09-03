"""Request/response schemas for applications and application versions.

Configuration policy: ``configuration`` fields carry **non-secret execution
configuration only** (e.g. prompt template ids, sampling parameters).
Secrets (API keys, credentials, tokens) are rejected at this boundary —
they are stripped before persistence, never stored, and never returned.
Provider credentials remain server-side environment configuration.

Phase 15 (generic application connections): ``connection`` carries the
validated non-secret HTTP connection configuration of a version, and
``secret`` is a **write-only** field — accepted on create/rotation, never
returned (responses expose only ``secret_configured``).
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from evalyx.application.connection import ConnectionConfig, ConnectionConfigError
from evalyx.evaluation.failures import ExecutionFailure
from evalyx.evaluation.regression.comparison import sanitize_configuration

#: Hard cap on the serialized configuration size (guard against accidental
#: huge payloads; documented intentional limit).
MAX_CONFIGURATION_JSON_CHARS = 20_000

#: Hard cap on plaintext credential size accepted by the API.
MAX_SECRET_CHARS = 4096


def clean_configuration(configuration: dict | None) -> dict:
    """Validate size, then strip secret-looking keys from configuration.

    One sanitizer implementation is reused (the regression engine's), so a
    secret can never take the API path that the comparison path would block.
    """
    payload = configuration or {}
    if len(str(payload)) > MAX_CONFIGURATION_JSON_CHARS:
        raise ValueError(
            f"configuration exceeds the {MAX_CONFIGURATION_JSON_CHARS}-character limit."
        )
    return sanitize_configuration(payload)


def validate_connection(connection: dict | None) -> dict | None:
    """Validate a version's connection configuration; return its sanitized
    storage form (secret-looking keys stripped before persistence).

    Raises :class:`ConnectionConfigError` (mapped to HTTP 422) with a
    rule-name message only — never echoing request values (which could
    contain secrets).
    """
    if connection is None:
        return None
    if not isinstance(connection, dict):
        raise ConnectionConfigError("connection must be a JSON object.")
    sanitized = sanitize_configuration(connection)
    try:
        return ConnectionConfig.model_validate(sanitized).model_dump(mode="json")
    except ConnectionConfigError as exc:
        raise ConnectionConfigError(str(exc)) from None
    except ValidationError as exc:
        # loc + msg only — never ``input`` (a client may have placed a
        # secret in a bad request; echoing it back would leak it).
        details = "; ".join(
            f"{'.'.join(str(part) for part in error.get('loc', ()))}: {error.get('msg', '')}"
            for error in exc.errors()
        )
        raise ConnectionConfigError(
            f"connection configuration is invalid: {details}"
        ) from None


class ApplicationCreate(BaseModel):
    """Request body for ``POST /api/v1/applications``."""

    name: str = Field(min_length=1, max_length=255, description="Unique application name.")
    description: str | None = Field(default=None, max_length=2000)
    connection_type: str = Field(
        default="mlgpt",
        pattern="^(http|mlgpt)$",
        description=(
            "'http' registers a generic user application driven by version "
            "connection configurations; 'mlgpt' (default, backwards "
            "compatible) is the Evalyx reference demo target."
        ),
    )
    #: Write-only credential; encrypted at rest, never returned.
    secret: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_SECRET_CHARS,
        description="Write-only application credential (encrypted at rest).",
    )


class ApplicationResponse(BaseModel):
    """A registered AI application/agent under evaluation.

    The credential is never part of any response — only whether one exists
    (``secret_configured``).
    """

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    description: str | None
    connection_type: str
    secret_configured: bool = False
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_application(cls, application: object) -> ApplicationResponse:
        """Build from the ORM row, exposing credential *existence* only."""
        return cls(
            id=application.id,  # type: ignore[attr-defined]
            name=application.name,  # type: ignore[attr-defined]
            description=application.description,  # type: ignore[attr-defined]
            connection_type=application.connection_type,  # type: ignore[attr-defined]
            secret_configured=application.encrypted_secret is not None,  # type: ignore[attr-defined]
            created_at=application.created_at,  # type: ignore[attr-defined]
            updated_at=application.updated_at,  # type: ignore[attr-defined]
        )


class ApplicationUpdate(BaseModel):
    """Request body for ``PATCH /api/v1/applications/{id}`` (metadata only)."""

    name: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=2000)


class ApplicationSecretUpdate(BaseModel):
    """Request body for credential rotation (``PATCH .../connection``).

    The previous secret is never returned and never required.
    """

    secret: str = Field(min_length=1, max_length=MAX_SECRET_CHARS)


class ApplicationSecretStateResponse(BaseModel):
    """Credential state: existence only, never the secret itself."""

    secret_configured: bool


class ApplicationVersionCreate(BaseModel):
    """Request body for ``POST /api/v1/applications/{id}/versions``."""

    version: str = Field(
        min_length=1,
        max_length=64,
        description="Version label, unique per application (immutable once created).",
    )
    description: str | None = Field(default=None, max_length=2000)
    configuration: dict = Field(
        default_factory=dict,
        description=(
            "Non-secret execution configuration snapshot. Secret-looking keys "
            "are stripped and never persisted."
        ),
    )
    connection: dict | None = Field(
        default=None,
        description=(
            "Non-secret HTTP connection configuration (generic applications "
            "only). Validated and stored immutably; secrets never belong here."
        ),
    )

    def sanitized_configuration(self) -> dict:
        return clean_configuration(self.configuration)

    def validated_connection(self) -> dict | None:
        return validate_connection(self.connection)


class ApplicationVersionResponse(BaseModel):
    """An immutable application configuration snapshot."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    application_id: UUID
    version: str
    description: str | None
    configuration: dict
    connection: dict | None = None
    created_at: datetime
    updated_at: datetime


class ConnectionTestRequest(BaseModel):
    """Request body for ``POST /api/v1/applications/{id}/test``.

    The endpoint always uses the stored connection configuration; callers
    may pin a version and choose a small, non-sensitive probe prompt.
    """

    version_id: UUID | None = Field(
        default=None,
        description=(
            "Optional application version to test (defaults to the latest "
            "version with a connection configuration)."
        ),
    )
    prompt: str = Field(
        default="Connection test: reply with the word ok.",
        min_length=1,
        max_length=2000,
        description="Small probe prompt (never use real or sensitive inputs).",
    )


class ConnectionTestResponse(BaseModel):
    """Safe structured connection-test result.

    Never contains authorization headers, API keys, sensitive headers, or
    full request payloads; ``preview`` is a truncated slice of the
    application's own answer.
    """

    success: bool
    latency_ms: int | None = None
    http_status: int | None = None
    preview: str | None = Field(
        default=None, description="Truncated answer preview (≤280 characters)."
    )
    failure: ExecutionFailure | None = Field(
        default=None, description="Typed failure classification when unsuccessful."
    )
