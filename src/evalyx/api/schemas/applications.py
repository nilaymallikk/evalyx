"""Request/response schemas for applications and application versions.

Configuration policy: ``configuration`` fields carry **non-secret execution
configuration only** (e.g. prompt template ids, sampling parameters).
Secrets (API keys, credentials, tokens) are rejected at this boundary —
they are stripped before persistence, never stored, and never returned.
Provider credentials remain server-side environment configuration.
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from evalyx.evaluation.regression.comparison import sanitize_configuration

#: Hard cap on the serialized configuration size (guard against accidental
#: huge payloads; documented intentional limit).
MAX_CONFIGURATION_JSON_CHARS = 20_000


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


class ApplicationCreate(BaseModel):
    """Request body for ``POST /api/v1/applications``."""

    name: str = Field(min_length=1, max_length=255, description="Unique application name.")
    description: str | None = Field(default=None, max_length=2000)


class ApplicationResponse(BaseModel):
    """A registered AI application/agent under evaluation."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    description: str | None
    created_at: datetime
    updated_at: datetime


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

    def sanitized_configuration(self) -> dict:
        return clean_configuration(self.configuration)


class ApplicationVersionResponse(BaseModel):
    """An immutable application configuration snapshot."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    application_id: UUID
    version: str
    description: str | None
    configuration: dict
    created_at: datetime
    updated_at: datetime
