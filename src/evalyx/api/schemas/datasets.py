"""Request/response schemas for datasets, dataset versions, and test cases."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class DatasetCreate(BaseModel):
    """Request body for ``POST /api/v1/datasets``."""

    name: str = Field(min_length=1, max_length=255, description="Unique dataset name.")
    description: str | None = Field(default=None, max_length=2000)


class DatasetResponse(BaseModel):
    """A logical collection of evaluation test cases."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    description: str | None
    created_at: datetime
    updated_at: datetime


class DatasetVersionCreate(BaseModel):
    """Request body for ``POST /api/v1/datasets/{id}/versions``.

    Version numbers are explicit and client-controlled; a version is an
    immutable snapshot — content is never updated after creation.
    """

    version: int = Field(ge=1, description="Integer version number, unique per dataset.")
    description: str | None = Field(default=None, max_length=2000)


class DatasetVersionResponse(BaseModel):
    """An immutable dataset version snapshot."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    dataset_id: UUID
    version: int
    description: str | None
    created_at: datetime
    updated_at: datetime


class TestCaseCreate(BaseModel):
    """Request body for creating a test case inside a dataset version."""

    name: str = Field(min_length=1, max_length=255)
    input: dict = Field(description="Structured input payload for the application under test.")
    expected_output: dict | None = None
    context: dict | None = None
    metadata: dict = Field(
        default_factory=dict,
        description="Free-form non-secret metadata (secret-looking keys are stripped).",
    )


class TestCaseResponse(BaseModel):
    """A single evaluation case belonging to one immutable dataset version."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    dataset_version_id: UUID
    name: str
    input: dict
    expected_output: dict | None
    context: dict | None
    metadata: dict
    created_at: datetime
    updated_at: datetime
