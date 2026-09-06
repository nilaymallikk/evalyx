"""Response schemas for ``GET /api/v1/me`` (display information only)."""

from pydantic import BaseModel, Field


class OrganizationSummary(BaseModel):
    """The local workspace."""

    organization_id: str
    name: str


class MeResponse(BaseModel):
    """The local operator identity and workspace context."""

    user_id: str
    active_organization: OrganizationSummary | None = None
    organizations: list[OrganizationSummary] = Field(default_factory=list)
