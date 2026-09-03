"""Response schemas for ``GET /api/v1/me`` (display information only)."""

from pydantic import BaseModel, Field


class OrganizationSummary(BaseModel):
    """One Clerk organization the caller belongs to."""

    clerk_organization_id: str
    name: str
    role: str | None = Field(
        default=None, description="Caller's role slug (e.g. 'admin'), if resolved."
    )


class MeResponse(BaseModel):
    """The authenticated caller's identity and organization context."""

    clerk_user_id: str
    email: str | None = Field(
        default=None, description="Primary email address (display only)."
    )
    active_organization: OrganizationSummary | None = None
    organizations: list[OrganizationSummary] = Field(default_factory=list)
