"""Local workspace description for ``GET /api/v1/me``.

Evalyx serves a single local workspace with no login. ``/me`` answers
"which workspace is this" for the CLI's ``evalyx whoami`` — display
information only, never secrets.
"""

from evalyx.api.schemas.me import MeResponse, OrganizationSummary


async def describe_caller() -> MeResponse:
    """Build the ``/me`` response for the local workspace."""
    active = OrganizationSummary(
        organization_id="local",
        name="local",
    )
    return MeResponse(
        user_id="local",
        active_organization=active,
        organizations=[active],
    )
