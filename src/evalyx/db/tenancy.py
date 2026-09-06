"""Workspace helpers shared by repositories and services.

Evalyx serves a single local workspace, mirrored as the
:class:`Organization` model. Repositories accept an ``organization_id`` on
every owned read/write and filter by it. All queries use SQLAlchemy ORM
expressions with bound parameters.

Style note: newer repository code uses ``session.scalars(...)`` for ORM
entity queries (scalar-mapping built in); the grandfathered
``session.execute(...)`` call sites behave identically.
"""

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from evalyx.db.models import Organization
from evalyx.db.repositories.errors import NotFoundError

#: The single local workspace's external id (unique here — provisioning is
#: idempotent on it).
LOCAL_ORGANIZATION_ID = "local"


class TenantError(Exception):
    """A referenced tenant does not exist / is not accessible (404-class)."""


async def get_organization_by_clerk_id(
    session: AsyncSession,
    clerk_organization_id: str,
) -> Organization | None:
    """The local organization mapped to a Clerk organization, if any.

    The lookup goes through the unique index on the Clerk organization id;
    the value travels as a bound ORM parameter.
    """
    result = await session.scalars(
        select(Organization).filter_by(clerk_organization_id=clerk_organization_id)
    )
    return result.first()


async def require_organization(
    session: AsyncSession,
    clerk_organization_id: str,
) -> Organization:
    """The local organization for an external organization id.

    The workspace row is created on first use. Provisioning is idempotent
    on the unique ``clerk_organization_id``.
    """
    organization = await get_organization_by_clerk_id(session, clerk_organization_id)
    if organization is None:
        organization = Organization(
            clerk_organization_id=clerk_organization_id,
            name=clerk_organization_id,
        )
        session.add(organization)
        await session.commit()
        await session.refresh(organization)
    return organization


async def require_local_organization(session: AsyncSession) -> Organization:
    """The single local workspace (auto-provisioned on first use)."""
    return await require_organization(session, LOCAL_ORGANIZATION_ID)


def tenant_not_found(resource: str, resource_id: uuid.UUID) -> NotFoundError:
    """Uniform 404 for both missing rows and other tenants' rows.

    Deliberately indistinguishable: error messages never reveal whether a
    resource exists under a different organization (IDOR hardening).
    """
    return NotFoundError(f"{resource} {resource_id} does not exist.")


__all__ = [
    "LOCAL_ORGANIZATION_ID",
    "TenantError",
    "get_organization_by_clerk_id",
    "require_local_organization",
    "require_organization",
    "tenant_not_found",
]
