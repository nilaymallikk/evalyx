"""Multi-tenancy helpers shared by repositories and services (Phase 14).

The tenant boundary is the Clerk organization, mirrored locally as the
:class:`Organization` model. Repositories accept an ``organization_id`` on
every tenant-owned read/write and filter by it, so a valid-id-wrong-tenant
lookup behaves exactly like a missing row (404, never another tenant's
data). All queries use SQLAlchemy ORM expressions with bound parameters.

Style note: newer repository code uses ``session.scalars(...)`` for ORM
entity queries (scalar-mapping built in); the grandfathered
``session.execute(...)`` call sites behave identically.
"""

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from evalyx.db.models import Organization
from evalyx.db.repositories.errors import NotFoundError


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
    """The local organization for a Clerk organization (auto-provisioning).

    The workspace row is created on first use so the API stays usable the
    moment a Clerk organization starts calling Evalyx. Provisioning is
    idempotent on the unique ``clerk_organization_id``.
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


def tenant_not_found(resource: str, resource_id: uuid.UUID) -> NotFoundError:
    """Uniform 404 for both missing rows and other tenants' rows.

    Deliberately indistinguishable: error messages never reveal whether a
    resource exists under a different organization (IDOR hardening).
    """
    return NotFoundError(f"{resource} {resource_id} does not exist.")


__all__ = [
    "TenantError",
    "get_organization_by_clerk_id",
    "require_organization",
    "tenant_not_found",
]
