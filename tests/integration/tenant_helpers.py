"""Shared multi-tenancy helpers for live-PostgreSQL test suites."""

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from evalyx.db.models import Organization

#: Well-known test tenant (rows are truncated per test via DOMAIN_TABLES).
TEST_CLERK_ORG_ID = "org_integration_test"


async def integration_organization_id(session: AsyncSession) -> uuid.UUID:
    """The local organization id for the well-known test tenant."""
    result = await session.scalars(
        select(Organization).filter_by(clerk_organization_id=TEST_CLERK_ORG_ID)
    )
    organization = result.first()
    if organization is None:
        organization = Organization(
            clerk_organization_id=TEST_CLERK_ORG_ID,
            name="Integration Test Org",
        )
        session.add(organization)
        await session.commit()
        await session.refresh(organization)
    return organization.id
