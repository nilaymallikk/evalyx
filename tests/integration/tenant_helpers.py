"""Shared workspace helper for live-PostgreSQL test suites."""

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from evalyx.db.tenancy import require_local_organization

#: The single local workspace (rows are truncated per test via DOMAIN_TABLES).
TEST_CLERK_ORG_ID = "local"


async def integration_organization_id(session: AsyncSession) -> uuid.UUID:
    """The local workspace id (auto-provisioned)."""
    organization = await require_local_organization(session)
    return organization.id
