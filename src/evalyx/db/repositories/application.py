"""Repository for applications and application versions."""

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from evalyx.db.models import Application, ApplicationVersion
from evalyx.db.repositories.errors import DuplicateVersionError, NotFoundError


class ApplicationRepository:
    """Async data access for :class:`Application` and its versions."""

    async def create(
        self,
        session: AsyncSession,
        *,
        organization_id: uuid.UUID,
        name: str,
        description: str | None = None,
    ) -> Application:
        application = Application(
            organization_id=organization_id, name=name, description=description
        )
        session.add(application)
        await session.commit()
        await session.refresh(application)
        return application

    async def get(
        self, session: AsyncSession, application_id: uuid.UUID
    ) -> Application | None:
        return await session.get(Application, application_id)

    async def get_in_organization(
        self,
        session: AsyncSession,
        application_id: uuid.UUID,
        *,
        organization_id: uuid.UUID,
    ) -> Application | None:
        """Tenant-scoped fetch: other tenants' applications read as missing."""
        result = await session.scalars(
            select(Application).filter_by(
                id=application_id, organization_id=organization_id
            )
        )
        return result.first()

    async def get_by_name(
        self, session: AsyncSession, *, organization_id: uuid.UUID, name: str
    ) -> Application | None:
        result = await session.scalars(
            select(Application).filter_by(
                organization_id=organization_id, name=name
            )
        )
        return result.first()

    async def create_version(
        self,
        session: AsyncSession,
        *,
        application_id: uuid.UUID,
        version: str,
        description: str | None = None,
        configuration: dict | None = None,
    ) -> ApplicationVersion:
        if await session.get(Application, application_id) is None:
            raise NotFoundError(f"Application {application_id} does not exist.")
        if await self.get_version(session, application_id, version) is not None:
            raise DuplicateVersionError(application_id, version)

        application_version = ApplicationVersion(
            application_id=application_id,
            version=version,
            description=description,
            configuration=configuration or {},
        )
        session.add(application_version)
        await session.commit()
        await session.refresh(application_version)
        return application_version

    async def get_version(
        self,
        session: AsyncSession,
        application_id: uuid.UUID,
        version: str,
    ) -> ApplicationVersion | None:
        result = await session.execute(
            select(ApplicationVersion).where(
                ApplicationVersion.application_id == application_id,
                ApplicationVersion.version == version,
            )
        )
        return result.scalar_one_or_none()

    async def list_versions(
        self,
        session: AsyncSession,
        application_id: uuid.UUID,
    ) -> list[ApplicationVersion]:
        """All versions of an application in creation order."""
        result = await session.execute(
            select(ApplicationVersion)
            .where(ApplicationVersion.application_id == application_id)
            .order_by(ApplicationVersion.created_at, ApplicationVersion.version)
        )
        return list(result.scalars().all())
