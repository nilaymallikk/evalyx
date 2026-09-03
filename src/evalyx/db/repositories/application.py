"""Repository for applications and application versions."""

import uuid

from sqlalchemy import func, select
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
        connection_type: str = "mlgpt",
        encrypted_secret: str | None = None,
        secret_metadata: dict | None = None,
    ) -> Application:
        application = Application(
            organization_id=organization_id,
            name=name,
            description=description,
            connection_type=connection_type,
            encrypted_secret=encrypted_secret,
            secret_metadata=secret_metadata,
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

    async def list_in_organization(
        self,
        session: AsyncSession,
        *,
        organization_id: uuid.UUID,
        limit: int,
        offset: int,
    ) -> tuple[list[Application], int]:
        """Tenant-scoped, creation-ordered page plus total count."""
        total = await session.scalar(
            select(func.count())
            .select_from(Application)
            .where(Application.organization_id == organization_id)
        )
        result = await session.scalars(
            select(Application)
            .where(Application.organization_id == organization_id)
            .order_by(Application.created_at, Application.id)
            .offset(offset)
            .limit(limit)
        )
        return list(result.all()), int(total or 0)

    async def update(
        self,
        session: AsyncSession,
        application: Application,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> Application:
        """Partial update (PATCH semantics: ``None`` leaves fields untouched)."""
        if name is not None:
            application.name = name
        if description is not None:
            application.description = description
        session.add(application)
        await session.commit()
        await session.refresh(application)
        return application

    async def delete(self, session: AsyncSession, application: Application) -> None:
        """Delete an application (its versions cascade; runs RESTRICT)."""
        await session.delete(application)
        await session.commit()

    async def set_secret(
        self,
        session: AsyncSession,
        application: Application,
        *,
        encrypted_secret: str,
        secret_metadata: dict,
    ) -> Application:
        """Credential rotation: replace the ciphertext (never read it back)."""
        application.encrypted_secret = encrypted_secret
        application.secret_metadata = secret_metadata
        session.add(application)
        await session.commit()
        await session.refresh(application)
        return application

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
        connection: dict | None = None,
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
            connection=connection,
        )
        session.add(application_version)
        await session.commit()
        await session.refresh(application_version)
        return application_version

    async def get_version_by_id(
        self,
        session: AsyncSession,
        *,
        application_id: uuid.UUID,
        version_id: uuid.UUID,
    ) -> ApplicationVersion | None:
        """Scoped by the (already tenant-checked) parent application."""
        result = await session.scalars(
            select(ApplicationVersion).filter_by(
                id=version_id, application_id=application_id
            )
        )
        return result.first()

    async def latest_version_with_connection(
        self,
        session: AsyncSession,
        application_id: uuid.UUID,
    ) -> ApplicationVersion | None:
        """The newest version that carries a connection configuration."""
        result = await session.scalars(
            select(ApplicationVersion)
            .where(
                ApplicationVersion.application_id == application_id,
                ApplicationVersion.connection.is_not(None),
            )
            .order_by(
                ApplicationVersion.created_at.desc(), ApplicationVersion.version.desc()
            )
            .limit(1)
        )
        return result.first()

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
