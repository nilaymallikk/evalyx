"""Repository for datasets, dataset versions, and test cases."""

import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from evalyx.db.models import Dataset, DatasetVersion, TestCase
from evalyx.db.repositories.errors import DuplicateVersionError, NotFoundError


class DatasetRepository:
    """Async data access for datasets, their immutable versions, and cases."""

    async def create(
        self,
        session: AsyncSession,
        *,
        organization_id: uuid.UUID,
        name: str,
        description: str | None = None,
    ) -> Dataset:
        dataset = Dataset(
            organization_id=organization_id, name=name, description=description
        )
        session.add(dataset)
        await session.commit()
        await session.refresh(dataset)
        return dataset

    async def get(self, session: AsyncSession, dataset_id: uuid.UUID) -> Dataset | None:
        return await session.get(Dataset, dataset_id)

    async def get_in_organization(
        self,
        session: AsyncSession,
        dataset_id: uuid.UUID,
        *,
        organization_id: uuid.UUID,
    ) -> Dataset | None:
        """Tenant-scoped fetch: other tenants' datasets read as missing."""
        result = await session.scalars(
            select(Dataset).filter_by(id=dataset_id, organization_id=organization_id)
        )
        return result.first()

    async def list_in_organization(
        self,
        session: AsyncSession,
        *,
        organization_id: uuid.UUID,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Dataset], int]:
        """Tenant-scoped dataset listing (creation order) with total count.

        Phase 16: the CLI's ``evalyx dataset list`` needs the same paginated
        surface applications already expose.
        """
        total = await session.scalar(
            select(func.count())
            .select_from(Dataset)
            .where(Dataset.organization_id == organization_id)
        )
        result = await session.scalars(
            select(Dataset)
            .filter_by(organization_id=organization_id)
            .order_by(Dataset.created_at.asc())
            .offset(offset)
            .limit(limit)
        )
        return list(result.all()), int(total or 0)

    async def get_by_name(
        self, session: AsyncSession, *, organization_id: uuid.UUID, name: str
    ) -> Dataset | None:
        result = await session.scalars(
            select(Dataset).filter_by(organization_id=organization_id, name=name)
        )
        return result.first()

    async def create_version(
        self,
        session: AsyncSession,
        *,
        dataset_id: uuid.UUID,
        version: int,
        description: str | None = None,
    ) -> DatasetVersion:
        """Create a new immutable dataset version.

        Raises :class:`DuplicateVersionError` if the version number already
        exists for the dataset; version rows are never overwritten.
        """
        if await session.get(Dataset, dataset_id) is None:
            raise NotFoundError(f"Dataset {dataset_id} does not exist.")
        if await self.get_version(session, dataset_id, version) is not None:
            raise DuplicateVersionError(dataset_id, version)

        dataset_version = DatasetVersion(
            dataset_id=dataset_id,
            version=version,
            description=description,
        )
        session.add(dataset_version)
        await session.commit()
        await session.refresh(dataset_version)
        return dataset_version

    async def get_version(
        self,
        session: AsyncSession,
        dataset_id: uuid.UUID,
        version: int,
    ) -> DatasetVersion | None:
        result = await session.execute(
            select(DatasetVersion).where(
                DatasetVersion.dataset_id == dataset_id,
                DatasetVersion.version == version,
            )
        )
        return result.scalar_one_or_none()

    async def get_version_by_id(
        self,
        session: AsyncSession,
        dataset_version_id: uuid.UUID,
    ) -> DatasetVersion | None:
        return await session.get(DatasetVersion, dataset_version_id)

    async def get_version_in_organization(
        self,
        session: AsyncSession,
        dataset_version_id: uuid.UUID,
        *,
        organization_id: uuid.UUID,
    ) -> DatasetVersion | None:
        """Tenant-scoped version fetch via its parent dataset's tenant."""
        result = await session.scalars(
            select(DatasetVersion)
            .join(Dataset, DatasetVersion.dataset_id == Dataset.id)
            .where(
                DatasetVersion.id == dataset_version_id,
                Dataset.organization_id == organization_id,
            )
        )
        return result.first()

    async def get_version_in_organization_via_dataset(
        self,
        session: AsyncSession,
        dataset_id: uuid.UUID,
        version: int,
        *,
        organization_id: uuid.UUID,
    ) -> DatasetVersion | None:
        """Tenant-scoped (dataset id + version number) fetch.

        The dataset must belong to the organization *and* the version must
        belong to that dataset — one query enforces both, so a guessed
        (dataset, version) pair from another tenant reads as missing.
        """
        result = await session.scalars(
            select(DatasetVersion)
            .join(Dataset, DatasetVersion.dataset_id == Dataset.id)
            .where(
                DatasetVersion.dataset_id == dataset_id,
                DatasetVersion.version == version,
                Dataset.organization_id == organization_id,
            )
        )
        return result.first()

    async def list_versions(
        self,
        session: AsyncSession,
        dataset_id: uuid.UUID,
    ) -> list[DatasetVersion]:
        result = await session.execute(
            select(DatasetVersion)
            .where(DatasetVersion.dataset_id == dataset_id)
            .order_by(DatasetVersion.version)
        )
        return list(result.scalars().all())

    async def add_test_case(
        self,
        session: AsyncSession,
        *,
        dataset_version_id: uuid.UUID,
        name: str,
        input: dict,
        expected_output: dict | None = None,
        context: dict | None = None,
        metadata: dict | None = None,
    ) -> TestCase:
        if await session.get(DatasetVersion, dataset_version_id) is None:
            raise NotFoundError(f"Dataset version {dataset_version_id} does not exist.")

        test_case = TestCase(
            dataset_version_id=dataset_version_id,
            name=name,
            input=input,
            expected_output=expected_output,
            context=context,
            metadata_=metadata or {},
        )
        session.add(test_case)
        await session.commit()
        await session.refresh(test_case)
        return test_case

    async def get_test_case(
        self,
        session: AsyncSession,
        test_case_id: uuid.UUID,
    ) -> TestCase | None:
        return await session.get(TestCase, test_case_id)

    async def list_test_cases(
        self,
        session: AsyncSession,
        dataset_version_id: uuid.UUID,
    ) -> list[TestCase]:
        result = await session.execute(
            select(TestCase)
            .where(TestCase.dataset_version_id == dataset_version_id)
            .order_by(TestCase.created_at)
        )
        return list(result.scalars().all())
