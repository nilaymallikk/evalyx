"""Repository for datasets, dataset versions, and test cases."""

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from evalyx.db.models import Dataset, DatasetVersion, TestCase
from evalyx.db.repositories.errors import DuplicateVersionError, NotFoundError


class DatasetRepository:
    """Async data access for datasets, their immutable versions, and cases."""

    async def create(
        self,
        session: AsyncSession,
        *,
        name: str,
        description: str | None = None,
    ) -> Dataset:
        dataset = Dataset(name=name, description=description)
        session.add(dataset)
        await session.commit()
        await session.refresh(dataset)
        return dataset

    async def get(self, session: AsyncSession, dataset_id: uuid.UUID) -> Dataset | None:
        return await session.get(Dataset, dataset_id)

    async def get_by_name(self, session: AsyncSession, name: str) -> Dataset | None:
        result = await session.execute(select(Dataset).where(Dataset.name == name))
        return result.scalar_one_or_none()

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
