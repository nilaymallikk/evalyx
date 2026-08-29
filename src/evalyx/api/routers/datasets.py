"""Dataset, dataset-version, and test-case endpoints.

Dataset versions are immutable: there is deliberately **no** update or
delete endpoint for version content — create a new version instead.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from evalyx.api.dependencies import get_session, pagination_params
from evalyx.api.schemas.common import Page
from evalyx.api.schemas.datasets import (
    DatasetCreate,
    DatasetResponse,
    DatasetVersionCreate,
    DatasetVersionResponse,
    TestCaseCreate,
    TestCaseResponse,
)
from evalyx.db.models import Dataset, DatasetVersion, TestCase
from evalyx.db.repositories import DatasetRepository, NotFoundError
from evalyx.evaluation.regression.comparison import sanitize_configuration

router = APIRouter(prefix="/datasets", tags=["datasets"])


def _repository() -> DatasetRepository:
    return DatasetRepository()


def _test_case_response(case: TestCase) -> TestCaseResponse:
    """Column-only mapping (``metadata_`` column → ``metadata`` field)."""
    return TestCaseResponse(
        id=case.id,
        dataset_version_id=case.dataset_version_id,
        name=case.name,
        input=case.input,
        expected_output=case.expected_output,
        context=case.context,
        metadata=case.metadata_,
        created_at=case.created_at,
        updated_at=case.updated_at,
    )


async def _get_version_or_404(
    session: AsyncSession, dataset_id: uuid.UUID, version: int
):
    dataset_version = await _repository().get_version(session, dataset_id, version)
    if dataset_version is None:
        raise NotFoundError(
            f"Dataset version {version} of dataset {dataset_id} does not exist."
        )
    return dataset_version


@router.post(
    "",
    response_model=DatasetResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a dataset",
    description="Register a logical dataset. Names are unique; duplicates get 409.",
)
async def create_dataset(
    payload: DatasetCreate,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> Dataset:
    return await _repository().create(
        session, name=payload.name, description=payload.description
    )


@router.get(
    "/{dataset_id}",
    response_model=DatasetResponse,
    summary="Retrieve a dataset",
    responses={404: {"description": "Dataset not found."}},
)
async def get_dataset(
    dataset_id: uuid.UUID,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> Dataset:
    dataset = await _repository().get(session, dataset_id)
    if dataset is None:
        raise NotFoundError(f"Dataset {dataset_id} does not exist.")
    return dataset


@router.get(
    "/{dataset_id}/versions",
    response_model=Page[DatasetVersionResponse],
    summary="List dataset versions",
    description="Immutable version snapshots. Ordered by version number ascending.",
)
async def list_dataset_versions(
    dataset_id: uuid.UUID,
    session: Annotated[AsyncSession, Depends(get_session)],
    pagination: Annotated[tuple[int, int], Depends(pagination_params)],
) -> Page[DatasetVersionResponse]:
    limit, offset = pagination
    repository = _repository()
    if await repository.get(session, dataset_id) is None:
        raise NotFoundError(f"Dataset {dataset_id} does not exist.")
    versions = await repository.list_versions(session, dataset_id)
    items = [
        DatasetVersionResponse.model_validate(version)
        for version in versions[offset : offset + limit]
    ]
    return Page[DatasetVersionResponse](
        items=items,
        total=len(versions),
        limit=limit,
        offset=offset,
    )


@router.post(
    "/{dataset_id}/versions",
    response_model=DatasetVersionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a dataset version",
    description=(
        "Create a new immutable version snapshot. Reusing a version number "
        "is rejected with 409. Add cases via the cases endpoints."
    ),
    responses={404: {"description": "Dataset not found."}, 409: {"description": "Duplicate version number."}},
)
async def create_dataset_version(
    dataset_id: uuid.UUID,
    payload: DatasetVersionCreate,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> DatasetVersion:
    return await _repository().create_version(
        session, dataset_id=dataset_id, version=payload.version, description=payload.description
    )


@router.post(
    "/{dataset_id}/versions/{version}/cases",
    response_model=TestCaseResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add a test case to a dataset version",
    description=(
        "Append one test case to an immutable dataset version. ``metadata`` "
        "must be non-secret; secret-looking keys are stripped."
    ),
    responses={404: {"description": "Dataset version not found."}},
)
async def add_test_case(
    dataset_id: uuid.UUID,
    version: int,
    payload: TestCaseCreate,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> TestCaseResponse:
    dataset_version = await _get_version_or_404(session, dataset_id, version)
    case = await _repository().add_test_case(
        session,
        dataset_version_id=dataset_version.id,
        name=payload.name,
        input=payload.input,
        expected_output=payload.expected_output,
        context=payload.context,
        metadata=sanitize_configuration(payload.metadata),
    )
    return _test_case_response(case)


@router.get(
    "/{dataset_id}/versions/{version}/cases",
    response_model=Page[TestCaseResponse],
    summary="List test cases in a dataset version",
    description="Ordered by creation time ascending (stable case order).",
    responses={404: {"description": "Dataset version not found."}},
)
async def list_test_cases(
    dataset_id: uuid.UUID,
    version: int,
    session: Annotated[AsyncSession, Depends(get_session)],
    pagination: Annotated[tuple[int, int], Depends(pagination_params)],
) -> Page[TestCaseResponse]:
    limit, offset = pagination
    dataset_version = await _get_version_or_404(session, dataset_id, version)
    cases = await _repository().list_test_cases(session, dataset_version.id)
    items = [_test_case_response(case) for case in cases[offset : offset + limit]]
    return Page[TestCaseResponse](
        items=items, total=len(cases), limit=limit, offset=offset
    )
