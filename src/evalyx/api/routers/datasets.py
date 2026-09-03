"""Dataset, dataset-version, and test-case endpoints.

Dataset versions are immutable: there is deliberately **no** update or
delete endpoint for version content — create a new version instead.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from evalyx.api.auth import AuthContext
from evalyx.api.dependencies import (
    get_quota_service,
    get_session,
    get_settings,
    pagination_params,
    require_organization,
)
from evalyx.api.schemas.common import Page
from evalyx.api.schemas.datasets import (
    DatasetCreate,
    DatasetResponse,
    DatasetVersionCreate,
    DatasetVersionResponse,
    TestCaseCreate,
    TestCaseResponse,
)
from evalyx.core.config import Settings
from evalyx.db.models import Dataset, DatasetVersion, Organization, TestCase
from evalyx.db.repositories import DatasetRepository, NotFoundError
from evalyx.evaluation.regression.comparison import sanitize_configuration
from evalyx.quotas import QuotaService
from evalyx.security.audit import (
    DATASET_CASE_ADD,
    DATASET_CREATE,
    DATASET_VERSION_CREATE,
    record_audit_event,
)

router = APIRouter(prefix="/datasets", tags=["datasets"])

#: Authenticated + tenant-resolved dependency (Clerk org → local workspace).
TenantContext = Annotated[tuple[AuthContext, Organization], Depends(require_organization)]


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
    session: AsyncSession,
    dataset_id: uuid.UUID,
    version: int,
    *,
    organization_id: uuid.UUID,
):
    """Tenant-scoped version fetch: the parent dataset must be the caller's."""
    dataset_version = await _repository().get_version_in_organization_via_dataset(
        session, dataset_id, version, organization_id=organization_id
    )
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
    settings: Annotated[Settings, Depends(get_settings)],
    context: TenantContext,
    quotas: Annotated[QuotaService, Depends(get_quota_service)],
) -> Dataset:
    auth, organization = context
    await quotas.admit_dataset_create(
        session,
        organization_id=organization.id,
        clerk_user_id=auth.clerk_user_id,
    )
    dataset = await _repository().create(
        session,
        organization_id=organization.id,
        name=payload.name,
        description=payload.description,
    )
    if settings.audit_enabled:
        await record_audit_event(
            session,
            organization_id=organization.id,
            clerk_user_id=auth.clerk_user_id,
            action=DATASET_CREATE,
            resource_type="dataset",
            resource_id=dataset.id,
            details={"name": payload.name},
        )
        await session.commit()
    return dataset


@router.get(
    "",
    response_model=Page[DatasetResponse],
    summary="List datasets",
    description=(
        "Datasets scoped to the authenticated organization, ordered by "
        "creation time ascending (Phase 16: paginated listing for the CLI)."
    ),
)
async def list_datasets(
    session: Annotated[AsyncSession, Depends(get_session)],
    pagination: Annotated[tuple[int, int], Depends(pagination_params)],
    context: TenantContext,
) -> Page[DatasetResponse]:
    limit, offset = pagination
    _auth, organization = context
    datasets, total = await _repository().list_in_organization(
        session, organization_id=organization.id, limit=limit, offset=offset
    )
    return Page[DatasetResponse](
        items=[
            DatasetResponse.model_validate(dataset) for dataset in datasets
        ],
        total=total,
        limit=limit,
        offset=offset,
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
    context: TenantContext,
) -> Dataset:
    _auth, organization = context
    dataset = await _repository().get_in_organization(
        session, dataset_id, organization_id=organization.id
    )
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
    context: TenantContext,
) -> Page[DatasetVersionResponse]:
    limit, offset = pagination
    _auth, organization = context
    repository = _repository()
    if (
        await repository.get_in_organization(
            session, dataset_id, organization_id=organization.id
        )
        is None
    ):
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
    settings: Annotated[Settings, Depends(get_settings)],
    context: TenantContext,
) -> DatasetVersion:
    auth, organization = context
    if (
        await _repository().get_in_organization(
            session, dataset_id, organization_id=organization.id
        )
        is None
    ):
        raise NotFoundError(f"Dataset {dataset_id} does not exist.")
    version = await _repository().create_version(
        session, dataset_id=dataset_id, version=payload.version, description=payload.description
    )
    if settings.audit_enabled:
        await record_audit_event(
            session,
            organization_id=organization.id,
            clerk_user_id=auth.clerk_user_id,
            action=DATASET_VERSION_CREATE,
            resource_type="dataset_version",
            resource_id=version.id,
            details={"dataset_id": str(dataset_id), "version": payload.version},
        )
        await session.commit()
    return version


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
    settings: Annotated[Settings, Depends(get_settings)],
    context: TenantContext,
    quotas: Annotated[QuotaService, Depends(get_quota_service)],
) -> TestCaseResponse:
    auth, organization = context
    dataset_version = await _get_version_or_404(
        session, dataset_id, version, organization_id=organization.id
    )
    await quotas.admit_case_add(
        session,
        organization_id=organization.id,
        clerk_user_id=auth.clerk_user_id,
        dataset_version_id=dataset_version.id,
    )
    case = await _repository().add_test_case(
        session,
        dataset_version_id=dataset_version.id,
        name=payload.name,
        input=payload.input,
        expected_output=payload.expected_output,
        context=payload.context,
        metadata=sanitize_configuration(payload.metadata),
    )
    if settings.audit_enabled:
        await record_audit_event(
            session,
            organization_id=organization.id,
            clerk_user_id=auth.clerk_user_id,
            action=DATASET_CASE_ADD,
            resource_type="test_case",
            resource_id=case.id,
            # Inputs/expected outputs can carry arbitrary user content —
            # record the case name only.
            details={"dataset_version_id": str(dataset_version.id), "name": payload.name},
        )
        await session.commit()
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
    context: TenantContext,
) -> Page[TestCaseResponse]:
    limit, offset = pagination
    _auth, organization = context
    dataset_version = await _get_version_or_404(
        session, dataset_id, version, organization_id=organization.id
    )
    cases = await _repository().list_test_cases(session, dataset_version.id)
    items = [_test_case_response(case) for case in cases[offset : offset + limit]]
    return Page[TestCaseResponse](
        items=items, total=len(cases), limit=limit, offset=offset
    )
