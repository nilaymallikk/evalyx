"""Application management endpoints (thin HTTP orchestration)."""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from evalyx.api.dependencies import get_session, pagination_params
from evalyx.api.schemas.applications import (
    ApplicationCreate,
    ApplicationResponse,
    ApplicationVersionCreate,
    ApplicationVersionResponse,
)
from evalyx.api.schemas.common import Page
from evalyx.db.models import Application, ApplicationVersion
from evalyx.db.repositories import ApplicationRepository, NotFoundError

router = APIRouter(prefix="/applications", tags=["applications"])

ORDER_NOTE = "Ordered by creation time ascending (creation order)."


def _repository() -> ApplicationRepository:
    return ApplicationRepository()


@router.post(
    "",
    response_model=ApplicationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create an application",
    description=(
        "Register an AI application/agent for evaluation. Names are unique; "
        "a duplicate name is rejected with 409."
    ),
)
async def create_application(
    payload: ApplicationCreate,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> Application:
    return await _repository().create(
        session, name=payload.name, description=payload.description
    )


@router.get(
    "/{application_id}",
    response_model=ApplicationResponse,
    summary="Retrieve an application",
    responses={404: {"description": "Application not found."}},
)
async def get_application(
    application_id: uuid.UUID,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> Application:
    application = await _repository().get(session, application_id)
    if application is None:
        raise NotFoundError(f"Application {application_id} does not exist.")
    return application


@router.get(
    "/{application_id}/versions",
    response_model=Page[ApplicationVersionResponse],
    summary="List application versions",
    description=f"Immutable configuration snapshots of one application. {ORDER_NOTE}",
)
async def list_application_versions(
    application_id: uuid.UUID,
    session: Annotated[AsyncSession, Depends(get_session)],
    pagination: Annotated[tuple[int, int], Depends(pagination_params)],
) -> Page[ApplicationVersionResponse]:
    limit, offset = pagination
    repository = _repository()
    if await repository.get(session, application_id) is None:
        raise NotFoundError(f"Application {application_id} does not exist.")
    versions = await repository.list_versions(session, application_id)
    items = [
        ApplicationVersionResponse.model_validate(version)
        for version in versions[offset : offset + limit]
    ]
    return Page[ApplicationVersionResponse](
        items=items,
        total=len(versions),
        limit=limit,
        offset=offset,
    )


@router.post(
    "/{application_id}/versions",
    response_model=ApplicationVersionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create an application version",
    description=(
        "Create an immutable application configuration snapshot. Reusing a "
        "version label is rejected with 409. Configuration must be "
        "non-secret execution configuration; secret-looking keys are "
        "stripped before persistence."
    ),
    responses={404: {"description": "Application not found."}, 409: {"description": "Duplicate version label."}},
)
async def create_application_version(
    application_id: uuid.UUID,
    payload: ApplicationVersionCreate,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> ApplicationVersion:
    return await _repository().create_version(
        session,
        application_id=application_id,
        version=payload.version,
        description=payload.description,
        configuration=payload.sanitized_configuration(),
    )
