"""Application management endpoints (thin HTTP orchestration).

Phase 15 extends the surface with generic application connections:

- full CRUD on applications (list / patch / delete added)
- version ``connection`` configuration (validated, immutable, non-secret)
- credential rotation (``PATCH .../connection``) — write-only secret
- connection testing (``POST .../test``) against the stored configuration

Security invariants enforced here:

- every lookup goes through ``get_in_organization`` — another tenant's
  application is indistinguishable from a missing one (404, no IDOR)
- the plaintext secret is accepted, immediately encrypted, and never
  echoed back in any response (only ``secret_configured``)
- connection tests use stored configuration and return only safe,
  structured fields (status code, latency, truncated answer preview)
"""

import uuid
from contextlib import suppress
from datetime import UTC, datetime
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
from evalyx.api.errors import ConnectionNotReadyError
from evalyx.api.schemas.applications import (
    ApplicationCreate,
    ApplicationResponse,
    ApplicationSecretStateResponse,
    ApplicationSecretUpdate,
    ApplicationUpdate,
    ApplicationVersionCreate,
    ApplicationVersionResponse,
    ConnectionTestRequest,
    ConnectionTestResponse,
)
from evalyx.api.schemas.common import Page
from evalyx.application.base import (
    ApplicationInvocationError,
    create_application_target,
)
from evalyx.application.connection import ConnectionConfig, ConnectionConfigError
from evalyx.application.resolve import build_http_target
from evalyx.core.config import Settings
from evalyx.core.encryption import CIPHERTEXT_VERSION, SecretEncryptor
from evalyx.core.metrics import metrics
from evalyx.db.models import Application, ApplicationVersion, Organization
from evalyx.db.repositories import ApplicationRepository, NotFoundError
from evalyx.evaluation.failures import classify_failure
from evalyx.quotas import QuotaService
from evalyx.security.audit import (
    APPLICATION_CREATE,
    APPLICATION_DELETE,
    APPLICATION_SECRET_ROTATE,
    APPLICATION_UPDATE,
    APPLICATION_VERSION_CREATE,
    record_audit_event,
)

router = APIRouter(prefix="/applications", tags=["applications"])

ORDER_NOTE = "Ordered by creation time ascending (creation order)."
#: Bound on the answer preview returned by the connection test.
PREVIEW_CHARS = 280


def _repository() -> ApplicationRepository:
    return ApplicationRepository()


def _require_http(application: Application) -> None:
    """Connection configuration only applies to generic HTTP applications."""
    if application.connection_type != "http":
        raise ConnectionConfigError(
            "connection configuration is only valid for applications with "
            "connection_type='http'."
        )


async def _require_application(
    session: AsyncSession, application_id: uuid.UUID, organization_id: uuid.UUID
) -> Application:
    """Tenant-scoped fetch; foreign applications read as missing (404)."""
    application = await _repository().get_in_organization(
        session, application_id, organization_id=organization_id
    )
    if application is None:
        raise NotFoundError(f"Application {application_id} does not exist.")
    return application


@router.post(
    "",
    response_model=ApplicationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create an application",
    description=(
        "Register an AI application/agent for evaluation. Names are unique; "
        "a duplicate name is rejected with 409. ``connection_type='http'`` "
        "registers a generic application driven by version connection "
        "configurations; the optional write-only ``secret`` is encrypted at "
        "rest and never returned."
    ),
)
async def create_application(
    payload: ApplicationCreate,
    session: Annotated[AsyncSession, Depends(get_session)],
    settings: Annotated[Settings, Depends(get_settings)],
    context: Annotated[tuple[AuthContext, Organization], Depends(require_organization)],
    quotas: Annotated[QuotaService, Depends(get_quota_service)],
) -> ApplicationResponse:
    auth, organization = context
    await quotas.admit_application_create(
        session,
        organization_id=organization.id,
        clerk_user_id=auth.clerk_user_id,
    )
    encrypted_secret = None
    secret_metadata = None
    if payload.secret is not None:
        encryptor = SecretEncryptor.from_settings(settings)
        encrypted_secret = encryptor.encrypt(payload.secret)
        secret_metadata = {
            "key_version": CIPHERTEXT_VERSION,
            "key_id": encryptor.current_key_id,
            "rotated_at": datetime.now(UTC).isoformat(),
        }
    application = await _repository().create(
        session,
        organization_id=organization.id,
        name=payload.name,
        description=payload.description,
        connection_type=payload.connection_type,
        encrypted_secret=encrypted_secret,
        secret_metadata=secret_metadata,
    )
    if settings.audit_enabled:
        await record_audit_event(
            session,
            organization_id=organization.id,
            clerk_user_id=auth.clerk_user_id,
            action=APPLICATION_CREATE,
            resource_type="application",
            resource_id=application.id,
            details={
                "name": payload.name,
                "connection_type": payload.connection_type,
                "secret_configured": encrypted_secret is not None,
            },
        )
        await session.commit()
    return ApplicationResponse.from_application(application)


@router.get(
    "",
    response_model=Page[ApplicationResponse],
    summary="List applications",
    description=f"Applications scoped to the authenticated organization. {ORDER_NOTE}",
)
async def list_applications(
    session: Annotated[AsyncSession, Depends(get_session)],
    pagination: Annotated[tuple[int, int], Depends(pagination_params)],
    context: Annotated[tuple[AuthContext, Organization], Depends(require_organization)],
) -> Page[ApplicationResponse]:
    limit, offset = pagination
    _auth, organization = context
    applications, total = await _repository().list_in_organization(
        session, organization_id=organization.id, limit=limit, offset=offset
    )
    return Page[ApplicationResponse](
        items=[ApplicationResponse.from_application(a) for a in applications],
        total=total,
        limit=limit,
        offset=offset,
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
    context: Annotated[tuple[AuthContext, Organization], Depends(require_organization)],
) -> ApplicationResponse:
    _auth, organization = context
    application = await _require_application(session, application_id, organization.id)
    return ApplicationResponse.from_application(application)


@router.patch(
    "/{application_id}",
    response_model=ApplicationResponse,
    summary="Update application metadata",
    responses={404: {"description": "Application not found."}},
)
async def update_application(
    application_id: uuid.UUID,
    payload: ApplicationUpdate,
    session: Annotated[AsyncSession, Depends(get_session)],
    settings: Annotated[Settings, Depends(get_settings)],
    context: Annotated[tuple[AuthContext, Organization], Depends(require_organization)],
) -> ApplicationResponse:
    auth, organization = context
    application = await _require_application(session, application_id, organization.id)
    updated = await _repository().update(
        session, application, name=payload.name, description=payload.description
    )
    if settings.audit_enabled:
        await record_audit_event(
            session,
            organization_id=organization.id,
            clerk_user_id=auth.clerk_user_id,
            action=APPLICATION_UPDATE,
            resource_type="application",
            resource_id=application.id,
            details={"name": updated.name},
        )
        await session.commit()
    return ApplicationResponse.from_application(updated)


@router.delete(
    "/{application_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete an application",
    description=(
        "Deletes the application and its versions. Applications with "
        "recorded evaluation runs cannot be deleted (409 conflict)."
    ),
    responses={
        404: {"description": "Application not found."},
        409: {"description": "Application has evaluation runs."},
    },
)
async def delete_application(
    application_id: uuid.UUID,
    session: Annotated[AsyncSession, Depends(get_session)],
    settings: Annotated[Settings, Depends(get_settings)],
    context: Annotated[tuple[AuthContext, Organization], Depends(require_organization)],
) -> None:
    auth, organization = context
    application = await _require_application(session, application_id, organization.id)
    application_name = application.name
    await _repository().delete(session, application)
    if settings.audit_enabled:
        await record_audit_event(
            session,
            organization_id=organization.id,
            clerk_user_id=auth.clerk_user_id,
            action=APPLICATION_DELETE,
            resource_type="application",
            resource_id=application_id,
            details={"name": application_name},
        )
        await session.commit()


@router.patch(
    "/{application_id}/connection",
    response_model=ApplicationSecretStateResponse,
    summary="Rotate the application credential",
    description=(
        "Replaces the stored encrypted credential. The previous secret is "
        "never returned and never required. Responses expose only "
        "``secret_configured``."
    ),
    responses={404: {"description": "Application not found."}},
)
async def rotate_application_secret(
    application_id: uuid.UUID,
    payload: ApplicationSecretUpdate,
    session: Annotated[AsyncSession, Depends(get_session)],
    settings: Annotated[Settings, Depends(get_settings)],
    context: Annotated[tuple[AuthContext, Organization], Depends(require_organization)],
) -> ApplicationSecretStateResponse:
    auth, organization = context
    application = await _require_application(session, application_id, organization.id)
    encryptor = SecretEncryptor.from_settings(settings)
    await _repository().set_secret(
        session,
        application,
        encrypted_secret=encryptor.encrypt(payload.secret),
        secret_metadata={
            "key_version": CIPHERTEXT_VERSION,
            "key_id": encryptor.current_key_id,
            "rotated_at": datetime.now(UTC).isoformat(),
        },
    )
    if settings.audit_enabled:
        await record_audit_event(
            session,
            organization_id=organization.id,
            clerk_user_id=auth.clerk_user_id,
            action=APPLICATION_SECRET_ROTATE,
            resource_type="application",
            resource_id=application.id,
            # The credential itself is never recorded — only which key
            # envelope version now protects it.
            details={"key_version": CIPHERTEXT_VERSION, "key_id": encryptor.current_key_id},
        )
        await session.commit()
    return ApplicationSecretStateResponse(secret_configured=True)


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
    context: Annotated[tuple[AuthContext, Organization], Depends(require_organization)],
) -> Page[ApplicationVersionResponse]:
    limit, offset = pagination
    _auth, organization = context
    repository = _repository()
    if (
        await repository.get_in_organization(
            session, application_id, organization_id=organization.id
        )
        is None
    ):
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


@router.get(
    "/{application_id}/versions/{version_id}",
    response_model=ApplicationVersionResponse,
    summary="Retrieve an application version",
    responses={404: {"description": "Application or version not found."}},
)
async def get_application_version(
    application_id: uuid.UUID,
    version_id: uuid.UUID,
    session: Annotated[AsyncSession, Depends(get_session)],
    context: Annotated[tuple[AuthContext, Organization], Depends(require_organization)],
) -> ApplicationVersionResponse:
    _auth, organization = context
    repository = _repository()
    if (
        await repository.get_in_organization(
            session, application_id, organization_id=organization.id
        )
        is None
    ):
        raise NotFoundError(f"Application {application_id} does not exist.")
    version = await repository.get_version_by_id(
        session, application_id=application_id, version_id=version_id
    )
    if version is None:
        raise NotFoundError(f"Application version {version_id} does not exist.")
    return ApplicationVersionResponse.model_validate(version)


@router.post(
    "/{application_id}/versions",
    response_model=ApplicationVersionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create an application version",
    description=(
        "Create an immutable application configuration snapshot. Reusing a "
        "version label is rejected with 409. Configuration must be "
        "non-secret execution configuration; secret-looking keys are "
        "stripped before persistence. Generic (``connection_type='http'``) "
        "applications store their validated, non-secret connection "
        "configuration on the version; reference applications must not "
        "carry one."
    ),
    responses={
        404: {"description": "Application not found."},
        409: {"description": "Duplicate version label."},
        422: {"description": "Invalid connection configuration."},
    },
)
async def create_application_version(
    application_id: uuid.UUID,
    payload: ApplicationVersionCreate,
    session: Annotated[AsyncSession, Depends(get_session)],
    settings: Annotated[Settings, Depends(get_settings)],
    context: Annotated[tuple[AuthContext, Organization], Depends(require_organization)],
) -> ApplicationVersion:
    auth, organization = context
    repository = _repository()
    application = await repository.get_in_organization(
        session, application_id, organization_id=organization.id
    )
    if application is None:
        raise NotFoundError(f"Application {application_id} does not exist.")
    if payload.connection is not None:
        _require_http(application)
    version = await repository.create_version(
        session,
        application_id=application_id,
        version=payload.version,
        description=payload.description,
        configuration=payload.sanitized_configuration(),
        connection=payload.validated_connection(),
    )
    if settings.audit_enabled:
        await record_audit_event(
            session,
            organization_id=organization.id,
            clerk_user_id=auth.clerk_user_id,
            action=APPLICATION_VERSION_CREATE,
            resource_type="application_version",
            resource_id=version.id,
            details={"application_id": str(application_id), "version": payload.version},
        )
        await session.commit()
    return version


@router.post(
    "/{application_id}/test",
    response_model=ConnectionTestResponse,
    summary="Test an application connection",
    description=(
        "Makes one controlled, bounded request to the stored application "
        "endpoint (never an arbitrary URL from the request) and returns a "
        "safe structured result: success, latency, HTTP status, and a "
        "truncated answer preview. Failures reuse the Phase 12 failure "
        "taxonomy. Headers, credentials, and request payloads are never "
        "returned."
    ),
    responses={
        404: {"description": "Application, version, or connection not found."},
        409: {"description": "Connection is not ready (e.g. credential missing)."},
    },
)
async def test_application_connection(
    application_id: uuid.UUID,
    payload: ConnectionTestRequest,
    session: Annotated[AsyncSession, Depends(get_session)],
    settings: Annotated[Settings, Depends(get_settings)],
    context: Annotated[tuple[AuthContext, Organization], Depends(require_organization)],
    quotas: Annotated[QuotaService, Depends(get_quota_service)],
) -> ConnectionTestResponse:
    auth, organization = context
    repository = _repository()
    application = await _require_application(session, application_id, organization.id)
    audit_event_id = await quotas.admit_connection_test(
        session,
        organization_id=organization.id,
        clerk_user_id=auth.clerk_user_id,
    )
    target = await _build_test_target(
        session, repository, application, payload.version_id, settings
    )
    metrics.increment("application_connection_tests_total", {"outcome": "attempted"})
    try:
        try:
            response = await target.invoke(payload.prompt)
        except ApplicationInvocationError as exc:
            failure = classify_failure(exc, attempts=exc.attempts)
            metrics.increment("application_connection_tests_total", {"outcome": "error"})
            await quotas.record_connection_test_outcome(
                session,
                event_id=audit_event_id,
                application_id=application.id,
                success=False,
            )
            await session.commit()
            return ConnectionTestResponse(
                success=False,
                http_status=failure.http_status,
                failure=failure,
            )
    finally:
        with suppress(Exception):
            await target.close()
    metrics.increment("application_connection_tests_total", {"outcome": "success"})
    await quotas.record_connection_test_outcome(
        session,
        event_id=audit_event_id,
        application_id=application.id,
        success=True,
    )
    await session.commit()
    return ConnectionTestResponse(
        success=True,
        latency_ms=response.latency_ms,
        http_status=response.status_code,
        preview=response.content[:PREVIEW_CHARS],
    )


async def _build_test_target(
    session: AsyncSession,
    repository: ApplicationRepository,
    application: Application,
    version_id: uuid.UUID | None,
    settings: Settings,
):
    """Build the target for a connection test from stored configuration.

    Raises 404-class errors for a missing connection configuration and
    409-class errors for a missing credential (a configuration state, not
    a missing resource).
    """
    if application.connection_type != "http":
        # Reference demo target: test it through its configured base URL.
        return create_application_target("mlgpt", settings)
    if version_id is not None:
        version = await repository.get_version_by_id(
            session, application_id=application.id, version_id=version_id
        )
    else:
        version = await repository.latest_version_with_connection(
            session, application.id
        )
    if version is None or not isinstance(version.connection, dict):
        raise NotFoundError(
            "Application has no version with a connection configuration."
        )
    if application.encrypted_secret is None and _connection_requires_secret(version):
        raise ConnectionNotReadyError(
            "The application credential is not configured; rotate it via "
            "PATCH /applications/{id}/connection first."
        )
    return build_http_target(application, version, settings)


def _connection_requires_secret(version: ApplicationVersion) -> bool:
    """Whether the stored connection's auth mode requires a credential."""
    try:
        return ConnectionConfig.model_validate(version.connection).auth.requires_secret
    except Exception:  # noqa: BLE001 — invalid stored config fails later anyway
        return False
