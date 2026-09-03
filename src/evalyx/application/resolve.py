"""Application target resolution (Phase 15).

Builds the :class:`evalyx.application.base.ApplicationTarget` for an
evaluation run from the database — the replacement for the hardcoded name
registry. Resolution rules (in order):

1. The run's ``agent_model`` must carry an ``application:<name>`` selector;
   otherwise the run is a raw-provider run (``None``).
2. ``connection_type == "http"``: the application's connection
   configuration on its pinned (or latest) version drives a generic
   :class:`HTTPApplicationTarget`. The credential is decrypted *here* — at
   the execution boundary — never in task arguments and never earlier.
3. Otherwise the legacy name registry applies (the ``mlgpt`` reference
   demo target), preserving Phase 8 behavior unchanged.

The evaluation engine stays ignorant of all of this: it only sees an
``ApplicationTarget``.
"""

import uuid

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from evalyx.application.base import (
    ApplicationInvocationError,
    ApplicationTarget,
    application_name_from_model,
    create_application_target,
)
from evalyx.application.connection import ConnectionConfig
from evalyx.application.generic_http import HTTPApplicationTarget
from evalyx.core.config import Settings
from evalyx.core.encryption import EncryptionError, SecretEncryptor
from evalyx.db.models import Application, ApplicationVersion
from evalyx.db.repositories import ApplicationRepository, EvaluationRepository


def build_http_target(
    application: Application,
    version: ApplicationVersion,
    settings: Settings,
    *,
    application_name: str | None = None,
) -> HTTPApplicationTarget:
    """Construct a generic HTTP target from stored, validated configuration.

    Decrypts the application credential only when the connection's auth
    mode requires one. Raises :class:`ApplicationInvocationError` (safe,
    secret-free messages) for missing configuration or credentials.
    """
    if not isinstance(version.connection, dict):
        raise ApplicationInvocationError(
            "Application version has no connection configuration.",
            category="unknown",
        )
    try:
        connection = ConnectionConfig.model_validate(version.connection)
    except Exception as exc:
        raise ApplicationInvocationError(
            "Application version connection configuration is invalid.",
            category="unknown",
        ) from exc
    secret: str | None = None
    if connection.auth.requires_secret:
        if not application.encrypted_secret:
            raise ApplicationInvocationError(
                "Application credential is not configured.",
                category="unknown",
            )
        try:
            encryptor = SecretEncryptor.from_settings(settings)
            secret = encryptor.decrypt(application.encrypted_secret)
        except EncryptionError:
            raise ApplicationInvocationError(
                "Application credential could not be decrypted.",
                category="unknown",
            ) from None
    return HTTPApplicationTarget(
        connection,
        secret=secret,
        application_name=application_name or application.name,
    )


async def resolve_run_target(
    session: AsyncSession, run, settings: Settings
) -> ApplicationTarget:
    """Resolve the target for a loaded evaluation run (DB-driven)."""
    name = application_name_from_model(run.agent_model)
    if name is None:
        raise ApplicationInvocationError(
            f"Evaluation run {run.id} does not select an application target.",
            category="unknown",
        )
    application = await session.get(Application, run.application_id)
    if application is None:
        raise ApplicationInvocationError(
            f"Evaluation run {run.id} references a missing application.",
            category="unknown",
        )
    if application.connection_type == "http":
        version = None
        if run.application_version_id is not None:
            version = await ApplicationRepository().get_version_by_id(
                session,
                application_id=application.id,
                version_id=run.application_version_id,
            )
        else:
            version = await ApplicationRepository().latest_version_with_connection(
                session, application.id
            )
        if version is None:
            raise ApplicationInvocationError(
                "Application has no version with a connection configuration.",
                category="unknown",
            )
        return build_http_target(application, version, settings)
    # Legacy registry (the mlgpt reference target); unknown names raise.
    return create_application_target(name, settings)


async def resolve_application_target(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    run_id: uuid.UUID,
    settings: Settings,
) -> ApplicationTarget:
    """Load the run and resolve its application target (worker entry point)."""
    async with session_factory() as session:
        run = await EvaluationRepository().get_run(session, run_id)
        if run is None:
            raise ApplicationInvocationError(
                f"Evaluation run {run_id} does not exist.",
                category="unknown",
            )
        return await resolve_run_target(session, run, settings)


__all__ = [
    "build_http_target",
    "resolve_application_target",
    "resolve_run_target",
]