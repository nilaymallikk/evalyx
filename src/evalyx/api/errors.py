"""Centralized API error handling: domain errors → consistent HTTP responses.

Every error response uses one JSON shape::

    {"error": {"code": "<stable_code>", "message": "<human-readable>"}}

Security rules enforced here:

- no stack traces, SQLAlchemy internals, filesystem paths, or provider
  payloads in any response body
- request-validation (422) responses deliberately exclude the offending
  ``input`` values — a client may have sent a secret in a bad request, and
  echoing it back would leak it
- unexpected exceptions are logged with full detail via structured logging
  and answered with a generic 500

Mapping:

- ``NotFoundError``                → 404 ``not_found``
- ``DuplicateVersionError``        → 409 ``duplicate_version``
- ``sqlalchemy IntegrityError``    → 409 ``conflict`` (e.g. duplicate name)
- ``RegressionValidationError``    → 400 ``invalid_comparison``
- ``EvaluationSubmissionError``    → 503 ``evaluation_enqueue_failed``
- request validation (FastAPI)     → 422 ``validation_error``
- anything else                    → 500 ``internal_error``
"""

import structlog
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.exc import IntegrityError

from evalyx.api.auth import (
    AuthenticationError,
    OrganizationRequiredError,
    OrganizationRoleError,
)
from evalyx.api.middleware import SCOPE_REQUEST_ID_KEY
from evalyx.core.context import get_request_id
from evalyx.db.repositories import DuplicateVersionError, NotFoundError
from evalyx.evaluation.regression.service import RegressionValidationError

logger = structlog.get_logger(__name__)


class ErrorBody(BaseModel):
    """The ``error`` object embedded in every error response."""

    code: str
    message: str


class ErrorResponse(BaseModel):
    """Consistent envelope for all non-2xx API responses."""

    error: ErrorBody


class EvaluationSubmissionError(Exception):
    """The evaluation run was persisted but could not be enqueued.

    The run is transitioned to ``failed`` (best effort) before this is
    raised, so PostgreSQL never shows a healthy pending run behind a job
    that was never queued.
    """

    def __init__(self, run_id: str, message: str) -> None:
        super().__init__(message)
        self.run_id = run_id
        self.code = "evaluation_enqueue_failed"


def _error_response(
    status_code: int,
    code: str,
    message: str,
    *,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    body = ErrorResponse(error=ErrorBody(code=code, message=message))
    return JSONResponse(status_code=status_code, content=body.model_dump(), headers=headers)


def register_error_handlers(app: FastAPI) -> None:
    """Install the centralized domain-error → HTTP-response handlers."""

    @app.exception_handler(RequestValidationError)
    async def _handle_request_validation(
        _: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # ``loc``/``msg`` only — never ``input`` (must not echo request data).
        details = [
            {"loc": ".".join(str(part) for part in error.get("loc", ())), "msg": error.get("msg", "")}
            for error in exc.errors()
        ]
        logger.info("request_validation_failed", errors=details)
        summary = "; ".join(f"{d['loc']}: {d['msg']}" for d in details) or "Invalid request."
        return _error_response(
            status.HTTP_422_UNPROCESSABLE_ENTITY, "validation_error", summary
        )

    @app.exception_handler(AuthenticationError)
    async def _handle_authentication(_: Request, exc: AuthenticationError) -> JSONResponse:
        # 401 with a kind-only message: no token contents, no Clerk details.
        return _error_response(
            status.HTTP_401_UNAUTHORIZED,
            "authentication_failed",
            str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        )

    @app.exception_handler(OrganizationRequiredError)
    async def _handle_organization_required(
        _: Request, exc: OrganizationRequiredError
    ) -> JSONResponse:
        return _error_response(
            status.HTTP_403_FORBIDDEN, "organization_required", str(exc)
        )

    @app.exception_handler(OrganizationRoleError)
    async def _handle_organization_role(
        _: Request, exc: OrganizationRoleError
    ) -> JSONResponse:
        return _error_response(
            status.HTTP_403_FORBIDDEN, "insufficient_role", str(exc)
        )

    @app.exception_handler(NotFoundError)
    async def _handle_not_found(_: Request, exc: NotFoundError) -> JSONResponse:
        return _error_response(status.HTTP_404_NOT_FOUND, "not_found", str(exc))

    @app.exception_handler(DuplicateVersionError)
    async def _handle_duplicate_version(
        _: Request, exc: DuplicateVersionError
    ) -> JSONResponse:
        return _error_response(
            status.HTTP_409_CONFLICT, "duplicate_version", str(exc)
        )

    @app.exception_handler(IntegrityError)
    async def _handle_integrity_error(_: Request, exc: IntegrityError) -> JSONResponse:
        # Uniqueness violations surfaced by the database (e.g. duplicate
        # application/dataset names). Details stay in the logs only.
        logger.info("integrity_error", error=type(exc.orig).__name__ if exc.orig else None)
        return _error_response(
            status.HTTP_409_CONFLICT,
            "conflict",
            "Resource conflict: the request violates a uniqueness constraint.",
        )

    @app.exception_handler(RegressionValidationError)
    async def _handle_regression_validation(
        _: Request, exc: RegressionValidationError
    ) -> JSONResponse:
        return _error_response(
            status.HTTP_400_BAD_REQUEST, "invalid_comparison", str(exc)
        )

    @app.exception_handler(EvaluationSubmissionError)
    async def _handle_submission_error(
        _: Request, exc: EvaluationSubmissionError
    ) -> JSONResponse:
        return _error_response(
            status.HTTP_503_SERVICE_UNAVAILABLE, exc.code, str(exc)
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        # The correlation contextvar is already cleared when this outer
        # handler runs (the observability middleware unwound first), so the
        # resolved id is read from the per-request ASGI scope instead.
        request_id = request.scope.get(SCOPE_REQUEST_ID_KEY) or get_request_id()
        logger.error(
            "unhandled_api_error",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            error=type(exc).__name__,
            exc_info=True,  # noqa: LOG014 — traceback goes to logs, never to the response
        )
        response = _error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "internal_error",
            "An unexpected internal error occurred.",
        )
        if request_id is not None:
            response.headers["X-Request-ID"] = request_id
        return response
