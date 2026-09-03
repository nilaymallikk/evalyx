"""CLI error hierarchy and exit codes.

Predictable exit codes (documented in the README) make the CLI safe for CI:

    0 success
    1 general failure
    2 usage/argument error
    3 authentication failure
    4 authorization failure
    5 resource not found
    6 connection/network failure
    7 evaluation completed with quality failures
    8 evaluation execution errors

Error messages are kind-only: they never echo tokens, secrets, or full
request payloads (the backend may have received them in a bad request).
"""

import httpx

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_AUTH = 3
EXIT_FORBIDDEN = 4
EXIT_NOT_FOUND = 5
EXIT_CONNECTION = 6
EXIT_QUALITY_FAILURES = 7
EXIT_EXECUTION_ERRORS = 8


class EvalyxCLIError(Exception):
    """Base class for CLI errors; carries the process exit code."""

    exit_code = EXIT_ERROR

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.hint = hint


class UsageError(EvalyxCLIError):
    exit_code = EXIT_USAGE


class AuthenticationError(EvalyxCLIError):
    exit_code = EXIT_AUTH


class AuthorizationError(EvalyxCLIError):
    exit_code = EXIT_FORBIDDEN


class NotFoundError(EvalyxCLIError):
    exit_code = EXIT_NOT_FOUND


class ValidationError(EvalyxCLIError):
    exit_code = EXIT_USAGE


class APIConnectionError(EvalyxCLIError):
    exit_code = EXIT_CONNECTION


class APIError(EvalyxCLIError):
    """A 4xx/5xx response that did not map to a more specific class."""


def normalize_http_error(exc: httpx.HTTPStatusError, subject: str) -> EvalyxCLIError:
    """Map one HTTP error response to the CLI exception hierarchy.

    The backend's error envelope is ``{"error": {"code", "message"}}``; any
    other shape degrades to a generic message with the status code only.
    """
    response = exc.response
    code = ""
    message = ""
    try:
        body = response.json()
        error = body.get("error", {})
        code = str(error.get("code", ""))
        message = str(error.get("message", ""))
    except Exception:  # noqa: BLE001, S110 — non-JSON error body
        pass

    hint = None
    if response.status_code == 401:
        return AuthenticationError(
            "Authentication failed or expired.", hint="Run: evalyx login"
        )
    if response.status_code == 403:
        if code == "organization_required":
            hint = "Set an active organization: evalyx org use <organization-id>"
        return AuthorizationError("Permission denied.", hint=hint)
    if response.status_code == 404:
        return NotFoundError(f"{subject} not found.")
    if response.status_code in (400, 409, 422):
        return ValidationError(message or f"{subject} request was rejected.")

    detail = f" (server said: {message})" if message else ""
    return APIError(f"Evalyx API error {response.status_code}{detail}")
