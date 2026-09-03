"""Generic HTTP application connection configuration (Phase 15).

A :class:`ConnectionConfig` describes *how Evalyx calls a user's AI
application over HTTP* — the immutable, non-secret part of an application
version. The credential itself is stored separately (encrypted at rest on
the application row); this module holds no secrets.

Deliberately small and safe (no workflow engine, no executable templates):

- **Request mapping** — two bounded modes: ``field`` (put the Evalyx test
  case input into one JSON field) and ``template`` (a small JSON body
  template whose string leaves may reference exactly one variable,
  ``{{input}}``). No arbitrary executable templates.
- **Response extraction** — a safe dotted JSON-path-like extractor
  (``answer``, ``data.response``, ``choices.0.message.content``). No
  JSONPath dependency: keys, numeric list indices, done.
- **Authentication** — ``none``, ``bearer`` (``Authorization: Bearer …``),
  or a custom API-key header (``X-API-Key: …`` by default).

Security rules enforced at validation time:

- endpoint must be http(s), credential-free, fragment-free, and — when it
  is a literal IP — public (full DNS validation happens per request; see
  :mod:`evalyx.application.ssrf`)
- user-supplied headers may not override hop-by-hop or security-sensitive
  headers (``Host``, ``Content-Length``, ``Authorization``, ...)
- template size/depth are bounded; timeouts and retry counts are bounded
"""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evalyx.application.ssrf import (
    FORBIDDEN_HEADER_NAMES,
    SSRFViolationError,
    assert_static_url_allowed,
)

#: Hard bounds (documented intentional limits, not magic numbers).
MAX_TEMPLATE_JSON_CHARS = 10_000
MAX_TEMPLATE_DEPTH = 10
MAX_EXTRA_FIELDS_JSON_CHARS = 10_000
MAX_HEADER_VALUE_CHARS = 1024
#: Extraction path: up to 10 segments of letters/digits/underscore.
RESPONSE_PATH_PATTERN = re.compile(r"^[A-Za-z0-9_]{1,64}(?:\.[A-Za-z0-9_]{1,64}){0,9}$")
_HEADER_NAME_PATTERN = re.compile(r"^[A-Za-z0-9-]{1,128}$")
_TEMPLATE_VARIABLE_PATTERN = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")
#: The single template variable (deliberately the only one).
TEMPLATE_INPUT_VARIABLE = "input"


class ConnectionConfigError(ValueError):
    """A connection configuration is invalid (maps to HTTP 422)."""


class AuthConfig(BaseModel):
    """Application authentication mode (the secret itself is NOT here)."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["none", "bearer", "api_key"] = "none"
    #: Header carrying the API key when ``type == "api_key"``.
    header_name: str = "X-API-Key"

    @property
    def requires_secret(self) -> bool:
        return self.type != "none"

    @model_validator(mode="after")
    def _validate(self) -> AuthConfig:
        if self.type == "api_key":
            name = self.header_name
            if not _HEADER_NAME_PATTERN.fullmatch(name):
                raise ConnectionConfigError(
                    "auth.header_name must be 1-128 letters, digits, or hyphens."
                )
            if name.lower() in FORBIDDEN_HEADER_NAMES:
                raise ConnectionConfigError(
                    "auth.header_name must not be a security-sensitive header."
                )
            if name.lower() == "authorization":
                raise ConnectionConfigError(
                    "use auth.type='bearer' for Authorization headers."
                )
        elif self.header_name != "X-API-Key":
            raise ConnectionConfigError(
                "auth.header_name is only valid when auth.type='api_key'."
            )
        return self


class RequestMapping(BaseModel):
    """How the Evalyx test-case input becomes the application's request body."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["field", "template"] = "field"
    #: Field mode: the JSON field receiving the test-case input.
    input_field: str = "input"
    #: Template mode: JSON body template; string leaves may use ``{{input}}``.
    body_template: dict | None = None
    #: Static extra fields merged into the request body (non-secret).
    extra_fields: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate(self) -> RequestMapping:
        if self.mode == "template":
            _validate_template(self)
        else:
            if self.body_template is not None:
                raise ConnectionConfigError(
                    "request.body_template is only valid when request.mode='template'."
                )
            if not self.input_field or len(self.input_field) > 128:
                raise ConnectionConfigError(
                    "request.input_field must be 1-128 characters."
                )
        if len(str(self.extra_fields)) > MAX_EXTRA_FIELDS_JSON_CHARS:
            raise ConnectionConfigError(
                f"request.extra_fields exceeds {MAX_EXTRA_FIELDS_JSON_CHARS} characters."
            )
        return self


def _validate_template(mapping: RequestMapping) -> None:
    if not isinstance(mapping.body_template, dict) or not mapping.body_template:
        raise ConnectionConfigError(
            "request.body_template is required when request.mode='template'."
        )
    if len(str(mapping.body_template)) > MAX_TEMPLATE_JSON_CHARS:
        raise ConnectionConfigError(
            f"request.body_template exceeds {MAX_TEMPLATE_JSON_CHARS} characters."
        )
    if _template_depth(mapping.body_template, 0) > MAX_TEMPLATE_DEPTH:
        raise ConnectionConfigError(
            f"request.body_template exceeds {MAX_TEMPLATE_DEPTH} levels of nesting."
        )
    variables = _template_variables(mapping.body_template)
    unknown = variables - {TEMPLATE_INPUT_VARIABLE}
    if unknown:
        raise ConnectionConfigError(
            f"request.body_template supports only the {{{{{TEMPLATE_INPUT_VARIABLE}}}}} variable."
        )
    if not variables:
        raise ConnectionConfigError(
            f"request.body_template must reference {{{{{TEMPLATE_INPUT_VARIABLE}}}}}."
        )


def _template_depth(value: object, depth: int) -> int:
    if depth > MAX_TEMPLATE_DEPTH + 1:
        return depth
    if isinstance(value, dict):
        return max((_template_depth(v, depth + 1) for v in value.values()), default=depth)
    if isinstance(value, list):
        return max((_template_depth(v, depth + 1) for v in value), default=depth)
    return depth


def _template_variables(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, str):
        found.update(_TEMPLATE_VARIABLE_PATTERN.findall(value))
    elif isinstance(value, dict):
        for item in value.values():
            found |= _template_variables(item)
    elif isinstance(value, list):
        for item in value:
            found |= _template_variables(item)
    return found


class ConnectionConfig(BaseModel):
    """Validated, non-secret configuration for calling an HTTP application.

    Stored (validated and sanitized) as the ``connection`` JSONB of an
    immutable application version. Secrets live separately, encrypted, on
    the application row.
    """

    model_config = ConfigDict(extra="forbid")

    endpoint: str = Field(min_length=1, max_length=2048)
    method: Literal["POST", "GET"] = "POST"
    auth: AuthConfig = Field(default_factory=AuthConfig)
    request: RequestMapping = Field(default_factory=RequestMapping)
    #: Dotted path to the model answer inside the JSON response.
    response_path: str = "answer"
    #: Static custom headers (validated against the forbidden set).
    headers: dict[str, str] = Field(default_factory=dict)
    #: Per-attempt read timeout (seconds, bounded).
    timeout_seconds: float = Field(default=30.0, ge=1.0, le=120.0)
    #: Transport attempts for transient failures (bounded).
    max_attempts: int = Field(default=3, ge=1, le=5)

    @model_validator(mode="after")
    def _validate(self) -> ConnectionConfig:
        try:
            assert_static_url_allowed(self.endpoint)
        except SSRFViolationError as exc:
            raise ConnectionConfigError(str(exc)) from None
        if not RESPONSE_PATH_PATTERN.fullmatch(self.response_path):
            raise ConnectionConfigError(
                "response_path must be a dotted path of up to 10 segments "
                "(letters, digits, underscore), e.g. 'choices.0.message.content'."
            )
        for name, value in self.headers.items():
            if not _HEADER_NAME_PATTERN.fullmatch(name):
                raise ConnectionConfigError(
                    "header names must be 1-128 letters, digits, or hyphens."
                )
            if name.lower() in FORBIDDEN_HEADER_NAMES:
                raise ConnectionConfigError(
                    f"header {name!r} may not be overridden (security-sensitive)."
                )
            if not isinstance(value, str) or not value or len(value) > MAX_HEADER_VALUE_CHARS:
                raise ConnectionConfigError(
                    "header values must be non-empty strings of at most "
                    f"{MAX_HEADER_VALUE_CHARS} characters."
                )
            if (
                self.auth.type == "api_key"
                and name.lower() == self.auth.header_name.lower()
            ):
                raise ConnectionConfigError(
                    f"header {name!r} conflicts with the configured auth header."
                )
        return self


# -- request mapping (runtime) -----------------------------------------------


def build_request_body(request: RequestMapping, prompt: str) -> dict:
    """Map the Evalyx test-case input into the application's request body.

    Bounded and side-effect free: field mode places the prompt in one JSON
    field; template mode substitutes ``{{input}}`` in string leaves of the
    validated template. No other templating exists (by design).
    """
    if request.mode == "template":
        assert request.body_template is not None  # schema-validated
        rendered = _render_template(request.body_template, prompt)
        assert isinstance(rendered, dict)  # the template root is a dict
        return rendered
    body: dict = dict(request.extra_fields)
    body[request.input_field] = prompt
    return body


def _render_template(template: object, prompt: str) -> object:
    if isinstance(template, str):
        return template.replace("{{" + TEMPLATE_INPUT_VARIABLE + "}}", prompt)
    if isinstance(template, dict):
        return {key: _render_template(value, prompt) for key, value in template.items()}
    if isinstance(template, list):
        return [_render_template(item, prompt) for item in template]
    return template


# -- response extraction (runtime) -------------------------------------------


class ResponseExtractionError(Exception):
    """The application's response did not contain the configured answer path.

    Safe by construction: messages name the failure kind only — never the
    response body (which may echo the prompt).
    """


def extract_answer(body: object, response_path: str) -> str:
    """Extract the model answer via a dotted path (``choices.0.message.content``).

    Deterministic JSON-walking: dict keys and numeric list indices only.
    Raises :class:`ResponseExtractionError` when the path is missing, the
    value is not a non-empty string, or the body is not JSON-shaped.
    """
    current = body
    for part in response_path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif (
            isinstance(current, list)
            and part.isdigit()
            and int(part) < len(current)
        ):
            current = current[int(part)]
        else:
            raise ResponseExtractionError(
                f"response path {response_path!r} not found in the application response."
            )
    if not isinstance(current, str) or not current.strip():
        raise ResponseExtractionError(
            f"response path {response_path!r} did not yield a non-empty string."
        )
    return current


__all__ = [
    "TEMPLATE_INPUT_VARIABLE",
    "AuthConfig",
    "ConnectionConfig",
    "ConnectionConfigError",
    "RequestMapping",
    "ResponseExtractionError",
    "build_request_body",
    "extract_answer",
]