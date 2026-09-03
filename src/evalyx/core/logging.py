"""Logging configuration for Evalyx.

Uses structlog for structured, level-filtered application logging. Human
readable output in development, JSON elsewhere. Secret values are protected
at the source by storing them as ``SecretStr`` in :mod:`evalyx.core.config`,
which means they can never be rendered by any log processor.
"""

import logging

import structlog

from evalyx.core.config import Settings

#: Event-dict keys whose values must never reach logs (case-insensitive
#: substring match). Applied as a structlog processor so a caller that
#: accidentally logs a credential gets "[redacted]" instead of the secret.
_SENSITIVE_SUBSTRINGS = (
    "secret",
    "token",
    "password",
    "api_key",
    "apikey",
    "authorization",
    "encryption_key",
    "clerk",
    "credential",
    "bearer",
)


def _redact_sensitive(
    _: object, __: str, event_dict: structlog.typing.EventDict
) -> structlog.typing.EventDict:
    for key in list(event_dict.keys()):
        lowered = str(key).lower()
        if any(part in lowered for part in _SENSITIVE_SUBSTRINGS):
            event_dict[key] = "[redacted]"
    return event_dict


def configure_logging(settings: Settings) -> None:
    """Configure stdlib + structlog logging from application settings.

    Safe to call multiple times (reconfigures with the latest settings).
    """
    level = logging.getLevelNamesMapping()[settings.log_level]

    logging.basicConfig(format="%(message)s", level=level, force=True)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            _redact_sensitive,
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer()
            if settings.is_development
            else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=False,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger."""
    return structlog.get_logger(name)
