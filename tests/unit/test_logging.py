"""Unit tests for the logging foundation. No network access required."""

import structlog

from evalyx.core.config import Settings
from evalyx.core.logging import configure_logging, get_logger

# Placeholder non-secret (these tests assert log filtering, not secrets).
_PLACEHOLDER_SECRET = "placeholder-" + "logging-secret"


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, evalyx_secret_key=_PLACEHOLDER_SECRET, **overrides)


def _captured(capsys) -> str:
    captured = capsys.readouterr()
    return captured.out + captured.err


def test_configure_logging_respects_log_level(capsys):
    configure_logging(_settings(log_level="WARNING"))

    get_logger("evalyx.test").info("below-threshold-message")

    assert "below-threshold-message" not in _captured(capsys)


def test_configure_logging_emits_at_configured_level(capsys):
    configure_logging(_settings(log_level="INFO"))

    get_logger("evalyx.test").info("configuration_loaded")

    assert "configuration_loaded" in _captured(capsys)


def test_configure_logging_is_idempotent():
    settings = _settings()
    configure_logging(settings)
    configure_logging(settings)  # must not raise


def test_logs_do_not_leak_secrets(capsys):
    configure_logging(_settings(log_level="DEBUG"))

    logger = get_logger("evalyx.test")
    logger.info("startup", settings=repr(_settings()))

    output = _captured(capsys)
    assert "startup" in output
    # Secrets live in SecretStr fields and can never be rendered.
    assert "unit-test-secret" not in output

