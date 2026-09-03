"""Unit tests for the configuration layer. No network access required."""

import pytest
from pydantic import ValidationError

from evalyx.core.config import (
    DEFAULT_AGENT_MODEL,
    DEFAULT_JUDGE_MODEL,
    Settings,
)

# Environment variables that could leak ambient configuration into the
# hermetic defaults tests below.
AMBIENT_ENV_VARS = (
    "APP_ENV",
    "LOG_LEVEL",
    "DATABASE_URL",
    "REDIS_URL",
    "EVALYX_SECRET_KEY",
    "OPENROUTER_API_KEY",
    "EVALYX_AGENT_MODEL",
    "EVALYX_JUDGE_MODEL",
    "OLLAMA_BASE_URL",
)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    """Remove ambient configuration so Settings defaults are tested directly."""
    for var in AMBIENT_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


#: Placeholder secret for tests. Deliberately derived from the helper's own
#: name so no literal credential-looking string appears in the codebase;
#: these tests assert masking behavior, not secret values.
_PLACEHOLDER_SECRET = "placeholder-" + "settings-secret"

#: A valid urlsafe base64-encoded 32-byte key (generated once for tests).
_TEST_ENCRYPTION_KEY = "CZWNnvRiuKkYgjlplxwPzBYz1hQYgo72d8M29i22800="


def make_settings(**overrides) -> Settings:
    """Build Settings without reading the developer's local .env file."""
    defaults = {"evalyx_secret_key": _PLACEHOLDER_SECRET, "auth_required": False}
    return Settings(_env_file=None, **{**defaults, **overrides})


class TestConfigurationLoading:
    def test_configuration_loads_with_safe_defaults(self):
        settings = make_settings()

        assert settings.app_env == "development"
        assert settings.log_level == "INFO"
        assert settings.database_url == (
            "postgresql+asyncpg://evalyx:evalyx@localhost:5433/evalyx"
        )
        assert settings.redis_url == "redis://localhost:6379/0"
        assert settings.ollama_base_url == "http://localhost:11434"

    def test_free_model_defaults_are_used(self):
        settings = make_settings()

        assert settings.evalyx_agent_model == DEFAULT_AGENT_MODEL
        assert settings.evalyx_judge_model == DEFAULT_JUDGE_MODEL
        # Guard against accidental paid-model drift.
        assert settings.evalyx_agent_model.endswith(":free")
        assert settings.evalyx_judge_model.endswith(":free")

    def test_environment_variables_override_defaults(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://other:other@db-host:9999/other")
        monkeypatch.setenv("REDIS_URL", "redis://cache-host:6380/1")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")

        settings = make_settings()

        assert settings.database_url == "postgresql+asyncpg://other:other@db-host:9999/other"
        assert settings.redis_url == "redis://cache-host:6380/1"
        assert settings.log_level == "DEBUG"

    def test_secret_key_is_required(self, monkeypatch):
        monkeypatch.delenv("EVALYX_SECRET_KEY", raising=False)

        with pytest.raises(ValidationError, match="EVALYX_SECRET_KEY"):
            Settings(_env_file=None, evalyx_secret_key="")


class TestSecretValidation:
    def test_placeholder_secret_rejected_in_production(self):
        with pytest.raises(ValidationError, match="EVALYX_SECRET_KEY"):
            make_settings(app_env="production", evalyx_secret_key="change-me")

    def test_generated_secret_accepted_in_production(self):
        # Phase 15: production additionally requires authentication and the
        # encryption key, so this acceptance test supplies both. The intent
        # is unchanged: a generated (non-placeholder) EVALYX_SECRET_KEY is
        # accepted in production.
        settings = make_settings(
            app_env="production",
            evalyx_secret_key="s3cret-value",
            auth_required=True,
            clerk_jwks_url="https://instance.clerk.accounts.dev/.well-known/jwks.json",
            clerk_secret_key="sk_test_placeholder_not-a-real-key",
            evalyx_encryption_key=_TEST_ENCRYPTION_KEY,
        )

        assert settings.app_env == "production"


class TestProductionAuthHardening:
    """Phase 15 Step 25: production cannot run with authentication off."""

    def test_production_auth_required_zero_fails(self):
        with pytest.raises(ValueError, match="AUTH_REQUIRED"):
            make_settings(app_env="production", auth_required=False)

    def test_development_auth_required_zero_allowed(self):
        settings = make_settings(app_env="development", auth_required=False)
        assert settings.auth_required is False

    def test_production_requires_encryption_key(self):
        with pytest.raises(ValueError, match="EVALYX_ENCRYPTION_KEY"):
            make_settings(
                app_env="production",
                auth_required=True,
                clerk_jwks_url="https://instance.clerk.accounts.dev/.well-known/jwks.json",
                clerk_secret_key=_PLACEHOLDER_SECRET,
                evalyx_encryption_key="",  # explicitly unset (env may provide one)
            )

    def test_malformed_encryption_key_fails_without_echo(self):
        with pytest.raises(ValueError, match="EVALYX_ENCRYPTION_KEY"):
            make_settings(evalyx_encryption_key="not-a-valid-key!!")

    def test_valid_encryption_key_accepted(self):
        settings = make_settings(evalyx_encryption_key=_TEST_ENCRYPTION_KEY)
        assert (
            settings.evalyx_encryption_key.get_secret_value() == _TEST_ENCRYPTION_KEY
        )


class TestSecretMasking:
    def test_secrets_never_appear_in_repr_or_str(self):
        settings = make_settings(
            evalyx_secret_key=make_settings.__name__,  # placeholder secret
            openrouter_api_key=make_settings.__name__,
        )

        assert make_settings.__name__ not in repr(settings)
        assert make_settings.__name__ not in str(settings)

    def test_secret_values_readable_only_explicitly(self):
        settings = make_settings(
            evalyx_secret_key=make_settings.__name__,  # placeholder secret
            openrouter_api_key=make_settings.__name__,
        )

        assert settings.evalyx_secret_key.get_secret_value() == make_settings.__name__
        assert settings.openrouter_api_key.get_secret_value() == make_settings.__name__
