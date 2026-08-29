"""Application configuration for Evalyx.

Configuration is loaded from environment variables (optionally via a local
`.env` file) and validated with pydantic-settings.

Rules:
- Secrets are stored as ``SecretStr`` so they never appear in logs or reprs.
- Development defaults are provided only for non-secret, local-only values.
- Components receive a ``Settings`` instance explicitly (dependency
  injection); :func:`get_settings` is the single cached accessor used for
  application startup.
"""

from functools import lru_cache
from typing import Literal

from typing import Literal

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["development", "test", "production"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

#: The free OpenRouter models used by Evalyx. Paid models are not allowed.
DEFAULT_AGENT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
DEFAULT_JUDGE_MODEL = "minimax/minimax-m3:free"

_INSECURE_SECRET_PLACEHOLDERS = frozenset({"", "change-me"})


class Settings(BaseSettings):
    """Typed, validated application settings.

    Values are read from environment variables first, then from a local
    `.env` file if present. Secrets must never be hard-coded anywhere.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application
    app_env: Environment = "development"
    log_level: LogLevel = "INFO"

    # LLM provider selection (azure_openai is a future provider)
    llm_provider: Literal["openrouter", "ollama"] = "openrouter"

    # Database (local Docker PostgreSQL: host 5433 -> container 5432)
    database_url: str = "postgresql+asyncpg://evalyx:evalyx@localhost:5433/evalyx"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Security
    evalyx_secret_key: SecretStr = SecretStr("")

    # OpenRouter
    openrouter_api_key: SecretStr = SecretStr("")
    evalyx_agent_model: str = DEFAULT_AGENT_MODEL
    evalyx_judge_model: str = DEFAULT_JUDGE_MODEL

    # Optional local Ollama
    ollama_base_url: str = "http://localhost:11434"

    # Background worker (Phase 7). Broker/backend URLs derive from REDIS_URL —
    # see celery_broker_url / celery_result_backend below. Concurrency stays
    # conservative: free OpenRouter models are rate-limited.
    worker_concurrency: int = 2
    worker_max_retries: int = 3
    worker_retry_backoff_seconds: float = 10.0
    worker_retry_max_backoff_seconds: float = 300.0
    worker_soft_time_limit_seconds: int = 3600
    worker_hard_time_limit_seconds: int = 3900
    worker_result_ttl_seconds: int = 86400

    @model_validator(mode="after")
    def _validate_worker_settings(self) -> Settings:
        if self.worker_hard_time_limit_seconds <= self.worker_soft_time_limit_seconds:
            raise ValueError(
                "WORKER_HARD_TIME_LIMIT_SECONDS must be greater than "
                "WORKER_SOFT_TIME_LIMIT_SECONDS."
            )
        if self.worker_max_retries < 0:
            raise ValueError("WORKER_MAX_RETRIES must be >= 0.")
        if self.worker_retry_backoff_seconds <= 0 or self.worker_retry_max_backoff_seconds <= 0:
            raise ValueError("Worker retry backoff values must be positive.")
        if self.worker_retry_max_backoff_seconds < self.worker_retry_backoff_seconds:
            raise ValueError(
                "WORKER_RETRY_MAX_BACKOFF_SECONDS must be >= "
                "WORKER_RETRY_BACKOFF_SECONDS."
            )
        return self

    @model_validator(mode="after")
    def _validate_secrets(self) -> "Settings":
        """Fail fast with clear errors on unsafe secret configuration."""
        if self.evalyx_secret_key.get_secret_value().strip() == "":
            raise ValueError(
                "EVALYX_SECRET_KEY must be set. Generate one with: "
                "python -c \"import secrets; print(secrets.token_urlsafe(32))\""
            )
        if (
            self.app_env == "production"
            and self.evalyx_secret_key.get_secret_value() in _INSECURE_SECRET_PLACEHOLDERS
        ):
            raise ValueError(
                "EVALYX_SECRET_KEY must be replaced with a real generated secret "
                "when APP_ENV=production."
            )
        return self

    @property
    def is_development(self) -> bool:
        return self.app_env == "development"

    @property
    def celery_broker_url(self) -> str:
        """Celery broker URL, derived from REDIS_URL (no duplicate setting)."""
        return self.redis_url

    @property
    def celery_result_backend(self) -> str:
        """Celery result backend URL, derived from REDIS_URL.

        Only operational task state (PENDING/STARTED/RETRY/...) lives here;
        PostgreSQL remains the source of truth for evaluation results.
        """
        return self.redis_url

    @property
    def worker_visibility_timeout_seconds(self) -> int:
        """Redis visibility timeout for Celery messages.

        Must stay well above the hard task time limit, or Redis would
        redeliver a message while its task is still executing.
        """
        return max(3600, self.worker_hard_time_limit_seconds * 2)


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings.

    Cached so startup wiring is consistent; components should still accept a
    ``Settings`` argument so they remain testable and injectable.
    """
    return Settings()
