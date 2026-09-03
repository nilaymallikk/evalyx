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

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from evalyx.core.encryption import decode_encryption_key

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
        env_file=(".env", ".env.local"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application
    app_env: Environment = "development"
    log_level: LogLevel = "INFO"
    # Observability: requests slower than this produce a structured
    # http_request_slow warning (milliseconds). Purely log-level — no
    # external monitoring dependency.
    slow_request_threshold_ms: int = Field(default=1000, ge=1)

    # API server (Phase 17 production deployment). Host/port are read by the
    # container entrypoint; workers=1 default keeps the in-memory rate
    # limiter correct — raise only behind an external shared limiter.
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, ge=1, le=65535)
    api_workers: int = Field(default=1, ge=1, le=32)

    # CORS (Phase 17). Empty = no CORS headers (default; CLI/TUI/API
    # clients do not need CORS). When set, must be explicit origins —
    # never "*".
    cors_allowed_origins: str = ""

    # Metrics (Phase 17). The /metrics endpoint requires authentication
    # and is never public; it exposes the in-process operational registry.
    metrics_enabled: bool = True

    # Rate limiting (Phase 18). Redis-backed, atomic fixed-window counters
    # shared across all API replicas — the Phase 17 process-local limiter
    # is gone. Keys are bounded (bucket × sanitized identifier × window)
    # with a 60 s TTL. When Redis is unavailable the configured policy
    # applies loudly (warning log + metric) — never a silent per-process
    # fallback.
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = Field(default=120, ge=1, le=100000)
    rate_limit_eval_per_minute: int = Field(default=20, ge=1, le=100000)
    rate_limit_test_per_minute: int = Field(default=10, ge=1, le=100000)
    rate_limit_redis_prefix: str = "evalyx:rl"
    rate_limit_on_redis_error: Literal["allow", "deny"] = "allow"

    # Organization quotas (Phase 18). Server-side admission enforced in
    # PostgreSQL under a per-organization row lock (race-safe); independent
    # from billing/subscriptions (there are none). Per-organization
    # overrides live in the organization_quota_overrides table.
    quota_enabled: bool = True
    quota_max_applications: int = Field(default=50, ge=1, le=100000)
    quota_max_datasets: int = Field(default=50, ge=1, le=100000)
    quota_max_cases_per_dataset_version: int = Field(default=5000, ge=1, le=100000)
    quota_max_evaluations_per_day: int = Field(default=100, ge=1, le=100000)
    quota_max_connection_tests_per_day: int = Field(default=200, ge=1, le=100000)
    quota_max_concurrent_evaluations: int = Field(default=5, ge=1, le=1000)
    #: Active runs older than this stop counting toward the concurrency
    #: quota (worker-loss safety valve; must exceed the worker hard limit).
    quota_stale_run_seconds: int = Field(default=7200, ge=600, le=86400)

    # Audit log (Phase 18). Durable tenant-scoped security/resource events.
    audit_enabled: bool = True
    audit_retention_days: int = Field(default=90, ge=1, le=3650)

    # Evaluation concurrency / abuse bounds (Phase 17; Phase 18 adds quotas).
    max_cases_per_evaluation: int = Field(default=500, ge=1, le=100000)
    max_request_body_bytes: int = Field(default=1_048_576, ge=1024)

    # PostgreSQL connection pooling (Phase 17 production deployment).
    db_pool_size: int = Field(default=5, ge=1, le=100)
    db_max_overflow: int = Field(default=10, ge=0, le=200)
    db_pool_timeout_seconds: int = Field(default=30, ge=1, le=300)

    # LLM provider selection (azure_openai is a future provider)
    llm_provider: Literal["openrouter", "ollama"] = "openrouter"

    # Database (local Docker PostgreSQL: host 5433 -> container 5432)
    database_url: str = "postgresql+asyncpg://evalyx:evalyx@localhost:5433/evalyx"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Application-under-test targets (reference demo: MLGPT RAG chatbot on
    # its documented default port). Server-side only — never sent to clients.
    mlgpt_base_url: str = "http://127.0.0.1:8001"

    # Clerk authentication (Phase 14). Clerk owns identity + organizations;
    # Evalyx owns the domain data and tenant scoping. The secret key is a
    # SecretStr (never logged/repr'd); the JWKS URL enables local public-key
    # verification of session tokens instead of Clerk API round-trips.
    # When clerk_jwks_url is empty, authentication is disabled (local dev
    # without Clerk) — auth_required below makes that explicit.
    clerk_secret_key: SecretStr = SecretStr("")
    clerk_jwks_url: str = ""
    clerk_authorized_parties: str = ""

    #: Master switch for API authentication. True in production; may be
    #: disabled in local development when Clerk is not configured.
    auth_required: bool = True

    # Application credential encryption (Phase 15). urlsafe base64 of a
    # 32-byte AES-GCM key (python -c "import secrets, os; \
    # print(secrets.token_urlsafe(32))"). Required in production; optional
    # in development until a secret is actually stored.
    evalyx_encryption_key: SecretStr = SecretStr("")
    # Previous encryption keys (Phase 18 rotation). Comma-separated urlsafe
    # base64 32-byte keys that remain able to *decrypt* (never encrypt).
    # During rotation the old key lives here while stored credentials are
    # re-encrypted to the current key; afterwards remove it.
    evalyx_previous_encryption_keys: SecretStr = SecretStr("")

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
    def _validate_secrets(self) -> Settings:
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

    @model_validator(mode="after")
    def _validate_production_safety(self) -> Settings:
        """Production may never run with authentication or encryption off."""
        if self.app_env == "production" and not self.auth_required:
            raise ValueError(
                "AUTH_REQUIRED cannot be disabled when APP_ENV=production. "
                "Every deployment must authenticate requests."
            )
        if self.app_env == "production" and self.evalyx_encryption_key.get_secret_value().strip() == "":
            raise ValueError(
                "EVALYX_ENCRYPTION_KEY must be set when APP_ENV=production "
                "(application credentials are encrypted at rest)."
            )
        return self

    @model_validator(mode="after")
    def _validate_encryption_key(self) -> Settings:
        """Fail fast on a malformed encryption key (never echo the value)."""
        key_value = self.evalyx_encryption_key.get_secret_value().strip()
        if key_value == "":
            return self  # optional outside production (see _validate_production_safety)
        try:
            decode_encryption_key(key_value)
        except ValueError:
            raise ValueError(
                "EVALYX_ENCRYPTION_KEY must be a urlsafe base64-encoded "
                "32-byte key. Generate one with: python -c "
                "\"import secrets; print(secrets.token_urlsafe(32))\""
            ) from None
        return self

    @model_validator(mode="after")
    def _validate_previous_encryption_keys(self) -> Settings:
        """Fail fast on malformed rotation keys (never echo values)."""
        for raw in self.previous_encryption_key_values:
            try:
                decode_encryption_key(raw)
            except ValueError:
                raise ValueError(
                    "EVALYX_PREVIOUS_ENCRYPTION_KEYS must be comma-separated "
                    "urlsafe base64-encoded 32-byte keys."
                ) from None
        return self

    @model_validator(mode="after")
    def _validate_clerk_settings(self) -> Settings:
        """Clerk must be fully configured when authentication is required."""
        if self.auth_required and self.clerk_jwks_url.strip() == "":
            raise ValueError(
                "Clerk configuration missing: AUTH_REQUIRED=1 requires "
                "CLERK_JWKS_URL (Clerk instance's .well-known/jwks.json URL). "
                "To run without Clerk locally, set AUTH_REQUIRED=0."
            )
        if self.auth_required and self.clerk_secret_key.get_secret_value().strip() == "":
            raise ValueError(
                "Clerk configuration missing: AUTH_REQUIRED=1 requires "
                "CLERK_SECRET_KEY."
            )
        return self

    @model_validator(mode="after")
    def _validate_production_config(self) -> Settings:
        """Production CORS sanity (never echo secrets)."""
        origins = [o.strip() for o in self.cors_allowed_origins.split(",") if o.strip()]
        if "*" in origins:
            raise ValueError(
                "CORS wildcard '*' is not allowed; configure explicit origins "
                "or leave CORS_ALLOWED_ORIGINS empty."
            )
        return self

    @property
    def cors_origins(self) -> list[str]:
        """Explicit CORS origins (empty = CORS disabled)."""
        return [o.strip() for o in self.cors_allowed_origins.split(",") if o.strip()]

    @property
    def previous_encryption_key_values(self) -> list[str]:
        """Previous encryption keys (decrypt-only rotation history)."""
        raw = self.evalyx_previous_encryption_keys.get_secret_value()
        return [part.strip() for part in raw.split(",") if part.strip()]

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
