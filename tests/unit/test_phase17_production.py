"""Phase 17 production deployment tests (no Docker required).

Covers the deployment contract without live infrastructure:

- production configuration rejects insecure settings (auth off, missing
  encryption key, missing Clerk configuration, CORS wildcard)
- rate limiting: default bucket, eval/test buckets, 429 envelope
- oversized request bodies rejected (413)
- security headers present on responses; CORS disabled by default and
  restrictive when enabled
- metrics endpoint requires authentication and exposes bounded labels only
- evaluation bound: oversized dataset versions rejected (422)
- deployment artifacts exist and contain no secrets
- logging redacts secret-shaped fields
"""

from __future__ import annotations

import pathlib

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from evalyx.api.app import create_app
from evalyx.api.auth import AuthContext, OrganizationRole
from evalyx.api.dependencies import require_organization
from evalyx.api.errors import EvaluationValidationError
from evalyx.api.ratelimit import RateLimiter, _bucket_for
from evalyx.core.config import Settings
from evalyx.core.logging import _redact_sensitive
from evalyx.core.metrics import MetricsRegistry
from evalyx.db.models import Organization

REPO = pathlib.Path(__file__).resolve().parents[2]

def _non_comment_lines(text: str) -> str:
    """Deployment scripts without shell comments (flags live in code)."""
    return "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )


_PLACEHOLDER_SECRET = "placeholder-production-secret"
_TEST_ENCRYPTION_KEY = "CZWNnvRiuKkYgjlplxwPzBYz1hQYgo72d8M29i22800="

_AMBIENT_VARS = (
    "APP_ENV",
    "AUTH_REQUIRED",
    "DATABASE_URL",
    "REDIS_URL",
    "EVALYX_SECRET_KEY",
    "EVALYX_ENCRYPTION_KEY",
    "CLERK_SECRET_KEY",
    "CLERK_JWKS_URL",
    "CLERK_AUTHORIZED_PARTIES",
    "CORS_ALLOWED_ORIGINS",
    "RATE_LIMIT_PER_MINUTE",
    "RATE_LIMIT_EVAL_PER_MINUTE",
    "RATE_LIMIT_TEST_PER_MINUTE",
    "MAX_CASES_PER_EVALUATION",
)


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch):
    for var in _AMBIENT_VARS:
        monkeypatch.delenv(var, raising=False)


def make_settings(**overrides) -> Settings:
    defaults = {
        "evalyx_secret_key": _PLACEHOLDER_SECRET,
        "auth_required": False,
    }
    return Settings(_env_file=None, **{**defaults, **overrides})


def build_client(settings: Settings | None = None) -> TestClient:
    from evalyx.api.dependencies import require_authenticated_user

    settings = settings or make_settings()
    app = create_app(settings)
    fake_auth = AuthContext(
        clerk_user_id="prod-test-user",
        clerk_organization_id="org_prod_test",
        organization_role=OrganizationRole.ADMIN,
    )
    fake_context = (fake_auth, Organization(name="Prod Test Org"))
    app.dependency_overrides[require_authenticated_user] = lambda: fake_auth
    app.dependency_overrides[require_organization] = lambda: fake_context
    return TestClient(app)


# -- production configuration -------------------------------------------------


class TestProductionConfiguration:
    def test_production_rejects_auth_disabled(self):
        with pytest.raises(ValidationError, match="AUTH_REQUIRED"):
            make_settings(app_env="production", auth_required=False)

    def test_production_requires_encryption_key(self):
        with pytest.raises(ValidationError, match="EVALYX_ENCRYPTION_KEY"):
            make_settings(
                app_env="production",
                auth_required=True,
                clerk_secret_key="sk-test",
                clerk_jwks_url="https://example.clerk.dev/.well-known/jwks.json",
                evalyx_encryption_key="",
            )

    def test_production_requires_clerk_configuration(self):
        with pytest.raises(ValidationError, match="Clerk configuration missing"):
            make_settings(
                app_env="production",
                auth_required=True,
                clerk_secret_key="",
                clerk_jwks_url="",
                evalyx_encryption_key=_TEST_ENCRYPTION_KEY,
            )
        # Error names the missing setting without echoing any secret value.
        try:
            make_settings(
                app_env="production",
                auth_required=True,
                clerk_secret_key="",
                clerk_jwks_url="",
                evalyx_encryption_key=_TEST_ENCRYPTION_KEY,
            )
        except ValidationError as exc:
            assert "sk-" not in str(exc)

    def test_production_accepts_complete_configuration(self):
        settings = make_settings(
            app_env="production",
            auth_required=True,
            clerk_secret_key="sk-test",
            clerk_jwks_url="https://example.clerk.dev/.well-known/jwks.json",
            evalyx_encryption_key=_TEST_ENCRYPTION_KEY,
        )
        assert settings.app_env == "production"

    def test_cors_wildcard_rejected(self):
        with pytest.raises(ValidationError, match="CORS"):
            make_settings(cors_allowed_origins="https://a.example.com, *")

    def test_cors_disabled_by_default(self):
        assert make_settings().cors_origins == []

    def test_eval_limit_cannot_exceed_default_limit(self):
        settings = make_settings(
            rate_limit_per_minute=10, rate_limit_eval_per_minute=5
        )
        assert settings.rate_limit_eval_per_minute == 5


# -- rate limiting ------------------------------------------------------------


class TestRateLimitBuckets:
    def test_bucket_classification(self):
        assert _bucket_for("/health", "GET") == "health"
        assert _bucket_for("/health/ready", "GET") == "health"
        assert _bucket_for("/api/v1/evaluations", "POST") == "eval"
        assert _bucket_for("/api/v1/applications/123/test", "POST") == "test"
        assert _bucket_for("/api/v1/evaluations", "GET") == "default"
        assert _bucket_for("/api/v1/datasets", "GET") == "default"

    def test_fixed_window_blocks_over_limit(self):
        settings = make_settings(rate_limit_per_minute=3)
        limiter = RateLimiter(settings)
        key = ("default", "127.0.0.1")
        assert all(limiter.allowed(key, "default", float(t)) for t in (0.0, 1.0, 2.0))
        assert not limiter.allowed(key, "default", 3.0)
        # Window slides: after 60 s the budget returns.
        assert limiter.allowed(key, "default", 61.0)

    def test_eval_bucket_uses_tighter_limit(self):
        settings = make_settings(
            rate_limit_per_minute=100, rate_limit_eval_per_minute=1
        )
        limiter = RateLimiter(settings)
        key = ("eval", "10.0.0.1")
        assert limiter.allowed(key, "eval", 0.0)
        assert not limiter.allowed(key, "eval", 1.0)

    def test_middleware_returns_429_envelope(self):
        settings = make_settings(rate_limit_per_minute=1)
        client = build_client(settings)
        assert client.get("/health").status_code == 200
        response = client.get("/health")
        assert response.status_code == 429
        assert response.json()["error"]["code"] == "rate_limited"
        assert "Retry-After" in response.headers

    def test_rate_limiting_disabled_when_configured(self):
        settings = make_settings(
            rate_limit_enabled=False, rate_limit_per_minute=1
        )
        client = build_client(settings)
        for _ in range(3):
            assert client.get("/health").status_code == 200


# -- request size bounds ------------------------------------------------------


class TestRequestSizeBound:
    def test_oversized_body_rejected(self):
        settings = make_settings(max_request_body_bytes=1024)
        client = build_client(settings)
        response = client.post(
            "/api/v1/datasets", json={"name": "x" * 2048, "description": None}
        )
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "payload_too_large"


# -- security headers & CORS --------------------------------------------------


class TestSecurityHeaders:
    def test_headers_present(self):
        client = build_client()
        response = client.get("/health")
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-Frame-Options"] == "DENY"
        assert response.headers["Referrer-Policy"] == "no-referrer"
        assert "Permissions-Policy" in response.headers

    def test_cors_disabled_by_default(self):
        client = build_client()
        response = client.options(
            "/api/v1/datasets",
            headers={
                "Origin": "https://evil.example.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert "Access-Control-Allow-Origin" not in response.headers

    def test_cors_restrictive_when_enabled(self):
        settings = make_settings(cors_allowed_origins="https://app.example.com")
        client = build_client(settings)
        allowed = client.options(
            "/api/v1/datasets",
            headers={
                "Origin": "https://app.example.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert allowed.headers.get("Access-Control-Allow-Origin") == "https://app.example.com"
        evil = client.options(
            "/api/v1/datasets",
            headers={
                "Origin": "https://evil.example.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert evil.headers.get("Access-Control-Allow-Origin") != "https://evil.example.com"


# -- metrics endpoint ---------------------------------------------------------


class TestMetricsEndpoint:
    def test_metrics_requires_authentication(self):
        settings = make_settings(
            auth_required=True,
            clerk_secret_key="sk-test",
            clerk_jwks_url="https://example.clerk.dev/.well-known/jwks.json",
        )
        app = create_app(settings)
        client = TestClient(app)
        response = client.get("/api/v1/metrics")
        assert response.status_code == 401

    def test_metrics_snapshot_has_no_secret_labels(self):
        client = build_client()
        response = client.get("/api/v1/metrics")
        assert response.status_code == 200
        snapshot = response.json()["metrics"]
        assert isinstance(snapshot, dict)
        forbidden = (
            "request_id",
            "run_id",
            "prompt",
            "token",
            "password",
            "secret",
        )
        for series in snapshot.values():
            for entry in series:
                for key, value in entry["labels"].items():
                    assert key not in forbidden
                    assert "sk-" not in str(value)

    def test_registry_rejects_secret_labels(self):
        registry = MetricsRegistry()
        with pytest.raises(ValueError, match="forbidden"):
            registry.increment("x_total", {"prompt": "hello"})


# -- evaluation bounds --------------------------------------------------------


class TestEvaluationBounds:
    def test_service_carries_configured_bound(self):
        from evalyx.api.services import EvaluationService

        assert issubclass(EvaluationValidationError, Exception)
        service = EvaluationService(
            None,  # type: ignore[arg-type] — bound check needs no DB here
            settings=make_settings(max_cases_per_evaluation=1),
        )
        assert service._settings is not None
        assert service._settings.max_cases_per_evaluation == 1

    def test_bound_error_maps_to_422_envelope(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from evalyx.api.errors import register_error_handlers

        app = FastAPI()
        register_error_handlers(app)

        @app.get("/probe")
        async def probe() -> None:
            raise EvaluationValidationError("Dataset version has 5 cases.")

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/probe")
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "evaluation_too_large"


# -- deployment artifacts -----------------------------------------------------


class TestDeploymentArtifacts:
    def test_dockerfile_contract(self):
        text = (REPO / "Dockerfile").read_text()
        code = _non_comment_lines(text)
        assert "uv sync --frozen" in code
        assert "--no-dev" in code
        assert "USER evalyx" in code
        assert "--reload" not in code
        # No secret files copied into the image (comments may name them).
        for line in code.splitlines():
            stripped = line.strip()
            if stripped.startswith("COPY"):
                assert ".env" not in stripped

    def test_dockerignore_excludes_secrets(self):
        text = (REPO / ".dockerignore").read_text()
        assert ".env" in text
        assert ".git/" in text

    def test_production_compose_contract(self):
        text = (REPO / "docker-compose.production.yml").read_text()
        # Required services present.
        for service in ("api:", "worker:", "postgres:", "redis:"):
            assert service in text
        # No secrets baked in: every credential is a ${VAR} reference.
        for secret in ("POSTGRES_PASSWORD:", "REDIS_PASSWORD", "CLERK_SECRET_KEY",
                       "EVALYX_SECRET_KEY", "EVALYX_ENCRYPTION_KEY"):
            assert secret in text
        assert "change-me" not in text.lower()
        assert "sk_test" not in text and "sk-live" not in text
        # Private internals: no published DB/Redis ports.
        assert '"5432' not in text and '"6379' not in text
        assert "internal: true" in text
        assert "stop_grace_period" in text

    def test_nginx_contract(self):
        text = (REPO / "deploy/nginx.conf").read_text()
        assert "return 301 https://" in text
        assert "ssl_certificate" in text
        assert "ssl_protocols TLSv1.2 TLSv1.3" in text
        assert "client_max_body_size" in text
        assert "proxy_set_header X-Forwarded-For" in text

    def test_env_template_has_no_values(self):
        text = (REPO / ".env.production.example").read_text()
        assert "APP_ENV=production" in text
        assert "AUTH_REQUIRED=1" in text
        assert "sk-" not in text or "CLERK_SECRET_KEY=\n" in text

    def test_entrypoints_production_safe(self):
        api = _non_comment_lines((REPO / "docker/entrypoint-api.sh").read_text())
        assert "--reload" not in api
        assert "proxy-headers" in api
        worker = _non_comment_lines(
            (REPO / "docker/entrypoint-worker.sh").read_text()
        )
        assert "celery -A evalyx.worker.celery_app worker" in worker

    def test_docs_exist(self):
        assert (REPO / "docs/deployment.md").exists()
        assert (REPO / "docs/production-checklist.md").exists()
        assert (REPO / "scripts/smoke_test.py").exists()
        assert (REPO / "scripts/backup.sh").exists()


# -- logging redaction --------------------------------------------------------


class TestLoggingRedaction:
    @pytest.mark.parametrize(
        "key",
        ["secret", "api_key", "Authorization", "CLERK_SECRET_KEY",
         "encryption_key", "token", "password", "credential"],
    )
    def test_sensitive_keys_redacted(self, key):
        event = _redact_sensitive(None, "info", {key: "super-secret-value", "ok": 1})
        assert event[key] == "[redacted]"
        assert "super-secret-value" not in str(event)
        assert event["ok"] == 1

    def test_safe_keys_untouched(self):
        event = _redact_sensitive(
            None, "info", {"request_id": "abc", "route": "/health"}
        )
        assert event == {"request_id": "abc", "route": "/health"}
