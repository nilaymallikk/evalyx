"""Phase 18 quota unit tests (hermetic).

Pure limit-merge logic plus the HTTP envelope. Database-backed admission
(lock, counts, races, release, staleness, overrides) is covered against
live PostgreSQL in tests/integration/test_phase18_quotas.py.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from evalyx.api.errors import QuotaExceededError, register_error_handlers
from evalyx.db.models.governance import OrganizationQuotaOverrides
from evalyx.quotas import QuotaService


def _settings(**overrides):
    from evalyx.core.config import Settings

    defaults = {"evalyx_secret_key": "placeholder", "auth_required": False}
    return Settings(_env_file=None, **{**defaults, **overrides})


def _service(**overrides):
    return QuotaService(None, _settings(**overrides))  # type: ignore[arg-type]


class TestMergeLimits:
    def test_defaults_without_overrides(self):
        limits = _service()._merge_limits(None)
        assert limits.max_applications == 50
        assert limits.max_datasets == 50
        assert limits.max_concurrent_evaluations == 5
        assert limits.max_evaluations_per_day == 100
        assert limits.max_connection_tests_per_day == 200

    def test_overrides_win_per_dimension(self):
        row = OrganizationQuotaOverrides(
            organization_id=uuid.uuid4(),
            max_applications=3,
            max_concurrent_evaluations=1,
        )
        limits = _service()._merge_limits(row)
        assert limits.max_applications == 3
        assert limits.max_concurrent_evaluations == 1
        assert limits.max_datasets == 50  # untouched dimensions keep defaults

    def test_settings_defaults_flow_through(self):
        service = _service(quota_max_applications=7)
        assert service._merge_limits(None).max_applications == 7


class TestQuotaEnvelope:
    def test_denial_maps_to_429_with_retry_after(self):
        app = FastAPI()
        register_error_handlers(app)

        @app.get("/probe")
        async def probe() -> None:
            raise QuotaExceededError("applications", "Organization application quota exceeded (2/2).")

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/probe")
        assert response.status_code == 429
        body = response.json()
        assert body["error"]["code"] == "quota_exceeded"
        assert "2/2" in body["error"]["message"]
        assert "Retry-After" in response.headers

    def test_error_carries_resource(self):
        error = QuotaExceededError("datasets", "full")
        assert error.resource == "datasets"
        assert error.code == "quota_exceeded"


class TestOverrideValidation:
    async def test_unknown_dimensions_rejected(self):
        service = _service()

        class _Session:
            async def scalar(self, *args, **kwargs):
                return None

        with pytest.raises(ValueError, match="Unknown quota dimensions"):
            await service.set_overrides(
                _Session(), uuid.uuid4(), max_planets=3  # type: ignore[arg-type]
            )

    async def test_non_positive_values_rejected(self):
        service = _service()

        class _Session:
            async def scalar(self, *args, **kwargs):
                return None

        with pytest.raises(ValueError, match="positive int"):
            await service.set_overrides(
                _Session(), uuid.uuid4(), max_applications=0  # type: ignore[arg-type]
            )
