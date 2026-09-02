"""API unit tests: routing, validation, error mapping, security (no services).

No live PostgreSQL/Redis: the app is created from test settings, handlers
that would touch infrastructure are either not invoked (request validation
fails first) or replaced with fakes via dependency overrides.
"""

import uuid
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.exc import IntegrityError

from evalyx.api.app import create_app
from evalyx.api.auth import AuthContext, OrganizationRole
from evalyx.api.dependencies import get_evaluation_service, require_organization
from evalyx.api.errors import EvaluationSubmissionError, register_error_handlers
from evalyx.core.config import Settings
from evalyx.db.models import Organization, RunStatus
from evalyx.db.repositories import DuplicateVersionError, NotFoundError

MAJOR_API_PATHS = (
    "/api/v1/applications",
    "/api/v1/applications/{application_id}",
    "/api/v1/applications/{application_id}/versions",
    "/api/v1/datasets",
    "/api/v1/datasets/{dataset_id}",
    "/api/v1/datasets/{dataset_id}/versions",
    "/api/v1/datasets/{dataset_id}/versions/{version}/cases",
    "/api/v1/evaluations",
    "/api/v1/evaluations/{run_id}",
    "/api/v1/evaluations/{run_id}/results",
    "/api/v1/evaluations/{run_id}/guardrails",
    "/api/v1/evaluations/{run_id}/regressions",
    "/api/v1/regressions",
    "/api/v1/regressions/{comparison_id}",
)


def build_client() -> TestClient:
    """Offline client: no lifespan (no connections), no DB-touching calls.

    Tenant authentication is bypassed with a fixed fake organization — these
    tests exercise validation/error mapping, not Clerk (dedicated suite:
    tests/unit/test_auth_api.py).
    """
    app = create_app(Settings(auth_required=False))
    fake_context = (
        AuthContext(
            clerk_user_id="unit-test-user",
            clerk_organization_id="org_unit_test",
            organization_role=OrganizationRole.ADMIN,
        ),
        Organization(name="Unit Test Org"),
    )
    app.dependency_overrides[require_organization] = lambda: fake_context
    return TestClient(app)


# -- application factory & OpenAPI ------------------------------------------------


def test_health_endpoints_unchanged():
    client = build_client()
    assert client.get("/health").status_code == 200
    assert client.get("/health").json() == {"status": "ok"}
    # /health/ready is exercised against live dependencies in integration tests;
    # here we only verify route registration (404 would mean it vanished).
    assert client.get("/health/ready").status_code in (200, 503)


def test_openapi_exposes_major_routes_and_schemas():
    client = build_client()
    spec = client.get("/openapi.json").json()
    paths = set(spec["paths"])
    for path in MAJOR_API_PATHS:
        assert path in paths, f"missing OpenAPI path {path}"
    schemas = set(spec["components"]["schemas"])
    for schema in (
        "ApplicationCreate",
        "ApplicationVersionCreate",
        "DatasetCreate",
        "DatasetVersionCreate",
        "TestCaseCreate",
        "EvaluationCreate",
        "EvaluationSubmissionResponse",
        "EvaluationRunSummary",
        "EvaluationCaseResultResponse",
        "GuardrailResultResponse",
        "RegressionCompareRequest",
        "RegressionReport",
    ):
        assert schema in schemas, f"missing OpenAPI schema {schema}"
    # Health endpoints stay outside the API version.
    assert "/health" in paths and "/health/ready" in paths


# -- error handling -----------------------------------------------------------------


def build_error_probe_app() -> TestClient:
    # ``raise_server_exceptions=False`` mirrors uvicorn's ServerErrorMiddleware
    # behavior for the catch-all 500 handler (TestClient re-raises by default).
    app = FastAPI()
    register_error_handlers(app)

    @app.get("/not-found")
    async def not_found():
        raise NotFoundError("Application 123 does not exist.")

    @app.get("/duplicate-version")
    async def duplicate_version():
        raise DuplicateVersionError(uuid.uuid4(), "v1")

    @app.get("/integrity")
    async def integrity():
        raise IntegrityError("stmt", {}, Exception("duplicate key"))

    @app.get("/bad-comparison")
    async def bad_comparison():
        from evalyx.evaluation.regression.service import RegressionValidationError

        raise RegressionValidationError("Runs cannot be compared.")

    @app.get("/enqueue-failed")
    async def enqueue_failed():
        raise EvaluationSubmissionError("run-1", "queue rejected the job")

    @app.get("/boom")
    async def boom():
        raise RuntimeError("secret-internal-traceback-nikto")

    return TestClient(app, raise_server_exceptions=False)


def test_error_envelope_shape():
    client = build_error_probe_app()
    body = client.get("/not-found").json()
    assert set(body) == {"error"}
    assert set(body["error"]) == {"code", "message"}


def test_domain_errors_map_to_http_statuses():
    client = build_error_probe_app()
    assert client.get("/not-found").status_code == 404
    assert client.get("/not-found").json()["error"]["code"] == "not_found"
    assert client.get("/duplicate-version").status_code == 409
    assert client.get("/duplicate-version").json()["error"]["code"] == "duplicate_version"
    assert client.get("/integrity").status_code == 409
    assert client.get("/integrity").json()["error"]["code"] == "conflict"
    assert client.get("/bad-comparison").status_code == 400
    assert client.get("/bad-comparison").json()["error"]["code"] == "invalid_comparison"
    assert client.get("/enqueue-failed").status_code == 503
    assert client.get("/enqueue-failed").json()["error"]["code"] == "evaluation_enqueue_failed"


def test_unexpected_error_returns_generic_500_without_traceback():
    client = build_error_probe_app()
    response = client.get("/boom")
    assert response.status_code == 500
    body = response.json()
    assert body["error"]["code"] == "internal_error"
    # Neither the exception text nor its type leaks to the client.
    assert "secret-internal-traceback-nikto" not in response.text
    assert "RuntimeError" not in response.text
    assert "Traceback" not in response.text


def test_request_validation_uses_error_envelope_and_never_echoes_input():
    client = build_client()
    fake_marker = "fake-" + "do-not-echo"
    response = client.post(
        "/api/v1/applications",
        # The invalid payload embeds a fake secret-shaped marker: the 422
        # body must not echo request data back.
        json={"name": "", "api_key": fake_marker},
    )
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "validation_error"
    assert fake_marker not in response.text
    assert "input" not in response.text


def test_invalid_uuid_path_parameter_returns_422():
    client = build_client()
    response = client.get("/api/v1/applications/not-a-uuid")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


# -- pagination validation -----------------------------------------------------------


def test_pagination_bounds_are_validated_before_handlers_run():
    client = build_client()
    assert client.get("/api/v1/evaluations", params={"limit": 0}).status_code == 422
    assert client.get("/api/v1/evaluations", params={"limit": 201}).status_code == 422
    assert client.get("/api/v1/evaluations", params={"limit": -5}).status_code == 422
    assert client.get("/api/v1/evaluations", params={"offset": -1}).status_code == 422
    # Boundary values accepted (validation only; the handler is not reached
    # without a database, but 422-vs-500 distinguishes validation from DB).
    limit_ok = client.get("/api/v1/evaluations", params={"limit": 200})
    assert limit_ok.status_code != 422


# -- evaluation submission (HTTP semantics, fake service) ------------------------------


class FakeEvaluationService:
    """Stands in for the real service; records the submitted payload."""

    def __init__(self, *, fail_enqueue: bool = False) -> None:
        self.submitted: list = []
        self._fail_enqueue = fail_enqueue

    async def submit(self, request, *, organization_id):
        self.submitted.append((request, organization_id))
        if self._fail_enqueue:
            raise EvaluationSubmissionError(
                str(uuid.uuid4()), "queue rejected the job"
            )
        run = SimpleNamespace(id=uuid.uuid4(), status=RunStatus.PENDING)
        return run, "task-abc-123"


VALID_EVALUATION_PAYLOAD = {
    "application_id": str(uuid.uuid4()),
    "dataset_version_id": str(uuid.uuid4()),
    "agent_model": "nvidia/nemotron-3-ultra-550b-a55b:free",
    "configuration_snapshot": {"temperature": 0.2, "max_tokens": 512},
}


def test_submit_evaluation_returns_202_with_task_identity():
    fake = FakeEvaluationService()
    client = build_client()
    client.app.dependency_overrides[get_evaluation_service] = lambda: fake

    response = client.post("/api/v1/evaluations", json=VALID_EVALUATION_PAYLOAD)

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "pending"
    assert body["task_id"] == "task-abc-123"
    assert body["status_url"] == f"/api/v1/evaluations/{body['run_id']}"
    assert uuid.UUID(body["run_id"])  # parses
    assert len(fake.submitted) == 1


def test_submit_evaluation_maps_enqueue_failure_to_503():
    fake = FakeEvaluationService(fail_enqueue=True)
    client = build_client()
    client.app.dependency_overrides[get_evaluation_service] = lambda: fake

    response = client.post("/api/v1/evaluations", json=VALID_EVALUATION_PAYLOAD)

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "evaluation_enqueue_failed"
    # The failure is the queue's, not a validation problem: the payload was
    # accepted by the service layer (which persisted and rolled back).
    assert len(fake.submitted) == 1


# -- request schema security -----------------------------------------------------------


def test_evaluation_request_strips_secret_looking_keys():
    from evalyx.api.schemas.evaluations import EvaluationCreate

    request = EvaluationCreate(
        application_id=uuid.uuid4(),
        dataset_version_id=uuid.uuid4(),
        agent_model="m",
        configuration_snapshot={
            "temperature": 0.2,
            "max_tokens": 512,
            "OPENROUTER_API_KEY": "sk-leak",
            "nested": {"auth_token": "t", "max_tokens": 512},
        },
    )
    config = request.configuration_snapshot
    assert config["temperature"] == 0.2
    assert config["max_tokens"] == 512
    assert "OPENROUTER_API_KEY" not in str(config)
    assert config["nested"] == {"max_tokens": 512}


def test_evaluation_request_rejects_implausible_known_parameters():
    import pytest
    from pydantic import ValidationError

    from evalyx.api.schemas.evaluations import EvaluationCreate

    with pytest.raises(ValidationError, match="temperature"):
        EvaluationCreate(
            application_id=uuid.uuid4(),
            dataset_version_id=uuid.uuid4(),
            agent_model="m",
            configuration_snapshot={"temperature": 9.5},
        )
    with pytest.raises(ValidationError, match="max_tokens"):
        EvaluationCreate(
            application_id=uuid.uuid4(),
            dataset_version_id=uuid.uuid4(),
            agent_model="m",
            configuration_snapshot={"max_tokens": 0},
        )


def test_application_version_request_sanitizes_configuration():
    from evalyx.api.schemas.applications import ApplicationVersionCreate

    payload = ApplicationVersionCreate(
        version="v1",
        configuration={"prompt_template": "tpl-7", "api_key": "sk-leak"},
    )
    config = payload.sanitized_configuration()
    assert config == {"prompt_template": "tpl-7"}
    assert "sk-leak" not in str(config)
