"""Unit tests for the CLI: client, commands, exit codes.

No live Evalyx server — the HTTP layer is exercised against a mocked
transport (``httpx.MockTransport``-style monkeypatching) and the Typer test
runner. Evalyx needs no login: the client sends plain requests.
"""

import json

import pytest
from typer.testing import CliRunner

from evalyx.cli import errors as cli_errors
from evalyx.cli.client import EvalyxClient
from evalyx.cli.config import Config, load_config
from evalyx.cli.main import app

runner = CliRunner()


# -- configuration -----------------------------------------------------------------


def test_config_precedence_flag_beats_env_and_file(tmp_path, monkeypatch):
    config_file = tmp_path / "config.toml"
    config_file.write_text('api_url = "http://file:1"\n')
    monkeypatch.setenv("EVALYX_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("EVALYX_API_URL", "http://env:2")
    monkeypatch.delenv("EVALYX_ORG", raising=False)
    loaded = load_config(api_url="http://flag:3")
    assert loaded.api_url == "http://flag:3"

    from evalyx.cli import config as cli_config

    monkeypatch.setattr(cli_config, "config_file", lambda: config_file)
    loaded = load_config()
    assert loaded.api_url == "http://env:2"


def test_config_default_api_url(monkeypatch, tmp_path):
    monkeypatch.setenv("EVALYX_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("EVALYX_API_URL", raising=False)
    loaded = load_config()
    assert loaded.api_url == "http://127.0.0.1:8000"


# -- errors & exit codes -------------------------------------------------------------


def test_exit_codes_are_stable():
    assert cli_errors.EXIT_OK == 0
    assert cli_errors.EXIT_ERROR == 1
    assert cli_errors.EXIT_USAGE == 2
    assert cli_errors.EXIT_AUTH == 3
    assert cli_errors.EXIT_FORBIDDEN == 4
    assert cli_errors.EXIT_NOT_FOUND == 5
    assert cli_errors.EXIT_CONNECTION == 6
    assert cli_errors.EXIT_QUALITY_FAILURES == 7
    assert cli_errors.EXIT_EXECUTION_ERRORS == 8


def test_normalize_http_error_maps_statuses():
    import httpx

    def _response(status: int, body: dict) -> httpx.Response:
        request = httpx.Request("GET", "http://api/x")
        return httpx.Response(status, json=body, request=request)

    def _normalize(status: int, body: dict):
        exc = httpx.HTTPStatusError(
            "x",
            request=httpx.Request("GET", "http://api/x"),
            response=_response(status, body),
        )
        return cli_errors.normalize_http_error(exc, "Application")

    assert isinstance(
        _normalize(401, {"error": {"code": "x", "message": "y"}}),
        cli_errors.AuthenticationError,
    )
    assert isinstance(
        _normalize(403, {"error": {"code": "x", "message": "y"}}),
        cli_errors.AuthorizationError,
    )
    assert isinstance(_normalize(404, {}), cli_errors.NotFoundError)
    assert isinstance(
        _normalize(422, {"error": {"code": "validation_error", "message": "bad"}}),
        cli_errors.ValidationError,
    )
    assert isinstance(_normalize(500, {}), cli_errors.APIError)


# -- client transport ---------------------------------------------------------------


class _MockAPI:
    """Minimal scripted Evalyx API for client tests (no login required)."""

    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls.append((request.method, request.url.path))
        path = request.url.path
        if path == "/api/v1/me":
            return httpx.Response(
                200,
                json={
                    "user_id": "local",
                    "active_organization": {
                        "organization_id": "local",
                        "name": "local",
                    },
                    "organizations": [{"organization_id": "local", "name": "local"}],
                },
            )
        if path == "/api/v1/applications" and request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "items": [
                        {
                            "id": "11111111-1111-1111-1111-111111111111",
                            "name": "a",
                            "connection_type": "http",
                            "secret_configured": True,
                        }
                    ],
                    "total": 1,
                    "limit": 50,
                    "offset": 0,
                },
            )
        if path == "/api/v1/applications/nope" and request.method == "GET":
            return httpx.Response(
                404, json={"error": {"code": "not_found", "message": "missing"}}
            )
        if path == "/api/v1/applications/err" and request.method == "GET":
            return httpx.Response(
                500, json={"error": {"code": "internal_error", "message": "boom"}}
            )
        return httpx.Response(
            404, json={"error": {"code": "not_found", "message": "missing"}}
        )


import httpx


def _client_with_mock(mock: _MockAPI) -> EvalyxClient:
    client = EvalyxClient(config=Config(api_url="http://mock", timeout=5))
    handler = mock.handler
    # Patch the module-level request function used by the client.
    import evalyx.cli.client as client_module

    def _request(method, url, **kw):
        kw.setdefault("timeout", client.config.timeout)
        request = httpx.Request(
            method,
            url,
            json=kw.get("json"),
            params=kw.get("params"),
            headers=kw.get("headers"),
        )
        return handler(request)

    client_module.httpx.request = _request  # type: ignore[assignment]
    return client


@pytest.fixture(autouse=True)
def _restore_httpx_request():
    original = httpx.request
    yield
    httpx.request = original


def test_client_sends_no_credentials():
    """Local-first: plain requests, no Authorization header, no org header."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update({k.lower(): v for k, v in request.headers.items()})
        return httpx.Response(200, json={"user_id": "local"})

    import evalyx.cli.client as client_module

    orig = client_module.httpx.request
    client_module.httpx.request = lambda method, url, **kw: handler(
        httpx.Request(method, url, json=kw.get("json"), headers=kw.get("headers"))
    )
    try:
        me = EvalyxClient(config=Config(api_url="http://mock")).me()
    finally:
        client_module.httpx.request = orig
    assert me["user_id"] == "local"
    assert "authorization" not in seen
    assert "x-dev-organization-id" not in seen


def test_client_me_returns_local_workspace():
    mock = _MockAPI()
    client = _client_with_mock(mock)
    me = client.me()
    assert me["user_id"] == "local"
    assert me["active_organization"]["organization_id"] == "local"
    assert ("GET", "/api/v1/me") in mock.calls


def test_client_404_raises_not_found():
    client = _client_with_mock(_MockAPI())
    with pytest.raises(cli_errors.NotFoundError):
        client.applications_get("nope")


def test_client_500_raises_api_error():
    client = _client_with_mock(_MockAPI())
    with pytest.raises(cli_errors.APIError):
        client.applications_get("err")


def test_client_connection_error_is_normalized(monkeypatch):
    client = EvalyxClient(config=Config(api_url="http://unreachable:1"))

    import httpx as httpx_mod

    def _raise(*args, **kwargs):
        raise httpx_mod.ConnectError("refused")

    import evalyx.cli.client as client_module

    monkeypatch.setattr(client_module.httpx, "request", _raise)
    with pytest.raises(cli_errors.APIConnectionError):
        client.applications_list()


def _patch_request(monkeypatch, handler) -> None:
    """Route every client request through the given httpx handler."""
    import evalyx.cli.client as client_module

    def _request(method, url, **kw):
        request = httpx.Request(
            method,
            url,
            json=kw.get("json"),
            params=kw.get("params"),
            headers=kw.get("headers"),
        )
        return handler(request)

    monkeypatch.setattr(client_module.httpx, "request", _request)


def test_client_get_retries_then_succeeds(monkeypatch):
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] < 3:
            return httpx.Response(
                503, json={"error": {"code": "unavailable", "message": "x"}}
            )
        return httpx.Response(200, json={"ok": True})

    _patch_request(monkeypatch, handler)
    client = EvalyxClient(config=Config(api_url="http://mock"))
    assert client.request("GET", "/api/v1/applications") == {"ok": True}
    assert attempts["n"] == 3


def test_client_writes_are_never_retried(monkeypatch):
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(503, json={"error": {"code": "x", "message": "y"}})

    _patch_request(monkeypatch, handler)
    client = EvalyxClient(config=Config(api_url="http://mock"))
    with pytest.raises(cli_errors.APIError):
        client.applications_create("n", "http")
    assert attempts["n"] == 1


# -- CLI commands ----------------------------------------------------------------------


def test_help_works():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "evalyx" in result.output.lower()


def test_version_flag():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.startswith("evalyx ")


def test_whoami_shows_local_workspace(monkeypatch):
    mock = _MockAPI()
    _patch_request(monkeypatch, mock.handler)
    result = runner.invoke(app, ["whoami"])
    assert result.exit_code == 0, result.output
    assert "local" in result.output


def test_app_list_json_output(monkeypatch):
    mock = _MockAPI()
    _patch_request(monkeypatch, mock.handler)
    result = runner.invoke(app, ["--json", "app", "list"], input="")
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["total"] == 1
    assert data["items"][0]["secret_configured"] is True


def test_app_list_human_output(monkeypatch):
    mock = _MockAPI()
    _patch_request(monkeypatch, mock.handler)
    result = runner.invoke(app, ["app", "list"])
    assert result.exit_code == 0, result.output
    assert "NAME" in result.output
    assert "11111111" in result.output


def test_json_output_is_pure_json(monkeypatch):
    mock = _MockAPI()
    _patch_request(monkeypatch, mock.handler)
    result = runner.invoke(app, ["--json", "app", "list"])
    # Strict parse of the entire stdout: no banners or progress lines.
    json.loads(result.output)


def test_eval_run_submit_json(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/evaluations"
        body = json.loads(request.content)
        assert body["application_id"] == "app-1"
        return httpx.Response(
            202,
            json={
                "run_id": "22222222-2222-2222-2222-222222222222",
                "status": "pending",
                "task_id": "task-1",
                "status_url": "/api/v1/evaluations/x",
            },
        )

    _patch_request(monkeypatch, handler)
    result = runner.invoke(
        app,
        [
            "--json",
            "eval",
            "run",
            "--application",
            "app-1",
            "--dataset-version",
            "dv-1",
            "--agent-model",
            "m",
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["status"] == "pending"


def test_eval_run_wait_completed_with_failures_exits_7(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/evaluations":
            return httpx.Response(
                202,
                json={
                    "run_id": "run-1",
                    "status": "pending",
                    "task_id": "t",
                    "status_url": "/s",
                },
            )
        if request.url.path == "/api/v1/evaluations/run-1":
            return httpx.Response(
                200,
                json={
                    "id": "run-1",
                    "status": "completed",
                    "counts": {
                        "total": 10,
                        "passed": 8,
                        "failed": 2,
                        "error": 0,
                        "executed": 0,
                    },
                },
            )
        return httpx.Response(
            404, json={"error": {"code": "not_found", "message": "x"}}
        )

    _patch_request(monkeypatch, handler)
    result = runner.invoke(
        app,
        [
            "eval",
            "run",
            "--application",
            "a",
            "--dataset-version",
            "d",
            "--agent-model",
            "m",
            "--wait",
        ],
    )
    assert result.exit_code == cli_errors.EXIT_QUALITY_FAILURES
    assert "Pass rate: 80.0%" in result.output


def test_reliability_command(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "total_cases": 120,
                "errored_cases": 4,
                "classified_failures": 4,
                "unclassified_execution_failures": 0,
                "retryable_failures": 2,
                "failure_breakdown": {
                    "timeout": 2,
                    "rate_limited": 1,
                    "connection_error": 1,
                },
            },
        )

    _patch_request(monkeypatch, handler)
    result = runner.invoke(app, ["reliability", "run-1"])
    assert result.exit_code == 0
    assert "timeout" in result.output
    assert "Error rate: 3.3%" in result.output


def test_regression_show_detected_exits_7(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "result": "REGRESSION_DETECTED",
                "regression_detected": True,
                "baseline_run_id": "b",
                "current_run_id": "c",
                "baseline": {"pass_rate": 94.2},
                "current": {"pass_rate": 87.1},
                "deltas": {"pass_rate_pp": -7.1},
                "threshold_violations": [{"detail": "pass rate dropped"}],
                "newly_failed_cases": [{"name": "testcase-042"}],
            },
        )

    _patch_request(monkeypatch, handler)
    result = runner.invoke(app, ["regression", "show", "cmp-1"])
    assert result.exit_code == cli_errors.EXIT_QUALITY_FAILURES
    assert "94.2" in result.output and "87.1" in result.output
    assert "testcase-042" in result.output


def test_delete_requires_confirmation_non_interactive():
    result = runner.invoke(app, ["app", "delete", "some-id"])
    assert result.exit_code == cli_errors.EXIT_USAGE


# -- quickstart ------------------------------------------------------------------------------


class _QuickstartAPI:
    """Scripted backend for the interactive quickstart flow."""

    def __init__(self) -> None:
        self.calls: list = []
        self.cases: list = []

    def applications_create(self, name, connection_type, description=None, secret=None):
        self.calls.append(("create_app", name, connection_type))
        return {"id": "aaaaaaaa-1111-1111-1111-111111111111", "name": name}

    def applications_create_version(self, app_id, version, connection=None, **kw):
        self.calls.append(("create_version", version, connection))
        return {"id": "v1", "version": version}

    def applications_test(self, app_id, prompt=None):
        self.calls.append(("test",))
        return {
            "success": True,
            "http_status": 200,
            "latency_ms": 42,
            "preview": "hello back",
        }

    def datasets_create(self, name, description=None):
        self.calls.append(("create_dataset", name))
        return {"id": "bbbbbbbb-2222-2222-2222-222222222222", "name": name}

    def datasets_create_version(self, ds_id, version, description=None):
        self.calls.append(("create_ds_version", version))
        return {"id": "cccccccc-3333-3333-3333-333333333333"}

    def datasets_add_case(self, ds_id, version, name, input, **kw):
        self.cases.append((name, input))
        return {"id": name}

    def evaluations_submit(self, app_id, dsv_id, agent_model, judge_model=None):
        self.calls.append(("submit", agent_model))
        assert agent_model == "application:demo"
        return {"run_id": "dddddddd-4444-4444-4444-444444444444", "status": "pending"}

    def evaluations_get(self, run_id):
        return {
            "id": run_id,
            "status": "completed",
            "counts": {"total": 1, "passed": 1, "failed": 0, "error": 0, "executed": 0},
        }


def _patch_quickstart(monkeypatch, fake: _QuickstartAPI) -> None:
    import evalyx.cli.main as main_module

    monkeypatch.setattr(main_module, "_is_interactive", lambda: True)

    def _bind(method: str):
        impl = getattr(fake, method)

        def _call(self, *args, **kwargs):
            return impl(*args, **kwargs)

        return _call

    for method in (
        "applications_create",
        "applications_create_version",
        "applications_test",
        "datasets_create",
        "datasets_create_version",
        "datasets_add_case",
        "evaluations_submit",
        "evaluations_get",
    ):
        monkeypatch.setattr(EvalyxClient, method, _bind(method))
    import time as time_module

    monkeypatch.setattr(time_module, "sleep", lambda s: None)


def test_quickstart_happy_path(monkeypatch):
    fake = _QuickstartAPI()
    _patch_quickstart(monkeypatch, fake)
    user_input = (
        "demo\n"  # app name
        "https://app.example.com/chat\n"  # endpoint
        "\n"  # auth mode (none)
        "\n"  # input field (question)
        "\n"  # response path (answer)
        "Say hello\n"  # question 1
        "\n"  # finish questions
    )
    result = runner.invoke(app, ["quickstart"], input=user_input)
    assert result.exit_code == 0, result.output
    assert "App registered: demo" in result.output
    assert "Connection works" in result.output
    assert "Added 1 test cases" in result.output
    assert "Passed:" in result.output
    assert ("submit", "application:demo") in fake.calls
    assert fake.cases == [("quickstart-1", {"prompt": "Say hello"})]


def test_quickstart_warns_on_localhost_and_continues(monkeypatch):
    fake = _QuickstartAPI()
    _patch_quickstart(monkeypatch, fake)
    user_input = (
        "demo\n"
        "http://localhost:8000/chat\n"  # warns (loopback), then continues
        "\n\n\n"  # defaults: auth, input field, response path
        "\n"  # no questions → defaults
    )
    result = runner.invoke(app, ["quickstart"], input=user_input)
    assert result.exit_code == 0, result.output
    assert "EVALYX_ALLOW_PRIVATE_ENDPOINTS" in result.output
    assert "Added 2 test cases" in result.output
    versions = [c for c in fake.calls if c[0] == "create_version"]
    assert versions[0][2]["endpoint"] == "http://localhost:8000/chat"


def test_quickstart_refuses_non_interactive():
    result = runner.invoke(app, ["quickstart"], input="")
    assert result.exit_code == cli_errors.EXIT_USAGE
    assert "interactive" in result.output


# -- TUI smoke ------------------------------------------------------------------------------


def test_tui_module_imports_and_binds():
    """The TUI module imports cleanly and declares the documented shortcuts."""
    from evalyx.cli.tui.app import EvalyxTUI

    binding_keys = {
        binding[0] if isinstance(binding, tuple) else binding.key
        for binding in EvalyxTUI.BINDINGS
    }
    assert {"q", "r", "escape", "a", "e", "d", "s", "g", "t"} <= binding_keys
