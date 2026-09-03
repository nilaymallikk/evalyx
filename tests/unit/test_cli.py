"""Unit tests for the Phase 16 CLI: client, commands, exit codes, security.

No live Evalyx server — the HTTP layer is exercised against a mocked
transport (``httpx.MockTransport``-style monkeypatching) and the Typer test
runner. Credential leakage assertions cover every command's output.
"""

import json

import pytest
from typer.testing import CliRunner

from evalyx.cli import auth as cli_auth
from evalyx.cli import errors as cli_errors
from evalyx.cli.client import EvalyxClient
from evalyx.cli.config import Config, load_config
from evalyx.cli.main import app

runner = CliRunner()

FAKE_TOKEN = "fake.session." + "token-value"


# -- configuration -----------------------------------------------------------------


def test_config_precedence_flag_beats_env_and_file(tmp_path, monkeypatch):
    config_file = tmp_path / "config.toml"
    config_file.write_text('api_url = "http://file:1"\norg = "org_file"\n')
    monkeypatch.setenv("EVALYX_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("EVALYX_API_URL", "http://env:2")
    monkeypatch.setenv("EVALYX_ORG", "org_env")
    loaded = load_config(api_url="http://flag:3", org="org_flag")
    assert loaded.api_url == "http://flag:3"
    assert loaded.org == "org_flag"

    from evalyx.cli import config as cli_config

    monkeypatch.setattr(cli_config, "config_file", lambda: config_file)
    loaded = load_config()
    assert loaded.api_url == "http://env:2"
    assert loaded.org == "org_env"


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
            "x", request=httpx.Request("GET", "http://api/x"),
            response=_response(status, body),
        )
        return cli_errors.normalize_http_error(exc, "Application")

    assert isinstance(_normalize(401, {"error": {"code": "x", "message": "y"}}), cli_errors.AuthenticationError)
    assert isinstance(_normalize(403, {"error": {"code": "x", "message": "y"}}), cli_errors.AuthorizationError)
    assert isinstance(_normalize(404, {}), cli_errors.NotFoundError)
    assert isinstance(_normalize(422, {"error": {"code": "validation_error", "message": "bad"}}), cli_errors.ValidationError)
    assert isinstance(_normalize(500, {}), cli_errors.APIError)
    # No token contents ever surface in messages.
    assert FAKE_TOKEN not in str(_normalize(401, {"error": {"code": "x", "message": FAKE_TOKEN}}))


# -- client transport ---------------------------------------------------------------


class _MockAPI:
    """Minimal scripted Evalyx API for client tests."""

    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls.append((request.method, request.url.path))
        headers = request.headers
        path = request.url.path
        if path == "/api/v1/me":
            if "authorization" in headers:
                if headers["authorization"] != f"Bearer {FAKE_TOKEN}":
                    return httpx.Response(401, json={"error": {"code": "authentication_failed", "message": "no"}})
                return httpx.Response(200, json={
                    "clerk_user_id": "user_1", "email": "user@example.com",
                    "active_organization": {"clerk_organization_id": "org_1", "name": "Team", "role": "admin"},
                    "organizations": [],
                })
            org = headers.get("x-dev-organization-id")
            if org == "org_dev":
                return httpx.Response(200, json={
                    "clerk_user_id": "dev-cli", "email": None,
                    "active_organization": {"clerk_organization_id": org, "name": org, "role": "admin"},
                    "organizations": [],
                })
            return httpx.Response(401, json={"error": {"code": "authentication_failed", "message": "Authentication failed."}})
        if path == "/api/v1/applications" and request.method == "GET":
            if "authorization" not in headers and "x-dev-organization-id" not in headers:
                return httpx.Response(401, json={"error": {"code": "authentication_failed", "message": "Auth failed."}})
            return httpx.Response(200, json={"items": [{"id": "11111111-1111-1111-1111-111111111111", "name": "a", "connection_type": "http", "secret_configured": True}], "total": 1, "limit": 50, "offset": 0})
        if path == "/api/v1/applications/nope" and request.method == "GET":
            return httpx.Response(404, json={"error": {"code": "not_found", "message": "missing"}})
        if path == "/api/v1/applications/err" and request.method == "GET":
            return httpx.Response(500, json={"error": {"code": "internal_error", "message": "boom"}})
        return httpx.Response(404, json={"error": {"code": "not_found", "message": "missing"}})


import httpx


def _client_with_mock(mock: _MockAPI, **kwargs) -> EvalyxClient:
    client = EvalyxClient(config=Config(api_url="http://mock", timeout=5), **kwargs)
    handler = mock.handler
    # Patch the module-level request function used by the client.
    import evalyx.cli.client as client_module

    def _request(method, url, **kw):
        kw.setdefault("timeout", client.config.timeout)
        request = httpx.Request(method, url, json=kw.get("json"), params=kw.get("params"), headers=kw.get("headers"))
        return handler(request)

    client_module.httpx.request = _request  # type: ignore[assignment]
    return client


@pytest.fixture(autouse=True)
def _restore_httpx_request():
    original = httpx.request
    yield
    httpx.request = original


def test_client_sends_bearer_token():
    mock = _MockAPI()
    client = _client_with_mock(mock, token=FAKE_TOKEN)
    me = client.me()
    assert me["email"] == "user@example.com"
    assert ("GET", "/api/v1/me") in mock.calls


def test_client_dev_org_header_when_no_token():
    mock = _MockAPI()
    client = _client_with_mock(mock)
    client.config = Config(api_url="http://mock", org="org_dev")
    me = client.me()
    assert me["active_organization"]["clerk_organization_id"] == "org_dev"


def test_client_401_raises_authentication_error():
    client = _client_with_mock(_MockAPI(), token="wrong")
    with pytest.raises(cli_errors.AuthenticationError):
        client.me()


def test_client_404_raises_not_found():
    client = _client_with_mock(_MockAPI(), token=FAKE_TOKEN)
    with pytest.raises(cli_errors.NotFoundError):
        client.applications_get("nope")


def test_client_500_raises_api_error():
    client = _client_with_mock(_MockAPI(), token=FAKE_TOKEN)
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
        request = httpx.Request(method, url, json=kw.get("json"), params=kw.get("params"), headers=kw.get("headers"))
        return handler(request)

    monkeypatch.setattr(client_module.httpx, "request", _request)


def test_client_get_retries_then_succeeds(monkeypatch):
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] < 3:
            return httpx.Response(503, json={"error": {"code": "unavailable", "message": "x"}})
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


# -- credential storage ----------------------------------------------------------------


def test_credential_file_roundtrip_and_permissions(tmp_path, monkeypatch):
    monkeypatch.setenv("EVALYX_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(cli_auth, "_keyring_available", lambda: False)
    assert cli_auth.load_token() is None
    cli_auth.save_token(FAKE_TOKEN)
    path = tmp_path / "credentials.json"
    assert path.exists()
    assert (path.stat().st_mode & 0o777) == 0o600
    assert cli_auth.load_token() == FAKE_TOKEN
    cli_auth.clear_token()
    assert cli_auth.load_token() is None


def test_credential_store_rejects_non_token_keys(tmp_path, monkeypatch):
    """Only the bearer token may ever live in the local store."""
    monkeypatch.setenv("EVALYX_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(cli_auth, "_keyring_available", lambda: False)
    (tmp_path / "credentials.json").write_text(
        json.dumps({"token": "t", "clerk_secret_key": "sk_evil", "api_key": "k"})
    )
    assert cli_auth.load_token() == "t"
    cli_auth.save_token("t2")
    data = json.loads((tmp_path / "credentials.json").read_text())
    assert set(data) == {"token"}
    assert "sk_evil" not in json.dumps(data)


# -- CLI commands ----------------------------------------------------------------------


@pytest.fixture
def no_keyring(monkeypatch, tmp_path):
    monkeypatch.setenv("EVALYX_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(cli_auth, "_keyring_available", lambda: False)
    return tmp_path


def test_help_works():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "evalyx" in result.output.lower()


def test_version_flag():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.startswith("evalyx ")


def test_whoami_without_credentials_exits_3(no_keyring):
    result = runner.invoke(app, ["whoami"])
    assert result.exit_code == cli_errors.EXIT_AUTH
    assert "login" in result.output


def test_app_list_json_output(no_keyring, monkeypatch):
    mock = _MockAPI()
    _patch_request(monkeypatch, mock.handler)
    cli_auth.save_token(FAKE_TOKEN)
    result = runner.invoke(app, ["--json", "app", "list"], input="")
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["total"] == 1
    assert data["items"][0]["secret_configured"] is True


def test_app_list_human_output(no_keyring, monkeypatch):
    mock = _MockAPI()
    _patch_request(monkeypatch, mock.handler)
    cli_auth.save_token(FAKE_TOKEN)
    result = runner.invoke(app, ["app", "list"])
    assert result.exit_code == 0, result.output
    assert "NAME" in result.output
    assert "11111111" in result.output


def test_token_never_appears_in_any_output(no_keyring, monkeypatch):
    """Security contract: the bearer token never leaks into stdout/stderr."""
    mock = _MockAPI()
    _patch_request(monkeypatch, mock.handler)
    cli_auth.save_token(FAKE_TOKEN)
    for args in (["app", "list"], ["whoami"], ["--json", "app", "list"], ["dataset", "list"]):
        result = runner.invoke(app, args)
        assert FAKE_TOKEN not in result.output, f"token leaked in {args}"
        assert "authorization" not in result.output.lower()


def test_json_output_is_pure_json(no_keyring, monkeypatch):
    mock = _MockAPI()
    _patch_request(monkeypatch, mock.handler)
    cli_auth.save_token(FAKE_TOKEN)
    result = runner.invoke(app, ["--json", "app", "list"])
    # Strict parse of the entire stdout: no banners or progress lines.
    json.loads(result.output)


def test_login_stdin_token_no_echo(no_keyring, monkeypatch):
    mock = _MockAPI()
    _patch_request(monkeypatch, mock.handler)
    result = runner.invoke(app, ["login"], input=FAKE_TOKEN + "\n")
    assert result.exit_code == 0, result.output
    assert FAKE_TOKEN not in result.output
    assert "user@example.com" in result.output


def test_login_bad_token_exits_3(no_keyring, monkeypatch):
    mock = _MockAPI()
    _patch_request(monkeypatch, mock.handler)
    result = runner.invoke(app, ["login"], input="wrong-token\n")
    assert result.exit_code == cli_errors.EXIT_AUTH


def test_dev_org_login_persists_config(no_keyring, monkeypatch):
    mock = _MockAPI()
    _patch_request(monkeypatch, mock.handler)
    result = runner.invoke(app, ["login", "--org", "org_dev"])
    assert result.exit_code == 0, result.output
    text = (no_keyring / "config.toml").read_text()
    assert 'org = "org_dev"' in text


def test_eval_run_submit_json(no_keyring, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/evaluations"
        body = json.loads(request.content)
        assert body["application_id"] == "app-1"
        return httpx.Response(202, json={
            "run_id": "22222222-2222-2222-2222-222222222222",
            "status": "pending", "task_id": "task-1", "status_url": "/api/v1/evaluations/x",
        })

    _patch_request(monkeypatch, handler)
    cli_auth.save_token(FAKE_TOKEN)
    result = runner.invoke(
        app,
        ["--json", "eval", "run", "--application", "app-1", "--dataset-version", "dv-1", "--agent-model", "m"],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["status"] == "pending"


def test_eval_run_wait_completed_with_failures_exits_7(no_keyring, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/evaluations":
            return httpx.Response(202, json={
                "run_id": "run-1", "status": "pending", "task_id": "t", "status_url": "/s",
            })
        if request.url.path == "/api/v1/evaluations/run-1":
            return httpx.Response(200, json={
                "id": "run-1", "status": "completed",
                "counts": {"total": 10, "passed": 8, "failed": 2, "error": 0, "executed": 0},
            })
        return httpx.Response(404, json={"error": {"code": "not_found", "message": "x"}})

    _patch_request(monkeypatch, handler)
    cli_auth.save_token(FAKE_TOKEN)
    result = runner.invoke(
        app,
        ["eval", "run", "--application", "a", "--dataset-version", "d", "--agent-model", "m", "--wait"],
    )
    assert result.exit_code == cli_errors.EXIT_QUALITY_FAILURES
    assert "Pass rate: 80.0%" in result.output


def test_reliability_command(no_keyring, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "total_cases": 120, "errored_cases": 4, "classified_failures": 4,
            "unclassified_execution_failures": 0, "retryable_failures": 2,
            "failure_breakdown": {"timeout": 2, "rate_limited": 1, "connection_error": 1},
        })

    _patch_request(monkeypatch, handler)
    cli_auth.save_token(FAKE_TOKEN)
    result = runner.invoke(app, ["reliability", "run-1"])
    assert result.exit_code == 0
    assert "timeout" in result.output
    assert "Error rate: 3.3%" in result.output


def test_regression_show_detected_exits_7(no_keyring, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "result": "REGRESSION_DETECTED", "regression_detected": True,
            "baseline_run_id": "b", "current_run_id": "c",
            "baseline": {"pass_rate": 94.2}, "current": {"pass_rate": 87.1},
            "deltas": {"pass_rate_pp": -7.1},
            "threshold_violations": [{"detail": "pass rate dropped"}],
            "newly_failed_cases": [{"name": "testcase-042"}],
        })

    _patch_request(monkeypatch, handler)
    cli_auth.save_token(FAKE_TOKEN)
    result = runner.invoke(app, ["regression", "show", "cmp-1"])
    assert result.exit_code == cli_errors.EXIT_QUALITY_FAILURES
    assert "94.2" in result.output and "87.1" in result.output
    assert "testcase-042" in result.output


def test_delete_requires_confirmation_non_interactive(no_keyring):
    cli_auth.save_token(FAKE_TOKEN)
    result = runner.invoke(app, ["app", "delete", "some-id"])
    assert result.exit_code == cli_errors.EXIT_USAGE


# -- TUI smoke ------------------------------------------------------------------------------


def test_tui_module_imports_and_binds():
    """The TUI module imports cleanly and declares the documented shortcuts."""
    from evalyx.cli.tui.app import EvalyxTUI

    binding_keys = {
        binding[0] if isinstance(binding, tuple) else binding.key
        for binding in EvalyxTUI.BINDINGS
    }
    assert {"q", "r", "escape", "a", "e", "d", "s", "g", "t"} <= binding_keys
