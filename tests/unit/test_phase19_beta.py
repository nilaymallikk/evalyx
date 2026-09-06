"""Phase 19 beta-readiness unit tests (hermetic: no network, no services).

Covers the beta usability fixes without touching production behavior:

- ``evalyx app version`` creates a metadata-only version for reference
  (mlgpt) apps without ``--endpoint``, and fails fast with a usage error
  for generic http apps;
- the TUI boots headless, loads every view from the client, and navigates
  without crashing.
"""

import pytest
from typer.testing import CliRunner

from evalyx.cli import errors as cli_errors
from evalyx.cli.client import EvalyxClient
from evalyx.cli.config import Config
from evalyx.cli.main import app
from evalyx.cli.tui.app import EvalyxTUI

runner = CliRunner()


# -- `evalyx app version` without --endpoint ---------------------------------------


class _VersionAPI:
    """Minimal fake backend for the version-creation flow."""

    def __init__(self, connection_type: str) -> None:
        self.connection_type = connection_type
        self.created_bodies: list[dict] = []

    def applications_get(self, application_id: str) -> dict:
        return {
            "id": application_id,
            "name": "demo",
            "connection_type": self.connection_type,
        }

    def applications_create_version(
        self,
        application_id: str,
        version: str,
        connection: dict | None = None,
        **kwargs: object,
    ) -> dict:
        self.created_bodies.append({"connection": connection})
        return {"id": "v1", "version": version, "application_id": application_id}


def _invoke_version(monkeypatch: pytest.MonkeyPatch, fake: _VersionAPI, *args: str):
    monkeypatch.setattr(
        EvalyxClient,
        "applications_get",
        lambda self, app_id: fake.applications_get(app_id),
    )
    monkeypatch.setattr(
        EvalyxClient,
        "applications_create_version",
        lambda self, app_id, version, connection=None, **kw: (
            fake.applications_create_version(
                app_id, version, connection=connection, **kw
            )
        ),
    )
    # No login needed: plain requests against the configured API URL.
    monkeypatch.delenv("EVALYX_API_URL", raising=False)
    return runner.invoke(app, ["app", "version", *args])


def test_app_version_without_endpoint_for_reference_app(monkeypatch):
    fake = _VersionAPI(connection_type="mlgpt")
    result = _invoke_version(monkeypatch, fake, "app-1", "v1")
    assert result.exit_code == 0, result.output
    assert fake.created_bodies == [{"connection": None}]


def test_app_version_without_endpoint_for_http_app_fails_fast(monkeypatch):
    fake = _VersionAPI(connection_type="http")
    result = _invoke_version(monkeypatch, fake, "app-1", "v1")
    assert result.exit_code == cli_errors.EXIT_USAGE
    assert "--endpoint" in result.output
    assert fake.created_bodies == []


def test_app_version_with_endpoint_still_sends_connection(monkeypatch):
    fake = _VersionAPI(connection_type="http")
    result = _invoke_version(
        monkeypatch, fake, "app-1", "v1", "--endpoint", "https://app.example.com/chat"
    )
    assert result.exit_code == 0, result.output
    assert len(fake.created_bodies) == 1
    assert (
        fake.created_bodies[0]["connection"]["endpoint"]
        == "https://app.example.com/chat"
    )


# -- TUI headless smoke test --------------------------------------------------------


@pytest.mark.asyncio
async def test_tui_boots_and_navigates_all_views(monkeypatch):
    """Every TUI view renders from stubbed client data (no network)."""
    monkeypatch.setattr(
        EvalyxClient,
        "applications_list",
        lambda self, limit=50, offset=0: {
            "items": [
                {
                    "id": "11111111-1111-1111-1111-111111111111",
                    "name": "demo",
                    "connection_type": "mlgpt",
                    "secret_configured": False,
                }
            ],
            "total": 1,
        },
    )
    monkeypatch.setattr(
        EvalyxClient,
        "datasets_list",
        lambda self, limit=50, offset=0: {"items": [], "total": 0},
    )
    monkeypatch.setattr(
        EvalyxClient,
        "evaluations_list",
        lambda self, limit=50, offset=0: {
            "items": [
                {
                    "id": "22222222-2222-2222-2222-222222222222",
                    "status": "completed",
                    "agent_model": "application:mlgpt",
                    "created_at": "2026-01-01T00:00:00Z",
                    "counts": {"failed": 0},
                }
            ],
            "total": 1,
        },
    )
    tui = EvalyxTUI(Config(api_url="http://mock"))
    statuses: list[str] = []
    tui._set_status = statuses.append  # type: ignore[method-assign]
    async with tui.run_test() as pilot:
        await pilot.pause()
        assert tui.query_one("#table") is not None
        for key in ("a", "s", "e", "g", "d", "r"):
            await pilot.press(key)
            await pilot.pause()
        assert statuses, "expected at least one status update"
        assert not any("Unexpected error" in s for s in statuses)
        await pilot.press("q")
