"""The interactive Evalyx TUI (Textual).

Layout: sidebar navigation + main content area. All data flows through
:class:`evalyx.cli.client.EvalyxClient` calls executed in worker threads
(``run_worker``) so the event loop never blocks on network I/O. The TUI is
render-only: it never executes evaluations, queries PostgreSQL, decrypts
credentials, or calls application endpoints — the REST API stays
authoritative.
"""

from __future__ import annotations

from typing import ClassVar

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Footer, Header, Label, Static

from evalyx.cli.auth import load_token
from evalyx.cli.client import EvalyxClient
from evalyx.cli.config import Config
from evalyx.cli.errors import EvalyxCLIError
from evalyx.cli.output import supports_unicode


class EvalyxTUI(App[None]):
    """The Evalyx developer TUI. Keyboard-first; no mouse required."""

    TITLE = "EVALYX"
    SUB_TITLE = "AI Evaluation & Reliability Platform"
    CSS = """
    #sidebar { width: 26; border-right: solid $primary; padding: 1; }
    #sidebar Label { padding: 0 1; margin-bottom: 1; }
    #sidebar Label:hover { background: $surface-lighten-2; }
    #sidebar .nav-selected { background: $primary; color: $text; }
    #content { padding: 1 2; }
    #content DataTable { height: 1fr; }
    #status { height: 1; color: $text-muted; padding: 0 2; }
    #detail { padding: 0 1; }
    """
    BINDINGS: ClassVar[list] = [
        ("q", "quit", "Quit"),
        ("r", "refresh", "Refresh"),
        ("d", "dashboard", "Dashboard"),
        ("a", "applications", "Applications"),
        ("s", "datasets", "Datasets"),
        ("e", "evaluations", "Evaluations"),
        ("g", "regressions", "Regressions"),
        ("escape", "back", "Back"),
        ("t", "test_connection", "Test connection"),
        ("c", "create", "Create"),
    ]

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        token = load_token()
        self.client = EvalyxClient(config, token=token)
        self._view_stack: list[str] = ["dashboard"]

    # -- layout ----------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Vertical(id="sidebar"):
                yield Label("Dashboard", id="nav-dashboard", classes="nav-selected")
                yield Label("Applications", id="nav-applications")
                yield Label("Datasets", id="nav-datasets")
                yield Label("Evaluations", id="nav-evaluations")
                yield Label("Regressions", id="nav-regressions")
            with Vertical(id="content"):
                yield Static("", id="detail")
                yield DataTable(id="table")
                yield Static("Loading…", id="status")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#table", DataTable)
        table.cursor_type = "row"
        self._select("dashboard")

    # -- helpers -----------------------------------------------------------------

    def _set_status(self, message: str) -> None:
        self.query_one("#status", Static).update(message)

    def _set_detail(self, text: str) -> None:
        self.query_one("#detail", Static).update(text)

    def _select(self, view: str) -> None:
        """Switch the active view; refresh its data asynchronously."""
        self._view_stack = [view]
        for nav in self.query("#sidebar Label"):
            nav.remove_class("nav-selected")
        self.query_one(f"#nav-{view}", Label).add_class("nav-selected")
        table = self.query_one("#table", DataTable)
        table.clear(columns=True)
        self._set_detail("")
        self._set_status("Loading…")
        self.run_worker(self._load_view(view), exclusive=True)

    # -- data loading (async, non-blocking) ----------------------------------------

    async def _load_view(self, view: str) -> None:
        table = self.query_one("#table", DataTable)
        try:
            if view == "dashboard":
                await self._load_dashboard(table)
            elif view == "applications":
                await self._load_applications(table)
            elif view == "datasets":
                await self._load_datasets(table)
            elif view == "evaluations":
                await self._load_evaluations(table)
            elif view == "regressions":
                await self._load_regressions(table)
            self._set_status("Loaded.")
        except EvalyxCLIError as exc:
            self._set_status(f"Error: {exc}")
            self._set_detail(
                f"{exc}\n\n{exc.hint or ''}".strip()
                + "\n\nStart the API and log in:\n  evalyx login\n  evalyx org use <org-id>"
            )
        except Exception as exc:  # noqa: BLE001 — the TUI never crashes on refresh
            self._set_status(f"Unexpected error: {type(exc).__name__}")

    async def _call(self, method: str, *args, **kwargs):
        """Run one blocking client call off the event loop."""
        import asyncio

        return await asyncio.to_thread(getattr(self.client, method), *args, **kwargs)

    def _columns(self, table: DataTable, names: list[str]) -> None:
        for name in names:
            table.add_column(name, key=name)

    async def _load_dashboard(self, table: DataTable) -> None:
        apps = await self._call("applications_list", limit=200)
        datasets = await self._call("datasets_list", limit=200)
        runs = await self._call("evaluations_list", limit=8)
        ok = "✓" if supports_unicode() else "[OK]"
        bad = "✗" if supports_unicode() else "[FAIL]"
        lines = [
            f"  Applications        {apps.get('total', len(apps.get('items', [])))}",
            f"  Datasets            {datasets.get('total', len(datasets.get('items', [])))}",
            f"  Evaluation runs     {runs.get('total', len(runs.get('items', [])))}",
            "",
            "  Recent evaluations",
        ]
        for run in runs.get("items", [])[:6]:
            counts = run.get("counts") or {}
            completed = run.get("status") == "completed"
            glyph = ok if completed and not counts.get("failed") else bad
            lines.append(
                f"  {glyph} {str(run.get('agent_model', ''))[:24].ljust(24)} {run.get('status', '')!s}"
            )
        self._set_detail("\n".join(lines))
        self._columns(table, ["RECENT RUNS", "STATUS"])
        for run in runs.get("items", []):
            table.add_row(
                str(run.get("id"))[:8] + "  " + str(run.get("agent_model", ""))[:24],
                str(run.get("status", "")),
            )

    async def _load_applications(self, table: DataTable) -> None:
        page = await self._call("applications_list", limit=200)
        self._columns(table, ["ID", "NAME", "TYPE", "CREDENTIAL"])
        for item in page.get("items", []):
            table.add_row(
                str(item["id"])[:8],
                item["name"],
                item.get("connection_type", "—").upper(),
                "configured" if item.get("secret_configured") else "—",
                key=str(item["id"]),
            )
        self._set_detail(
            "Select a row and press enter for details, t to test the connection."
        )

    async def _load_datasets(self, table: DataTable) -> None:
        page = await self._call("datasets_list", limit=200)
        self._columns(table, ["ID", "NAME", "DESCRIPTION"])
        for item in page.get("items", []):
            table.add_row(
                str(item["id"])[:8], item["name"], (item.get("description") or "—")[:40],
                key=str(item["id"]),
            )

    async def _load_evaluations(self, table: DataTable) -> None:
        page = await self._call("evaluations_list", limit=50)
        self._columns(table, ["RUN", "STATUS", "AGENT MODEL", "CREATED"])
        for item in page.get("items", []):
            table.add_row(
                str(item["id"])[:8],
                str(item.get("status", "")),
                str(item.get("agent_model", ""))[:32],
                str(item.get("created_at", ""))[:19],
                key=str(item["id"]),
            )
        self._set_detail("Press enter on a run for its summary; r to refresh.")

    async def _load_regressions(self, table: DataTable) -> None:
        self._columns(table, ["INFO"])
        table.add_row("Compare two runs with the CLI:")
        table.add_row("  evalyx regression run --baseline <id> --current <id>")
        self._set_detail("Regression artifacts are listed per run from the CLI.")

    # -- actions --------------------------------------------------------------------

    def action_dashboard(self) -> None:
        self._select("dashboard")

    def action_applications(self) -> None:
        self._select("applications")

    def action_datasets(self) -> None:
        self._select("datasets")

    def action_evaluations(self) -> None:
        self._select("evaluations")

    def action_regressions(self) -> None:
        self._select("regressions")

    def action_refresh(self) -> None:
        if self._view_stack:
            self._select(self._view_stack[0])

    async def action_back(self) -> None:
        if len(self._view_stack) > 1:
            self._view_stack.pop()
            self._select(self._view_stack[-1])

    async def action_test_connection(self) -> None:
        table = self.query_one("#table", DataTable)
        row_key = table.cursor_row
        try:
            keyed_row = table.get_row_at(row_key)
        except Exception:  # noqa: BLE001 — empty table
            return
        if self._view_stack[0] != "applications" or not keyed_row:
            return
        app_id = keyed_row[0]
        self._set_status("Testing connection…")
        self.run_worker(self._do_test(app_id), exclusive=True)

    async def _do_test(self, application_id: str) -> None:
        try:
            full = await self._call("applications_get", application_id)
            result = await self._call("applications_test", full["id"])
        except EvalyxCLIError as exc:
            self._set_status(f"Test failed: {exc}")
            return
        if result.get("success"):
            self._set_status(
                f"Connection OK — {result.get('http_status')} in {result.get('latency_ms')} ms"
            )
        else:
            failure = result.get("failure") or {}
            self._set_status(
                f"Connection failed — {failure.get('category', 'unknown')}: {failure.get('reason', '')}"
            )

    def action_create(self) -> None:
        self._set_status("Create from the CLI for now: evalyx app create")


def run_tui(context) -> None:
    """Launch the TUI (blocking)."""
    EvalyxTUI(context.config).run()
