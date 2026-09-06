"""The ``evalyx`` command-line interface (Typer).

Entry point for every non-interactive command plus the TUI launcher. All
network access flows through :class:`evalyx.cli.client.EvalyxClient`; the
CLI never touches PostgreSQL, Redis, Celery, or application endpoints.
"""

from __future__ import annotations

import sys
import time
from typing import Annotated, Any

import typer

from evalyx.cli import errors as err
from evalyx.cli.client import EvalyxClient
from evalyx.cli.config import Config, load_config
from evalyx.cli.output import (
    arrow_down,
    arrow_up,
    emit_json,
    human_table,
    info,
    kv,
    mark,
)
from evalyx.cli.tui.app import run_tui

app = typer.Typer(
    name="evalyx",
    help="Evalyx — AI evaluation & reliability platform (terminal client).",
    no_args_is_help=True,
    add_completion=False,
)
app_cli = typer.Typer(
    help="Manage applications under evaluation.", invoke_without_command=False
)
app_dataset = typer.Typer(help="Manage datasets and test cases.")
app_eval = typer.Typer(help="Submit and inspect evaluation runs.")
app.add_typer(app_cli, name="app")
app.add_typer(app_dataset, name="dataset")
app.add_typer(app_eval, name="eval")

#: Flag-style options shared by commands (defined once).
JsonOpt = Annotated[
    bool,
    typer.Option("--json", help="Machine-readable JSON output (CI-safe)."),
]
QuietOpt = Annotated[
    bool, typer.Option("--quiet", help="Suppress progress output (stderr).")
]
VerboseOpt = Annotated[
    bool,
    typer.Option("--verbose", help="Developer diagnostics (secrets stay redacted)."),
]
ApiUrlOpt = Annotated[
    str | None,
    typer.Option("--api-url", envvar="EVALYX_API_URL", help="Evalyx API base URL."),
]


class Context:
    """Per-invocation state resolved from global options."""

    def __init__(
        self,
        api_url: str | None,
        json_mode: bool,
        quiet: bool,
        verbose: bool,
    ) -> None:
        self.config: Config = load_config(api_url=api_url, quiet=quiet, verbose=verbose)
        self.json_mode = json_mode
        self.verbose = verbose

    @property
    def client(self) -> EvalyxClient:
        return EvalyxClient(self.config)

    def require_client(self) -> EvalyxClient:
        """The API client (no login required — Evalyx is local-first)."""
        return self.client

    @property
    def json(self) -> bool:
        return self.json_mode


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    api_url: ApiUrlOpt = None,
    json_mode: JsonOpt = False,
    quiet: QuietOpt = False,
    verbose: VerboseOpt = False,
    version: Annotated[
        bool, typer.Option("--version", help="Print the CLI version and exit.")
    ] = False,
) -> None:
    """Evalyx terminal client. Run `evalyx` with no command to launch the TUI."""
    if version:
        from evalyx.cli import __version__

        typer.echo(f"evalyx {__version__}")
        raise typer.Exit(err.EXIT_OK)

    ctx.obj = Context(api_url, json_mode, quiet, verbose)
    if ctx.invoked_subcommand is None:
        run_tui(ctx.obj)
        raise typer.Exit(err.EXIT_OK)


def _run(ctx: typer.Context, action):
    """Execute a command body, mapping CLI errors to exit codes.

    Tracebacks are never printed unless --verbose, and even then only the
    exception type chain — messages, never Authorization headers or tokens.
    """
    try:
        action()
    except err.EvalyxCLIError as exc:
        _print_error(str(exc), exc.hint, ctx.obj)
        raise typer.Exit(exc.exit_code) from None
    except KeyboardInterrupt:
        raise typer.Exit(err.EXIT_ERROR) from None


def _print_error(message: str, hint: str | None, context: Context) -> None:
    prefix = mark(False) if sys.stdout.isatty() and not context.json_mode else "✗"
    typer.secho(f"{prefix} {message}", fg=typer.colors.RED, err=True)
    if hint:
        typer.secho(f"  {hint}", fg=typer.colors.YELLOW, err=True)
    if context.verbose:
        typer.secho(
            "(verbose mode: secrets and tokens are always redacted)",
            fg=typer.colors.CYAN,
            err=True,
        )


# -- whoami ------------------------------------------------------------------------


@app.command()
def whoami(
    ctx: typer.Context,
    json_mode: JsonOpt = False,
    api_url: ApiUrlOpt = None,
) -> None:
    """Show the local workspace and API URL (no login required)."""
    context = ctx.ensure_object(Context) if ctx.obj is None else ctx.obj
    if api_url:
        context.config = load_config(api_url=api_url or context.config.api_url)

    def action() -> None:
        client = context.require_client()
        me = client.me()
        if json_mode or context.json_mode:
            emit_json(me)
            return
        active = me.get("active_organization") or {}
        kv(
            [
                ("User:", me.get("user_id", "unknown")),
                ("Workspace:", active.get("name") or "—"),
                ("API:", context.config.api_url),
            ]
        )

    _run(ctx, action)


# -- applications -------------------------------------------------------------------


@app_cli.command("list")
def app_list(
    ctx: typer.Context,
    json_mode: JsonOpt = False,
    limit: int = typer.Option(50, help="Page size."),
) -> None:
    """List applications."""
    context = ctx.ensure_object(Context) if ctx.obj is None else ctx.obj

    def action() -> None:
        page = context.require_client().applications_list(limit=limit)
        if json_mode or context.json_mode:
            emit_json(page)
            return
        rows = [
            [
                str(item["id"])[:8],
                item["name"],
                item.get("connection_type", "—").upper(),
                "secret" if item.get("secret_configured") else "no-secret",
            ]
            for item in page.get("items", [])
        ]
        human_table(rows, ["ID", "NAME", "TYPE", "CREDENTIAL"])

    _run(ctx, action)


@app_cli.command("create")
def app_create(
    ctx: typer.Context,
    name: str | None = typer.Argument(None),
    connection_type: str = typer.Option(
        "http", "--type", help="Connection type: http (generic) or mlgpt (reference)."
    ),
    description: str | None = typer.Option(None, help="Human description."),
    json_mode: JsonOpt = False,
) -> None:
    """Create an application (interactive when no name is given)."""
    context = ctx.ensure_object(Context) if ctx.obj is None else ctx.obj

    def action() -> None:
        resolved_name = name
        if resolved_name is None:
            if not sys.stdin.isatty():
                raise err.UsageError(
                    "Application name is required in non-interactive mode."
                )
            resolved_name = typer.prompt("Application name")
        created = context.require_client().applications_create(
            resolved_name, connection_type, description
        )
        if json_mode or context.json_mode:
            emit_json(created)
            return
        typer.echo(
            f"{mark(True)} Application created: {created['name']} ({created['id']})"
        )
        typer.echo(
            "Next: add a version with a connection configuration, then evalyx app test."
        )

    _run(ctx, action)


@app_cli.command("show")
def app_show(
    ctx: typer.Context, application_id: str, json_mode: JsonOpt = False
) -> None:
    """Show one application (metadata + versions, never secrets)."""
    context = ctx.ensure_object(Context) if ctx.obj is None else ctx.obj

    def action() -> None:
        client = context.require_client()
        application = client.applications_get(application_id)
        versions = client.applications_versions(application_id)
        if json_mode or context.json_mode:
            emit_json({"application": application, "versions": versions})
            return
        kv(
            [
                ("ID:", application["id"]),
                ("Name:", application["name"]),
                ("Description:", application.get("description") or "—"),
                ("Type:", application.get("connection_type", "—")),
                (
                    "Credential:",
                    "configured"
                    if application.get("secret_configured")
                    else "not configured",
                ),
                ("Created:", application.get("created_at", "—")),
            ]
        )
        print()
        rows = [
            [
                v["version"],
                (v.get("description") or "—")[:40],
                str(bool(v.get("connection"))),
            ]
            for v in versions.get("items", [])
        ]
        human_table(rows, ["VERSION", "DESCRIPTION", "CONNECTION"])

    _run(ctx, action)


# -- quickstart ----------------------------------------------------------------------


def _is_interactive() -> bool:
    """True when stdin is a terminal (separate function for testability)."""
    return sys.stdin.isatty()


def _quickstart_reject_local_host(url: str) -> str | None:
    """Fast client-side check: loopback/private URLs are rejected server-side.

    Returns a human reason, or None when the URL looks acceptable. The
    server remains authoritative (full DNS validation happens there).
    """
    import ipaddress
    from urllib.parse import urlparse

    try:
        parts = urlparse(url.strip())
    except Exception:  # noqa: BLE001 — unparseable input is simply invalid
        return "that URL could not be parsed."
    if parts.scheme not in ("http", "https"):
        return "the URL must start with http:// or https://."
    host = (parts.hostname or "").lower()
    if host in ("localhost",) or host.endswith(".localhost"):
        return (
            "that URL is localhost, which default servers reject "
            "(SSRF protection). It works only if the server runs with "
            "EVALYX_ALLOW_PRIVATE_ENDPOINTS=1 — otherwise expose your app "
            "on a public URL or tunnel, or register a local MLGPT with: "
            "evalyx app create <name> --type mlgpt"
        )
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return None  # hostname: the server validates DNS at request time
    if ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_reserved:
        return (
            "that IP is not public, which default servers reject "
            "(SSRF protection). It works only if the server runs with "
            "EVALYX_ALLOW_PRIVATE_ENDPOINTS=1."
        )
    return None


@app.command()
def quickstart(ctx: typer.Context) -> None:
    """Interactive first run: endpoint → app → dataset → evaluation.

    Asks for your app's endpoint, registers it, creates a small dataset
    from your own sample questions, runs an evaluation, and prints the
    summary. Requires an interactive terminal.
    """
    context = ctx.ensure_object(Context) if ctx.obj is None else ctx.obj

    def action() -> None:
        if not _is_interactive():
            raise err.UsageError(
                "quickstart is interactive — run it in a terminal.",
                hint="Or script the same steps: app create → app version → dataset create → eval run.",
            )
        client = context.require_client()

        typer.echo("Welcome to Evalyx — let's evaluate your app.")
        name = typer.prompt("App name", default="my-app").strip() or "my-app"

        endpoint = typer.prompt("Your app's chat endpoint URL").strip()
        reason = _quickstart_reject_local_host(endpoint)
        if reason is not None:
            # Advisory only: servers running with
            # EVALYX_ALLOW_PRIVATE_ENDPOINTS=1 accept local URLs, and the
            # server re-validates everything. Warn, then let them proceed.
            typer.secho(f"  Note: {reason}", fg=typer.colors.YELLOW)
            typer.secho(
                "  Continuing anyway — if the server rejects it, expose the "
                "app on a public URL or set EVALYX_ALLOW_PRIVATE_ENDPOINTS=1 "
                "on the server and restart it.",
                fg=typer.colors.YELLOW,
            )

        auth = typer.prompt("Auth mode", default="none")
        auth = auth.strip().lower() or "none"
        if auth not in ("none", "bearer", "api_key"):
            raise err.UsageError(
                f"Unknown auth mode {auth!r}.",
                hint="Choose one of: none, bearer, api_key.",
            )
        input_field = (
            typer.prompt("Request field for the test input", default="question").strip()
            or "question"
        )
        response_path = (
            typer.prompt("Response field holding the answer", default="answer").strip()
            or "answer"
        )

        created = client.applications_create(name, "http")
        application_id = str(created["id"])
        typer.echo(f"{mark(True)} App registered: {name} ({application_id[:8]})")

        client.applications_create_version(
            application_id,
            "v1",
            connection={
                "endpoint": endpoint,
                "method": "POST",
                "auth": {"type": auth},
                "request": {"mode": "field", "input_field": input_field},
                "response_path": response_path,
                "timeout_seconds": 30.0,
            },
        )
        typer.echo(f"{mark(True)} Version v1 saved")

        if auth in ("bearer", "api_key"):
            secret = typer.prompt("App credential", hide_input=True)
            if not secret:
                raise err.UsageError("No credential provided.")
            client.applications_rotate_secret(application_id, secret)
            typer.echo(f"{mark(True)} Credential stored (encrypted, never shown)")

        probe = client.applications_test(application_id, "Hello — connection check.")
        if not probe.get("success"):
            failure = probe.get("failure") or {}
            raise err.APIError(
                f"Connection test failed: {failure.get('reason', 'unknown')}.",
                hint="Fix the endpoint and re-run: evalyx app test "
                + application_id[:8],
            )
        typer.echo(
            f"{mark(True)} Connection works "
            f"(HTTP {probe.get('http_status')}, {probe.get('latency_ms')} ms)"
        )

        dataset = client.datasets_create(f"{name}-quickstart")
        dataset_id = str(dataset["id"])
        version = client.datasets_create_version(dataset_id, 1)
        dataset_version_id = str(version["id"])
        typer.echo(f"{mark(True)} Dataset created")

        typer.echo("Give 1–3 sample questions for your app (blank line finishes):")
        questions: list[str] = []
        for i in range(1, 4):
            answer = typer.prompt(f"  Question {i}", default="").strip()
            if not answer:
                break
            questions.append(answer)
        if not questions:
            questions = [
                "Say hello in one sentence.",
                "What can you help me with?",
            ]
        for i, question in enumerate(questions, 1):
            client.datasets_add_case(
                dataset_id, 1, f"quickstart-{i}", {"prompt": question}
            )
        typer.echo(f"{mark(True)} Added {len(questions)} test cases")

        submission = client.evaluations_submit(
            application_id, dataset_version_id, f"application:{name}", None
        )
        run_id = str(submission["run_id"])
        typer.echo("Evaluation running…")
        final = _poll_run(client, context, run_id, False)
        _emit_final_run(context, final, False)

        typer.echo("")
        typer.echo("Next steps:")
        typer.echo(f"  evalyx eval results {run_id[:8]}      # per-case outcomes")
        typer.echo(f"  evalyx eval guardrails {run_id[:8]}   # why cases passed/failed")
        typer.echo(f"  evalyx eval reliability {run_id[:8]}  # execution failures")

    _run(ctx, action)


# -- applications -------------------------------------------------------------------
def app_update(
    ctx: typer.Context,
    application_id: str,
    name: str | None = typer.Option(None, help="New name."),
    description: str | None = typer.Option(None, help="New description."),
    json_mode: JsonOpt = False,
) -> None:
    """Update application metadata."""
    context = ctx.ensure_object(Context) if ctx.obj is None else ctx.obj

    def action() -> None:
        updated = context.require_client().applications_update(
            application_id, name, description
        )
        if json_mode or context.json_mode:
            emit_json(updated)
            return
        typer.echo(f"{mark(True)} Updated {updated['name']}")

    _run(ctx, action)


@app_cli.command("delete")
def app_delete(
    ctx: typer.Context,
    application_id: str,
    yes: bool = typer.Option(False, "--yes", help="Skip confirmation."),
) -> None:
    """Delete an application (requires confirmation)."""
    context = ctx.ensure_object(Context) if ctx.obj is None else ctx.obj
    if not yes:
        if sys.stdin.isatty():
            typer.confirm(f"Delete application {application_id}?", abort=True)
        else:
            _print_error(
                "Refusing to delete without --yes in non-interactive mode.",
                "Re-run with --yes to confirm.",
                context,
            )
            raise typer.Exit(err.EXIT_USAGE)

    def action() -> None:
        context.require_client().applications_delete(application_id)
        if not context.json_mode:
            typer.echo(f"{mark(True)} Deleted {application_id}")

    _run(ctx, action)


@app_cli.command("test")
def app_test(
    ctx: typer.Context,
    application_id: str,
    prompt: str | None = typer.Option(None, help="Small non-sensitive probe prompt."),
    json_mode: JsonOpt = False,
) -> None:
    """Test an application connection through the Evalyx API."""
    context = ctx.ensure_object(Context) if ctx.obj is None else ctx.obj

    def action() -> None:
        result = context.require_client().applications_test(application_id, prompt)
        if json_mode or context.json_mode:
            emit_json(result)
            return
        if result.get("success"):
            typer.echo(f"{mark(True)} Connection successful")
            kv(
                [
                    ("HTTP status", result.get("http_status") or "—"),
                    ("Latency", f"{result.get('latency_ms') or '—'} ms"),
                    ("Response", (result.get("preview") or "—")[:120]),
                ]
            )
        else:
            failure = result.get("failure") or {}
            typer.echo(f"{mark(False)} Connection failed")
            kv(
                [
                    ("Category", failure.get("category", "unknown")),
                    ("Reason", failure.get("reason", "—")),
                    (
                        "HTTP status",
                        failure.get("http_status") or result.get("http_status") or "—",
                    ),
                ]
            )

    _run(ctx, action)


@app_cli.command("secret")
def app_secret(ctx: typer.Context, application_id: str) -> None:
    """Rotate the application credential (prompted, never echoed)."""
    context = ctx.ensure_object(Context) if ctx.obj is None else ctx.obj
    if not sys.stdin.isatty():
        secret = sys.stdin.readline().rstrip("\n")
    else:
        secret = typer.prompt("Secret", hide_input=True, confirmation_prompt=True)
    if not secret:
        raise err.UsageError("No secret provided.")

    def action() -> None:
        context.require_client().applications_rotate_secret(application_id, secret)
        if not context.json_mode:
            typer.echo(f"{mark(True)} Credential updated for {application_id}")

    _run(ctx, action)


@app_cli.command("version")
def app_version(
    ctx: typer.Context,
    application_id: str,
    version: str = typer.Argument(..., help="Version label."),
    endpoint: str | None = typer.Option(
        None,
        help="Application endpoint URL (https). Required for generic http apps; "
        "omit for reference (mlgpt) apps to create a metadata-only version.",
    ),
    method: str = typer.Option("POST", help="HTTP method (POST/GET)."),
    auth: str = typer.Option("none", help="Auth mode: none | bearer | api_key."),
    input_field: str = typer.Option(
        "input", help="Request field receiving the case input."
    ),
    response_path: str = typer.Option(
        "answer", help="Dotted response extraction path."
    ),
    timeout_seconds: float = typer.Option(30.0, min=1.0, max=120.0),
    json_mode: JsonOpt = False,
) -> None:
    """Create an immutable application version with a connection configuration."""
    context = ctx.ensure_object(Context) if ctx.obj is None else ctx.obj

    def action() -> None:
        client = context.require_client()
        connection: dict | None = None
        if endpoint is not None:
            connection = {
                "endpoint": endpoint,
                "method": method.upper(),
                "auth": {"type": auth},
                "request": {"mode": "field", "input_field": input_field},
                "response_path": response_path,
                "timeout_seconds": timeout_seconds,
            }
        else:
            # Metadata-only versions are valid for reference (mlgpt) apps
            # whose connection is server-configured; generic http apps need
            # an endpoint — fail fast instead of creating a version the
            # worker cannot execute.
            application = client.applications_get(application_id)
            if application.get("connection_type", "http") == "http":
                raise err.UsageError(
                    "Generic http applications require --endpoint.",
                    hint="evalyx app version <id> <label> --endpoint https://...",
                )
        created = client.applications_create_version(
            application_id, version, connection=connection
        )
        if json_mode or context.json_mode:
            emit_json(created)
            return
        typer.echo(
            f"{mark(True)} Version {created['version']} created for {application_id}"
        )

    _run(ctx, action)


# -- datasets -----------------------------------------------------------------------


@app_dataset.command("list")
def dataset_list(
    ctx: typer.Context, json_mode: JsonOpt = False, limit: int = typer.Option(50)
) -> None:
    """List datasets."""
    context = ctx.ensure_object(Context) if ctx.obj is None else ctx.obj

    def action() -> None:
        page = context.require_client().datasets_list(limit=limit)
        if json_mode or context.json_mode:
            emit_json(page)
            return
        rows = [
            [str(item["id"])[:8], item["name"], (item.get("description") or "—")[:40]]
            for item in page.get("items", [])
        ]
        human_table(rows, ["ID", "NAME", "DESCRIPTION"])

    _run(ctx, action)


@app_dataset.command("create")
def dataset_create(
    ctx: typer.Context,
    name: str | None = typer.Argument(None),
    description: str | None = typer.Option(None),
    json_mode: JsonOpt = False,
) -> None:
    """Create a dataset."""
    context = ctx.ensure_object(Context) if ctx.obj is None else ctx.obj

    def action() -> None:
        resolved = name or (
            typer.prompt("Dataset name") if sys.stdin.isatty() else None
        )
        if not resolved:
            raise err.UsageError("Dataset name is required.")
        created = context.require_client().datasets_create(resolved, description)
        if json_mode or context.json_mode:
            emit_json(created)
            return
        typer.echo(f"{mark(True)} Dataset created: {created['name']} ({created['id']})")

    _run(ctx, action)


@app_dataset.command("show")
def dataset_show(
    ctx: typer.Context, dataset_id: str, json_mode: JsonOpt = False
) -> None:
    """Show a dataset and its versions."""
    context = ctx.ensure_object(Context) if ctx.obj is None else ctx.obj

    def action() -> None:
        client = context.require_client()
        dataset = client.datasets_get(dataset_id)
        versions = client.datasets_versions(dataset_id)
        if json_mode or context.json_mode:
            emit_json({"dataset": dataset, "versions": versions})
            return
        kv(
            [
                ("ID:", dataset["id"]),
                ("Name:", dataset["name"]),
                ("Description:", dataset.get("description") or "—"),
            ]
        )
        print()
        rows = [
            [str(v["version"]), str(v["id"])[:8], (v.get("description") or "—")[:40]]
            for v in versions.get("items", [])
        ]
        human_table(rows, ["VERSION", "ID", "DESCRIPTION"])

    _run(ctx, action)


@app_dataset.command("add-case")
def dataset_add_case(
    ctx: typer.Context,
    dataset_id: str,
    version: int,
    name: str = typer.Option(..., help="Case name."),
    input_json: str = typer.Option(..., "--input", help="Case input as JSON."),
    expected_json: str | None = typer.Option(
        None, "--expected", help="Expected output as JSON."
    ),
) -> None:
    """Add a test case to a dataset version."""
    context = ctx.ensure_object(Context) if ctx.obj is None else ctx.obj
    import json as _json

    try:
        case_input = _json.loads(input_json)
        expected = _json.loads(expected_json) if expected_json else None
    except _json.JSONDecodeError as exc:
        raise err.UsageError(f"Invalid JSON: {exc}") from None

    def action() -> None:
        created = context.require_client().datasets_add_case(
            dataset_id, version, name, case_input, expected
        )
        if context.json_mode:
            emit_json(created)
            return
        typer.echo(f"{mark(True)} Case added: {created['name']}")

    _run(ctx, action)


# -- evaluations ------------------------------------------------------------------------


@app_eval.command("run")
def eval_run(
    ctx: typer.Context,
    application: str = typer.Option(..., help="Application id."),
    dataset_version: str = typer.Option(..., help="Dataset version id."),
    agent_model: str = typer.Option(
        ..., "--agent-model", help="Model identifier for the agent under test."
    ),
    judge_model: str | None = typer.Option(
        None, "--judge-model", help="Judge model (default: server EVALYX_JUDGE_MODEL)."
    ),
    wait: bool = typer.Option(
        False, "--wait", help="Poll until the run finishes (bounded)."
    ),
    json_mode: JsonOpt = False,
) -> None:
    """Submit an evaluation run (asynchronous by design)."""
    context = ctx.ensure_object(Context) if ctx.obj is None else ctx.obj

    def action() -> None:
        client = context.require_client()
        submission = client.evaluations_submit(
            application, dataset_version, agent_model, judge_model
        )
        run_id = str(submission["run_id"])
        if not wait:
            if json_mode or context.json_mode:
                emit_json(submission)
                return
            typer.echo("Evaluation submitted")
            typer.echo(f"Run ID: {run_id}")
            typer.echo(f"Status: {submission.get('status', 'pending')}")
            return

        final = _poll_run(client, context, run_id, json_mode or context.json_mode)
        _emit_final_run(context, final, json_mode or context.json_mode)

    _run(ctx, action)


def _poll_run(
    client: EvalyxClient, context: Context, run_id: str, json_mode: bool
) -> dict:
    """Bounded status polling via the REST API (never Celery/Redis)."""
    deadline = time.monotonic() + context.config.poll_timeout
    terminal = {"completed", "failed", "cancelled"}
    last = None
    while time.monotonic() < deadline:
        run = client.evaluations_get(run_id)
        status = str(run.get("status"))
        if not json_mode and status != last:
            info(f"  {arrow_down()} {status}")
            last = status
        if status in terminal:
            return run
        time.sleep(context.config.poll_interval)
    raise err.APIError(
        "Polling timed out before the evaluation finished.",
        hint="Check the run later with: evalyx eval show " + run_id,
    )


def _emit_final_run(context: Context, run: dict, json_mode: bool) -> None:
    counts = run.get("counts") or {}
    status = run.get("status")
    if json_mode:
        emit_json(run)
        return
    kv(
        [
            ("Run:", str(run.get("id"))),
            ("Status:", str(status)),
            ("Cases:", counts.get("total", 0)),
            ("Passed:", counts.get("passed", 0)),
            ("Failed:", counts.get("failed", 0)),
            ("Errors:", counts.get("error", 0)),
        ]
    )
    if status == "completed":
        total = counts.get("total", 0)
        passed = counts.get("passed", 0)
        if total:
            typer.echo(f"Pass rate: {passed / total * 100:.1f}%")
        if counts.get("failed", 0):
            raise typer.Exit(err.EXIT_QUALITY_FAILURES)
        if counts.get("error", 0):
            raise typer.Exit(err.EXIT_EXECUTION_ERRORS)


@app_eval.command("list")
def eval_list(
    ctx: typer.Context,
    json_mode: JsonOpt = False,
    limit: int = typer.Option(20),
    application_id: str | None = typer.Option(None, "--application-id"),
) -> None:
    """List evaluation runs (newest first)."""
    context = ctx.ensure_object(Context) if ctx.obj is None else ctx.obj

    def action() -> None:
        page = context.require_client().evaluations_list(
            limit=limit, application_id=application_id
        )
        if json_mode or context.json_mode:
            emit_json(page)
            return
        rows = [
            [
                str(item["id"])[:8],
                str(item.get("status", "—")),
                str(item.get("agent_model", "—"))[:32],
                (item.get("created_at") or "—")[:19],
            ]
            for item in page.get("items", [])
        ]
        human_table(rows, ["RUN", "STATUS", "AGENT MODEL", "CREATED"])

    _run(ctx, action)


@app_eval.command("show")
def eval_show(ctx: typer.Context, run_id: str, json_mode: JsonOpt = False) -> None:
    """Show one evaluation run summary."""
    context = ctx.ensure_object(Context) if ctx.obj is None else ctx.obj

    def action() -> None:
        run = context.require_client().evaluations_get(run_id)
        if json_mode or context.json_mode:
            emit_json(run)
            return
        counts = run.get("counts") or {}
        total = counts.get("total", 0)
        passed = counts.get("passed", 0)
        kv(
            [
                ("Run:", str(run.get("id"))),
                ("Status:", str(run.get("status"))),
                ("Agent model:", run.get("agent_model") or "—"),
                ("Judge model:", run.get("judge_model") or "—"),
            ]
        )
        print()
        kv(
            [
                ("Cases:", total),
                ("Passed:", passed),
                ("Failed:", counts.get("failed", 0)),
                ("Errors:", counts.get("error", 0)),
            ]
        )
        if total:
            print(f"Pass rate: {passed / total * 100:.1f}%")
        if run.get("started_at") and run.get("completed_at"):
            print(
                f"Duration: {_format_duration(run['started_at'], run['completed_at'])}"
            )

    _run(ctx, action)


def _format_duration(started: str, completed: str) -> str:
    try:
        from datetime import datetime

        start = datetime.fromisoformat(started.replace("Z", ""))
        end = datetime.fromisoformat(completed.replace("Z", ""))
        seconds = int((end - start).total_seconds())
        minutes, secs = divmod(max(seconds, 0), 60)
        return f"{minutes}m {secs}s" if minutes else f"{secs}s"
    except Exception:  # noqa: BLE001 — display-only fallback
        return "—"


@app_eval.command("results")
def eval_results(
    ctx: typer.Context,
    run_id: str,
    json_mode: JsonOpt = False,
    limit: int = typer.Option(50),
    failures_only: bool = typer.Option(
        False, "--failures-only", help="Only failed/error cases."
    ),
) -> None:
    """Inspect case-level results; failures and errors stay distinct."""
    context = ctx.ensure_object(Context) if ctx.obj is None else ctx.obj

    def action() -> None:
        page = context.require_client().evaluations_results(run_id, limit=limit)
        items = page.get("items", [])
        if failures_only:
            items = [
                item
                for item in items
                if item.get("status") in ("failed", "error", "executed")
            ]
        if json_mode or context.json_mode:
            emit_json(
                {
                    "items": items,
                    "total": page.get("total"),
                    "limit": limit,
                    "offset": 0,
                }
            )
            return

        failed = [item for item in items if item.get("status") == "failed"]
        errored = [
            item for item in items if item.get("status") in ("error", "executed")
        ]

        if failed:
            typer.secho("FAILED TESTS", bold=True)
            for item in failed:
                failure = item.get("failure") or {}
                guardrail_reasons = [
                    g.get("reason") or g.get("name")
                    for g in item.get("guardrail_results", [])
                    if not g.get("passed")
                ]
                kv(
                    [
                        (str(item.get("test_case_id"))[:8], ""),
                        ("  Status:", item.get("status")),
                        (
                            "  Category:",
                            failure.get("category")
                            or "; ".join(guardrail_reasons)
                            or "quality",
                        ),
                        ("  Expected:", _preview(item.get("expected_output"))),
                        ("  Actual:", _preview(item.get("actual_output"))),
                    ]
                )
                print()
        if errored:
            typer.secho("ERRORS", bold=True)
            for item in errored:
                failure = item.get("failure") or {}
                kv(
                    [
                        (str(item.get("test_case_id"))[:8], ""),
                        (
                            "  Category:",
                            failure.get("category") or item.get("error") or "unknown",
                        ),
                        ("  Retryable:", str(failure.get("retryable", "—"))),
                    ]
                )
                print()
        if not failed and not errored:
            typer.echo(f"{mark(True)} No failures or errors on this page.")

    _run(ctx, action)


def _preview(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, dict):
        import json as _json

        text = _json.dumps(value, default=str)
    else:
        text = str(value)
    return text[:120]


@app_eval.command("guardrails")
def eval_guardrails(
    ctx: typer.Context, run_id: str, json_mode: JsonOpt = False
) -> None:
    """Inspect guardrail outcomes for a run."""
    context = ctx.ensure_object(Context) if ctx.obj is None else ctx.obj

    def action() -> None:
        page = context.require_client().evaluations_guardrails(run_id)
        if json_mode or context.json_mode:
            emit_json(page)
            return
        rows = [
            [
                g.get("name", "—"),
                str(g.get("status", "—")),
                mark(bool(g.get("passed"))),
                f"{g['score']:.2f}"
                if isinstance(g.get("score"), (int, float))
                else "—",
                (g.get("reason") or "—")[:48],
            ]
            for g in page.get("items", [])
        ]
        human_table(rows, ["GUARDRAIL", "STATUS", "OK", "SCORE", "REASON"])

    _run(ctx, action)


@app_eval.command("reliability")
def eval_reliability(
    ctx: typer.Context, run_id: str, json_mode: JsonOpt = False
) -> None:
    """Show the Phase 12 execution-reliability report."""
    _reliability(ctx, run_id, json_mode)


@app.command()
def reliability(ctx: typer.Context, run_id: str, json_mode: JsonOpt = False) -> None:
    """Show the execution-reliability report for a run."""
    _reliability(ctx, run_id, json_mode)


def _reliability(ctx: typer.Context, run_id: str, json_mode: bool) -> None:
    context = ctx.ensure_object(Context) if ctx.obj is None else ctx.obj

    def action() -> None:
        report = context.require_client().evaluations_reliability(run_id)
        if json_mode or context.json_mode:
            emit_json(report)
            return
        total = report.get("total_cases", 0)
        errored = report.get("errored_cases", 0)
        typer.echo("Reliability")
        print()
        kv(
            [
                ("Total requests", total),
                ("Successful", total - errored),
                ("Errors", errored),
            ]
        )
        print()
        breakdown = report.get("failure_breakdown") or {}
        if breakdown:
            typer.echo("Failure categories:")
            for category, count in breakdown.items():
                typer.echo(f"  {category.ljust(24)} {count}")
        else:
            typer.echo("Failure categories: none")
        if total:
            typer.echo(f"Error rate: {errored / total * 100:.1f}%")

    _run(ctx, action)


# -- regressions -------------------------------------------------------------------------


regression_app = typer.Typer(help="Regression comparisons.")
app.add_typer(regression_app, name="regression")


@regression_app.command("run")
def regression_run(
    ctx: typer.Context,
    baseline: str = typer.Option(..., "--baseline", help="Baseline run id."),
    current: str = typer.Option(..., "--current", help="Current run id."),
    json_mode: JsonOpt = False,
) -> None:
    """Compare two completed runs (idempotent artifact)."""
    _regression_compare(ctx, baseline, current, json_mode)


@regression_app.command("show")
def regression_show(
    ctx: typer.Context, comparison_id: str, json_mode: JsonOpt = False
) -> None:
    """Show a persisted regression report."""
    context = ctx.ensure_object(Context) if ctx.obj is None else ctx.obj

    def action() -> None:
        report = context.require_client().regressions_get(comparison_id)
        _print_regression(context, report, json_mode or context.json_mode)

    _run(ctx, action)


def _regression_compare(
    ctx: typer.Context, baseline: str, current: str, json_mode: bool
) -> None:
    context = ctx.ensure_object(Context) if ctx.obj is None else ctx.obj

    def action() -> None:
        report = context.require_client().regressions_compare(baseline, current)
        _print_regression(context, report, json_mode or context.json_mode)

    _run(ctx, action)


def _print_regression(context: Context, report: dict, json_mode: bool) -> None:
    if json_mode:
        emit_json(report)
        return
    current_metrics = report.get("current") or {}
    baseline_metrics = report.get("baseline") or {}
    deltas = report.get("deltas") or {}
    typer.echo(f"Regression: {report.get('result')}")
    print()
    kv(
        [
            ("Baseline", str(report.get("baseline_run_id"))[:8]),
            ("Current", str(report.get("current_run_id"))[:8]),
        ]
    )
    print()
    pass_delta = deltas.get("pass_rate_pp")
    if pass_delta is not None:
        direction = arrow_down() if pass_delta < 0 else arrow_up()
        typer.echo(
            f"Pass rate   {baseline_metrics.get('pass_rate')}% → {current_metrics.get('pass_rate')}%   {direction} {pass_delta:+.1f} pp"
        )
    error_delta = deltas.get("error_rate_pp")
    if error_delta is not None:
        direction = arrow_up() if error_delta > 0 else arrow_down()
        typer.echo(
            f"Error rate  {baseline_metrics.get('error_rate')}% → {current_metrics.get('error_rate')}%   {direction} {error_delta:+.1f} pp"
        )
    violations = report.get("threshold_violations") or []
    if violations:
        print()
        typer.secho("Threshold violations:", bold=True)
        for violation in violations:
            typer.echo(f"  - {violation.get('detail')}")
    newly_failed = report.get("newly_failed_cases") or []
    if newly_failed:
        print()
        typer.secho("Newly failed:", bold=True)
        for finding in newly_failed:
            typer.echo(f"  {finding.get('name') or finding.get('identity')}")
    if report.get("regression_detected"):
        raise typer.Exit(err.EXIT_QUALITY_FAILURES)


def run() -> None:
    """Console entrypoint."""
    app()


if __name__ == "__main__":
    run()
