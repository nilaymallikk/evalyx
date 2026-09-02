"""Evalyx × MLGPT end-to-end evaluation demo.

Orchestrates the complete demonstration through Evalyx's **public REST API**
(no database access, no Evalyx internals):

    MLGPT (system under test)
        ↓ HTTP
    Evalyx API → Celery worker → guardrails → scoring → PostgreSQL
        ↓
    regression comparison (baseline vs current)

Workflow:

1. verify MLGPT and Evalyx availability
2. register the MLGPT application + two versions (idempotent via a small
   local state file; the API has no by-name lookup by design)
3. create the demo dataset/version/cases (idempotent, stable case names)
4. submit the BASELINE evaluation (202 Accepted) and poll to completion
5. apply the controlled regression: temporarily swap MLGPT's RAG system
   prompt for a deliberately degraded one (original restored in ``finally``)
6. submit the CURRENT evaluation and poll
7. compare baseline vs current with the Phase 8 regression engine
8. print a recruiter-friendly summary (no prompts, outputs, PII, or keys)

Usage (from the Evalyx repository root, with MLGPT + Evalyx API + worker
+ PostgreSQL + Redis running):

    uv run python examples/mlgpt_demo/run_demo.py             # full demo
    uv run python examples/mlgpt_demo/run_demo.py --resume    # reuse the completed
                                                              # baseline run from the
                                                              # state file; run the
                                                              # degraded evaluation only
    uv run python examples/mlgpt_demo/run_demo.py --restore   # restore prompt only

Environment: EVALYX_API_URL (default http://127.0.0.1:8000),
MLGPT_BASE_URL (default http://127.0.0.1:8001), MLGPT_ROOT (default
/home/nilaymallik/MLGPT). No secrets are read, printed, or transmitted.

Safety: the prompt swap is temporary and always restored (try/finally plus
the ``--restore`` fallback). MLGPT source code is never modified.
"""

import argparse
import ipaddress
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import DATASET_VERSION, DEGRADED_RAG_PROMPT, DEMO_CASES


def _assert_local(base_url: str) -> None:
    """SSRF guard: only http(s) to explicitly-loopback hosts is allowed.

    The demo talks to the operator's own MLGPT and Evalyx instances, which
    run on this machine by design. Any non-http(s) scheme or a host that
    does not resolve to loopback is rejected before a request is made.
    """
    from urllib.parse import urlparse

    parsed = urlparse(base_url)
    if parsed.scheme not in ("http", "https"):
        raise SystemExit(f"Refusing non-http(s) base URL: {parsed.scheme!r}")
    host = parsed.hostname or ""
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError as error:
        raise SystemExit(f"Cannot resolve {host!r}: {error}") from error
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if address.is_loopback:
            return
    raise SystemExit(
        f"Refusing non-loopback base URL {base_url!r} — this demo is "
        "restricted to the operator's own local services."
    )


EVALYX_API_URL = os.environ.get("EVALYX_API_URL", "http://127.0.0.1:8000").rstrip("/")
MLGPT_BASE_URL = os.environ.get("MLGPT_BASE_URL", "http://127.0.0.1:8001").rstrip("/")
_assert_local(EVALYX_API_URL)
_assert_local(MLGPT_BASE_URL)
MLGPT_ROOT = Path(os.environ.get("MLGPT_ROOT", "/home/nilaymallik/MLGPT"))
RAG_PROMPT_PATH = MLGPT_ROOT / "prompts" / "rag_prompt.txt"
STATE_PATH = Path(__file__).resolve().parent / ".demo_state.json"

POLL_INTERVAL_SECONDS = 5.0
POLL_TIMEOUT_SECONDS = 20 * 60.0

APPLICATION_NAME = "MLGPT"
BASELINE_VERSION = "1.0.0-grounded"
DEGRADED_VERSION = "1.1.0-degraded-demo"
DATASET_NAME = "mlgpt-support-v1"
AGENT_MODEL_SELECTOR = "application:mlgpt"  # bounded selector, not a model name
JUDGE_MODEL = "minimax/minimax-m3:free"  # Evalyx's judge (free-model policy)


# -- tiny HTTP helpers over the public API -------------------------------------------


class ApiError(RuntimeError):
    pass


def api(method: str, path: str, body: dict | None = None):
    """Call the Evalyx REST API; return (status_code, parsed_json)."""
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        EVALYX_API_URL + "/api/v1" + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8"))


def expect(code: int, payload, action: str, wanted: tuple[int, ...]) -> dict:
    if code not in wanted:
        raise ApiError(f"{action} failed (HTTP {code}): {payload}")
    return payload


def wait_for_terminal(run_id: str) -> dict:
    """Poll GET /evaluations/{run_id} until a terminal, fully-counted status.

    The run status may flip to ``completed`` a moment before the scoring
    engine's per-case status updates become visible; after reaching a
    terminal status, wait until the counts actually reflect final scoring
    (every case is passed/failed/error — no residual ``executed``).
    """
    deadline = time.monotonic() + POLL_TIMEOUT_SECONDS
    while True:
        if time.monotonic() > deadline:
            raise ApiError(
                f"Evaluation {run_id} did not finish within "
                f"{POLL_TIMEOUT_SECONDS:.0f}s — is the Celery worker running?"
            )
        time.sleep(POLL_INTERVAL_SECONDS)
        code, run = api("GET", f"/evaluations/{run_id}")
        if code != 200:
            raise ApiError(f"Failed to fetch run {run_id}: HTTP {code}")
        if run["status"] in ("completed", "failed", "cancelled"):
            counts = run.get("counts") or {}
            settled = (
                counts.get("total", 0)
                == counts.get("passed", 0)
                + counts.get("failed", 0)
                + counts.get("error", 0)
            )
            if settled:
                return run
            # Terminal but counts not final yet — keep polling quietly.
            continue
        counts = run.get("counts") or {}
        print(
            f"    … {run['status']}"
            f" (passed={counts.get('passed', 0)} failed={counts.get('failed', 0)}"
            f" error={counts.get('error', 0)} executed={counts.get('executed', 0)})"
        )


# -- idempotent resource setup (local state file; API has no by-name lookup) ----------


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {}


def save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def ensure_application(state: dict) -> str:
    """Return the application id, creating it once."""
    if "application_id" in state:
        code, body = api("GET", f"/applications/{state['application_id']}")
        if code == 200:
            return body["id"]
        print("    (stored application no longer exists — re-creating)")
    code, body = api("POST", "/applications", {"name": APPLICATION_NAME})
    if code == 409:  # created by an earlier run whose state file was lost
        raise ApiError(
            f"An application named {APPLICATION_NAME!r} already exists but is "
            "not in the local state file. Delete it via the API or remove "
            f"{STATE_PATH} after cleaning the database."
        )
    expect(code, body, "Application creation", (201,))
    state["application_id"] = body["id"]
    save_state(state)
    return body["id"]


def ensure_application_version(state: dict, version: str) -> str:
    """Return the application-version id, creating the version once.

    Stored ids are validated against the live API first — the state file can
    go stale when the database is reset (e.g. after integration tests
    truncate the domain tables).
    """
    key = f"version::{version}"
    app_id = state["application_id"]
    code, versions = api("GET", f"/applications/{app_id}/versions?limit=200")
    expect(code, versions, "Version listing", (200,))
    listed = {item["version"]: item["id"] for item in versions["items"]}
    if key in state and state[key] in listed.values():
        return state[key]
    if version in listed:  # exists under this application — re-adopt it
        state[key] = listed[version]
        save_state(state)
        return listed[version]
    code, created = api(
        "POST",
        f"/applications/{app_id}/versions",
        {
            "version": version,
            "description": (
                "Grounded reference behavior (original rag_prompt.txt)"
                if version == BASELINE_VERSION
                else "Deliberately degraded behavior (demo regression lever)"
            ),
            "configuration": {
                "endpoint_path": "/v1/chat",
                "evaluation_target": "mlgpt",
            },
        },
    )
    expect(code, created, f"Version {version} creation", (201,))
    state[key] = created["id"]
    save_state(state)
    return created["id"]


def ensure_dataset(state: dict) -> None:
    """Create the dataset + version + all cases once; reuse afterwards.

    Version-aware: bumping ``DATASET_VERSION`` in dataset.py creates a new
    immutable version under the same dataset on the next run.
    """
    version_key = f"dataset_version_id::v{DATASET_VERSION}"
    if version_key in state:
        # Validate the stored ids against the live API: the state file goes
        # stale when the database is reset (e.g. integration-test truncation).
        dataset_ok = (
            state.get("dataset_id") is not None
            and api("GET", f"/datasets/{state['dataset_id']}")[0] == 200
        )
        if dataset_ok:
            code, versions = api(
                "GET", f"/datasets/{state['dataset_id']}/versions?limit=200"
            )
            dataset_ok = (
                code == 200
                and any(v["id"] == state[version_key] for v in versions["items"])
            )
        if dataset_ok:
            return
        print("    (stored dataset version no longer exists — re-creating)")
        state.pop("dataset_id", None)
        state.pop(version_key, None)
    if "dataset_id" in state:
        code, body = api("GET", f"/datasets/{state['dataset_id']}")
        if code != 200:
            print("    (stored dataset no longer exists — re-creating)")
            state.pop("dataset_id", None)
    if "dataset_id" not in state:
        code, created = api("POST", "/datasets", {"name": DATASET_NAME})
        if code == 409:
            raise ApiError(
                f"A dataset named {DATASET_NAME!r} already exists but is not "
                f"in the local state file ({STATE_PATH})."
            )
        expect(code, created, "Dataset creation", (201,))
        state["dataset_id"] = created["id"]
    else:
        created = {"id": state["dataset_id"]}

    code, version = api(
        "POST",
        f"/datasets/{created['id']}/versions",
        {
            "version": DATASET_VERSION,
            "description": "Reference demo: RAG behavior suite "
            "(normal, instruction-following, injection, fake PII, safety, "
            "hallucination, edge cases)",
        },
    )
    expect(code, version, "Dataset version creation", (201,))
    state[version_key] = version["id"]

    for case in DEMO_CASES:
        code, body = api(
            "POST",
            f"/datasets/{created['id']}/versions/{DATASET_VERSION}/cases",
            case,
        )
        expect(code, body, f"Case {case['name']} creation", (201,))
    save_state(state)


# -- MLGPT pre-flight ------------------------------------------------------------------


def preflight_mlgpt() -> None:
    """One real /v1/chat request before submitting a run.

    The reference MLGPT wraps every pipeline failure as an opaque HTTP 500;
    when its upstream model is down or overloaded, a full run would burn
    the free-model daily quota on retries for nothing. Failing fast with a
    clear message protects the quota.
    """
    payload = json.dumps(
        {
            "question": "Reply with the single word: pong",
            "user_id": "00000000-0000-4000-8000-000000000001",
            "conversation_id": None,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        MLGPT_BASE_URL + "/v1/chat",
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            if response.status != 200:
                raise ApiError(f"MLGPT pre-flight returned HTTP {response.status}.")
            body = json.loads(response.read().decode("utf-8"))
            if not str(body.get("answer", "")).strip():
                raise ApiError("MLGPT pre-flight returned an empty answer.")
    except urllib.error.HTTPError as error:
        raise ApiError(
            "MLGPT pre-flight failed: its upstream model backend appears "
            f"unavailable (HTTP {error.code}). Try again later — this "
            "protects the free-model daily quota."
        ) from error


# -- evaluation submission -------------------------------------------------------------


def submit_evaluation(state: dict, app_version_id: str) -> str:
    code, body = api(
        "POST",
        "/evaluations",
        {
            "application_id": state["application_id"],
            "application_version_id": app_version_id,
            "dataset_version_id": state[f"dataset_version_id::v{DATASET_VERSION}"],
            "agent_model": AGENT_MODEL_SELECTOR,
            "judge_model": JUDGE_MODEL,
            "configuration_snapshot": {"demo": "mlgpt-reference"},
        },
    )
    expect(code, body, "Evaluation submission", (202,))
    print(f"    submitted → run {body['run_id']} (task {body.get('task_id')})")
    return str(body["run_id"])


def compare(baseline_run_id: str, current_run_id: str) -> dict:
    code, report = api(
        "POST",
        "/regressions",
        {"baseline_run_id": baseline_run_id, "current_run_id": current_run_id},
    )
    expect(code, report, "Regression comparison", (200,))
    return report


# -- MLGPT prompt swap (the controlled, reversible regression lever) -------------------


def apply_degraded_prompt() -> None:
    """Temporarily replace MLGPT's RAG system prompt (backup kept beside it)."""
    original = RAG_PROMPT_PATH.read_text(encoding="utf-8")
    backup = RAG_PROMPT_PATH.with_name("rag_prompt.txt.evalyx-backup")
    backup.write_text(original, encoding="utf-8")
    RAG_PROMPT_PATH.write_text(DEGRADED_RAG_PROMPT, encoding="utf-8")
    print(f"[regression] degraded RAG prompt applied (backup: {backup.name})")


def restore_prompt() -> None:
    """Restore the original prompt; safe to run repeatedly."""
    backup = RAG_PROMPT_PATH.with_name("rag_prompt.txt.evalyx-backup")
    if backup.exists():
        RAG_PROMPT_PATH.write_text(
            backup.read_text(encoding="utf-8"), encoding="utf-8"
        )
        backup.unlink()
        print("[restore] original RAG prompt restored")
    else:
        print("[restore] no backup found — nothing to restore")


# -- output -----------------------------------------------------------------------------


def banner(title: str) -> None:
    print(f"\n{'=' * 12} {title} {'=' * 12}")


def show_run(label: str, run: dict) -> None:
    counts = run.get("counts") or {}
    print(
        f"{label}: run {run['id']}  status={run['status']}  "
        f"cases: total={counts.get('total', 0)} passed={counts.get('passed', 0)} "
        f"failed={counts.get('failed', 0)} error={counts.get('error', 0)} "
        f"executed={counts.get('executed', 0)}"
    )


def show_regression(report: dict) -> None:
    print(f"Result:           {report['result']}")
    print(f"Regression:       {report['regression_detected']}")
    b, c, d = report["baseline"], report["current"], report["deltas"]
    print(
        f"Pass rate:        {fmt_rate(b['pass_rate'])} → {fmt_rate(c['pass_rate'])} "
        f"(Δ {fmt_delta(d['pass_rate_pp'])} pp)"
    )
    print(
        f"Error rate:       {fmt_rate(b['error_rate'])} → {fmt_rate(c['error_rate'])} "
        f"(Δ {fmt_delta(d['error_rate_pp'])} pp)"
    )
    lat_b = b["latency"]["average_ms"]
    lat_c = c["latency"]["average_ms"]
    print(f"Avg latency (ms): {fmt_ms(lat_b)} → {fmt_ms(lat_c)}")

    violations = report.get("threshold_violations") or []
    if violations:
        print("Threshold violations:")
        for v in violations:
            print(f"  - {v['metric']}: Δ{v['delta']} vs threshold {v['threshold']} "
                  f"{v['unit']} — {v['detail']}")
    sections = [
        ("Newly failed cases", "newly_failed_cases"),
        ("Newly errored cases", "newly_errored_cases"),
        ("Fixed cases", "fixed_cases"),
        ("Recovered cases", "recovered_cases"),
    ]
    for title, key in sections:
        findings = report.get(key) or []
        if not findings:
            continue
        print(f"\n{title} ({len(findings)}):")
        for f in findings:
            guardrails = f.get("new_guardrail_failures") or []
            extra = f"  [{', '.join(guardrails)}]" if guardrails else ""
            print(f"  - {f['name']}{extra}")
    stable = report.get("stable_failures") or []
    if stable:
        print(f"\nStable failures: {len(stable)} (unchanged between runs)")
    comparisons = report.get("guardrail_comparison") or []
    degraded = [g for g in comparisons if (g.get("failure_rate_delta_pp") or 0) > 0]
    if degraded:
        print("\nGuardrail failure-rate increases:")
        for g in degraded:
            b = g["baseline"]["failure_rate"]
            c = g["current"]["failure_rate"]
            print(
                f"  - {g['name']}: {fmt_rate(b)} → {fmt_rate(c)} "
                f"(Δ {fmt_delta(g['failure_rate_delta_pp'])} pp)"
            )
    print(f"\nMatched cases: {report.get('matched_cases')}")
    print(f"Comparison id: {report.get('comparison_id')}")


def fmt_rate(value) -> str:
    return "n/a" if value is None else f"{value:.1f}%"


def fmt_delta(value) -> str:
    return "n/a" if value is None else f"{value:+.1f}"


def fmt_ms(value) -> str:
    return "n/a" if value is None else f"{value:.0f}"


# -- main --------------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--restore",
        action="store_true",
        help="only restore MLGPT's original RAG prompt, then exit",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse the completed baseline run recorded in the state file "
        "and only execute the degraded (current) evaluation — useful when "
        "the free-model daily quota is nearly exhausted",
    )
    args = parser.parse_args()
    if args.restore:
        restore_prompt()
        return 0

    try:
        banner("Evalyx × MLGPT Evaluation Demo")

        # 1. availability (public, unauthenticated endpoints)
        with urllib.request.urlopen(f"{MLGPT_BASE_URL}/health", timeout=10) as r:
            mlgpt = json.loads(r.read().decode("utf-8"))
        with urllib.request.urlopen(f"{EVALYX_API_URL}/health/ready", timeout=10) as r:
            evalyx = json.loads(r.read().decode("utf-8"))
        print(f"MLGPT:       {mlgpt.get('status')} (redis: {mlgpt.get('redis')})")
        print(f"Evalyx API:  {evalyx}")

        # 2-3. idempotent setup
        state = load_state()
        app_id = ensure_application(state)
        baseline_version_id = ensure_application_version(state, BASELINE_VERSION)
        ensure_dataset(state)
        print(f"Application: {APPLICATION_NAME} ({app_id})")
        print(
            f"Dataset:     {DATASET_NAME} v{DATASET_VERSION} "
            f"({len(DEMO_CASES)} cases)"
        )

        # 4. baseline run against grounded MLGPT (or reuse a completed one)
        baseline_run_id = state.get("baseline_run_id")
        if args.resume and baseline_run_id:
            code, run = api("GET", f"/evaluations/{baseline_run_id}")
            if code == 200 and run.get("status") == "completed":
                print(f"\n(resuming — reusing baseline run {baseline_run_id})")
            else:
                raise ApiError(
                    f"--resume: stored baseline run {baseline_run_id} is not "
                    f"a completed run (HTTP {code})."
                )
        else:
            banner("Baseline evaluation (grounded MLGPT)")
            preflight_mlgpt()
            baseline_run_id = submit_evaluation(state, baseline_version_id)
            baseline_run = wait_for_terminal(baseline_run_id)
            show_run("Baseline", baseline_run)
            if baseline_run["status"] != "completed":
                raise ApiError("Baseline run did not complete — aborting.")
            state["baseline_run_id"] = baseline_run_id
            save_state(state)

        # 5-6. controlled regression: temporary configuration-only change
        banner("Applying controlled regression")
        try:
            apply_degraded_prompt()
            degraded_version_id = ensure_application_version(
                state, DEGRADED_VERSION
            )
            banner("Current evaluation (degraded MLGPT behavior)")
            preflight_mlgpt()
            current_run_id = submit_evaluation(state, degraded_version_id)
            current_run = wait_for_terminal(current_run_id)
        finally:
            restore_prompt()

        show_run("Current", current_run)
        if current_run["status"] != "completed":
            raise ApiError("Current run did not complete — cannot compare.")

        # 7. Phase 8 regression comparison
        banner("Regression comparison")
        report = compare(baseline_run_id, current_run_id)
        show_regression(report)

        banner("Done")
        print("Evidence (public API, no DB access):")
        print(f"  GET /api/v1/evaluations/{baseline_run_id}/results")
        print(f"  GET /api/v1/evaluations/{current_run_id}/results")
        print(f"  GET /api/v1/regressions/{report.get('comparison_id')}")
        return 0
    except ApiError as error:
        print(f"\nDEMO FAILED: {error}", file=sys.stderr)
        restore_prompt()  # never leave MLGPT degraded
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
