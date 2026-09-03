"""Production smoke test for Evalyx (Phase 17).

Verifies the deployment path without committing any secrets: every
credential comes from the environment.

Required:
    EVALYX_API_URL  base URL of the deployment (https://... in production)
    EVALYX_TOKEN    Clerk session token for an operator account
                    (production). For local AUTH_REQUIRED=0 servers, set
                    EVALYX_ORG instead (e.g. org_smoketest — lowercase
                    alphanumerics after the `org_` prefix).

Optional:
    EVALYX_SMOKE_SUBMIT=1  also submit one evaluation + poll status
    EVALYX_APPLICATION_ID / EVALYX_DATASET_VERSION_ID for the submit step

Checks:
    /health, /health/ready, authenticated /api/v1/me, application listing,
    dataset listing, evaluation submission (opt-in), evaluation status,
    reliability endpoint. Exits non-zero on the first failure; prints one
    line per check (PASS/FAIL) with safe details only (no tokens, no
    secrets, no connection strings).
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request

API_URL = os.environ.get("EVALYX_API_URL", "").rstrip("/")
TOKEN = os.environ.get("EVALYX_TOKEN", "")
# Local development servers (AUTH_REQUIRED=0) honor the bounded dev
# organization header instead of Clerk tokens — same contract as the CLI.
DEV_ORG = os.environ.get("EVALYX_ORG", "")
SUBMIT = os.environ.get("EVALYX_SMOKE_SUBMIT", "") == "1"

FAILURES = 0


def _request(
    method: str, path: str, *, body: dict | None = None, timeout: int = 30
) -> tuple[int, object]:
    url = f"{API_URL}{path}"
    data = json.dumps(body).encode() if body is not None else None
    if TOKEN:
        auth = {"Authorization": f"Bearer {TOKEN}"}
    elif DEV_ORG:
        auth = {"X-Dev-Organization-Id": DEV_ORG}
    else:
        auth = {}
    request = urllib.request.Request(
        url, data=data, method=method,
        headers={"Accept": "application/json", **auth},
    )
    if body is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", "replace")
            return response.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as exc:
        try:
            payload: object = json.loads(exc.read().decode("utf-8", "replace"))
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            payload = None
        return exc.code, payload


def check(name: str, ok: bool, detail: str = "") -> None:
    global FAILURES
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES += 1


def main() -> int:
    if not API_URL:
        print("EVALYX_API_URL must be set (e.g. https://api.example.com).")
        return 2
    if not TOKEN and not DEV_ORG:
        print("Set EVALYX_TOKEN (production Clerk session token) or "
              "EVALYX_ORG (local AUTH_REQUIRED=0 server).")
        return 2

    status, body = _request("GET", "/health")
    check("liveness /health", status == 200 and body == {"status": "ok"},
          f"HTTP {status}")

    status, body = _request("GET", "/health/ready")
    ready = status == 200 and isinstance(body, dict) and body.get("status") == "ok"
    check("readiness /health/ready", ready, f"HTTP {status}: {body}")

    status, body = _request("GET", "/api/v1/me")
    check("authenticated /api/v1/me", status == 200, f"HTTP {status}")

    status, body = _request("GET", "/api/v1/applications?limit=1")
    check("application listing", status == 200, f"HTTP {status}")

    status, body = _request("GET", "/api/v1/datasets?limit=1")
    check("dataset listing", status == 200, f"HTTP {status}")

    if SUBMIT:
        app_id = os.environ.get("EVALYX_APPLICATION_ID", "")
        dsv_id = os.environ.get("EVALYX_DATASET_VERSION_ID", "")
        if not app_id or not dsv_id:
            print("EVALYX_SMOKE_SUBMIT=1 needs EVALYX_APPLICATION_ID and "
                  "EVALYX_DATASET_VERSION_ID.")
            return 2
        status, body = _request("POST", "/api/v1/evaluations", body={
            "application_id": app_id,
            "dataset_version_id": dsv_id,
            "agent_model": "smoke-test",
        })
        ok = status == 202 and isinstance(body, dict) and "run_id" in body
        check("evaluation submission (202)", ok, f"HTTP {status}")
        if ok:
            assert isinstance(body, dict)
            run_id = body["run_id"]
            seen = None
            for _ in range(12):
                time.sleep(5)
                status, body = _request("GET", f"/api/v1/evaluations/{run_id}")
                if isinstance(body, dict):
                    seen = body.get("status")
                    if seen in ("completed", "failed", "cancelled"):
                        break
            check("evaluation status poll", seen is not None, f"status={seen}")
            status, _ = _request("GET", f"/api/v1/evaluations/{run_id}/reliability")
            check("reliability endpoint", status == 200, f"HTTP {status}")

    print("smoke test: " + ("ALL CHECKS PASSED" if FAILURES == 0 else f"{FAILURES} FAILURES"))
    return 0 if FAILURES == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
