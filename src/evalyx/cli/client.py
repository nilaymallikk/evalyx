"""Typed HTTP client for the Evalyx REST API (Phase 16).

One client, used by every CLI command and the TUI. Responsibilities:

- attach the stored bearer credential (Clerk session token in production;
  the dev organization header in ``AUTH_REQUIRED=0`` local mode)
- normalize backend errors into :mod:`evalyx.cli.errors` exceptions
- bounded retries for idempotent GET requests on transient failures
- never log, print, or embed the Authorization header or token anywhere

All resource access mirrors the real API contract (``/api/v1/...``): the
client is a thin, typed wrapper — no evaluation, scoring, or tenancy logic.
"""

import json
import time
from typing import Any

import httpx

from evalyx.cli import errors
from evalyx.cli.config import Config, load_config
from evalyx.cli.errors import EvalyxCLIError


class EvalyxClient:
    """Typed client for the Evalyx API. The CLI/TUI's single network path."""

    def __init__(self, config: Config | None = None, token: str | None = None) -> None:
        self.config = config or load_config()
        self._token = token

    # -- transport ------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        if self._token:
            return {"Authorization": f"Bearer {self._token}"}
        # Dev mode: an explicit organization preference travels in the
        # bounded X-Dev-Organization-Id header (honored only by a
        # backend running with AUTH_REQUIRED=0; production requires a
        # Clerk bearer token and ignores it entirely).
        if self.config.org:
            return {"X-Dev-Organization-Id": self.config.org}
        return {}

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: Any | None = None,
        params: dict[str, Any] | None = None,
        retries: int = 2,
    ) -> Any:
        """One API call; returns decoded JSON (or ``None`` for 204).

        GET retries are bounded and only for transient transport failures or
        5xx responses — never for 4xx, never for writes.
        """
        url = f"{self.config.api_url}{path}"
        attempts = 0
        while True:
            attempts += 1
            try:
                response = httpx.request(
                    method,
                    url,
                    headers=self._headers(),
                    json=json_body,
                    params=params,
                    timeout=self.config.timeout,
                )
            except httpx.TimeoutException as exc:
                if method == "GET" and attempts <= retries:
                    time.sleep(min(0.5 * attempts, 2.0))
                    continue
                raise errors.APIConnectionError(
                    "Request to the Evalyx API timed out.",
                    hint=f"Is the API running at {self.config.api_url}?",
                ) from exc
            except httpx.HTTPError as exc:
                if method == "GET" and attempts <= retries:
                    time.sleep(min(0.5 * attempts, 2.0))
                    continue
                raise errors.APIConnectionError(
                    "Unable to connect to the Evalyx API.",
                    hint=f"Check --api-url (currently {self.config.api_url}).",
                ) from exc

            if response.status_code >= 500 and method == "GET" and attempts <= retries:
                time.sleep(min(0.5 * attempts, 2.0))
                continue

            if response.status_code == 204:
                return None
            if response.status_code >= 400:
                raise self._error_from(response, method, path)
            if not response.content:
                return None
            try:
                return response.json()
            except json.JSONDecodeError as exc:
                raise errors.APIError(
                    "The Evalyx API returned a malformed response."
                ) from exc

    @staticmethod
    def _error_from(response: httpx.Response, method: str, path: str) -> EvalyxCLIError:
        """Normalize one error response without exposing headers or payloads."""
        subject = path.strip("/").split("/")[-1] or "resource"
        exc = httpx.HTTPStatusError(
            f"{response.status_code}", request=httpx.Request(method, path), response=response
        )
        return errors.normalize_http_error(exc, subject)

    # -- health -----------------------------------------------------------------

    def health(self) -> dict:
        return self.request("GET", "/health")

    def me(self) -> dict:
        return self.request("GET", "/api/v1/me")

    # -- applications -----------------------------------------------------------

    def applications_list(self, limit: int = 50, offset: int = 0) -> dict:
        return self.request(
            "GET", "/api/v1/applications", params={"limit": limit, "offset": offset}
        )

    def applications_get(self, application_id: str) -> dict:
        return self.request("GET", f"/api/v1/applications/{application_id}")

    def applications_create(
        self,
        name: str,
        connection_type: str,
        description: str | None = None,
        secret: str | None = None,
    ) -> dict:
        body: dict[str, Any] = {"name": name, "connection_type": connection_type}
        if description is not None:
            body["description"] = description
        if secret is not None:
            body["secret"] = secret
        return self.request("POST", "/api/v1/applications", json_body=body)

    def applications_update(
        self, application_id: str, name: str | None = None, description: str | None = None
    ) -> dict:
        body: dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if description is not None:
            body["description"] = description
        return self.request(
            "PATCH", f"/api/v1/applications/{application_id}", json_body=body
        )

    def applications_delete(self, application_id: str) -> None:
        self.request("DELETE", f"/api/v1/applications/{application_id}")

    def applications_versions(self, application_id: str) -> dict:
        return self.request("GET", f"/api/v1/applications/{application_id}/versions")

    def applications_create_version(
        self,
        application_id: str,
        version: str,
        connection: dict | None = None,
        configuration: dict | None = None,
        description: str | None = None,
    ) -> dict:
        body: dict[str, Any] = {"version": version}
        if connection is not None:
            body["connection"] = connection
        if configuration is not None:
            body["configuration"] = configuration
        if description is not None:
            body["description"] = description
        return self.request(
            "POST", f"/api/v1/applications/{application_id}/versions", json_body=body
        )

    def applications_rotate_secret(self, application_id: str, secret: str) -> dict:
        return self.request(
            "PATCH",
            f"/api/v1/applications/{application_id}/connection",
            json_body={"secret": secret},
        )

    def applications_test(
        self, application_id: str, prompt: str | None = None
    ) -> dict:
        body: dict[str, Any] = {}
        if prompt is not None:
            body["prompt"] = prompt
        return self.request(
            "POST", f"/api/v1/applications/{application_id}/test", json_body=body
        )

    # -- datasets ----------------------------------------------------------------

    def datasets_list(self, limit: int = 50, offset: int = 0) -> dict:
        return self.request(
            "GET", "/api/v1/datasets", params={"limit": limit, "offset": offset}
        )

    def datasets_get(self, dataset_id: str) -> dict:
        return self.request("GET", f"/api/v1/datasets/{dataset_id}")

    def datasets_create(self, name: str, description: str | None = None) -> dict:
        body: dict[str, Any] = {"name": name}
        if description is not None:
            body["description"] = description
        return self.request("POST", "/api/v1/datasets", json_body=body)

    def datasets_versions(self, dataset_id: str) -> dict:
        return self.request("GET", f"/api/v1/datasets/{dataset_id}/versions")

    def datasets_create_version(
        self, dataset_id: str, version: int, description: str | None = None
    ) -> dict:
        body: dict[str, Any] = {"version": version}
        if description is not None:
            body["description"] = description
        return self.request(
            "POST", f"/api/v1/datasets/{dataset_id}/versions", json_body=body
        )

    def datasets_add_case(
        self,
        dataset_id: str,
        version: int,
        name: str,
        input: dict,
        expected_output: dict | None = None,
        context: dict | None = None,
    ) -> dict:
        body: dict[str, Any] = {"name": name, "input": input}
        if expected_output is not None:
            body["expected_output"] = expected_output
        if context is not None:
            body["context"] = context
        return self.request(
            "POST",
            f"/api/v1/datasets/{dataset_id}/versions/{version}/cases",
            json_body=body,
        )

    # -- evaluations ---------------------------------------------------------------

    def evaluations_list(
        self,
        limit: int = 20,
        offset: int = 0,
        application_id: str | None = None,
    ) -> dict:
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if application_id is not None:
            params["application_id"] = application_id
        return self.request("GET", "/api/v1/evaluations", params=params)

    def evaluations_get(self, run_id: str) -> dict:
        return self.request("GET", f"/api/v1/evaluations/{run_id}")

    def evaluations_status(self, run_id: str) -> str:
        return str(self.request("GET", f"/api/v1/evaluations/{run_id}/status"))

    def evaluations_submit(
        self,
        application_id: str,
        dataset_version_id: str,
        agent_model: str,
        judge_model: str | None = None,
        configuration_snapshot: dict | None = None,
        application_version_id: str | None = None,
    ) -> dict:
        body: dict[str, Any] = {
            "application_id": application_id,
            "dataset_version_id": dataset_version_id,
            "agent_model": agent_model,
        }
        if judge_model is not None:
            body["judge_model"] = judge_model
        if configuration_snapshot:
            body["configuration_snapshot"] = configuration_snapshot
        if application_version_id is not None:
            body["application_version_id"] = application_version_id
        return self.request("POST", "/api/v1/evaluations", json_body=body)

    def evaluations_results(
        self, run_id: str, limit: int = 50, offset: int = 0
    ) -> dict:
        return self.request(
            "GET",
            f"/api/v1/evaluations/{run_id}/results",
            params={"limit": limit, "offset": offset},
        )

    def evaluations_guardrails(self, run_id: str, limit: int = 100) -> dict:
        return self.request(
            "GET", f"/api/v1/evaluations/{run_id}/guardrails", params={"limit": limit}
        )

    def evaluations_reliability(self, run_id: str) -> dict:
        return self.request("GET", f"/api/v1/evaluations/{run_id}/reliability")

    # -- regressions -----------------------------------------------------------------

    def regressions_compare(
        self,
        baseline_run_id: str,
        current_run_id: str,
        thresholds: dict | None = None,
    ) -> dict:
        body: dict[str, Any] = {
            "baseline_run_id": baseline_run_id,
            "current_run_id": current_run_id,
        }
        if thresholds is not None:
            body["thresholds"] = thresholds
        return self.request("POST", "/api/v1/regressions", json_body=body)

    def regressions_get(self, comparison_id: str) -> dict:
        return self.request("GET", f"/api/v1/regressions/{comparison_id}")


__all__ = ["EvalyxCLIError", "EvalyxClient"]
