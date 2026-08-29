"""Request/response schemas for evaluation submission, status, and results."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evalyx.api.schemas.applications import clean_configuration
from evalyx.db.models import CaseStatus, RunStatus


class EvaluationCreate(BaseModel):
    """Request body for ``POST /api/v1/evaluations``.

    Provider credentials are **never** accepted here: ``LLM_PROVIDER`` and
    ``OPENROUTER_API_KEY`` are server-side environment configuration. Any
    secret-looking keys inside ``configuration_snapshot`` are stripped.
    """

    application_id: UUID
    application_version_id: UUID | None = Field(
        default=None,
        description="Optional application version pinned to this run (reproducibility).",
    )
    dataset_version_id: UUID
    agent_model: str = Field(
        min_length=1,
        max_length=255,
        description="Arbitrary provider model identifier for the agent under test.",
    )
    judge_model: str | None = Field(
        default=None,
        max_length=255,
        description="Optional judge model identifier (LLM-as-a-judge scoring).",
    )
    configuration_snapshot: dict = Field(
        default_factory=dict,
        description=(
            "Non-secret execution configuration (temperature, max tokens, ...). "
            "Secret-looking keys are stripped; secrets are never accepted."
        ),
    )

    @model_validator(mode="after")
    def _validate_configuration(self) -> EvaluationCreate:
        # Size cap + secret stripping happen once, at the boundary.
        self.configuration_snapshot = clean_configuration(self.configuration_snapshot)
        return self

    @model_validator(mode="after")
    def _validate_known_execution_parameters(self) -> EvaluationCreate:
        """Light sanity bounds for well-known parameters, when present.

        The API validates request plausibility; domain services remain
        authoritative. Unknown keys pass through untouched.
        """
        config = self.configuration_snapshot
        temperature = config.get("temperature")
        if temperature is not None and not (
            isinstance(temperature, (int, float)) and 0 <= float(temperature) <= 2
        ):
            raise ValueError("configuration_snapshot.temperature must be a number in [0, 2].")
        max_tokens = config.get("max_tokens")
        if max_tokens is not None and not (
            isinstance(max_tokens, int) and 1 <= max_tokens <= 1_000_000
        ):
            raise ValueError("configuration_snapshot.max_tokens must be an integer in [1, 1000000].")
        return self


class EvaluationSubmissionResponse(BaseModel):
    """Response for a queued evaluation (HTTP 202 Accepted).

    ``task_id`` is the Celery task identifier (operational metadata). The
    authoritative evaluation state is the PostgreSQL run — poll ``status_url``.
    """

    run_id: UUID
    status: RunStatus
    task_id: str | None = Field(description="Celery task id, when enqueueing succeeded.")
    status_url: str = Field(description="Relative URL to poll for run status.")


class CaseStatusCounts(BaseModel):
    """Case-result counts by outcome for one run."""

    total: int = 0
    passed: int = 0
    failed: int = 0
    error: int = 0
    executed: int = 0


class EvaluationRunSummary(BaseModel):
    """Run status/metadata without case outputs (summary-first design).

    ``counts`` is populated on the single-run endpoint; list views omit it
    to keep responses small (it stays ``null`` there).
    """

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    status: RunStatus
    application_id: UUID
    application_version_id: UUID | None
    dataset_version_id: UUID
    agent_model: str
    judge_model: str | None
    configuration_snapshot: dict
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime
    counts: CaseStatusCounts | None = None


class GuardrailResultResponse(BaseModel):
    """A single guardrail check outcome (safe fields only).

    Guardrail metadata contains categories/counts by design — the platform
    never persists raw PII matches, API keys, or provider payloads.
    """

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    evaluation_case_result_id: UUID
    name: str
    type: str | None
    status: str
    passed: bool
    score: float | None
    reason: str | None
    metadata: dict
    created_at: datetime


class EvaluationCaseResultResponse(BaseModel):
    """One test-case execution result, with its guardrail outcomes."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    test_case_id: UUID
    input: dict
    expected_output: dict | None
    actual_output: str | None
    status: CaseStatus
    latency_ms: int | None
    error: str | None
    metrics: dict | None
    guardrail_results: list[GuardrailResultResponse]
    created_at: datetime
