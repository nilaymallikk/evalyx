"""Request/response schemas for the regression API.

The response model **is** the domain ``RegressionReport`` (a typed Pydantic
model, not an ORM object) — the authoritative comparison implementation
lives in :mod:`evalyx.evaluation.regression`; the API only forwards it.
"""

from uuid import UUID

from pydantic import BaseModel, Field

from evalyx.evaluation.regression.models import RegressionThresholds


class RegressionCompareRequest(BaseModel):
    """Request body for ``POST /api/v1/regressions``.

    ``thresholds`` is optional; omitted fields fall back to the defaults of
    :class:`RegressionThresholds` (2.0 pp rate thresholds, 20 % latency).
    """

    baseline_run_id: UUID = Field(description="Reference (older/known-good) run.")
    current_run_id: UUID = Field(description="Run under evaluation.")
    thresholds: RegressionThresholds = Field(
        default_factory=RegressionThresholds,
        description="Explicit threshold policy; snapshot-persisted with the artifact.",
    )
