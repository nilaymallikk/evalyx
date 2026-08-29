"""Regression comparison endpoints.

All comparison logic lives in :class:`evalyx.evaluation.regression.service.
RegressionService` — this router only forwards requests and responses. A
detected regression is a **successful HTTP 200** whose body says
``result: REGRESSION_DETECTED``; it is a finding, not a server error.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends

from evalyx.api.dependencies import get_regression_service
from evalyx.api.schemas.regressions import RegressionCompareRequest
from evalyx.evaluation.regression.models import RegressionReport
from evalyx.evaluation.regression.service import RegressionService

router = APIRouter(prefix="/regressions", tags=["regressions"])


@router.post(
    "",
    response_model=RegressionReport,
    summary="Compare two completed runs for regressions",
    description=(
        "Synchronous, deterministic comparison (no LLM involved). Idempotent: "
        "the same run pair with the same threshold policy returns the "
        "original persisted artifact. A detected regression is still a 200 "
        "response — inspect `result` and `threshold_violations`."
    ),
    responses={
        200: {"description": "Comparison persisted (or existing artifact returned)."},
        400: {"description": "The runs cannot be compared (status, application, or dataset mismatch)."},
        404: {"description": "A referenced run does not exist."},
        422: {"description": "Invalid request body."},
    },
)
async def compare_runs(
    payload: RegressionCompareRequest,
    service: Annotated[RegressionService, Depends(get_regression_service)],
) -> RegressionReport:
    return await service.compare_runs(
        payload.baseline_run_id, payload.current_run_id, payload.thresholds
    )


@router.get(
    "/{comparison_id}",
    response_model=RegressionReport,
    summary="Retrieve a persisted regression report",
    responses={404: {"description": "Comparison not found."}},
)
async def get_regression(
    comparison_id: uuid.UUID,
    service: Annotated[RegressionService, Depends(get_regression_service)],
) -> RegressionReport:
    return await service.get_report(comparison_id)
