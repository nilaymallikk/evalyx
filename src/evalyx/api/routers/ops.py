"""Operational endpoints (Phase 17): authenticated metrics snapshot.

``GET /api/v1/metrics`` returns the in-process operational registry
(counters + timing aggregates). Authenticated only — never public, never
scraped anonymously. Metric labels are bounded by design (the registry
rejects correlation ids, URLs, payloads, and secrets as labels), so the
snapshot is safe to expose to authenticated operators.
"""

from typing import Annotated

from fastapi import APIRouter, Depends

from evalyx.api.dependencies import require_authenticated_user
from evalyx.core.metrics import metrics

router = APIRouter(tags=["operations"])


@router.get(
    "/metrics",
    summary="Operational metrics snapshot (authenticated)",
    description=(
        "In-process counters and timing aggregates for the API process: "
        "HTTP request counts/latency, evaluation submissions, connection "
        "tests, application invocations. Requires authentication. "
        "Worker-process metrics live in the worker logs; queue depth is "
        "observed via the worker/Celery inspection (see deployment docs)."
    ),
)
async def get_metrics(
    _auth: Annotated[object, Depends(require_authenticated_user)],
) -> dict:
    return {"metrics": metrics.snapshot()}
