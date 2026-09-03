"""Operational endpoints (Phase 17/18): authenticated metrics snapshot.

``GET /api/v1/metrics`` returns the in-process operational registry
(counters + timing aggregates). Authenticated only — never public, never
scraped anonymously. Metric labels are bounded by design (the registry
rejects correlation ids, URLs, payloads, and secrets as labels), so the
snapshot is safe to expose to authenticated operators.

Multi-replica note: every API replica exposes its own in-process snapshot
under its own ``instance`` id — operators aggregate by *summing* counters
(and count-weighted averaging timings) across replicas. No cross-replica
shared state is needed for metrics; distributed coordination (rate limits,
quota windows) lives in Redis/PostgreSQL, not here.
"""

import socket
from typing import Annotated

from fastapi import APIRouter, Depends

from evalyx.api.dependencies import require_authenticated_user
from evalyx.core.metrics import metrics

router = APIRouter(tags=["operations"])


def _instance_id() -> str:
    """Bounded replica identity for the metrics envelope (not a label)."""
    return socket.gethostname()[:64] or "unknown"


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
    return {"instance": _instance_id(), "metrics": metrics.snapshot()}
