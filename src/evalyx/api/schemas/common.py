"""Shared API schemas: pagination envelope."""

from pydantic import BaseModel, ConfigDict

#: Maximum page size for every paginated endpoint (DoS guard).
MAX_PAGE_SIZE = 200
#: Default page size.
DEFAULT_PAGE_SIZE = 50


class Page[ItemT](BaseModel):
    """Stable pagination envelope used by all list endpoints.

    ``total`` is the full (unpaginated) item count so clients can compute
    pages; ``limit``/``offset`` echo the request. Ordering is documented per
    endpoint and is deterministic.
    """

    model_config = ConfigDict(from_attributes=True)

    items: list[ItemT]
    total: int
    limit: int
    offset: int
