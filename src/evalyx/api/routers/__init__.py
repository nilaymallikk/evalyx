"""API routers package: the versioned v1 surface."""

from fastapi import APIRouter

from evalyx.api.routers import applications, datasets, evaluations, regressions

#: The complete /api/v1 surface. Health endpoints stay outside the version.
api_router = APIRouter(prefix="/api/v1")
api_router.include_router(applications.router)
api_router.include_router(datasets.router)
api_router.include_router(evaluations.router)
api_router.include_router(regressions.router)

__all__ = ["api_router"]
