"""Local workspace endpoint: ``GET /api/v1/me``.

Answers "which workspace is this" — consumed by the CLI's ``evalyx
whoami``. No login required. Display information only.
"""

from fastapi import APIRouter

from evalyx.api.schemas.me import MeResponse

router = APIRouter(tags=["me"])


@router.get(
    "/me",
    response_model=MeResponse,
    summary="Describe the local workspace",
)
async def get_me() -> MeResponse:
    from evalyx.api.me import describe_caller

    return await describe_caller()


__all__ = ["router"]
