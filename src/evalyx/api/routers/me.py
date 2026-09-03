"""Caller identity endpoint (Phase 16): ``GET /api/v1/me``.

A single authenticated endpoint answering "who am I, which organizations
can I use" — the CLI's ``evalyx whoami`` and ``evalyx login`` consume it.
Identity always comes from the verified token; the endpoint adds only the
Clerk-Frontend-API lookups (email address, organization list) that a
session token payload does not carry. Safe fields only — no token
contents, no Clerk secrets, no raw membership payloads.
"""

from typing import Annotated

from fastapi import APIRouter, Depends

from evalyx.api.auth import AuthContext, TokenVerifier
from evalyx.api.dependencies import get_token_verifier, require_authenticated_user
from evalyx.api.schemas.me import MeResponse, OrganizationSummary

router = APIRouter(tags=["me"])


@router.get(
    "/me",
    response_model=MeResponse,
    summary="Describe the authenticated caller",
    description=(
        "Identity from the verified Clerk session token plus the caller's "
        "Clerk organizations and the active organization's role. Contains "
        "only display information (email, org names/roles) — never tokens "
        "or secrets."
    ),
    responses={401: {"description": "Authentication failed."}},
)
async def get_me(
    auth: Annotated[AuthContext, Depends(require_authenticated_user)],
    verifier: Annotated[TokenVerifier, Depends(get_token_verifier)],
) -> MeResponse:
    from evalyx.api.me import describe_caller

    return await describe_caller(auth, verifier)


__all__ = ["OrganizationSummary", "router"]
