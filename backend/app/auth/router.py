from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from backend.app.core.config import settings
from backend.app.core.security import TokenData, create_access_token, get_current_user
from backend.app.auth.schemas import DevTokenRequest, TokenResponse, WhoAmIResponse

router = APIRouter(prefix="/auth", tags=["Auth"])


@router.post("/dev-token", response_model=TokenResponse)
async def issue_dev_token(payload: DevTokenRequest):
    """
    Mint an access token for local development / testing ONLY.

    There is no user store in this platform yet, so this endpoint trusts
    whatever (sub, tenant_id, roles, permissions) it is given and signs a
    token for it. It is disabled whenever DEBUG=False so it can never be
    reached in a real deployment - swap it out for real login/SSO that
    verifies credentials before calling `create_access_token(...)`.
    """
    if not settings.DEBUG:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Not found",
        )

    token = create_access_token(
        sub=payload.sub,
        tenant_id=payload.tenant_id,
        roles=payload.roles,
        permissions=payload.permissions,
    )
    return TokenResponse(
        access_token=token,
        expires_in_minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES,
    )


@router.get("/whoami", response_model=WhoAmIResponse)
async def whoami(user: Annotated[TokenData, Depends(get_current_user)]):
    """Returns the identity encoded in the caller's bearer token."""
    return WhoAmIResponse(
        sub=user.sub,
        tenant_id=user.tenant_id,
        roles=user.roles,
        permissions=user.permissions,
    )
