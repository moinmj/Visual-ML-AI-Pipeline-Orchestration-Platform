from typing import Annotated
 
from fastapi import APIRouter, Depends
 
from backend.app.core.security import TokenData, get_current_user

from backend.app.auth.schemas import WhoAmIResponse
 
router = APIRouter(prefix="/auth", tags=["Auth"])
 
# NOTE: the dev-token endpoint is gone. Authentication is now session based and

# owned by the User-Management service — this service only VALIDATES the UM

# session cookie (`sid`) against public.user_sessions (see core/security.py).

# To get a session, log in through User-Management; the browser then carries the

# HttpOnly `sid` cookie to this API automatically.
 
 
@router.get("/whoami", response_model=WhoAmIResponse)

async def whoami(user: Annotated[TokenData, Depends(get_current_user)]):

    """Returns the identity resolved from the caller's UM session."""

    return WhoAmIResponse(

        sub=user.sub,

        tenant_id=user.tenant_id,

        roles=user.roles,

        permissions=user.permissions,

    )
 