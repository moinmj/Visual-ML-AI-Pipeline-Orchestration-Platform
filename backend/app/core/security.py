# """
# Authentication & Authorization.

# Verifies Bearer JWTs issued by an identity provider (or, in development,
# by the /api/v1/auth/dev-token endpoint) and exposes FastAPI dependencies
# for pulling the authenticated caller (and their tenant) out of the token,
# plus role/permission gates for protecting individual routes.

# Adapted from the platform's existing auth pattern - see
# `backend/app/core/config.py` for JWT_SECRET_KEY / JWT_ALGORITHM.
# """
# from datetime import datetime, timedelta, timezone
# from typing import Annotated, Any, Optional

# from fastapi import Depends, HTTPException, status
# from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
# from jose import ExpiredSignatureError, JWTError, jwt

# from backend.app.core.config import settings

# _bearer = HTTPBearer(auto_error=False)


# class TokenData:
#     """Decoded/validated representation of the caller's access token."""

#     def __init__(self, sub: str, tenant_id: int, roles: list[str], permissions: list[str]):
#         self.sub = sub
#         self.tenant_id = tenant_id
#         self.roles = roles
#         self.permissions = permissions

#     def has_role(self, *roles: str) -> bool:
#         return any(r in self.roles for r in roles)

#     def has_permission(self, *perms: str) -> bool:
#         return any(p in self.permissions for p in perms)


# def create_access_token(
#     sub: str,
#     tenant_id: int,
#     roles: Optional[list[str]] = None,
#     permissions: Optional[list[str]] = None,
#     expires_minutes: Optional[int] = None,
# ) -> str:
#     """
#     Issues a signed access token.

#     NOTE: There is currently no user/credential store in this platform, so
#     this helper does not authenticate anyone - it only signs a token for a
#     (sub, tenant_id, roles, permissions) tuple that the caller has already
#     verified some other way. It's used by the dev-only login endpoint in
#     `backend/app/auth/router.py`, and is the function you'd call from real
#     login logic once one exists.
#     """
#     now = datetime.now(timezone.utc)
#     expire = now + timedelta(minutes=expires_minutes or settings.ACCESS_TOKEN_EXPIRE_MINUTES)
#     payload: dict[str, Any] = {
#         "sub": sub,
#         "tenant_id": tenant_id,
#         "roles": roles or [],
#         "permissions": permissions or [],
#         "type": "access",
#         "iat": now,
#         "exp": expire,
#     }
#     return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


# def _decode_token(token: str) -> TokenData:
#     credentials_error = HTTPException(
#         status_code=status.HTTP_401_UNAUTHORIZED,
#         detail="Could not validate credentials",
#         headers={"WWW-Authenticate": "Bearer"},
#     )
#     try:
#         payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
#     except ExpiredSignatureError:
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Token has expired",
#             headers={"WWW-Authenticate": "Bearer"},
#         )
#     except JWTError:
#         raise credentials_error

#     sub = payload.get("sub") or payload.get("user_id") or payload.get("id") or payload.get("email") or payload.get("username")
#     tenant_id = payload.get("tenant_id") or payload.get("tenantId") or payload.get("tenant") or 1
#     roles = payload.get("roles", []) or payload.get("role", [])
#     perms = payload.get("permissions", []) or payload.get("perms", [])

#     if sub is None:
#         raise credentials_error

#     # Reject refresh tokens used as access tokens.
#     if payload.get("type") and payload.get("type") != "access":
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Invalid token type — use access token",
#             headers={"WWW-Authenticate": "Bearer"},
#         )

#     return TokenData(
#         sub=str(sub),
#         tenant_id=int(tenant_id),
#         roles=roles if isinstance(roles, list) else [roles],
#         permissions=perms if isinstance(perms, list) else [],
#     )


# def get_current_user(
#     credentials: Annotated[Optional[HTTPAuthorizationCredentials], Depends(_bearer)],
# ) -> TokenData:
#     """FastAPI dependency: validates the Bearer token, returns the caller.

#     Downstream code should always read `tenant_id` from this object (never
#     from a path/query param supplied by the client) so a caller can never
#     request another tenant's data by simply changing an ID in the URL.
#     """
#     dev_fallback = TokenData(
#         sub="dev-user",
#         tenant_id=1,
#         roles=["Tenant Admin", "Data Scientist", "ML Engineer"],
#         permissions=["workflow:read", "workflow:write", "workflow:execute", "workflow:delete", "can_ingest_data"],
#     )

#     if credentials is None:
#         if settings.DEBUG or settings.ENVIRONMENT == "development":
#             return dev_fallback
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Authentication credentials were not provided",
#             headers={"WWW-Authenticate": "Bearer"},
#         )

#     try:
#         user = _decode_token(credentials.credentials)
#         if (settings.DEBUG or settings.ENVIRONMENT == "development") and not user.roles:
#             user.roles = ["Tenant Admin", "Data Scientist", "ML Engineer"]
#         return user
#     except HTTPException:
#         if settings.DEBUG or settings.ENVIRONMENT == "development":
#             return dev_fallback
#         raise


# def require_role(*allowed_roles: str):
#     """
#     Authorization dependency factory.
#         @router.post("/x", dependencies=[Depends(require_role("Tenant Admin"))])
#     """
#     def _check(user: Annotated[TokenData, Depends(get_current_user)]) -> TokenData:
#         if not user.has_role(*allowed_roles):
#             raise HTTPException(
#                 status_code=status.HTTP_403_FORBIDDEN,
#                 detail=f"Required role not found. Allowed roles: {list(allowed_roles)}. Your roles: {user.roles}",
#             )
#         return user
#     return _check


# def require_permission(*allowed_perms: str):
#     """
#     Authorization dependency factory.
#         @router.post("/x", dependencies=[Depends(require_permission("workflow:execute"))])
#     """
#     def _check(user: Annotated[TokenData, Depends(get_current_user)]) -> TokenData:
#         if not user.has_permission(*allowed_perms):
#             raise HTTPException(
#                 status_code=status.HTTP_403_FORBIDDEN,
#                 detail=f"Required permission not found. Allowed permissions: {list(allowed_perms)}. Your permissions: {user.permissions}",
#             )
#         return user
#     return _check




"""
Authentication & Authorization.

Verifies Bearer JWTs issued by an identity provider (or, in development,
by the /api/v1/auth/dev-token endpoint) and exposes FastAPI dependencies
for pulling the authenticated caller (and their tenant) out of the token,
plus role/permission gates for protecting individual routes.

Adapted from the platform's existing auth pattern - see
`backend/app/core/config.py` for JWT_SECRET_KEY / JWT_ALGORITHM.
"""
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import ExpiredSignatureError, JWTError, jwt

from backend.app.core.config import settings

_bearer = HTTPBearer(auto_error=False)


class TokenData:
    """Decoded/validated representation of the caller's access token."""

    def __init__(self, sub: str, tenant_id: int, roles: list[str], permissions: list[str]):
        self.sub = sub
        self.tenant_id = tenant_id
        self.roles = roles
        self.permissions = permissions

    def has_role(self, *roles: str) -> bool:
        return any(r in self.roles for r in roles)

    def has_permission(self, *perms: str) -> bool:
        return any(p in self.permissions for p in perms)


def create_access_token(
    sub: str,
    tenant_id: int,
    roles: Optional[list[str]] = None,
    permissions: Optional[list[str]] = None,
    expires_minutes: Optional[int] = None,
) -> str:
    """
    Issues a signed access token.

    NOTE: There is currently no user/credential store in this platform, so
    this helper does not authenticate anyone - it only signs a token for a
    (sub, tenant_id, roles, permissions) tuple that the caller has already
    verified some other way. It's used by the dev-only login endpoint in
    `backend/app/auth/router.py`, and is the function you'd call from real
    login logic once one exists.
    """
    now = datetime.now(timezone.utc)
    expire = now + timedelta(minutes=expires_minutes or settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    payload: dict[str, Any] = {
        "sub": sub,
        "tenant_id": tenant_id,
        "roles": roles or [],
        "permissions": permissions or [],
        "type": "access",
        "iat": now,
        "exp": expire,
    }
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def _decode_token(token: str) -> TokenData:
    credentials_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
    except ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except JWTError:
        raise credentials_error

    sub = payload.get("sub") or payload.get("user_id") or payload.get("id") or payload.get("email") or payload.get("username")
    tenant_id = payload.get("tenant_id") or payload.get("tenantId") or payload.get("tenant") or 1
    roles = payload.get("roles", []) or payload.get("role", [])
    perms = payload.get("permissions", []) or payload.get("perms", [])

    if sub is None:
        raise credentials_error

    # Reject refresh tokens used as access tokens.
    if payload.get("type") and payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type — use access token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return TokenData(
        sub=str(sub),
        tenant_id=int(tenant_id),
        roles=roles if isinstance(roles, list) else [roles],
        permissions=perms if isinstance(perms, list) else [],
    )


def get_current_user(
    credentials: Annotated[Optional[HTTPAuthorizationCredentials], Depends(_bearer)],
) -> TokenData:
    """FastAPI dependency: validates the Bearer token, returns the caller.

    Downstream code should always read `tenant_id` from this object (never
    from a path/query param supplied by the client) so a caller can never
    request another tenant's data by simply changing an ID in the URL.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials were not provided",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return _decode_token(credentials.credentials)


def require_role(*allowed_roles: str):
    """
    Authorization dependency factory.
        @router.post("/x", dependencies=[Depends(require_role("Tenant Admin"))])
    """
    def _check(user: Annotated[TokenData, Depends(get_current_user)]) -> TokenData:
        if not user.has_role(*allowed_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Required role not found. Allowed roles: {list(allowed_roles)}. Your roles: {user.roles}",
            )
        return user
    return _check


def require_permission(*allowed_perms: str):
    """
    Authorization dependency factory.
        @router.post("/x", dependencies=[Depends(require_permission("workflow:execute"))])
    """
    def _check(user: Annotated[TokenData, Depends(get_current_user)]) -> TokenData:
        if not user.has_permission(*allowed_perms):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Required permission not found. Allowed permissions: {list(allowed_perms)}. Your permissions: {user.permissions}",
            )
        return user
    return _check