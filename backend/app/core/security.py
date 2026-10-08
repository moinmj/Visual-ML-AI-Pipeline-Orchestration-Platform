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

import ipaddress

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import ExpiredSignatureError, JWTError, jwt

from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.core.tenant_context import set_current_tenant_id

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
    # No fallback tenant: a token without a tenant claim is rejected, otherwise
    # every such token would silently land in (and see the data of) tenant 1.
    tenant_id = payload.get("tenant_id") or payload.get("tenantId") or payload.get("tenant")
    roles = payload.get("roles", []) or payload.get("role", [])
    perms = payload.get("permissions", []) or payload.get("perms", [])

    if sub is None or tenant_id is None:
        raise credentials_error
    try:
        tenant_id = int(tenant_id)
    except (TypeError, ValueError):
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
        tenant_id=tenant_id,
        roles=roles if isinstance(roles, list) else [roles],
        permissions=perms if isinstance(perms, list) else [],
    )


async def get_current_user(
    credentials: Annotated[Optional[HTTPAuthorizationCredentials], Depends(_bearer)],
) -> TokenData:
    """FastAPI dependency: validates the Bearer token, returns the caller.

    Downstream code should always read `tenant_id` from this object (never
    from a path/query param supplied by the client) so a caller can never
    request another tenant's data by simply changing an ID in the URL.

    No dev fallback: a missing, malformed, or expired token always raises
    401, in every environment.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials were not provided",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = _decode_token(credentials.credentials)
    # Make the caller's tenant available to the DB tenant guard, job manager and
    # executors for the rest of this request. (async on purpose: a sync
    # dependency runs in a worker thread and the value would not reach the endpoint.)
    set_current_tenant_id(user.tenant_id)
    return user


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

def _csv_setting(value: Optional[str]) -> list[str]:
    return [v.strip() for v in (value or "").split(",") if v.strip()]


def client_ip(request: Request) -> str:
    """
    The caller's IP. X-Forwarded-For is only believed when the direct peer is one of
    TRUSTED_PROXY_IPS (our own reverse proxy); the entry used is the right-most one that
    is not itself a trusted proxy, so a client cannot spoof it by sending its own header.
    """
    peer = request.client.host if request.client else ""
    trusted = set(_csv_setting(settings.TRUSTED_PROXY_IPS))
    if peer in trusted:
        forwarded = [p.strip() for p in request.headers.get("x-forwarded-for", "").split(",") if p.strip()]
        for hop in reversed(forwarded):
            if hop not in trusted:
                return hop
    return peer


def ip_allowed(ip: str) -> bool:
    """True when EXPORT_ALLOWED_IPS is empty, or the IP is inside one of its addresses / ranges."""
    entries = _csv_setting(settings.EXPORT_ALLOWED_IPS)
    if not entries:
        return True
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    for entry in entries:
        try:
            if addr in ipaddress.ip_network(entry, strict=False):
                return True
        except ValueError:
            continue
    return False


async def require_export_access(
    request: Request,
    user: Annotated[TokenData, Depends(get_current_user)],
) -> TokenData:
    """
    Gate for every endpoint that exports / downloads data:
      - the caller needs one of EXPORT_ALLOWED_ROLES, and
      - when EXPORT_ALLOWED_IPS is set, the request must come from an allowed IP.
    Every allowed export is written to the log (who, from where, what).
    """
    allowed_roles = _csv_setting(settings.EXPORT_ALLOWED_ROLES)
    if not user.has_role(*allowed_roles):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Exporting data requires one of these roles: {allowed_roles}. Your roles: {user.roles}",
        )
    ip = client_ip(request)
    if not ip_allowed(ip):
        logger.warning(f"Export blocked for '{user.sub}' (tenant {user.tenant_id}): IP {ip} not in allow-list")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Exporting data is not allowed from your network.",
        )
    logger.info(f"EXPORT user='{user.sub}' tenant={user.tenant_id} ip={ip} {request.method} {request.url.path}")
    return user