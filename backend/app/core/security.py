"""
Authentication & Authorization — SESSION based (matches User-Management).

UM no longer issues JWTs. It creates a row in public.user_sessions and sets an
HttpOnly cookie `sid` (value = secrets.token_urlsafe(32)); only sha256(sid) is
stored (session_hash). This service validates a request by looking that row up
DIRECTLY in the UM database (read-only), exactly the way UM does:

  live  ⟺  row exists AND revoked_at IS NULL
           AND expires_at > now
           AND last_seen_at + idle(30m) > now
           AND user is active / not deleted
           AND user.token_version == session.token_version
           AND (tenant is active, when the session has a tenant)

roles/permissions come straight off the session row (jsonb), so nothing is
re-resolved. CSRF (unsafe methods, cookie flow) = X-CSRF-Token must equal
HMAC(SESSION_SECRET_KEY, "csrf:" + sid) — the same scheme UM uses.

Config needed (backend/app/core/config.py Settings + .env):
    UM_DATABASE_URL      DSN to user_management_db (asyncpg), e.g.
                         postgresql+asyncpg://dbuser:Classic%40123@127.0.0.1:5432/user_management_db
                         (URL-encode specials in the password: '@' → '%40')
    SESSION_SECRET_KEY   MUST equal UM's SECRET_KEY (used for the CSRF HMAC)
    SESSION_IDLE_MINUTES = 30
    SESSION_CSRF_ENABLED = True
Requires: pip install asyncpg

Public API (unchanged from the JWT version so routers keep working):
    TokenData, get_current_user, require_role, require_permission,
    require_export_access, client_ip, ip_allowed
New helper for the WS / media-download flows (they pass the sid themselves):
    resolve_session(sid) -> TokenData
"""
import hashlib
import hmac
import ipaddress
from datetime import datetime, timedelta, timezone
from typing import Annotated, List, Optional, Tuple

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.core.tenant_context import set_current_tenant_id, set_current_user_id

COOKIE_NAME    = "sid"
CSRF_HEADER    = "X-CSRF-Token"
SESSION_HEADER = "X-Session-Id"          # service clients (Airflow) send the sid here
SAFE_METHODS   = {"GET", "HEAD", "OPTIONS"}
IDLE_TIMEOUT   = timedelta(minutes=int(getattr(settings, "SESSION_IDLE_MINUTES", 30)))
_DB_TOUCH_INTERVAL = timedelta(seconds=60)

# ── read-only async connection to the User-Management database ────────────────
_um_sessionmaker: Optional[async_sessionmaker] = None


def _um_sm() -> async_sessionmaker:
    global _um_sessionmaker
    if _um_sessionmaker is None:
        url = getattr(settings, "UM_DATABASE_URL", None)
        if not url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="UM_DATABASE_URL is not configured.",
            )
        engine = create_async_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=5)
        _um_sessionmaker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    return _um_sessionmaker


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _csrf_token_for(sid: str) -> str:
    return hmac.new(settings.SESSION_SECRET_KEY.encode(), f"csrf:{sid}".encode(),
                    hashlib.sha256).hexdigest()


class TokenData:
    """Decoded/validated representation of the caller's session."""

    def __init__(self, sub: str, tenant_id: int, roles: list[str], permissions: list[str],
                 email: Optional[str] = None, sid: Optional[str] = None):
        self.sub = sub
        self.tenant_id = tenant_id
        self.roles = roles
        self.permissions = permissions
        self.email = email
        self.sid = sid

    def has_role(self, *roles: str) -> bool:
        return any(r in self.roles for r in roles)

    def has_permission(self, *perms: str) -> bool:
        return any(p in self.permissions for p in perms)


_SESSION_SQL = text("""
    SELECT s.id                AS session_id,
           s.user_id           AS user_id,
           s.tenant_id         AS tenant_id,
           s.roles             AS roles,
           s.permissions       AS permissions,
           s.token_version     AS session_token_version,
           s.revoked_at        AS revoked_at,
           s.expires_at        AS expires_at,
           s.last_seen_at      AS last_seen_at,
           u.is_active         AS user_active,
           u.is_deleted        AS user_deleted,
           u.token_version     AS user_token_version,
           u.email             AS email,
           t.is_active         AS tenant_active
    FROM user_sessions s
    JOIN user_details u  ON u.user_id   = s.user_id
    LEFT JOIN tenant_master t ON t.tenant_id = s.tenant_id
    WHERE s.session_hash = :h
""")

_TOUCH_SQL = text(
    "UPDATE user_sessions SET last_seen_at = :now "
    "WHERE session_hash = :h AND revoked_at IS NULL"
)


async def _load_session(sid: str) -> Optional[dict]:
    """Return the validated session row (as a dict) for a raw cookie value, or None."""
    h = _hash(sid)
    now = _utcnow()
    try:
        async with _um_sm()() as db:
            row = (await db.execute(_SESSION_SQL, {"h": h})).mappings().first()
            if row is None:
                return None
            if row["revoked_at"] is not None:
                return None
            if row["expires_at"] <= now:
                return None
            if row["last_seen_at"] + IDLE_TIMEOUT <= now:
                return None
            if not row["user_active"] or row["user_deleted"]:
                return None
            if row["user_token_version"] != row["session_token_version"]:
                return None
            if row["tenant_id"] is not None and row["tenant_active"] is False:
                return None

            # keep the shared session alive across services (throttled write)
            if now - row["last_seen_at"] >= _DB_TOUCH_INTERVAL:
                await db.execute(_TOUCH_SQL, {"now": now, "h": h})
                await db.commit()
            return dict(row)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Session lookup against UM DB failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication backend unavailable.",
        )


def _unauth(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail)


def _row_to_token_data(row: dict) -> TokenData:
    tenant_id = row["tenant_id"]
    if tenant_id is None:
        # Same stance as before: a session without a tenant is rejected so it can
        # never silently land in (and see) another tenant's data.
        raise _unauth("Tenant missing on session.")
    roles = row["roles"] or []
    perms = row["permissions"] or []
    return TokenData(
        sub=str(row["user_id"]),
        tenant_id=int(tenant_id),
        roles=list(roles) if isinstance(roles, list) else [roles],
        permissions=list(perms) if isinstance(perms, list) else [],
        email=row["email"],
        sid=str(row["session_id"]),
    )


async def resolve_session(sid: str, *, set_context: bool = True) -> TokenData:
    """
    Validate a RAW session id (the cookie value) and return the caller.

    Used by flows that carry the sid themselves instead of as the normal cookie:
    the execute WebSocket and the media-preview download. No CSRF here (those are
    not cookie-borne browser form posts). Raises 401 if the session isn't live.
    """
    if not sid:
        raise _unauth("Authentication credentials were not provided")
    row = await _load_session(sid)
    if row is None:
        raise _unauth("Session expired or not logged in.")
    user = _row_to_token_data(row)
    if set_context:
        set_current_tenant_id(user.tenant_id)
        set_current_user_id(user.sub)
    return user


def _extract_sid(request: Request) -> Tuple[Optional[str], bool]:
    """Return (sid, from_cookie). Cookie first (browser), then header / Bearer (services)."""
    sid = request.cookies.get(COOKIE_NAME)
    if sid:
        return sid, True
    hdr = request.headers.get(SESSION_HEADER)
    if hdr:
        return hdr.strip(), False
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip(), False
    return None, False


async def get_current_user(request: Request) -> TokenData:
    """
    FastAPI dependency: validates the session cookie/header, returns the caller.

    Downstream code always reads tenant_id from this object (never from a path/
    query param) so a caller can't request another tenant's data by changing an ID.
    """
    sid, from_cookie = _extract_sid(request)
    if not sid:
        raise _unauth("Authentication credentials were not provided")

    user = await resolve_session(sid)   # loads, validates, sets tenant/user context

    # CSRF — only for cookie-borne unsafe requests (browser). Service clients that
    # send the sid via header/Bearer aren't subject to CSRF.
    if (from_cookie
            and getattr(settings, "SESSION_CSRF_ENABLED", True)
            and request.method not in SAFE_METHODS):
        sent = request.headers.get(CSRF_HEADER) or ""
        if not hmac.compare_digest(sent, _csrf_token_for(sid)):
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Missing or invalid CSRF token.")

    return user


def require_role(*allowed_roles: str):
    """
    Authorization dependency factory.
        @router.post("/x", dependencies=[Depends(require_role("Tenant Admin"))])
    """
    async def _check(user: Annotated[TokenData, Depends(get_current_user)]) -> TokenData:
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
    async def _check(user: Annotated[TokenData, Depends(get_current_user)]) -> TokenData:
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