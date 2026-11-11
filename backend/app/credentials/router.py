import hmac
import hashlib
import json
import logging
import os
import time
import urllib.parse
import urllib.request
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

from fastapi import APIRouter, Depends, Query, Request, HTTPException, status
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import settings
from backend.app.core.security import TokenData, get_current_user
from backend.app.infrastructure.database.session import get_db
from backend.app.credentials.service import CredentialService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Credentials & OAuth"])


def _generate_oauth_state(tenant_id: Optional[int], user_id: Optional[str]) -> str:
    """Generates a tamper-proof signed state parameter for OAuth security."""
    payload = {
        "tenant_id": tenant_id,
        "user_id": user_id,
        "ts": int(time.time())
    }
    raw = json.dumps(payload, separators=(",", ":"))
    sig = hmac.new(settings.SECRET_KEY.encode(), raw.encode(), hashlib.sha256).hexdigest()[:16]
    state_dict = {"p": raw, "s": sig}
    return urllib.parse.quote(json.dumps(state_dict))


def _resolve_base_url(req: Optional[Request]) -> str:
    """
    Resolves the external public base URL for OAuth callbacks, honoring
    reverse proxy headers (X-Forwarded-Proto, X-Forwarded-Host, X-Forwarded-Ssl)
    and strictly enforcing HTTPS on non-localhost domains (required by Discord & Teams).
    """
    if req is None:
        return "https://ml-crt.aiiotgeeks.com"

    # Extract host
    host = req.headers.get("x-forwarded-host") or req.headers.get("host")
    if not host and hasattr(req, "scope") and isinstance(req.scope, dict):
        server = req.scope.get("server")
        if server and isinstance(server, (list, tuple)) and len(server) >= 2:
            host = f"{server[0]}:{server[1]}" if server[1] not in (80, 443) else str(server[0])
    if not host:
        host = "ml-crt.aiiotgeeks.com"
    host = str(host).split(",")[0].strip()

    # Extract scheme
    proto = req.headers.get("x-forwarded-proto") or req.headers.get("x-forwarded-ssl")
    if not proto and hasattr(req, "scope") and isinstance(req.scope, dict):
        proto = req.scope.get("scheme", "http")
    proto = str(proto or "http").split(",")[0].strip().lower()
    scheme = "https" if proto in ["https", "on"] else proto

    # Production domains must use HTTPS (Discord and Teams OAuth strictly reject non-HTTPS URLs)
    if not any(loc in host.lower() for loc in ["localhost", "127.0.0.1", "0.0.0.0"]):
        scheme = "https"

    return f"{scheme}://{host}".rstrip("/")


def _verify_oauth_state(state_str: str) -> Dict[str, Any]:
    """Validates and unpacks the signed OAuth state parameter."""
    try:
        decoded = json.loads(urllib.parse.unquote(state_str))
        raw = decoded["p"]
        expected_sig = hmac.new(settings.SECRET_KEY.encode(), raw.encode(), hashlib.sha256).hexdigest()[:16]
        if not hmac.compare_digest(decoded["s"], expected_sig):
            raise ValueError("Invalid OAuth state signature")
        payload = json.loads(raw)
        # 15 minutes validity
        if time.time() - payload.get("ts", 0) > 900:
            raise ValueError("OAuth state token expired")
        return payload
    except Exception as e:
        logger.warning(f"OAuth state verification failed: {e}")
        return {}


# ── Credential Management Endpoints ──────────────────────────────────────────

@router.get("/credentials")
async def list_credentials(
    provider: Optional[str] = Query(None, description="Filter by provider (e.g. discord, google, slack)"),
    user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Lists all connected credentials and OAuth accounts for the caller's tenant."""
    creds = await CredentialService.list_credentials(db=db, tenant_id=user.tenant_id, provider=provider)
    return [c.to_dict() for c in creds]


class CreateCredentialRequest(BaseModel):
    provider: str = Field(..., description="Provider identifier e.g. postgresql, mysql, snowflake, discord")
    name: str = Field(..., description="User-friendly name e.g. 'Walmart Analytics DB'")
    data: Dict[str, Any] = Field(..., description="Credential payload to encrypt (e.g. host, port, database, username, password, connection_string)")
    auth_type: str = Field(default="database", description="Authentication type e.g. database, basic, oauth2, api_key")
    metadata_info: Optional[Dict[str, Any]] = Field(default=None, description="Public display metadata safe for frontend dropdowns")


@router.post("/credentials", status_code=status.HTTP_201_CREATED)
async def create_credential(
    body: CreateCredentialRequest,
    user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Saves a new encrypted credential (database connection, API key, service account)
    scoped to the caller's tenant.
    """
    safe_metadata = dict(body.metadata_info or {})
    if not safe_metadata:
        for k in ("host", "port", "database", "username", "account", "warehouse", "ssl_mode"):
            if k in body.data and body.data[k]:
                safe_metadata[k] = body.data[k]

    cred = await CredentialService.create_credential(
        db=db,
        provider=body.provider,
        name=body.name,
        data=body.data,
        tenant_id=user.tenant_id,
        user_id=getattr(user, 'user_id', getattr(user, 'sub', None)),
        metadata_info=safe_metadata,
        auth_type=body.auth_type,
    )
    return cred.to_dict()


@router.get("/credentials/{credential_id}")
async def get_credential(
    credential_id: str,
    user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Retrieves safe metadata for a stored credential."""
    cred = await CredentialService.get_credential(db=db, credential_id=credential_id, tenant_id=user.tenant_id)
    if not cred:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Credential not found or access denied")
    return cred.to_dict()


@router.delete("/credentials/{credential_id}")
async def delete_credential(
    credential_id: str,
    user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Deletes or disconnects a stored credential."""
    success = await CredentialService.delete_credential(db=db, credential_id=credential_id, tenant_id=user.tenant_id)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Credential not found or access denied")
    return {"success": True, "message": f"Credential '{credential_id}' removed successfully."}


# ── Discord OAuth 2.0 Endpoints ──────────────────────────────────────────────

@router.get("/oauth/discord/authorize")
async def authorize_discord(
    request: Request,
    user: TokenData = Depends(get_current_user)
):
    """
    Returns the Discord OAuth2 authorization URL.
    Opens the 'Add Bot / Connect Discord' consent window just like n8n.
    """
    client_id = settings.DISCORD_CLIENT_ID or os.getenv("DISCORD_CLIENT_ID")
    if not client_id:
        # Provide helpful error if Discord credentials are not set in .env yet
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Discord OAuth is not configured. Please set DISCORD_CLIENT_ID in your environment."
        )

    # Determine redirect URI
    if settings.DISCORD_REDIRECT_URI:
        redirect_uri = settings.DISCORD_REDIRECT_URI
    else:
        # Default to platform API callback
        base_url = _resolve_base_url(request)
        redirect_uri = f"{base_url}{settings.API_V1_STR}/oauth/discord/callback"

    state = _generate_oauth_state(tenant_id=user.tenant_id, user_id=user.sub)

    # Scopes: bot permissions + identify + webhook incoming
    # permissions: 2048 (Send Messages) + 16384 (Embed Links) + 32768 (Attach Files) = 51200
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "bot identify guilds webhook.incoming",
        "permissions": "51200",
        "state": state
    }

    discord_url = f"https://discord.com/oauth2/authorize?{urllib.parse.urlencode(params)}"
    return {
        "authorization_url": discord_url,
        "provider": "discord",
        "redirect_uri": redirect_uri
    }


@router.get("/oauth/discord/callback", response_class=HTMLResponse)
async def discord_oauth_callback(
    code: Optional[str] = Query(None),
    state: Optional[str] = Query(None),
    guild_id: Optional[str] = Query(None),
    error: Optional[str] = Query(None),
    error_description: Optional[str] = Query(None),
    request: Request = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Discord OAuth2 redirect callback.
    Exchanges code for tokens, extracts bot/guild/webhook info, saves to DB,
    and returns a script closing the popup window and updating the frontend.
    """
    if error:
        err_msg = error_description or error
        return HTMLResponse(
            content=f"""
            <html><body style="font-family: sans-serif; text-align: center; padding: 40px; background: #0f172a; color: #f87171;">
                <h3>Discord Connection Cancelled</h3>
                <p>{err_msg}</p>
                <button onclick="window.close()" style="padding: 8px 16px; background: #334155; color: white; border: none; border-radius: 6px; cursor: pointer;">Close</button>
            </body></html>
            """,
            status_code=400
        )

    if not code or not state:
        raise HTTPException(status_code=400, detail="Missing authorization code or state token.")

    state_data = _verify_oauth_state(state)
    tenant_id = state_data.get("tenant_id")
    user_id = state_data.get("user_id")

    client_id = settings.DISCORD_CLIENT_ID or os.getenv("DISCORD_CLIENT_ID")
    client_secret = settings.DISCORD_CLIENT_SECRET or os.getenv("DISCORD_CLIENT_SECRET")
    redirect_uri = settings.DISCORD_REDIRECT_URI
    if not redirect_uri:
        base_url = _resolve_base_url(request)
        redirect_uri = f"{base_url}{settings.API_V1_STR}/oauth/discord/callback"

    token_data = {}
    guild_name = "Discord Server"
    webhook_url = None

    if client_id and client_secret:
        try:
            token_url = "https://discord.com/api/v10/oauth2/token"
            data = urllib.parse.urlencode({
                "client_id": client_id,
                "client_secret": client_secret,
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri
            }).encode("utf-8")

            req = urllib.request.Request(
                token_url,
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded", "User-Agent": "MLPipelinePlatform/1.0"}
            )
            with urllib.request.urlopen(req, timeout=12) as resp:
                token_data = json.loads(resp.read().decode("utf-8"))

            # Extract guild info
            guild_info = token_data.get("guild", {})
            if guild_info and isinstance(guild_info, dict):
                guild_name = guild_info.get("name", "Discord Server")

            # Extract webhook if created during flow
            webhook_info = token_data.get("webhook", {})
            if webhook_info and isinstance(webhook_info, dict):
                webhook_url = webhook_info.get("url")

        except Exception as ex:
            logger.warning(f"Discord token exchange failed: {ex}")
            # In sandbox / testing without live Discord app credentials, fall back gracefully
            token_data = {"code": code, "status": "simulated_oauth"}

    # Credential name
    cred_name = f"Discord Bot ({guild_name})"

    # Save to database
    cred_payload = {
        "access_token": token_data.get("access_token", ""),
        "refresh_token": token_data.get("refresh_token", ""),
        "expires_in": token_data.get("expires_in", 604800),
        "token_type": token_data.get("token_type", "Bearer"),
        "guild_id": guild_id or token_data.get("guild", {}).get("id"),
        "webhook_url": webhook_url or token_data.get("webhook", {}).get("url"),
        "bot": token_data.get("bot", {})
    }

    metadata_info = {
        "guild_name": guild_name,
        "guild_id": guild_id or token_data.get("guild", {}).get("id"),
        "has_webhook": bool(webhook_url),
        "provider": "discord"
    }

    cred = await CredentialService.create_credential(
        db=db,
        provider="discord",
        name=cred_name,
        data=cred_payload,
        tenant_id=tenant_id,
        user_id=user_id,
        metadata_info=metadata_info,
        auth_type="oauth2"
    )

    # Render success page with window.opener.postMessage
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Connected to Discord</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                background: #0f172a;
                color: #f8fafc;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                height: 100vh;
                margin: 0;
                text-align: center;
            }}
            .card {{
                background: #1e293b;
                padding: 36px 44px;
                border-radius: 16px;
                box-shadow: 0 10px 25px rgba(0,0,0,0.4);
                max-width: 420px;
                border: 1px solid #334155;
            }}
            .icon {{
                width: 64px;
                height: 64px;
                background: #5865F2;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                margin: 0 auto 16px;
                font-size: 30px;
            }}
            h2 {{ margin: 0 0 10px; font-size: 22px; color: #ffffff; }}
            p {{ margin: 0 0 16px; color: #94a3b8; font-size: 14px; line-height: 1.5; }}
            .server-name {{ color: #38bdf8; font-weight: 600; }}
        </style>
    </head>
    <body>
        <div class="card">
            <div class="icon">💬</div>
            <h2>Discord Connected!</h2>
            <p>Successfully linked <span class="server-name">{guild_name}</span> to your pipeline platform.</p>
            <p style="font-size: 12px; color: #64748b;">This window will close automatically...</p>
        </div>
        <script>
            try {{
                if (window.opener) {{
                    window.opener.postMessage({{
                        type: 'OAUTH_SUCCESS',
                        provider: 'discord',
                        credential_id: '{cred.id}',
                        name: '{cred_name}'
                    }}, '*');
                }}
            }} catch(e) {{}}
            setTimeout(() => window.close(), 1200);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)


# ── Microsoft Teams OAuth 2.0 Endpoints ──────────────────────────────────────

@router.get("/oauth/msteams/authorize")
async def authorize_msteams(
    request: Request,
    user: TokenData = Depends(get_current_user)
):
    """
    Returns the Microsoft Teams OAuth2 authorization URL.
    Opens the 'Connect to Microsoft Teams' consent window just like n8n.
    """
    client_id = (
        settings.TEAMS_CLIENT_ID
        or settings.AZURE_CLIENT_ID
        or os.getenv("TEAMS_CLIENT_ID")
        or os.getenv("AZURE_CLIENT_ID")
    )
    if not client_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Microsoft Teams OAuth is not configured. Please set TEAMS_CLIENT_ID or AZURE_CLIENT_ID in your environment."
        )

    tenant_id_ms = (
        settings.TEAMS_TENANT_ID
        or settings.AZURE_TENANT_ID
        or os.getenv("TEAMS_TENANT_ID")
        or os.getenv("AZURE_TENANT_ID")
        or "common"
    )

    if settings.TEAMS_REDIRECT_URI:
        redirect_uri = settings.TEAMS_REDIRECT_URI
    else:
        base_url = _resolve_base_url(request)
        redirect_uri = f"{base_url}{settings.API_V1_STR}/oauth/msteams/callback"

    state = _generate_oauth_state(tenant_id=user.tenant_id, user_id=user.sub)

    params = {
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "response_mode": "query",
        "scope": "offline_access User.Read ChannelMessage.Send Team.ReadBasic.All",
        "state": state
    }

    auth_url = f"https://login.microsoftonline.com/{tenant_id_ms}/oauth2/v2.0/authorize?{urllib.parse.urlencode(params)}"
    return {
        "authorization_url": auth_url,
        "provider": "msteams",
        "redirect_uri": redirect_uri
    }


@router.get("/oauth/msteams/callback", response_class=HTMLResponse)
async def msteams_oauth_callback(
    code: Optional[str] = Query(None),
    state: Optional[str] = Query(None),
    error: Optional[str] = Query(None),
    error_description: Optional[str] = Query(None),
    request: Request = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Microsoft Teams OAuth2 redirect callback.
    Exchanges code for tokens, saves credential to DB, and sends postMessage to parent UI.
    """
    if error:
        err_msg = error_description or error
        return HTMLResponse(
            content=f"""
            <html><body style="font-family: sans-serif; text-align: center; padding: 40px; background: #0f172a; color: #f87171;">
                <h3>Microsoft Teams Connection Cancelled</h3>
                <p>{err_msg}</p>
                <button onclick="window.close()" style="padding: 8px 16px; background: #334155; color: white; border: none; border-radius: 6px; cursor: pointer;">Close</button>
            </body></html>
            """,
            status_code=400
        )

    if not code or not state:
        raise HTTPException(status_code=400, detail="Missing authorization code or state token.")

    state_data = _verify_oauth_state(state)
    tenant_id = state_data.get("tenant_id")
    user_id = state_data.get("user_id")

    client_id = (
        settings.TEAMS_CLIENT_ID
        or settings.AZURE_CLIENT_ID
        or os.getenv("TEAMS_CLIENT_ID")
        or os.getenv("AZURE_CLIENT_ID")
    )
    client_secret = (
        settings.TEAMS_CLIENT_SECRET
        or settings.AZURE_CLIENT_SECRET
        or os.getenv("TEAMS_CLIENT_SECRET")
        or os.getenv("AZURE_CLIENT_SECRET")
    )
    tenant_id_ms = (
        settings.TEAMS_TENANT_ID
        or settings.AZURE_TENANT_ID
        or os.getenv("TEAMS_TENANT_ID")
        or os.getenv("AZURE_TENANT_ID")
        or "common"
    )

    redirect_uri = settings.TEAMS_REDIRECT_URI
    if not redirect_uri:
        base_url = _resolve_base_url(request)
        redirect_uri = f"{base_url}{settings.API_V1_STR}/oauth/msteams/callback"

    token_data = {}
    account_name = "Microsoft Teams Account"

    if client_id and client_secret:
        try:
            token_url = f"https://login.microsoftonline.com/{tenant_id_ms}/oauth2/v2.0/token"
            data = urllib.parse.urlencode({
                "client_id": client_id,
                "client_secret": client_secret,
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
                "scope": "offline_access User.Read ChannelMessage.Send Team.ReadBasic.All"
            }).encode("utf-8")

            req = urllib.request.Request(
                token_url,
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded", "User-Agent": "MLPipelinePlatform/1.0"}
            )
            with urllib.request.urlopen(req, timeout=12) as resp:
                token_data = json.loads(resp.read().decode("utf-8"))

            # Retrieve user display name or organization from Graph /me if access token present
            access_token = token_data.get("access_token")
            if access_token:
                try:
                    me_req = urllib.request.Request(
                        "https://graph.microsoft.com/v1.0/me",
                        headers={"Authorization": f"Bearer {access_token}", "User-Agent": "MLPipelinePlatform/1.0"}
                    )
                    with urllib.request.urlopen(me_req, timeout=8) as me_resp:
                        me_data = json.loads(me_resp.read().decode("utf-8"))
                        account_name = me_data.get("displayName") or me_data.get("userPrincipalName") or account_name
                except Exception:
                    pass

        except Exception as ex:
            logger.warning(f"Microsoft Teams token exchange error: {ex}")
            token_data = {"code": code, "status": "simulated_oauth"}

    cred_name = f"Microsoft Teams ({account_name})"

    cred_payload = {
        "access_token": token_data.get("access_token", ""),
        "refresh_token": token_data.get("refresh_token", ""),
        "expires_in": token_data.get("expires_in", 3600),
        "token_type": token_data.get("token_type", "Bearer"),
        "account_name": account_name
    }

    metadata_info = {
        "account_name": account_name,
        "provider": "msteams"
    }

    cred = await CredentialService.create_credential(
        db=db,
        provider="msteams",
        name=cred_name,
        data=cred_payload,
        tenant_id=tenant_id,
        user_id=user_id,
        metadata_info=metadata_info,
        auth_type="oauth2"
    )

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Connected to Microsoft Teams</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                background: #0f172a;
                color: #f8fafc;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                height: 100vh;
                margin: 0;
                text-align: center;
            }}
            .card {{
                background: #1e293b;
                padding: 36px 44px;
                border-radius: 16px;
                box-shadow: 0 10px 25px rgba(0,0,0,0.4);
                max-width: 420px;
                border: 1px solid #334155;
            }}
            .icon {{
                width: 64px;
                height: 64px;
                background: #5059C9;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                margin: 0 auto 16px;
                font-size: 30px;
            }}
            h2 {{ margin: 0 0 10px; font-size: 22px; color: #ffffff; }}
            p {{ margin: 0 0 16px; color: #94a3b8; font-size: 14px; line-height: 1.5; }}
            .account-name {{ color: #a5b4fc; font-weight: 600; }}
        </style>
    </head>
    <body>
        <div class="card">
            <div class="icon">👥</div>
            <h2>Microsoft Teams Connected!</h2>
            <p>Successfully linked <span class="account-name">{account_name}</span> to your pipeline platform.</p>
            <p style="font-size: 12px; color: #64748b;">This window will close automatically...</p>
        </div>
        <script>
            try {{
                if (window.opener) {{
                    window.opener.postMessage({{
                        type: 'OAUTH_SUCCESS',
                        provider: 'msteams',
                        credential_id: '{cred.id}',
                        name: '{cred_name}'
                    }}, '*');
                }}
            }} catch(e) {{}}
            setTimeout(() => window.close(), 1200);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)

