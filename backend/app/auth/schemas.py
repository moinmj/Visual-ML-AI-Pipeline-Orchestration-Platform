from typing import List, Optional
from pydantic import BaseModel, Field


class DevTokenRequest(BaseModel):
    """
    Dev-only helper for minting an access token without a real identity
    provider wired up yet. In production, replace this endpoint with real
    login/SSO and call `create_access_token(...)` from
    `backend.app.core.security` once credentials have been verified.
    """
    sub: str = Field(..., description="Subject / user identifier to embed in the token")
    tenant_id: int = Field(..., description="Tenant the caller belongs to")
    roles: List[str] = Field(default_factory=list, description="e.g. ['Tenant Admin']")
    permissions: List[str] = Field(default_factory=list, description="e.g. ['can_ingest_data']")


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_minutes: int


class WhoAmIResponse(BaseModel):
    sub: str
    tenant_id: int
    roles: List[str]
    permissions: List[str]
