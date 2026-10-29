import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, DateTime, JSON, Text, Boolean, Integer
from backend.app.infrastructure.database.session import Base


class Credential(Base):
    """
    Stores encrypted third-party credentials (OAuth tokens, API keys, Webhooks)
    for connected integrations like Discord, Google Workspace, Slack, etc.
    Scoped per tenant and creator.
    """
    __tablename__ = "credentials"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(Integer, nullable=True, index=True)
    user_id = Column(String(255), nullable=True, index=True)

    # Provider and Display
    provider = Column(String(50), nullable=False, index=True)  # e.g. "discord", "google", "slack"
    name = Column(String(255), nullable=False)                 # e.g. "Discord Bot (My Community Server)"
    auth_type = Column(String(50), nullable=False, default="oauth2")  # "oauth2", "api_key", "service_account"

    # Encrypted payload containing tokens, refresh tokens, bot secrets, webhook URLs
    encrypted_data = Column(Text, nullable=False)

    # Public metadata safe to display in UI dropdowns (e.g. guild_id, guild_name, account_email)
    metadata_info = Column(JSON, nullable=False, default=dict)

    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "provider": self.provider,
            "name": self.name,
            "auth_type": self.auth_type,
            "metadata_info": self.metadata_info or {},
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
