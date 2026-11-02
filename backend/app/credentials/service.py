import base64
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import uuid

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import delete

from backend.app.core.config import settings
from backend.app.credentials.models import Credential

logger = logging.getLogger(__name__)

# Encryption helpers
try:
    from cryptography.fernet import Fernet
    _fernet = Fernet(settings.FERNET_KEY.encode() if isinstance(settings.FERNET_KEY, str) else Fernet.generate_key())
except Exception:
    _fernet = None


def encrypt_payload(payload: Dict[str, Any]) -> str:
    """Encrypts a credential dictionary (tokens, secrets) into a secure string."""
    raw_json = json.dumps(payload).encode("utf-8")
    if _fernet:
        return _fernet.encrypt(raw_json).decode("utf-8")
    # Base64 fallback if Fernet key is unavailable in dev
    return base64.b64encode(raw_json).decode("utf-8")


def decrypt_payload(encrypted_str: str) -> Dict[str, Any]:
    """Decrypts a secure string back into the credential dictionary."""
    if not encrypted_str:
        return {}
    if _fernet:
        try:
            decrypted = _fernet.decrypt(encrypted_str.encode("utf-8"))
            return json.loads(decrypted.decode("utf-8"))
        except Exception:
            pass
    try:
        raw = base64.b64decode(encrypted_str.encode("utf-8"))
        return json.loads(raw.decode("utf-8"))
    except Exception as e:
        logger.error(f"Failed to decrypt credential payload: {e}")
        return {}


class CredentialService:
    @classmethod
    async def create_credential(
        cls,
        db: AsyncSession,
        provider: str,
        name: str,
        data: Dict[str, Any],
        tenant_id: Optional[int] = None,
        user_id: Optional[str] = None,
        metadata_info: Optional[Dict[str, Any]] = None,
        auth_type: str = "oauth2"
    ) -> Credential:
        """Saves a new encrypted credential in the database."""
        encrypted_data = encrypt_payload(data)
        cred = Credential(
            id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            user_id=user_id,
            provider=provider.lower().strip(),
            name=name.strip(),
            auth_type=auth_type,
            encrypted_data=encrypted_data,
            metadata_info=metadata_info or {},
            is_active=True
        )
        db.add(cred)
        await db.commit()
        await db.refresh(cred)
        logger.info(f"Created credential '{cred.id}' for provider '{provider}' (tenant: {tenant_id})")
        return cred

    @classmethod
    async def get_credential(
        cls,
        db: AsyncSession,
        credential_id: str,
        tenant_id: Optional[int] = None
    ) -> Optional[Credential]:
        """Fetches a credential model by ID with tenant isolation."""
        stmt = select(Credential).where(Credential.id == credential_id, Credential.is_active == True)
        if tenant_id is not None:
            stmt = stmt.where(Credential.tenant_id == tenant_id)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    @classmethod
    async def get_credential_data(
        cls,
        db: AsyncSession,
        credential_id: str,
        tenant_id: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """Fetches and decrypts the underlying credential tokens/keys."""
        cred = await cls.get_credential(db=db, credential_id=credential_id, tenant_id=tenant_id)
        if not cred:
            return None
        return decrypt_payload(cred.encrypted_data)

    @classmethod
    async def list_credentials(
        cls,
        db: AsyncSession,
        tenant_id: Optional[int] = None,
        provider: Optional[str] = None
    ) -> List[Credential]:
        """Lists active credentials for a tenant, optionally filtered by provider."""
        stmt = select(Credential).where(Credential.is_active == True)
        if tenant_id is not None:
            stmt = stmt.where(Credential.tenant_id == tenant_id)
        if provider:
            stmt = stmt.where(Credential.provider == provider.lower().strip())
        stmt = stmt.order_by(Credential.created_at.desc())
        result = await db.execute(stmt)
        return list(result.scalars().all())

    @classmethod
    async def delete_credential(
        cls,
        db: AsyncSession,
        credential_id: str,
        tenant_id: Optional[int] = None
    ) -> bool:
        """Deactivates / deletes a credential."""
        cred = await cls.get_credential(db=db, credential_id=credential_id, tenant_id=tenant_id)
        if not cred:
            return False
        cred.is_active = False
        await db.commit()
        logger.info(f"Deactivated credential '{credential_id}'")
        return True
