import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import uuid

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from backend.app.credentials.models import Credential
from backend.app.core.crypto import (
    encrypt_payload,
    decrypt_payload,
    encrypt_credential,
    decrypt_credential,
    mask_credential_payload,
    rotate_credential_token,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CredentialService",
    "encrypt_payload",
    "decrypt_payload",
    "encrypt_credential",
    "decrypt_credential",
    "mask_credential_payload",
    "rotate_credential_token",
]


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
        """
        Encrypts sensitive credentials (passwords, tokens, keys) with MultiFernet
        and securely persists the credential model in the database.
        """
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
        logger.info(f"Created encrypted credential '{cred.id}' for provider '{provider}' (tenant: {tenant_id})")
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
        """
        Fetches and decrypts the underlying credential payload (tokens, passwords, keys)
        strictly in memory for connector execution or backend service authentication.
        """
        cred = await cls.get_credential(db=db, credential_id=credential_id, tenant_id=tenant_id)
        if not cred:
            return None
        return decrypt_payload(cred.encrypted_data)

    @classmethod
    async def get_credential_masked(
        cls,
        db: AsyncSession,
        credential_id: str,
        tenant_id: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieves credential information with sensitive keys (passwords, tokens)
        masked with '********' for safe API representation.
        """
        cred = await cls.get_credential(db=db, credential_id=credential_id, tenant_id=tenant_id)
        if not cred:
            return None
        data = decrypt_payload(cred.encrypted_data)
        masked_data = mask_credential_payload(data)
        res = cred.to_dict()
        res["safe_data"] = masked_data
        return res

    @classmethod
    async def update_credential(
        cls,
        db: AsyncSession,
        credential_id: str,
        data: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        metadata_info: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[int] = None
    ) -> Optional[Credential]:
        """
        Updates an existing credential. If new data is provided, it is re-encrypted with
        the active MultiFernet key before saving.
        """
        cred = await cls.get_credential(db=db, credential_id=credential_id, tenant_id=tenant_id)
        if not cred:
            return None

        if name is not None:
            cred.name = name.strip()
        if metadata_info is not None:
            merged_meta = dict(cred.metadata_info or {})
            merged_meta.update(metadata_info)
            cred.metadata_info = merged_meta
        if data is not None:
            cred.encrypted_data = encrypt_payload(data)

        cred.updated_at = datetime.now(timezone.utc)
        await db.commit()
        await db.refresh(cred)
        logger.info(f"Updated credential '{credential_id}'")
        return cred

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
        """Deactivates / soft-deletes a credential."""
        cred = await cls.get_credential(db=db, credential_id=credential_id, tenant_id=tenant_id)
        if not cred:
            return False
        cred.is_active = False
        cred.updated_at = datetime.now(timezone.utc)
        await db.commit()
        logger.info(f"Deactivated credential '{credential_id}'")
        return True

    @classmethod
    async def rotate_all_credentials(
        cls,
        db: AsyncSession,
        tenant_id: Optional[int] = None
    ) -> int:
        """
        Re-encrypts all active credentials with the current primary active encryption key.
        Run this operation after updating APP_ENCRYPTION_KEYS in production.
        """
        stmt = select(Credential).where(Credential.is_active == True)
        if tenant_id is not None:
            stmt = stmt.where(Credential.tenant_id == tenant_id)
        result = await db.execute(stmt)
        creds = list(result.scalars().all())

        rotated_count = 0
        for cred in creds:
            try:
                rotated_data = rotate_credential_token(cred.encrypted_data)
                if rotated_data != cred.encrypted_data:
                    cred.encrypted_data = rotated_data
                    cred.updated_at = datetime.now(timezone.utc)
                    rotated_count += 1
            except Exception as rot_err:
                logger.error(f"Failed to rotate credential '{cred.id}': {rot_err}")

        if rotated_count > 0:
            await db.commit()
            logger.info(f"Successfully rotated encryption on {rotated_count} credential records.")
        return rotated_count
