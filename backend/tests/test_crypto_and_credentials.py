"""
Comprehensive tests for MultiFernet symmetric encryption, key rotation,
safe credential masking, and CredentialService storage/retrieval.
"""

import os
import pytest
from unittest.mock import patch
from cryptography.fernet import Fernet

from backend.app.core import crypto
from backend.app.core.crypto import (
    encrypt_credential,
    decrypt_credential,
    rotate_credential_token,
    encrypt_payload,
    decrypt_payload,
    mask_credential_payload,
    reset_cipher_cache,
)
from backend.app.workflows.router import sanitize_vaulted_node_configs


def _account_manager_aes_encrypt(plain: str, secret: str) -> str:
    """Verbatim copy of Account Manager's aes_encrypt (app/services/connection_adapters.py)."""
    import base64
    import hashlib
    from cryptography.hazmat.primitives import padding
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    key = hashlib.sha256(secret.encode()).digest()
    padder = padding.PKCS7(128).padder()
    data = padder.update(plain.encode()) + padder.finalize()
    enc = Cipher(algorithms.AES(key), modes.CBC(key[:16])).encryptor()
    return base64.b64encode(enc.update(data) + enc.finalize()).decode()


def test_single_key_encryption_decryption():
    """Verify that encrypt_credential encrypts plaintext into ciphertext and decrypts faithfully."""
    reset_cipher_cache()
    plain = "SuperSecret_Password_2026!"
    cipher = encrypt_credential(plain)

    # Ciphertext must be non-empty and not contain the plaintext
    assert cipher != plain
    assert plain not in cipher
    assert not cipher.startswith("gAAAAA")

    # Decrypt must recover the original plaintext exactly
    decrypted = decrypt_credential(cipher)
    assert decrypted == plain


def test_matches_account_manager_format():
    """Ciphertext must be byte-identical to Account Manager's, and theirs must decrypt here."""
    with patch.dict(os.environ, {"SECRET_KEY": "shared-secret-key"}):
        plain = "P@ssw0rd-for-postgres"
        theirs = _account_manager_aes_encrypt(plain, "shared-secret-key")
        assert encrypt_credential(plain) == theirs
        assert decrypt_credential(theirs) == plain


def test_legacy_fernet_tokens_still_decrypt_and_migrate():
    """Values saved by the old Fernet version still read, and rotation moves them to AES."""
    fernet_key = Fernet.generate_key().decode()
    with patch.dict(os.environ, {"FERNET_KEY": fernet_key, "APP_ENCRYPTION_KEYS": ""}):
        reset_cipher_cache()
        old_token = Fernet(fernet_key.encode()).encrypt(b"OldFernetSecret").decode()
        assert decrypt_credential(old_token) == "OldFernetSecret"
        migrated = rotate_credential_token(old_token)
        assert not migrated.startswith("gAAAAA")
        assert decrypt_credential(migrated) == "OldFernetSecret"
    reset_cipher_cache()


def test_payload_dictionary_encryption_decryption():
    """Verify that encrypt_payload and decrypt_payload securely handle nested dictionaries."""
    reset_cipher_cache()
    payload = {
        "host": "mysql.internal.corp",
        "port": 3306,
        "database": "analytics",
        "username": "etl_service",
        "password": "CorrectHorseBatteryStaple#99",
        "extra_tokens": ["token_abc", "token_xyz"],
        "ssl": {"enabled": True, "ca_cert": "MIIB..."},
    }

    encrypted_str = encrypt_payload(payload)
    assert encrypted_str != ""
    assert "CorrectHorseBatteryStaple" not in encrypted_str

    decrypted_dict = decrypt_payload(encrypted_str)
    assert decrypted_dict == payload
    assert decrypted_dict["password"] == "CorrectHorseBatteryStaple#99"
    assert decrypted_dict["port"] == 3306


def test_secret_key_rotation_with_legacy_keys():
    """
    Same rotation model as Account Manager:
    SECRET_KEY = new key, DBS_LEGACY_SECRET_KEYS = old keys still accepted for decryption.
    """
    plain_text = "DatabaseSecretForRotation"
    with patch.dict(os.environ, {"SECRET_KEY": "key-a", "DBS_LEGACY_SECRET_KEYS": ""}):
        cipher_under_a = encrypt_credential(plain_text)

    with patch.dict(os.environ, {"SECRET_KEY": "key-b", "DBS_LEGACY_SECRET_KEYS": "key-a"}):
        assert decrypt_credential(cipher_under_a) == plain_text
        rotated = rotate_credential_token(cipher_under_a)
        assert rotated != cipher_under_a
        assert rotate_credential_token(rotated) == rotated       # already current → unchanged

    with patch.dict(os.environ, {"SECRET_KEY": "key-b", "DBS_LEGACY_SECRET_KEYS": "", "FERNET_KEY": "",
                                 "APP_ENCRYPTION_KEYS": ""}):
        assert decrypt_credential(rotated) == plain_text
        with pytest.raises(ValueError, match="Decryption failed"):
            decrypt_credential(cipher_under_a)


def test_masked_values_never_saved():
    """Masked placeholders sent back by the UI keep the stored secret, or are dropped — never stored."""
    from backend.app.core.crypto import encrypt_node_config_secrets, merge_masked_secrets

    assert merge_masked_secrets({"password": "********", "host": "h"}, {"password": "real"}) == {
        "password": "real", "host": "h"}
    assert merge_masked_secrets({"password": "••••••••"}, {}) == {}
    assert "password" not in encrypt_node_config_secrets({"password": "*****"})

    class _Wf:
        node_configs = {}
        nodes = [{"id": "n1", "data": {"config": {"smtp_password": encrypt_credential("realpwd")}}}]

    nodes = [{"id": "n1", "data": {"config": {"smtp_password": "••••••••"}}}]
    clean_nodes, _ = sanitize_vaulted_node_configs(nodes, None, existing_wf=_Wf())
    assert decrypt_credential(clean_nodes[0]["data"]["config"]["smtp_password"]) == "realpwd"


def test_legacy_base64_backward_compatibility():
    """Verify that legacy base64 encoded strings from dev fallback are decrypted gracefully."""
    import base64
    import json

    reset_cipher_cache()
    legacy_dict = {"username": "legacy_user", "password": "LegacyPassword123"}
    raw_b64 = base64.b64encode(json.dumps(legacy_dict).encode("utf-8")).decode("utf-8")

    # Decrypting legacy payload should succeed without crashing
    recovered = decrypt_payload(raw_b64)
    assert recovered == legacy_dict
    assert recovered["password"] == "LegacyPassword123"


def test_mask_credential_payload():
    """Verify that mask_credential_payload masks sensitive secrets while preserving non-sensitive metadata."""
    payload = {
        "host": "prod-mysql.internal",
        "port": 3306,
        "database": "customer_data",
        "username": "app_user",
        "password": "RawPassword!",
        "client_secret": "VerySecretClientKey",
        "access_token": "ya29.secret_token",
        "refresh_token": "1//refresh_token_xyz",
        "api_key": "AIzaSy...",
        "ssl_mode": "REQUIRED",
        "nested": {
            "token": "nested_secret",
            "safe_label": "production_cluster"
        }
    }

    masked = mask_credential_payload(payload)

    # Sensitive fields masked
    assert masked["password"] == "********"
    assert masked["client_secret"] == "********"
    assert masked["access_token"] == "********"
    assert masked["refresh_token"] == "********"
    assert masked["api_key"] == "********"
    assert masked["nested"]["token"] == "********"

    # Non-sensitive fields preserved
    assert masked["host"] == "prod-mysql.internal"
    assert masked["port"] == 3306
    assert masked["database"] == "customer_data"
    assert masked["username"] == "app_user"
    assert masked["ssl_mode"] == "REQUIRED"
    assert masked["nested"]["safe_label"] == "production_cluster"


def test_sanitize_vaulted_node_configs():
    """Verify that sanitize_vaulted_node_configs strips passwords from nodes when credential_id is set."""
    node_configs = {
        "node_1": {
            "connection_mode": "credential",
            "credential_id": "vault-cred-abc",
            "host": "db.corp.internal",
            "port": 5432,
            "password": "PlaintextPasswordThatShouldBeScrubbed",
            "secret_key": "PlaintextSecret",
        },
        "node_2": {
            "connection_mode": "host_port",
            "host": "open.corp.internal",
            "port": 8080,
            "query": "SELECT * FROM test",
        }
    }

    nodes = [
        {
            "id": "node_1",
            "data": {
                "config": {
                    "credential_id": "vault-cred-abc",
                    "password": "PlaintextPasswordThatShouldBeScrubbed",
                    "host": "db.corp.internal"
                }
            }
        }
    ]

    clean_nodes, clean_configs = sanitize_vaulted_node_configs(nodes, node_configs)

    # In node_1 (has credential_id), password and secret_key are stripped
    assert "password" not in clean_configs["node_1"]
    assert "secret_key" not in clean_configs["node_1"]
    assert clean_configs["node_1"]["credential_id"] == "vault-cred-abc"
    assert clean_configs["node_1"]["host"] == "db.corp.internal"

    # In clean_nodes, node_1 data.config password is also stripped
    assert "password" not in clean_nodes[0]["data"]["config"]
    assert clean_nodes[0]["data"]["config"]["credential_id"] == "vault-cred-abc"
    assert clean_nodes[0]["data"]["config"]["host"] == "db.corp.internal"

    # node_2 was unaffected
    assert clean_configs["node_2"]["host"] == "open.corp.internal"


def test_inline_node_secret_automatic_encryption_and_decryption():
    """Verify that inline node secrets (like smtp_password) are automatically encrypted on save and decrypted at runtime."""
    from backend.app.core.crypto import encrypt_node_config_secrets, decrypt_node_config_secrets, is_encrypted_token
    from backend.app.workflows.router import mask_workflow_configs_for_frontend

    node_configs = {
        "processor_gmail_2": {
            "smtp_host": "smtp.gmail.com",
            "smtp_port": 587,
            "smtp_user": "khanalisak1113@gmail.com",
            "smtp_password": "pdaa cezs achk xtin ",
            "subject": "Pipeline Report",
            "save_credential": False
        }
    }

    nodes = [
        {
            "id": "processor_gmail_2",
            "data": {
                "config": {
                    "smtp_password": "pdaa cezs achk xtin ",
                    "smtp_user": "khanalisak1113@gmail.com"
                }
            }
        }
    ]

    # 1. On saving to DB: inline smtp_password is encrypted
    clean_nodes, clean_configs = sanitize_vaulted_node_configs(nodes, node_configs)

    encrypted_pwd = clean_configs["processor_gmail_2"]["smtp_password"]
    assert encrypted_pwd != "pdaa cezs achk xtin "
    assert is_encrypted_token(encrypted_pwd)

    # In clean_nodes data.config, smtp_password is also encrypted
    assert is_encrypted_token(clean_nodes[0]["data"]["config"]["smtp_password"])

    # 2. At runtime execution: decrypt_node_config_secrets recovers the plaintext in memory
    runtime_cfg = decrypt_node_config_secrets(clean_configs["processor_gmail_2"])
    assert runtime_cfg["smtp_password"] == "pdaa cezs achk xtin "

    # 3. Masking before sending to frontend: ciphertext is converted to '••••••••'
    masked_nodes, masked_configs = mask_workflow_configs_for_frontend(clean_nodes, clean_configs)
    assert masked_configs["processor_gmail_2"]["smtp_password"] == "••••••••"
    assert masked_nodes[0]["data"]["config"]["smtp_password"] == "••••••••"


@pytest.mark.asyncio
async def test_credential_service_create_and_decrypt():
    """Verify CredentialService end-to-end: stores encrypted payload, prevents plaintext leakage, and decrypts."""
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker
    from backend.app.infrastructure.database.session import Base
    from backend.app.credentials.service import CredentialService

    # Use in-memory SQLite for isolated async test
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with async_session() as session:
        cred = await CredentialService.create_credential(
            db=session,
            provider="mysql",
            name="Corp Analytics MySQL",
            data={
                "host": "mysql.analytics.internal",
                "port": 3306,
                "database": "sales",
                "username": "etl_user",
                "password": "MySuperSecretPassword#123",
            },
            tenant_id=42,
            metadata_info={"host": "mysql.analytics.internal", "database": "sales"},
            auth_type="database"
        )

        assert cred.id is not None
        # Plaintext password MUST NOT appear in the database column
        assert "MySuperSecretPassword#123" not in cred.encrypted_data
        assert decrypt_credential(cred.encrypted_data).startswith("{")

        # to_dict must indicate has_secret is True and never expose encrypted_data
        cred_dict = cred.to_dict()
        assert cred_dict["has_secret"] is True
        assert "encrypted_data" not in cred_dict
        assert "password" not in cred_dict

        # Decrypt via get_credential_data
        resolved_data = await CredentialService.get_credential_data(
            db=session,
            credential_id=cred.id,
            tenant_id=42
        )
        assert resolved_data is not None
        assert resolved_data["password"] == "MySuperSecretPassword#123"
        assert resolved_data["username"] == "etl_user"

        # Safe masked retrieval
        masked_info = await CredentialService.get_credential_masked(
            db=session,
            credential_id=cred.id,
            tenant_id=42
        )
        assert masked_info is not None
        assert masked_info["safe_data"]["password"] == "********"
        assert masked_info["safe_data"]["username"] == "etl_user"

    await engine.dispose()
