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
    decrypt_with_ttl,
    rotate_credential_token,
    encrypt_payload,
    decrypt_payload,
    mask_credential_payload,
    reset_cipher_cache,
)
from backend.app.workflows.router import sanitize_vaulted_node_configs


def test_single_key_encryption_decryption():
    """Verify that encrypt_credential encrypts plaintext into ciphertext and decrypts faithfully."""
    reset_cipher_cache()
    plain = "SuperSecret_Password_2026!"
    cipher = encrypt_credential(plain)

    # Ciphertext must be non-empty and not contain the plaintext
    assert cipher != plain
    assert plain not in cipher
    assert cipher.startswith("gAAAAA")

    # Decrypt must recover the original plaintext exactly
    decrypted = decrypt_credential(cipher)
    assert decrypted == plain


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


def test_multifernet_key_rotation():
    """
    Verify key rotation:
    1. Encrypt with Key A.
    2. Add Key B as primary (Key A becomes fallback).
    3. Ciphertext from Key A can still be decrypted.
    4. New encryptions use Key B.
    5. Existing tokens can be rotated in-place to Key B.
    """
    key_a = Fernet.generate_key().decode()
    key_b = Fernet.generate_key().decode()

    # Step 1: Encrypt with Key A only
    with patch.dict(os.environ, {"APP_ENCRYPTION_KEYS": key_a}):
        reset_cipher_cache()
        plain_text = "DatabaseSecretForRotation"
        cipher_under_a = encrypt_credential(plain_text)

        # Sanity: decrypt under key A works
        assert decrypt_credential(cipher_under_a) == plain_text

    # Step 2: Rotate key: Key B is primary (active), Key A is fallback
    with patch.dict(os.environ, {"APP_ENCRYPTION_KEYS": f"{key_b},{key_a}"}):
        reset_cipher_cache()

        # Step 3: Old ciphertext under Key A must still decrypt successfully!
        assert decrypt_credential(cipher_under_a) == plain_text

        # Step 4: New encryption uses Key B
        cipher_under_b = encrypt_credential(plain_text)
        assert cipher_under_b != cipher_under_a
        assert decrypt_credential(cipher_under_b) == plain_text

        # Step 5: Rotate token in-place to Key B
        rotated_token = rotate_credential_token(cipher_under_a)
        assert rotated_token != cipher_under_a
        assert decrypt_credential(rotated_token) == plain_text

    # Verify that rotated token decrypts with Key B alone (Key A removed)
    with patch.dict(os.environ, {"APP_ENCRYPTION_KEYS": key_b}):
        reset_cipher_cache()
        assert decrypt_credential(rotated_token) == plain_text

        # Old unrotated cipher_under_a fails if Key A is completely removed
        with pytest.raises(ValueError, match="Decryption failed"):
            decrypt_credential(cipher_under_a)


def test_decrypt_with_ttl():
    """Verify that decrypt_with_ttl honors expiration limits."""
    reset_cipher_cache()
    token = encrypt_credential("TemporarySecret")

    # Within valid TTL -> succeeds
    valid_decrypted = decrypt_with_ttl(token, max_age_seconds=60)
    assert valid_decrypted == "TemporarySecret"

    # Expired TTL (advance time beyond max_age_seconds) -> returns None
    import time
    with patch("time.time", return_value=time.time() + 100):
        expired = decrypt_with_ttl(token, max_age_seconds=10)
        assert expired is None


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
    assert encrypted_pwd.startswith("gAAAAA")

    # In clean_nodes data.config, smtp_password is also encrypted
    assert clean_nodes[0]["data"]["config"]["smtp_password"].startswith("gAAAAA")

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
        assert cred.encrypted_data.startswith("gAAAAA")

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
