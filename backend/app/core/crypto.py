"""
Cryptographic utility module for symmetric credential encryption, decryption,
key rotation, and secure payload handling using cryptography.fernet.MultiFernet.

Blueprint adapted from Datalect / User Management architecture.
Supports:
- Primary key encryption with multi-key fallback decryption (seamless key rotation)
- APP_ENCRYPTION_KEYS (comma-separated keys in env / settings)
- Fallback to FERNET_KEY and deterministic SHA-256 derivation from SECRET_KEY
- TTL-based time-bound decryption
- Graceful legacy base64 fallback for backward compatibility
- Credential masking for safe API serialization
"""

import base64
import hashlib
import json
import logging
import os
from functools import lru_cache
from typing import Dict, Any, List, Optional, Union

from cryptography.fernet import Fernet, MultiFernet, InvalidToken

logger = logging.getLogger(__name__)

# List of sensitive keys that should be redacted/masked in logs and safe display payloads
SENSITIVE_FIELD_NAMES = {
    "password",
    "secret",
    "client_secret",
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
    "private_key",
    "auth_token",
    "bearer_token",
    "sasl_password",
    "smtp_password",
    "secret_key",
    "service_account_key",
    "webhook_url",
    "token",
}


def _derive_fernet_key(source_secret: str) -> bytes:
    """
    Derives a valid 32-byte URL-safe base64-encoded Fernet key from an arbitrary string.
    Ensures that development/demo setups never crash even if APP_ENCRYPTION_KEYS is omitted.
    """
    digest = hashlib.sha256(source_secret.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


def _validate_or_derive_fernet_key(raw_key: Union[str, bytes]) -> bytes:
    """
    Validates that a key is a valid 32-byte base64-encoded Fernet key.
    If it's already a valid Fernet key, returns its bytes.
    Otherwise, derives a 32-byte key from it.
    """
    if isinstance(raw_key, str):
        raw_key = raw_key.strip().encode("utf-8")
    else:
        raw_key = raw_key.strip()

    try:
        decoded = base64.urlsafe_b64decode(raw_key)
        if len(decoded) == 32:
            return raw_key
    except Exception:
        pass

    return _derive_fernet_key(raw_key.decode("utf-8", errors="ignore"))


def get_fernet_keys() -> List[bytes]:
    """
    Gathers encryption keys in prioritized order:
    1. APP_ENCRYPTION_KEYS (comma-separated list, e.g. "new_key,old_key")
    2. FERNET_KEY
    3. Derived from SECRET_KEY
    """
    keys: List[bytes] = []

    # 1. Check APP_ENCRYPTION_KEYS in settings or os.environ
    raw_multi = os.getenv("APP_ENCRYPTION_KEYS")
    if not raw_multi:
        try:
            from backend.app.core.config import settings
            raw_multi = getattr(settings, "APP_ENCRYPTION_KEYS", None)
        except Exception:
            raw_multi = None

    if raw_multi:
        for k in str(raw_multi).split(","):
            cleaned = k.strip()
            if cleaned:
                keys.append(_validate_or_derive_fernet_key(cleaned))

    # 2. Check FERNET_KEY
    if not keys:
        raw_fernet = os.getenv("FERNET_KEY")
        if not raw_fernet:
            try:
                from backend.app.core.config import settings
                raw_fernet = getattr(settings, "FERNET_KEY", None)
            except Exception:
                raw_fernet = None

        if raw_fernet:
            for k in str(raw_fernet).split(","):
                cleaned = k.strip()
                if cleaned:
                    keys.append(_validate_or_derive_fernet_key(cleaned))

    # 3. Fallback derivation from SECRET_KEY
    if not keys:
        secret_key = os.getenv("SECRET_KEY")
        if not secret_key:
            try:
                from backend.app.core.config import settings
                secret_key = getattr(settings, "SECRET_KEY", "default-platform-crypto-secret")
            except Exception:
                secret_key = "default-platform-crypto-secret"
        keys.append(_derive_fernet_key(secret_key))

    return keys


@lru_cache(maxsize=1)
def get_cipher() -> MultiFernet:
    """
    Returns a MultiFernet instance initialized with the configured encryption keys.
    MultiFernet encrypts using keys[0] (primary) and decrypts trying all keys in sequence.
    """
    keys = get_fernet_keys()
    fernets = [Fernet(k) for k in keys]
    return MultiFernet(fernets)


def reset_cipher_cache() -> None:
    """Clears the cached cipher instance (useful during tests or key rotation)."""
    get_cipher.cache_clear()


def encrypt_credential(plain_text: str) -> str:
    """
    Encrypts a plaintext string and returns a URL-safe base64 ciphertext string.
    Always encrypts with the primary (first) key in MultiFernet.
    """
    if plain_text is None or plain_text == "":
        return ""
    cipher = get_cipher()
    return cipher.encrypt(plain_text.encode("utf-8")).decode("utf-8")


def decrypt_credential(cipher_text: str) -> str:
    """
    Decrypts a ciphertext string back into plaintext.
    Attempts decryption against all keys in MultiFernet.
    Includes backward-compatibility fallback for legacy unencrypted/base64 strings.
    """
    if not cipher_text:
        return ""

    cipher = get_cipher()
    try:
        return cipher.decrypt(cipher_text.encode("utf-8")).decode("utf-8")
    except (InvalidToken, Exception):
        pass

    # Legacy fallback: try base64 decode if it was stored using old dev fallback
    try:
        raw = base64.b64decode(cipher_text.encode("utf-8")).decode("utf-8")
        # Validate that decoded content looks like utf-8 string or JSON
        if raw:
            return raw
    except Exception:
        pass

    raise ValueError("Decryption failed: invalid token, bad key, or corrupted ciphertext.")


def decrypt_with_ttl(cipher_text: str, max_age_seconds: int) -> Optional[str]:
    """
    Decrypts ciphertext and verifies that it is not older than max_age_seconds.
    Returns None if expired, tampered, or corrupted.
    """
    if not cipher_text:
        return None
    cipher = get_cipher()
    try:
        return cipher.decrypt(cipher_text.encode("utf-8"), ttl=max_age_seconds).decode("utf-8")
    except (InvalidToken, Exception):
        return None


def rotate_credential_token(cipher_text: str) -> str:
    """
    Re-encrypts an existing ciphertext token with the primary (active) key.
    Useful when rotating keys in production so older ciphertext is updated to the newest key.
    """
    if not cipher_text:
        return ""
    cipher = get_cipher()
    try:
        return cipher.rotate(cipher_text.encode("utf-8")).decode("utf-8")
    except Exception:
        # Fallback: full decrypt and re-encrypt
        decrypted = decrypt_credential(cipher_text)
        return encrypt_credential(decrypted)


def encrypt_payload(payload: Dict[str, Any]) -> str:
    """
    Serializes a dictionary (containing tokens, credentials, connection configs)
    into a JSON string and encrypts it with MultiFernet.
    """
    if not payload:
        return ""
    raw_json = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    return encrypt_credential(raw_json)


def decrypt_payload(cipher_text: str) -> Dict[str, Any]:
    """
    Decrypts an encrypted payload string and parses it back into a dictionary.
    Returns {} on empty or failed decryption.
    """
    if not cipher_text:
        return {}
    try:
        decrypted_str = decrypt_credential(cipher_text)
        if not decrypted_str:
            return {}
        return json.loads(decrypted_str)
    except Exception as e:
        logger.error(f"Failed to decrypt credential payload: {e}")
        return {}


def mask_credential_payload(payload: Dict[str, Any], mask_str: str = "********") -> Dict[str, Any]:
    """
    Recursively masks sensitive fields (passwords, tokens, keys) in a dictionary
    so it can be safely returned in public API responses or logged.
    """
    if not isinstance(payload, dict):
        return payload

    masked = {}
    for k, v in payload.items():
        k_lower = str(k).lower().strip()
        is_sensitive = k_lower in SENSITIVE_FIELD_NAMES or any(s in k_lower for s in ("password", "secret", "token", "api_key", "apikey"))
        if is_sensitive and isinstance(v, (str, int, float, bool)):
            masked[k] = mask_str
        elif isinstance(v, dict):
            masked[k] = mask_credential_payload(v, mask_str=mask_str)
        elif isinstance(v, list):
            masked[k] = [
                mask_credential_payload(item, mask_str=mask_str) if isinstance(item, dict) else item
                for item in v
            ]
        else:
            masked[k] = v
    return masked


def is_encrypted_token(val: Any) -> bool:
    """Checks if a string looks like a Fernet token (starts with gAAAAA and valid base64)."""
    if isinstance(val, str) and val.startswith("gAAAAA") and len(val) >= 60:
        return True
    return False


def encrypt_node_config_secrets(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Scans a node configuration dictionary and symmetrically encrypts any plaintext
    passwords, tokens, or secrets using MultiFernet before saving to the database.
    """
    if not isinstance(config, dict):
        return config
    cleaned = dict(config)
    for k, v in list(cleaned.items()):
        k_lower = str(k).lower().strip()
        is_secret = k_lower in SENSITIVE_FIELD_NAMES or any(s in k_lower for s in ("password", "secret", "token", "api_key", "apikey"))
        if is_secret and isinstance(v, str) and v.strip() and not is_encrypted_token(v):
            if v != "********" and v != "••••••••":
                cleaned[k] = encrypt_credential(v)
    return cleaned


def decrypt_node_config_secrets(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Scans a node configuration dictionary in memory and decrypts any MultiFernet-encrypted
    fields so recipes can authenticate at runtime.
    """
    if not isinstance(config, dict):
        return config
    decrypted = dict(config)
    for k, v in list(decrypted.items()):
        if isinstance(v, str) and is_encrypted_token(v):
            try:
                decrypted[k] = decrypt_credential(v)
            except Exception as e:
                logger.warning(f"Failed to decrypt node config field '{k}': {e}")
    return decrypted

