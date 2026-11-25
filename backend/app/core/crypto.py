"""
Symmetric credential encryption — same scheme as the Account Manager module
(app/services/connection_adapters.py: aes_encrypt / aes_decrypt), so both modules
write and read the same ciphertext format:

    key     = sha256(SECRET_KEY)            (32 bytes → AES-256)
    mode    = AES-CBC, IV = key[:16]
    padding = PKCS7 (128-bit block)
    output  = standard base64 (not URL-safe)

Decryption also tries DBS_LEGACY_SECRET_KEYS and the IV variants / '*' padding the
Account Manager accepts (Superset-created rows), then falls back to:
  - legacy Fernet tokens ("gAAAAA...") written by the previous version of this module
    (keys from APP_ENCRYPTION_KEYS / FERNET_KEY / SECRET_KEY), and
  - legacy plain base64 strings from the old dev fallback.
New values are always written in the AES format.
"""

import base64
import hashlib
import json
import logging
import os
from functools import lru_cache
from typing import Dict, Any, List, Optional, Union

from cryptography.fernet import Fernet, MultiFernet, InvalidToken
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

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

_MASK_CHARS = set("*•●")
_MASK_LITERALS = {"XXXXXXXXXX"}     # Account Manager's MASK inside sqlalchemy_uri


def is_masked_value(val: Any) -> bool:
    """True for placeholder values the UI shows instead of a secret ('********', '••••••••', ...)."""
    if not isinstance(val, str):
        return False
    s = val.strip()
    if s in _MASK_LITERALS:
        return True
    return len(s) >= 3 and set(s) <= _MASK_CHARS


# ── Settings ────────────────────────────────────────────────────────────────

def _setting(name: str) -> Optional[str]:
    """Env var first (so tests / deployments can override), then app settings."""
    val = os.getenv(name)
    if not val:
        try:
            from backend.app.core.config import settings
            val = getattr(settings, name, None)
        except Exception:
            val = None
    return str(val).strip() if val else None


def _secret_key() -> str:
    key = _setting("SECRET_KEY")
    if not key:
        raise RuntimeError("SECRET_KEY is not configured.")
    return key


def _decrypt_keys() -> List[str]:
    """SECRET_KEY first, then DBS_LEGACY_SECRET_KEYS (comma separated) — same order as Account Manager."""
    keys = [_secret_key()]
    extra = _setting("DBS_LEGACY_SECRET_KEYS") or ""
    keys += [k.strip() for k in extra.split(",") if k.strip() and k.strip() not in keys]
    return keys


def _readable(text: str) -> bool:
    return bool(text) and all(ch.isprintable() for ch in text)


# ── AES-256-CBC (Account Manager format) ────────────────────────────────────

def aes_encrypt(plain: str) -> str:
    """Byte-for-byte equivalent of Account Manager's aes_encrypt."""
    key = hashlib.sha256(_secret_key().encode()).digest()
    padder = padding.PKCS7(128).padder()
    data = padder.update(plain.encode()) + padder.finalize()
    enc = Cipher(algorithms.AES(key), modes.CBC(key[:16])).encryptor()
    return base64.b64encode(enc.update(data) + enc.finalize()).decode()


def aes_decrypt(value: Union[str, bytes, memoryview, None], strict: bool = False) -> str:
    """
    Equivalent of Account Manager's aes_decrypt. Raises ValueError if it can't decrypt.
    strict=True accepts only PKCS7-padded output (used to detect ciphertext, not to read Superset rows).
    """
    if value is None:
        return ""
    raw = bytes(value) if isinstance(value, (bytes, memoryview)) else str(value).encode()
    if not raw:
        return ""
    try:
        data = base64.b64decode(raw.decode().strip(), validate=True)
    except Exception:
        raise ValueError("cannot decrypt")
    if not data or len(data) % 16:
        raise ValueError("cannot decrypt")
    naive = None
    for secret in _decrypt_keys():
        key = hashlib.sha256(secret.encode()).digest()
        for iv in (key[:16], key[16:], b"\x00" * 16):
            dec = Cipher(algorithms.AES(key), modes.CBC(iv)).decryptor()
            out = dec.update(data) + dec.finalize()
            pad = out[-1]
            if 1 <= pad <= 16 and out[-pad:] == bytes([pad]) * pad:          # PKCS7
                try:
                    text = out[:-pad].decode("utf-8")
                    if _readable(text):
                        return text
                except UnicodeDecodeError:
                    pass
            if not strict and naive is None and out.endswith(b"*"):          # '*' padding (Superset)
                try:
                    text = out.rstrip(b"*").decode("utf-8")
                    if _readable(text):
                        naive = text
                except UnicodeDecodeError:
                    pass
    if naive is not None:
        return naive
    raise ValueError("cannot decrypt")


# ── Legacy Fernet (read-only, for values written before the switch) ─────────

def _derive_fernet_key(source_secret: str) -> bytes:
    digest = hashlib.sha256(source_secret.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


def _validate_or_derive_fernet_key(raw_key: Union[str, bytes]) -> bytes:
    raw_key = raw_key.strip().encode("utf-8") if isinstance(raw_key, str) else raw_key.strip()
    try:
        if len(base64.urlsafe_b64decode(raw_key)) == 32:
            return raw_key
    except Exception:
        pass
    return _derive_fernet_key(raw_key.decode("utf-8", errors="ignore"))


def _legacy_fernet_keys() -> List[bytes]:
    keys: List[bytes] = []
    for name in ("APP_ENCRYPTION_KEYS", "FERNET_KEY", "SECRET_KEY"):
        raw = _setting(name)
        if raw:
            parts = [raw] if name == "SECRET_KEY" else raw.split(",")
            for k in parts:
                k = k.strip()
                if k:
                    fk = _derive_fernet_key(k) if name == "SECRET_KEY" else _validate_or_derive_fernet_key(k)
                    if fk not in keys:
                        keys.append(fk)
    return keys


@lru_cache(maxsize=1)
def _legacy_cipher() -> Optional[MultiFernet]:
    keys = _legacy_fernet_keys()
    return MultiFernet([Fernet(k) for k in keys]) if keys else None


def reset_cipher_cache() -> None:
    """Clears the cached legacy Fernet cipher (useful during tests or key changes)."""
    _legacy_cipher.cache_clear()


def _is_fernet_token(val: Any) -> bool:
    return isinstance(val, str) and val.startswith("gAAAAA") and len(val) >= 60


# ── Public API ──────────────────────────────────────────────────────────────

def encrypt_credential(plain_text: str) -> str:
    """Encrypts plaintext with AES-256-CBC(sha256(SECRET_KEY)) — the Account Manager format."""
    if plain_text is None or plain_text == "":
        return ""
    return aes_encrypt(plain_text)


def decrypt_credential(cipher_text: str) -> str:
    """
    Decrypts a stored value. Tries, in order: legacy Fernet token, AES (Account Manager format),
    legacy plain base64. Raises ValueError if none work.
    """
    if not cipher_text:
        return ""

    if _is_fernet_token(cipher_text):
        cipher = _legacy_cipher()
        if cipher is not None:
            try:
                return cipher.decrypt(cipher_text.encode("utf-8")).decode("utf-8")
            except (InvalidToken, Exception):
                pass

    try:
        return aes_decrypt(cipher_text)
    except ValueError:
        pass

    # Legacy fallback: plain base64 stored by the old dev fallback
    try:
        raw = base64.b64decode(cipher_text.encode("utf-8"), validate=True).decode("utf-8")
        if _readable(raw):
            return raw
    except Exception:
        pass

    raise ValueError("Decryption failed: invalid token, bad key, or corrupted ciphertext.")


def rotate_credential_token(cipher_text: str) -> str:
    """
    Re-encrypts a stored value with the current SECRET_KEY in the AES format.
    Migrates legacy Fernet tokens and values encrypted with DBS_LEGACY_SECRET_KEYS.
    Returns the same string if it is already current.
    """
    if not cipher_text:
        return ""
    return encrypt_credential(decrypt_credential(cipher_text))


def encrypt_payload(payload: Dict[str, Any]) -> str:
    """Serializes a dict to JSON and encrypts it."""
    if not payload:
        return ""
    raw_json = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    return encrypt_credential(raw_json)


def decrypt_payload(cipher_text: str) -> Dict[str, Any]:
    """Decrypts an encrypted payload back into a dict. Returns {} on empty or failed decryption."""
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


def merge_masked_secrets(incoming: Dict[str, Any], existing: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns `incoming` with every masked placeholder replaced by the value from `existing`
    (recursively). Masked fields with no existing value are dropped, so a mask is never saved.
    """
    merged: Dict[str, Any] = {}
    existing = existing if isinstance(existing, dict) else {}
    for k, v in (incoming or {}).items():
        if is_masked_value(v):
            if k in existing:
                merged[k] = existing[k]
        elif isinstance(v, dict):
            merged[k] = merge_masked_secrets(v, existing.get(k) if isinstance(existing.get(k), dict) else {})
        else:
            merged[k] = v
    return merged


def is_encrypted_token(val: Any) -> bool:
    """True if the value is ciphertext this module can decrypt (AES format or legacy Fernet)."""
    if not isinstance(val, str) or not val:
        return False
    if _is_fernet_token(val):
        return True
    if len(val) < 24:          # one AES block base64-encoded is 24 chars
        return False
    try:
        aes_decrypt(val, strict=True)
        return True
    except Exception:
        return False


_SECRET_KEY_PARTS = ("password", "secret", "token", "api_key", "apikey")
# Names that contain one of the words above but hold a setting, not a secret.
_NON_SECRET_KEYS = {"max_tokens", "max_new_tokens", "tokenizer"}
_NON_SECRET_SUFFIXES = ("_url", "_endpoint", "_header", "_path", "_type")
# A node linked to a vaulted credential (credential_id) must not keep its own copy of these.
_VAULTED_KEY_PARTS = (
    "password", "secret", "api_key", "apikey", "access_token", "refresh_token",
    "private_key", "auth_token", "bearer_token", "service_account_key",
)
MASK_PLACEHOLDER = "••••••••"


def is_secret_config_key(key: Any) -> bool:
    """True when a node config setting with this name holds a credential."""
    k = str(key).lower().strip()
    if k in SENSITIVE_FIELD_NAMES:
        return True
    if k in _NON_SECRET_KEYS or k.endswith(_NON_SECRET_SUFFIXES):
        return False
    return any(p in k for p in _SECRET_KEY_PARTS)


def encrypt_node_config_secrets(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Encrypts plaintext passwords / tokens / secrets in a node config before saving, at any
    depth (nested settings such as {"config": {...}} or {"fallback_values": {...}} and dicts
    inside lists). Masked placeholders are dropped instead of being stored. Returns a copy.
    """
    if not isinstance(config, dict):
        return config
    cleaned: Dict[str, Any] = {}
    for k, v in config.items():
        if isinstance(v, str) and v.strip() and is_secret_config_key(k):
            if is_masked_value(v):
                continue
            cleaned[k] = v if is_encrypted_token(v) else encrypt_credential(v)
        elif isinstance(v, dict):
            cleaned[k] = encrypt_node_config_secrets(v)
        elif isinstance(v, list):
            cleaned[k] = [encrypt_node_config_secrets(i) if isinstance(i, dict) else i for i in v]
        else:
            cleaned[k] = v
    return cleaned


def _map_encrypted(config: Dict[str, Any], fn, top: bool) -> Dict[str, Any]:
    """
    Applies fn to every encrypted value of a node config: any top-level value (as before), and
    nested values whose setting name is a secret (mirrors encrypt_node_config_secrets).
    """
    out: Dict[str, Any] = {}
    for k, v in config.items():
        if isinstance(v, str):
            if (top or is_secret_config_key(k)) and is_encrypted_token(v):
                v = fn(k, v)
        elif isinstance(v, dict):
            v = _map_encrypted(v, fn, top=False)
        elif isinstance(v, list):
            v = [_map_encrypted(i, fn, top=False) if isinstance(i, dict) else i for i in v]
        out[k] = v
    return out


def decrypt_node_config_secrets(config: Dict[str, Any]) -> Dict[str, Any]:
    """Decrypts encrypted fields of a node config in memory (nested ones too) so recipes can authenticate at runtime."""
    if not isinstance(config, dict):
        return config

    def _decrypt(k: Any, v: str) -> str:
        try:
            return decrypt_credential(v)
        except Exception as e:
            logger.warning(f"Failed to decrypt node config field '{k}': {e}")
            return v

    return _map_encrypted(config, _decrypt, top=True)


def mask_node_config_secrets(config: Dict[str, Any]) -> Dict[str, Any]:
    """Replaces encrypted fields of a node config (nested ones too) with a placeholder for the frontend."""
    if not isinstance(config, dict):
        return config
    return _map_encrypted(config, lambda k, v: MASK_PLACEHOLDER, top=True)


def _protect_node_config(cfg: Any) -> Any:
    if not isinstance(cfg, dict):
        return cfg
    if cfg.get("credential_id"):
        cfg = {k: v for k, v in cfg.items()
               if not any(p in str(k).lower() for p in _VAULTED_KEY_PARTS)}
    inner = cfg.get("config")
    if isinstance(inner, dict) and inner.get("credential_id"):
        cfg = {**cfg, "config": _protect_node_config(inner)}
    return encrypt_node_config_secrets(cfg)


def protect_graph_secrets(nodes: Any, node_configs: Any):
    """
    Copy of a stored graph (nodes list, node_configs dict) that is safe to write to the database:
    inline credentials encrypted at any depth, masked placeholders dropped, and secrets removed
    from nodes that use a vaulted credential (credential_id). Handles node configs at
    node["data"]["config"] and node["config"], and node_configs values that are flat or
    wrapped as {"config": {...}}. Anything that isn't a list / dict is returned unchanged.
    """
    new_configs = node_configs
    if isinstance(node_configs, dict):
        new_configs = {nid: _protect_node_config(c) for nid, c in node_configs.items()}
    new_nodes = nodes
    if isinstance(nodes, list):
        new_nodes = []
        for n in nodes:
            if isinstance(n, dict):
                n = dict(n)
                if isinstance(n.get("config"), dict):
                    n["config"] = _protect_node_config(n["config"])
                data = n.get("data")
                if isinstance(data, dict) and isinstance(data.get("config"), dict):
                    n["data"] = {**data, "config": _protect_node_config(data["config"])}
            new_nodes.append(n)
    return new_nodes, new_configs


# (nodes key, node_configs key) pairs a stored execution dict (workflows.last_execution) may hold
_EXECUTION_GRAPH_KEYS = (("snapshot_nodes", "snapshot_node_configs"), ("nodes", "node_configs"))


def protect_execution_secrets(execution: Any) -> Any:
    """
    Copy of a stored execution dict (workflows.last_execution) with protect_graph_secrets applied
    to the graph snapshot inside it. Run results (metrics, node outputs, logs) are left as they are.
    """
    if not isinstance(execution, dict):
        return execution
    out = dict(execution)
    for nodes_key, configs_key in _EXECUTION_GRAPH_KEYS:
        if nodes_key in out or configs_key in out:
            out[nodes_key], out[configs_key] = protect_graph_secrets(out.get(nodes_key), out.get(configs_key))
            for key in (nodes_key, configs_key):     # don't add keys that weren't there
                if key not in execution:
                    out.pop(key, None)
    return out
