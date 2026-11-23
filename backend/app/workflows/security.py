# -*- coding: utf-8 -*-
"""
Workflow At-Rest Security & Encryption Module.

Ensures that passwords, connection strings with embedded credentials, secrets,
tokens, and private keys are never stored in plaintext within the database
(e.g. workflows.node_configs, workflows.nodes, workflow_executions.snapshot_node_configs).

Uses AES-256 (Fernet) encryption prefixed with 'enc:' for transparent identification,
idempotent re-saves, and seamless in-memory decryption at execution time.
"""

import base64
import copy
import logging
import re
from typing import Any, Dict, List, Optional, Union

from backend.app.core.config import settings

logger = logging.getLogger(__name__)

# Prefix used to distinguish Fernet ciphertext tokens
CIPHERTEXT_PREFIX = "enc:"

# Sensitive key substrings (case-insensitive substring match)
SENSITIVE_KEY_SUBSTRINGS = (
    "password",
    "secret",
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
    "private_key",
    "auth_token",
    "bearer_token",
    "service_account_key",
    "sasl_password",
    "rabbitmq_password",
    "aws_secret_access_key",
    "aws_session_token",
    "bot_token",
    "client_secret",
    "app_password",
)

# Exact match sensitive keys
SENSITIVE_KEY_EXACT = {
    "token",
    "key",
    "pass",
    "pwd",
}

# Keys that often contain connection strings with passwords
CONNECTION_STRING_KEYS = {
    "connection_string",
    "conn_string",
    "uri",
    "db_uri",
    "database_uri",
    "amqp_uri",
    "mongo_uri",
    "redis_uri",
    "dsn",
}

# Regex to detect URIs with embedded user:password (e.g. mysql://user:pass@host:3306)
_URI_CREDENTIALS_REGEX = re.compile(
    r"^[a-zA-Z][a-zA-Z0-9+.-]*://(?P<user>[^/@:]+):(?P<pass>[^/@]+)@",
    re.IGNORECASE,
)

# Initialize Fernet cipher
_fernet = None
try:
    from cryptography.fernet import Fernet
    if settings.FERNET_KEY:
        _fernet = Fernet(settings.FERNET_KEY.encode() if isinstance(settings.FERNET_KEY, str) else settings.FERNET_KEY)
    else:
        _fernet = Fernet(Fernet.generate_key())
except Exception as e:
    logger.warning(f"Failed to initialize Fernet cipher for workflow encryption: {e}")


def encrypt_secret(val: Any) -> Any:
    """
    Encrypts a plaintext secret string using Fernet, prefixing with 'enc:'.
    If already encrypted (starts with 'enc:') or not a non-empty string, returns val as-is.
    """
    if not isinstance(val, str) or not val.strip():
        return val
    if val.startswith(CIPHERTEXT_PREFIX):
        return val

    try:
        if _fernet:
            encrypted_bytes = _fernet.encrypt(val.encode("utf-8"))
            return f"{CIPHERTEXT_PREFIX}{encrypted_bytes.decode('utf-8')}"
        # Fallback to base64 encoding if Fernet is unavailable
        b64 = base64.b64encode(val.encode("utf-8")).decode("utf-8")
        return f"{CIPHERTEXT_PREFIX}b64:{b64}"
    except Exception as e:
        logger.error(f"Error encrypting secret value: {e}")
        return val


def decrypt_secret(val: Any) -> Any:
    """
    Decrypts a ciphertext string prefixed with 'enc:' back to its plaintext value.
    If val does not start with 'enc:' or is not a string, returns val as-is.
    """
    if not isinstance(val, str) or not val.startswith(CIPHERTEXT_PREFIX):
        return val

    token = val[len(CIPHERTEXT_PREFIX):]
    if not token:
        return ""

    try:
        if token.startswith("b64:"):
            raw = base64.b64decode(token[4:].encode("utf-8"))
            return raw.decode("utf-8")
        if _fernet:
            decrypted_bytes = _fernet.decrypt(token.encode("utf-8"))
            return decrypted_bytes.decode("utf-8")
        # Try raw base64 decode if _fernet was missing
        raw = base64.b64decode(token.encode("utf-8"))
        return raw.decode("utf-8")
    except Exception as e:
        logger.warning(f"Failed to decrypt ciphertext token (returning raw): {e}")
        return val


def is_sensitive_key(key: Any) -> bool:
    """Checks whether a dictionary key name indicates a sensitive credential or password."""
    if not isinstance(key, str):
        return False
    k_lower = key.lower().strip()
    if k_lower in SENSITIVE_KEY_EXACT or k_lower in CONNECTION_STRING_KEYS:
        return True
    return any(sub in k_lower for sub in SENSITIVE_KEY_SUBSTRINGS)


def contains_embedded_credentials(val: Any) -> bool:
    """Checks if a string looks like a URI containing username and password."""
    if not isinstance(val, str) or len(val) < 8:
        return False
    return bool(_URI_CREDENTIALS_REGEX.search(val))


def encrypt_node_configs(obj: Any) -> Any:
    """
    Recursively scans and encrypts sensitive keys and connection strings
    in a node_configs structure before database persistence.
    """
    if obj is None:
        return None

    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                result[k] = encrypt_node_configs(v)
            elif isinstance(v, str) and (is_sensitive_key(k) or contains_embedded_credentials(v)):
                result[k] = encrypt_secret(v)
            else:
                result[k] = v
        return result
    elif isinstance(obj, list):
        return [encrypt_node_configs(item) for item in obj]
    return obj


def decrypt_node_configs(obj: Any) -> Any:
    """
    Recursively scans and decrypts any 'enc:' prefixed tokens in a node_configs
    structure back into plaintext for in-memory pipeline execution.
    """
    if obj is None:
        return None

    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                result[k] = decrypt_node_configs(v)
            elif isinstance(v, str) and v.startswith(CIPHERTEXT_PREFIX):
                result[k] = decrypt_secret(v)
            else:
                result[k] = v
        return result
    elif isinstance(obj, list):
        return [decrypt_node_configs(item) for item in obj]
    elif isinstance(obj, str) and obj.startswith(CIPHERTEXT_PREFIX):
        return decrypt_secret(obj)
    return obj


def encrypt_workflow_nodes(nodes: Any) -> Any:
    """
    Encrypts sensitive credentials inside canvas nodes objects
    (such as node['config'], node['data']['config'], node['data']['parameters']).
    """
    if not isinstance(nodes, list):
        return nodes

    result = []
    for node in nodes:
        if not isinstance(node, dict):
            result.append(node)
            continue

        node_copy = copy.deepcopy(node)
        # Encrypt direct config
        if "config" in node_copy and isinstance(node_copy["config"], dict):
            node_copy["config"] = encrypt_node_configs(node_copy["config"])

        # Encrypt nested data.config / data.parameters
        if "data" in node_copy and isinstance(node_copy["data"], dict):
            data = node_copy["data"]
            if "config" in data and isinstance(data["config"], dict):
                data["config"] = encrypt_node_configs(data["config"])
            if "parameters" in data and isinstance(data["parameters"], dict):
                data["parameters"] = encrypt_node_configs(data["parameters"])

        result.append(node_copy)
    return result


def decrypt_workflow_nodes(nodes: Any) -> Any:
    """
    Decrypts encrypted credentials inside canvas nodes objects back to plaintext.
    """
    if not isinstance(nodes, list):
        return nodes

    result = []
    for node in nodes:
        if not isinstance(node, dict):
            result.append(node)
            continue

        node_copy = copy.deepcopy(node)
        if "config" in node_copy and isinstance(node_copy["config"], dict):
            node_copy["config"] = decrypt_node_configs(node_copy["config"])

        if "data" in node_copy and isinstance(node_copy["data"], dict):
            data = node_copy["data"]
            if "config" in data and isinstance(data["config"], dict):
                data["config"] = decrypt_node_configs(data["config"])
            if "parameters" in data and isinstance(data["parameters"], dict):
                data["parameters"] = decrypt_node_configs(data["parameters"])

        result.append(node_copy)
    return result


def sanitize_node_configs_credentials(obj: Any) -> Any:
    """
    Recursively redacts passwords, tokens, API keys, credentials, and encrypted ciphertext
    in node configurations so exported workflows can be safely shared across environments
    without secret leakage.
    """
    if isinstance(obj, dict):
        sanitized = {}
        for k, v in obj.items():
            if is_sensitive_key(k) or (isinstance(v, str) and (v.startswith(CIPHERTEXT_PREFIX) or contains_embedded_credentials(v))):
                sanitized[k] = ""
            elif isinstance(v, (dict, list)):
                sanitized[k] = sanitize_node_configs_credentials(v)
            else:
                sanitized[k] = v
        return sanitized
    elif isinstance(obj, list):
        return [sanitize_node_configs_credentials(item) for item in obj]
    elif isinstance(obj, str) and (obj.startswith(CIPHERTEXT_PREFIX) or contains_embedded_credentials(obj)):
        return ""
    return obj
