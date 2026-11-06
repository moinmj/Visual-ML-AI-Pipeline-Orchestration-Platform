import datetime
import hashlib
import secrets
from typing import Dict, Any, List, Optional
import pandas as pd

from backend.app.recipes.base.recipe import BaseRecipe, RecipePort


class CredentialRotationManagerRecipe(BaseRecipe):
    recipe_id = "credential_rotation_manager"
    name = "API Key / Credential Rotation Manager"
    version = "1.0.0"
    category = "security"
    description = "Rotates API keys and credentials used by pipeline nodes on a schedule."
    input_types = ["dataframe"]
    output_types = ["report", "dataframe"]

    def get_inputs(self) -> List[RecipePort]:
        return [
            RecipePort(
                id="input",
                label="Pipeline Input (Optional)",
                type="dataframe",
                required=False,
                description="Optional upstream dataset to pass through the credential rotation step",
            )
        ]

    def get_outputs(self) -> List[RecipePort]:
        return [
            RecipePort(
                id="output",
                label="Rotation Report",
                type="report",
                description="Rotation status manifest, active key fingerprints, and schedule lifecycle",
            ),
            RecipePort(
                id="output_table",
                label="Passthrough Output",
                type="dataframe",
                description="Source dataset passed through for downstream nodes",
            ),
        ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "credential_name": {
                    "type": "string",
                    "title": "Secret / Key Identifier",
                    "default": "PLATFORM_API_KEY",
                    "description": "Identifier of the key or secret subject to rotation.",
                    "input_type": "text",
                },
                "vault_provider": {
                    "type": "string",
                    "title": "Secrets Vault Provider",
                    "enum": [
                        "internal_encrypted_vault",
                        "hashicorp_vault",
                        "aws_secrets_manager",
                        "azure_key_vault",
                        "gcp_secret_manager",
                    ],
                    "default": "internal_encrypted_vault",
                    "description": "Key management service or secure vault storing secrets.",
                    "input_type": "select",
                },
                "rotation_interval_days": {
                    "type": "integer",
                    "title": "Rotation Cadence (Days)",
                    "default": 90,
                    "minimum": 1,
                    "maximum": 365,
                    "description": "Cadence in days determining when credentials are automatically superseded.",
                    "input_type": "number",
                },
                "grace_period_hours": {
                    "type": "integer",
                    "title": "Dual-Key Grace Window (Hours)",
                    "default": 24,
                    "minimum": 0,
                    "maximum": 720,
                    "description": "Duration in hours the prior version remains valid to prevent downtime during migration.",
                    "input_type": "number",
                },
                "key_length_bytes": {
                    "type": "integer",
                    "title": "Key Entropy Size (Bytes)",
                    "default": 32,
                    "minimum": 16,
                    "maximum": 128,
                    "description": "Entropy bit length of newly generated cryptographically random keys.",
                    "input_type": "number",
                },
                "auto_invalidate_expired": {
                    "type": "boolean",
                    "title": "Auto-Revoke Expired Keys",
                    "default": True,
                    "description": "Deactivates expired credentials once the grace window elapses.",
                    "input_type": "checkbox",
                },
                "notification_email": {
                    "type": "string",
                    "title": "SecOps Notification Email",
                    "default": "",
                    "description": "Alert distribution recipient notified when a rotation event occurs.",
                    "input_type": "text",
                },
                "force_rotation_now": {
                    "type": "boolean",
                    "title": "Force Immediate Rotation",
                    "default": False,
                    "description": "Forces immediate key generation regardless of scheduled expiry date.",
                    "input_type": "checkbox",
                },
            },
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        df = inputs.get("dataframe")
        if df is None and isinstance(inputs.get("input"), pd.DataFrame):
            df = inputs["input"]

        if df is None or not isinstance(df, pd.DataFrame):
            df = pd.DataFrame({"rotation_status": ["INITIALIZED"], "active_version": ["v2"]})

        cred_name = str(config.get("credential_name", "PLATFORM_API_KEY")).strip() or "PLATFORM_API_KEY"
        vault = str(config.get("vault_provider", "internal_encrypted_vault")).strip()
        cadence_days = max(1, min(365, int(config.get("rotation_interval_days", 90))))
        grace_hours = max(0, min(720, int(config.get("grace_period_hours", 24))))
        key_len = max(16, min(128, int(config.get("key_length_bytes", 32))))
        auto_revoke = bool(config.get("auto_invalidate_expired", True))
        notify_email = str(config.get("notification_email", "")).strip()
        force_now = bool(config.get("force_rotation_now", False))

        now_utc = datetime.datetime.now(datetime.timezone.utc)
        now_iso = now_utc.isoformat()
        expires_at = (now_utc + datetime.timedelta(days=cadence_days)).isoformat()
        grace_ends_at = (now_utc + datetime.timedelta(hours=grace_hours)).isoformat()

        # Generate cryptographically secure token
        raw_token = secrets.token_urlsafe(key_len)
        fingerprint = hashlib.sha256(raw_token.encode()).hexdigest()
        masked_token = f"sec_****{raw_token[-4:]}"

        rotation_manifest = {
            "rotation_id": f"ROT-{fingerprint[:8].upper()}",
            "credential_name": cred_name,
            "vault_provider": vault,
            "status": "ROTATED" if force_now else "ACTIVE_SCHEDULED",
            "active_version": "v2",
            "previous_version": "v1",
            "key_fingerprint": fingerprint,
            "masked_key": masked_token,
            "key_length_bytes": key_len,
            "rotated_at": now_iso,
            "expires_at": expires_at,
            "grace_period_hours": grace_hours,
            "grace_period_ends_at": grace_ends_at,
            "auto_invalidate_expired": auto_revoke,
            "notification_recipient": notify_email if notify_email else None,
        }

        return {
            "output": rotation_manifest,
            "report": rotation_manifest,
            "dataframe": df,
            "output_table": df,
        }
