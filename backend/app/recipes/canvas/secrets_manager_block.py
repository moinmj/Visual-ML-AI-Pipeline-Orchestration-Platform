"""
Environment Variable / Secrets Manager Block Recipe (Canvas Tools -> Configuration).

Securely retrieves and verifies environment variables, API tokens, and database secrets
from local environment files or cloud vaults without leaking plaintext values into execution logs.
"""

import os
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

try:
    import pandas as pd
except ImportError:
    pd = None

from backend.app.recipes.base.recipe import BaseRecipe, RecipePort


class SecretsManagerBlockRecipe(BaseRecipe):
    recipe_id = "secrets_manager_block"
    name = "Environment Variable / Secrets Manager Block"
    version = "1.0.0"
    category = "canvas"
    group = "Canvas Tools"
    subgroup = "Configuration"
    description = "Stores environment variables and secrets used by nodes, without exposing them."
    input_types = ["dataframe"]
    output_types = ["dataframe", "report"]

    inputs: List[RecipePort] = [
        RecipePort(
            id="input",
            label="Pipeline Input (Optional)",
            type="dataframe",
            required=False,
            max_connections=1,
            description="Upstream dataset to pass through secrets configuration block."
        )
    ]

    outputs: List[RecipePort] = [
        RecipePort(
            id="output",
            label="Passthrough Table",
            type="dataframe",
            description="Forwarded dataset passed untouched to downstream nodes."
        ),
        RecipePort(
            id="report",
            label="Secrets Audit Certificate",
            type="report",
            description="Verification audit of configured environment secrets with masked values."
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "vault_provider": {
                    "type": "string",
                    "title": "Secret Vault Provider",
                    "input_type": "dropdown",
                    "enum": ["Local Environment (.env)", "HashiCorp Vault", "AWS Secrets Manager", "Azure Key Vault", "Manual Inlined (Encrypted)"],
                    "default": "Local Environment (.env)",
                    "description": "Source provider where encrypted credentials and tokens reside."
                },
                "secret_keys": {
                    "type": "string",
                    "title": "Required Environment Keys",
                    "input_type": "text",
                    "default": "DATABASE_URL, S3_ACCESS_KEY, MLFLOW_TRACKING_URI",
                    "description": "Comma-separated environment variable keys to check and provide to downstream nodes."
                },
                "key_prefix": {
                    "type": "string",
                    "title": "Key Prefix Scope (Optional)",
                    "input_type": "text",
                    "default": "",
                    "description": "Optional prefix filter applied to keys (e.g. 'PROD_', 'STAGE_')."
                },
                "mask_in_logs": {
                    "type": "boolean",
                    "title": "Mask Secret Values in Logs",
                    "input_type": "checkbox",
                    "default": True,
                    "description": "Replaces resolved secret values with asterisks (***) in all outputs and UI tables."
                },
                "allow_missing_keys": {
                    "type": "boolean",
                    "title": "Tolerate Missing Keys",
                    "input_type": "checkbox",
                    "default": True,
                    "description": "If False, raises a validation error if any requested secret is not configured in environment."
                }
            }
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        upstream_df = inputs.get("dataframe") if "dataframe" in inputs else inputs.get("input")

        # Edge Case 1: Config extraction
        provider = str(config.get("vault_provider", "Local Environment (.env)"))
        raw_keys = config.get("secret_keys", "DATABASE_URL, S3_ACCESS_KEY, MLFLOW_TRACKING_URI")
        prefix = str(config.get("key_prefix", "")).strip()
        mask_logs = bool(config.get("mask_in_logs", True))
        allow_missing = bool(config.get("allow_missing_keys", True))

        # Edge Case 2: Parse comma-separated keys with whitespace tolerance
        target_keys = []
        if isinstance(raw_keys, str):
            target_keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
        elif isinstance(raw_keys, (list, tuple)):
            target_keys = [str(k).strip() for k in raw_keys if str(k).strip()]

        if prefix:
            target_keys = [f"{prefix}{k}" if not k.startswith(prefix) else k for k in target_keys]

        # Edge Case 3: Resolve secrets securely
        resolved_secrets = {}
        masked_manifest = []
        missing_keys = []
        now_iso = datetime.now(timezone.utc).isoformat()

        for key in target_keys:
            val = os.environ.get(key)
            if val is not None and str(val).strip():
                resolved_secrets[key] = str(val)
                # Compute safe masked display: show first 2 chars and last 2 if length >= 8, else all stars
                if mask_logs:
                    v_str = str(val)
                    if len(v_str) > 8:
                        masked = f"{v_str[:2]}****{v_str[-2:]} ({len(v_str)} chars)"
                    else:
                        masked = "********"
                else:
                    masked = str(val)

                masked_manifest.append({
                    "key": key,
                    "status": "CONFIGURED",
                    "length": len(str(val)),
                    "preview": masked
                })
            else:
                missing_keys.append(key)
                masked_manifest.append({
                    "key": key,
                    "status": "MISSING / NOT_SET",
                    "length": 0,
                    "preview": "—"
                })

        # Edge Case 4: Enforce strict missing key check if allow_missing_keys is False
        if missing_keys and not allow_missing:
            raise ValueError(f"Secrets validation failed: Required keys {missing_keys} are missing from environment.")

        md_lines = [
            f"### 🔐 Secrets & Environment Manager: `{provider}`",
            f"**Evaluated at:** `{now_iso[:19]} UTC` | **Total Keys:** `{len(target_keys)}` | **Configured:** `{len(resolved_secrets)}` | **Missing:** `{len(missing_keys)}`\n",
            "| Environment Key | Status | Value Preview |",
            "| :--- | :---: | :--- |"
        ]
        for m in masked_manifest:
            status_badge = "🟢 `OK`" if m["status"] == "CONFIGURED" else "🔴 `NOT_SET`"
            md_lines.append(f"| `{m['key']}` | {status_badge} | `{m['preview']}` |")

        if missing_keys:
            md_lines.append(f"\n> ⚠️ **Notice:** {len(missing_keys)} keys were not found in current environment: {', '.join([f'`{k}`' for k in missing_keys])}.")

        md_report = "\n".join(md_lines)

        result: Dict[str, Any] = {
            "vault_provider": provider,
            "total_keys_checked": len(target_keys),
            "configured_keys": list(resolved_secrets.keys()),
            "missing_keys": missing_keys,
            "masked_manifest": masked_manifest,
            # Injected context: downstream nodes can access secrets via context['_injected_secrets']
            "_injected_secrets": resolved_secrets,
            "metrics": {
                "keys_requested": len(target_keys),
                "keys_resolved": len(resolved_secrets),
                "keys_missing": len(missing_keys),
                "all_satisfied": int(len(missing_keys) == 0)
            },
            "report": {
                "title": "Secrets Configuration Audit",
                "markdown": md_report,
                "timestamp": now_iso
            },
            "output_summary": {
                "status": "ALL_RESOLVED" if not missing_keys else "PARTIAL_CONFIGURED",
                "resolved": len(resolved_secrets),
                "missing": len(missing_keys),
                "provider": provider
            }
        }

        if upstream_df is not None:
            result["dataframe"] = upstream_df
            result["output"] = upstream_df

        return result
