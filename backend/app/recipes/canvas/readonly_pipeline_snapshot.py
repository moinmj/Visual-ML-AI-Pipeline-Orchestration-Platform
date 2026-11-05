"""
Shareable Read-Only Pipeline Snapshot Recipe (Canvas Tools -> Versioning & Reuse).

Generates an immutable, shareable read-only snapshot manifest of a pipeline DAG,
with automated credential/token redaction, watermark branding, access controls, and expiration tokens.
"""

import hashlib
import json
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional

try:
    import pandas as pd
except ImportError:
    pd = None

from backend.app.recipes.base.recipe import BaseRecipe, RecipePort


class ReadonlyPipelineSnapshotRecipe(BaseRecipe):
    recipe_id = "readonly_pipeline_snapshot"
    name = "Shareable Read-Only Pipeline Snapshot"
    version = "1.0.0"
    category = "canvas"
    group = "Canvas Tools"
    subgroup = "Versioning & Reuse"
    description = "Shares a read-only view of the pipeline via a link."
    input_types = ["dataframe"]
    output_types = ["dataframe", "report"]

    inputs: List[RecipePort] = [
        RecipePort(
            id="input",
            label="Pipeline Table (Optional)",
            type="dataframe",
            required=False,
            max_connections=1,
            description="Upstream dataset to include in read-only snapshot preview."
        )
    ]

    outputs: List[RecipePort] = [
        RecipePort(
            id="output",
            label="Passthrough Table",
            type="dataframe",
            description="Preserved table passed unchanged to downstream nodes."
        ),
        RecipePort(
            id="report",
            label="Shareable Snapshot Manifest",
            type="report",
            description="Public read-only link, access token, and redacted pipeline summary."
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "snapshot_title": {
                    "type": "string",
                    "title": "Shareable Link Headline",
                    "input_type": "text",
                    "default": "Executive Pipeline Review View",
                    "description": "Public title presented on the read-only canvas banner."
                },
                "access_level": {
                    "type": "string",
                    "title": "Access Authorization Level",
                    "input_type": "dropdown",
                    "enum": ["Public Read-Only", "Organization Only", "Restricted Key"],
                    "default": "Organization Only",
                    "description": "Security access boundary required to open this read-only link."
                },
                "mask_sensitive_credentials": {
                    "type": "boolean",
                    "title": "Auto-Redact Passwords & Secrets",
                    "input_type": "checkbox",
                    "default": True,
                    "description": "Automatically mask database passwords, API tokens, and secret keys with asterisks."
                },
                "watermark_text": {
                    "type": "string",
                    "title": "Canvas Watermark Text",
                    "input_type": "text",
                    "default": "INTERNAL PREVIEW - NOT FOR EXTERNAL REDISTRIBUTION",
                    "description": "Subtle background watermark stamped across read-only canvas viewport."
                },
                "expiration_days": {
                    "type": "integer",
                    "title": "Link Expiration (Days)",
                    "input_type": "number",
                    "default": 30,
                    "minimum": 1,
                    "maximum": 365,
                    "description": "Number of days before the shared read-only link automatically expires."
                },
                "allow_export_data": {
                    "type": "boolean",
                    "title": "Allow Viewer Data CSV Export",
                    "input_type": "checkbox",
                    "default": False,
                    "description": "Whether viewers can download CSV previews of intermediate node outputs."
                }
            }
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        upstream_df = inputs.get("dataframe") if "dataframe" in inputs else inputs.get("input")

        # Edge Case 1: Config extraction & safe boundaries
        title = str(config.get("snapshot_title", "Executive Pipeline Review View")).strip() or "Read-Only View"
        access_level = str(config.get("access_level", "Organization Only"))
        mask_creds = bool(config.get("mask_sensitive_credentials", True))
        watermark = str(config.get("watermark_text", "INTERNAL PREVIEW")).strip() or "CONFIDENTIAL"
        allow_export = bool(config.get("allow_export_data", False))

        try:
            exp_days = max(1, min(365, int(config.get("expiration_days", 30))))
        except (ValueError, TypeError):
            exp_days = 30

        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(days=exp_days)
        share_token = f"share_{uuid.uuid4().hex[:16]}"
        share_url = f"https://platform.datalect.internal/viewer/{share_token}"

        # Edge Case 2: Deep Credential Redaction function
        SENSITIVE_KEYWORDS = ["pass", "secret", "token", "key", "cred", "auth", "pwd", "hash"]
        redacted_count = 0

        def _redact_dict(obj: Any) -> Any:
            nonlocal redacted_count
            if isinstance(obj, dict):
                cleaned = {}
                for k, v in obj.items():
                    k_str = str(k).lower()
                    if mask_creds and any(kw in k_str for kw in SENSITIVE_KEYWORDS) and isinstance(v, (str, int, float)):
                        cleaned[k] = "********"
                        redacted_count += 1
                    else:
                        cleaned[k] = _redact_dict(v)
                return cleaned
            elif isinstance(obj, (list, tuple)):
                return [_redact_dict(item) for item in obj]
            return obj

        sanitized_context = _redact_dict(context) if context else {"nodes": [], "edges": []}

        # Data shape summary
        data_preview_rows = len(upstream_df) if upstream_df is not None and pd is not None and isinstance(upstream_df, pd.DataFrame) else 0

        md_report = (
            f"### 🔗 Shareable Read-Only Link: **{title}**\n"
            f"**Access Level:** `{access_level}` | **Status:** `ACTIVE`\n\n"
            f"**Share URL:**\n`{share_url}`\n\n"
            f"- **Token ID:** `{share_token}`\n"
            f"- **Created:** `{now.isoformat()[:19]} UTC`\n"
            f"- **Expires:** `{expires_at.isoformat()[:10]} ({exp_days} days remaining)`\n"
            f"- **Watermark:** `{watermark}`\n"
            f"- **Sensitive Keys Redacted:** `{redacted_count}` values masked with `********`\n"
            f"- **Data Export Allowed:** `{'Yes' if allow_export else 'No (Protected)'}`"
        )

        result: Dict[str, Any] = {
            "snapshot_title": title,
            "share_token": share_token,
            "share_url": share_url,
            "access_level": access_level,
            "expires_at": expires_at.isoformat(),
            "redacted_credentials_count": redacted_count,
            "watermark": watermark,
            "allow_export_data": allow_export,
            "sanitized_pipeline_manifest": sanitized_context,
            "metrics": {
                "expiration_days": exp_days,
                "redacted_keys": redacted_count,
                "preview_rows": data_preview_rows,
                "allow_export": int(allow_export)
            },
            "report": {
                "title": f"Share: {title}",
                "share_url": share_url,
                "markdown": md_report,
                "timestamp": now.isoformat()
            },
            "output_summary": {
                "status": "LINK_ACTIVE",
                "access": access_level,
                "expires": expires_at.isoformat()[:10],
                "token": share_token[:10] + "..."
            }
        }

        if upstream_df is not None:
            result["dataframe"] = upstream_df
            result["output"] = upstream_df

        return result
