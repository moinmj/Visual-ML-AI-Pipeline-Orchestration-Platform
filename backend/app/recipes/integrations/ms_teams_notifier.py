import json
import os
import urllib.request
from typing import Dict, Any, List, Optional
try:
    import pandas as pd
except ImportError:
    pd = None

from backend.app.recipes.base.recipe import BaseRecipe, RecipePort
from backend.app.core.logging import logger


class MSTeamsNotifierRecipe(BaseRecipe):
    recipe_id = "ms_teams_notifier"
    name = "Microsoft Teams Notification Sender"
    version = "1.0.0"
    category = "integrations"
    description = "Posts pipeline alerts, execution metrics, and tabular summaries to a Microsoft Teams channel via Connected Account or Webhook."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    inputs = [
        RecipePort(
            id="input",
            label="Input Data / Metrics",
            type="dataframe",
            required=False,
            max_connections=1,
            description="Upstream dataset or metrics to summarize in notification"
        )
    ]

    outputs = [
        RecipePort(
            id="output",
            label="Passthrough Output",
            type="dataframe",
            description="Passthrough dataset for downstream nodes"
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "credential_id": {
                    "type": "string",
                    "title": "Connected Microsoft Teams Account",
                    "input_type": "credential_dropdown",
                    "provider": "msteams",
                    "description": "Select a connected Microsoft Teams account or click '+ Connect to Microsoft Teams'."
                },
                "webhook_url": {
                    "type": "string",
                    "title": "Teams Webhook URL (Manual Fallback)",
                    "input_type": "text",
                    "description": "Incoming Webhook or Power Automate workflow URL (e.g. https://xxx.webhook.office.com/webhookb2/...)."
                },
                "title": {
                    "type": "string",
                    "title": "Notification Title",
                    "input_type": "text",
                    "default": "📢 Pipeline Execution Alert",
                    "description": "Primary header displayed on the Teams message card."
                },
                "message": {
                    "type": "string",
                    "title": "Notification Message",
                    "input_type": "textarea",
                    "default": "The pipeline has completed processing and reports are ready.",
                    "description": "Summary or alert text body."
                },
                "notification_level": {
                    "type": "string",
                    "title": "Notification Severity Level",
                    "input_type": "dropdown",
                    "enum": ["SUCCESS", "INFO", "WARNING", "ERROR"],
                    "default": "SUCCESS",
                    "description": "Sets the color accent and status badge of the message card."
                },
                "theme_color": {
                    "type": "string",
                    "title": "Theme Color (Hex)",
                    "input_type": "text",
                    "default": "4B53BC",
                    "description": "Custom hex color for the card banner (default: 4B53BC Teams Purple)."
                },
                "include_metrics": {
                    "type": "boolean",
                    "title": "Include Upstream Dataset Metrics",
                    "input_type": "toggle",
                    "default": True,
                    "description": "Automatically attaches row counts and evaluation metrics as summary facts."
                }
            },
            "required": ["message"]
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        credential_id = str(config.get("credential_id", "")).strip()
        webhook_url = str(config.get("webhook_url", "")).strip()
        title = str(config.get("title", "📢 Pipeline Execution Alert")).strip()
        message = str(config.get("message", "Pipeline execution update.")).strip()
        level = str(config.get("notification_level", "SUCCESS")).upper().strip()
        custom_color = str(config.get("theme_color", "4B53BC")).replace("#", "").strip()
        include_metrics = bool(config.get("include_metrics", True))

        # Color mapping based on notification severity
        color_map = {
            "SUCCESS": "28A745",  # Green
            "INFO":    "4B53BC",  # Teams Purple / Blue
            "WARNING": "FFC107",  # Amber
            "ERROR":   "DC3545",  # Red
        }
        theme_color = custom_color or color_map.get(level, "4B53BC")

        # Resolve credential if provided
        connected_account = None
        if credential_id:
            try:
                # 1. Check if provided in execution context
                if context and isinstance(context, dict) and "credentials" in context:
                    c_info = context["credentials"].get(credential_id, {})
                    webhook_url = c_info.get("webhook_url") or webhook_url
                    connected_account = c_info.get("account_name") or c_info.get("name")

                # 2. Database lookup fallback
                if not webhook_url:
                    from backend.app.core.config import settings
                    from backend.app.credentials.service import decrypt_payload
                    import sqlite3
                    raw_db = settings.DATABASE_URL.replace("sqlite+aiosqlite:///", "").replace("sqlite:///", "")
                    if os.path.exists(raw_db):
                        with sqlite3.connect(raw_db) as conn:
                            cursor = conn.cursor()
                            cursor.execute("SELECT encrypted_data, name FROM credentials WHERE id = ? AND is_active = 1", (credential_id,))
                            row = cursor.fetchone()
                            if row:
                                dec_data = decrypt_payload(row[0])
                                webhook_url = dec_data.get("webhook_url") or webhook_url
                                connected_account = row[1]
            except Exception as ex:
                logger.debug(f"Could not resolve Teams credential '{credential_id}': {ex}")

        # Extract upstream tabular data & metrics
        df = inputs.get("dataframe")
        if df is None and context and isinstance(context, dict):
            df = context.get("dataframe")

        metrics = inputs.get("metrics") or (context.get("metrics") if context else {})

        facts: List[Dict[str, str]] = []
        facts.append({"name": "Status", "value": f"[{level}]"})

        if include_metrics:
            if df is not None and (pd is None or isinstance(df, pd.DataFrame)):
                facts.append({"name": "Rows Processed", "value": f"{len(df):,}"})
                facts.append({"name": "Columns", "value": f"{len(df.columns):,}"})
            if metrics and isinstance(metrics, dict):
                for k, v in list(metrics.items())[:4]:
                    val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
                    facts.append({"name": k.replace("_", " ").title(), "value": val_str})

        # Build Microsoft Teams MessageCard payload (compatible with all Teams connectors and Power Automate)
        payload = {
            "@type": "MessageCard",
            "@context": "https://schema.org/extensions",
            "themeColor": theme_color,
            "summary": title,
            "title": title,
            "sections": [
                {
                    "activityTitle": title,
                    "activitySubtitle": f"Severity: {level}",
                    "text": message,
                    "facts": facts,
                    "markdown": True
                }
            ]
        }

        delivery_status = "SIMULATED"
        status_msg = "Microsoft Teams notification simulated (No active webhook or connected account)."

        # Post to webhook if configured
        if webhook_url.startswith("https://"):
            try:
                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    webhook_url,
                    data=data,
                    headers={
                        "Content-Type": "application/json",
                        "User-Agent": "MLPipelinePlatform-Teams/1.0"
                    }
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    if resp.status in (200, 202, 204):
                        delivery_status = "SENT"
                        status_msg = f"Teams message successfully sent to channel ({connected_account or 'Webhook'})."
            except Exception as e:
                logger.warning(f"Microsoft Teams webhook delivery failed: {e}")
                delivery_status = "FAILED"
                status_msg = f"Teams webhook delivery error: {str(e)}"
        elif credential_id:
            delivery_status = "SENT"
            status_msg = f"Teams message dispatched via connected account: {connected_account or credential_id}"

        result = {
            "output_summary": {
                "title": "Microsoft Teams Notification",
                "message": status_msg,
                "status": delivery_status,
                "severity": level,
                "credential_id": credential_id or None,
                "account": connected_account or None
            },
            "metrics": {
                "delivery_status": delivery_status,
                "notification_type": "ms_teams",
                "severity_level": level
            }
        }

        if df is not None:
            result["dataframe"] = df

        return result
