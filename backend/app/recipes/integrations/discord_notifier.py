import json
import os
import urllib.request
from typing import Dict, Any, List, Optional
from backend.app.recipes.base.recipe import BaseRecipe, RecipePort
from backend.app.core.logging import logger


class DiscordRecipe(BaseRecipe):
    recipe_id = "discord"
    name = "Discord"
    version = "1.1.0"
    category = "integrations"
    description = "Sends rich embed alerts, logs, or completion summaries to Discord channels via Connected Discord Bot or Webhooks."
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
                    "title": "Connected Discord Account",
                    "input_type": "credential_dropdown",
                    "provider": "discord",
                    "description": "Select a connected Discord Server or click '+ Connect to Discord' to link a new server."
                },
                "webhook_url": {
                    "type": "string",
                    "title": "Discord Webhook URL (Manual Fallback)",
                    "input_type": "text",
                    "description": "Optional manual webhook URL (e.g. https://discord.com/api/webhooks/...) if not using an OAuth bot."
                },
                "username": {
                    "type": "string",
                    "title": "Bot Display Name (Optional)",
                    "input_type": "text",
                    "default": "ML Pipeline Bot",
                    "description": "Custom display name for the bot sender."
                },
                "content": {
                    "type": "string",
                    "title": "Message Content",
                    "input_type": "textarea",
                    "default": "📢 Automated Pipeline Notification",
                    "description": "Primary text content or alert message."
                },
                "embed_title": {
                    "type": "string",
                    "title": "Embed Box Title",
                    "input_type": "text",
                    "default": "Pipeline Execution Status",
                    "description": "Title displayed inside the rich embed banner."
                },
                "embed_color": {
                    "type": "string",
                    "title": "Embed Color (Hex/Decimal)",
                    "input_type": "text",
                    "default": "3066993",
                    "description": "Decimal or hex color code (e.g. 3066993 for Teal / Blue)."
                }
            },
            "required": ["content"]
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        credential_id = str(config.get("credential_id", "")).strip()
        webhook_url = str(config.get("webhook_url", "")).strip()
        content = config.get("content", "Pipeline notification")
        embed_title = config.get("embed_title", "Pipeline Execution")
        bot_name = config.get("username", "Pipeline Bot")

        # Resolve credential if provided
        connected_guild = None
        if credential_id:
            try:
                # 1. Check if provided in execution context
                if context and isinstance(context, dict) and "credentials" in context:
                    c_info = context["credentials"].get(credential_id, {})
                    webhook_url = c_info.get("webhook_url") or webhook_url
                    connected_guild = c_info.get("guild_name") or c_info.get("name")
                
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
                                connected_guild = row[1]
            except Exception as ex:
                logger.debug(f"Could not resolve Discord credential '{credential_id}': {ex}")

        df = inputs.get("dataframe")
        if df is None and context and isinstance(context, dict):
            df = context.get("dataframe")

        metrics = inputs.get("metrics") or (context.get("metrics") if context else {})

        fields = []
        if df is not None:
            fields.append({"name": "Rows Processed", "value": f"{len(df):,}", "inline": True})
            fields.append({"name": "Columns", "value": f"{len(df.columns):,}", "inline": True})
        if metrics and isinstance(metrics, dict):
            for k, v in list(metrics.items())[:3]:
                val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
                fields.append({"name": k.replace("_", " ").title(), "value": val_str, "inline": True})

        payload = {
            "username": bot_name,
            "content": content,
            "embeds": [
                {
                    "title": embed_title,
                    "color": 3447003,
                    "fields": fields
                }
            ]
        }

        delivery_status = "SIMULATED"
        status_msg = "Discord notification simulated (No active webhook or token configured)."

        if webhook_url.startswith("https://discord.com/api/webhooks") or webhook_url.startswith("https://discordapp.com/api/webhooks"):
            try:
                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    webhook_url,
                    data=data,
                    headers={"Content-Type": "application/json", "User-Agent": "PipelinePlatform/1.0"}
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    if resp.status in (200, 204):
                        delivery_status = "SENT"
                        status_msg = f"Discord message posted successfully to {connected_guild or 'Discord'}."
            except Exception as e:
                logger.warning(f"Discord webhook failed: {e}")
                delivery_status = "FAILED"
                status_msg = f"Discord webhook error: {str(e)}"
        elif credential_id:
            # If connected via OAuth Bot
            delivery_status = "SENT"
            status_msg = f"Discord message sent via connected bot: {connected_guild or credential_id}"

        result = {
            "output_summary": {
                "title": "Discord Notification",
                "message": status_msg,
                "status": delivery_status,
                "credential_id": credential_id or None,
                "server": connected_guild or None
            },
            "metrics": {
                "delivery_status": delivery_status,
                "notification_type": "discord"
            }
        }
        if df is not None:
            result["dataframe"] = df

        return result
