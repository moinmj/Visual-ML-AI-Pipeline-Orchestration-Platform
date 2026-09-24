import json
import urllib.request
from typing import Dict, Any, List, Optional
from backend.app.recipes.base.recipe import BaseRecipe, RecipePort
from backend.app.core.logging import logger


class DiscordRecipe(BaseRecipe):
    recipe_id = "discord"
    name = "Discord"
    version = "1.0.0"
    category = "integrations"
    description = "Sends rich embed alerts, logs, or completion summaries to Discord channels via Discord Webhooks."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    inputs = [
        RecipePort(
            id="input",
            label="Input Data / Metrics",
            type="dataframe",
            required=False,
            max_connections=1,
            description="Upstream dataset or metrics to summarize"
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
                "webhook_url": {
                    "type": "string",
                    "title": "Discord Webhook URL",
                    "description": "Discord channel webhook URL (e.g. https://discord.com/api/webhooks/...)."
                },
                "username": {
                    "type": "string",
                    "title": "Bot Display Name (Optional)",
                    "default": "ML Pipeline Bot"
                },
                "content": {
                    "type": "string",
                    "title": "Message Content",
                    "default": "📢 Automated Pipeline Notification"
                },
                "embed_title": {
                    "type": "string",
                    "title": "Embed Box Title",
                    "default": "Pipeline Execution Status"
                },
                "embed_color": {
                    "type": "string",
                    "title": "Embed Color (Hex)",
                    "default": "3066993",
                    "description": "Decimal or hex color code (e.g. 3066993 for Teal / Blue)."
                }
            },
            "required": ["content"]
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        webhook_url = str(config.get("webhook_url", "")).strip()
        content = config.get("content", "Pipeline notification")
        embed_title = config.get("embed_title", "Pipeline Execution")
        bot_name = config.get("username", "Pipeline Bot")

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
        status_msg = "Discord notification simulated (No Webhook URL configured)."

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
                        status_msg = "Discord message posted successfully."
            except Exception as e:
                logger.warning(f"Discord webhook failed: {e}")
                delivery_status = "FAILED"
                status_msg = f"Discord webhook error: {str(e)}"

        result = {
            "output_summary": {
                "title": "Discord Notification",
                "message": status_msg,
                "status": delivery_status
            },
            "metrics": {
                "delivery_status": delivery_status,
                "notification_type": "discord"
            }
        }
        if df is not None:
            result["dataframe"] = df

        return result
