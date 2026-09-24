import json
import urllib.request
from typing import Dict, Any, List, Optional
from backend.app.recipes.base.recipe import BaseRecipe, RecipePort
from backend.app.core.logging import logger


class TelegramRecipe(BaseRecipe):
    recipe_id = "telegram"
    name = "Telegram"
    version = "1.0.0"
    category = "integrations"
    description = "Sends instant alert messages, metrics summaries, or pipeline status to a Telegram chat/channel via Bot API."
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
                "bot_token": {
                    "type": "string",
                    "title": "Telegram Bot Token",
                    "description": "Bot API token obtained from @BotFather (e.g. 123456789:ABCdefGhIJKlmNoPQRsTUVwxyZ)."
                },
                "chat_id": {
                    "type": "string",
                    "title": "Chat / Channel ID",
                    "description": "Telegram user ID, group ID, or @channel username."
                },
                "message": {
                    "type": "string",
                    "title": "Message Text",
                    "default": "🤖 *Pipeline Notification*\nExecution finished successfully!",
                    "description": "Message to send (supports Markdown or HTML formatting)."
                },
                "parse_mode": {
                    "type": "string",
                    "title": "Parse Mode",
                    "enum": ["Markdown", "HTML"],
                    "default": "Markdown"
                }
            },
            "required": ["message"]
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        bot_token = str(config.get("bot_token", "")).strip()
        chat_id = str(config.get("chat_id", "")).strip()
        message = str(config.get("message", "Pipeline Notification"))
        parse_mode = config.get("parse_mode", "Markdown")

        df = inputs.get("dataframe")
        if df is None and context and isinstance(context, dict):
            df = context.get("dataframe")

        delivery_status = "SIMULATED"
        status_msg = "Telegram notification simulated (Token or Chat ID not provided)."

        if bot_token and chat_id:
            try:
                url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
                payload = {
                    "chat_id": chat_id,
                    "text": message,
                    "parse_mode": parse_mode
                }
                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    url,
                    data=data,
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    if resp.status == 200:
                        delivery_status = "SENT"
                        status_msg = f"Telegram message sent to chat {chat_id}."
            except Exception as e:
                logger.warning(f"Telegram send failed: {e}")
                delivery_status = "FAILED"
                status_msg = f"Telegram error: {str(e)}"

        result = {
            "output_summary": {
                "title": "Telegram Notification",
                "message": status_msg,
                "status": delivery_status
            },
            "metrics": {
                "delivery_status": delivery_status,
                "notification_type": "telegram"
            }
        }
        if df is not None:
            result["dataframe"] = df

        return result
