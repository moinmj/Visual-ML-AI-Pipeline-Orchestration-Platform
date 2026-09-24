import json
import urllib.request
from typing import Dict, Any, List, Optional
from backend.app.recipes.base.recipe import BaseRecipe, RecipePort
from backend.app.core.logging import logger


class SlackRecipe(BaseRecipe):
    recipe_id = "slack"
    name = "Slack"
    version = "1.0.0"
    category = "integrations"
    description = "Posts rich pipeline alerts, anomaly notices, or execution summaries to Slack channels via Incoming Webhook."
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
                    "title": "Slack Webhook URL",
                    "description": "Incoming Webhook URL from your Slack App (e.g. https://hooks.slack.com/services/...)."
                },
                "channel": {
                    "type": "string",
                    "title": "Channel Override (Optional)",
                    "description": "Optional channel override (e.g. #ml-alerts)."
                },
                "message": {
                    "type": "string",
                    "title": "Message Text",
                    "default": "🚀 Pipeline execution completed successfully! Metrics and summary are attached.",
                    "description": "Custom notification message text. Supports standard Slack markdown."
                },
                "include_metrics_summary": {
                    "type": "boolean",
                    "title": "Include Metrics & Data Summary",
                    "default": True,
                    "description": "If enabled, automatically appends upstream row count, metrics, and duration."
                }
            },
            "required": ["message"]
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        webhook_url = str(config.get("webhook_url", "")).strip()
        message_text = config.get("message", "Pipeline update")
        include_summary = bool(config.get("include_metrics_summary", True))

        df = inputs.get("dataframe")
        if df is None and context and isinstance(context, dict):
            df = context.get("dataframe")

        metrics = inputs.get("metrics") or (context.get("metrics") if context else {})

        # Build Slack blocks payload
        blocks: List[Dict[str, Any]] = [
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*{message_text}*"}
            }
        ]

        if include_summary:
            fields = []
            if df is not None:
                fields.append({"type": "mrkdwn", "text": f"*Processed Rows:* {len(df):,}"})
                fields.append({"type": "mrkdwn", "text": f"*Columns:* {len(df.columns):,}"})
            if metrics and isinstance(metrics, dict):
                for k, v in list(metrics.items())[:4]:
                    val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
                    fields.append({"type": "mrkdwn", "text": f"*{k.replace('_', ' ').title()}:* {val_str}"})
            if fields:
                blocks.append({"type": "section", "fields": fields})

        payload = {"text": message_text, "blocks": blocks}
        if config.get("channel"):
            payload["channel"] = config["channel"]

        delivery_status = "SIMULATED"
        status_msg = "Slack notification simulated (No Webhook URL provided)."

        if webhook_url.startswith("https://"):
            try:
                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    webhook_url,
                    data=data,
                    headers={"Content-Type": "application/json", "User-Agent": "PipelinePlatform/1.0"}
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    if resp.status == 200:
                        delivery_status = "SENT"
                        status_msg = "Slack message posted successfully."
            except Exception as e:
                logger.warning(f"Slack webhook send failed: {e}")
                delivery_status = "FAILED"
                status_msg = f"Slack webhook error: {str(e)}"

        result = {
            "output_summary": {
                "title": "Slack Notification",
                "message": status_msg,
                "status": delivery_status
            },
            "metrics": {
                "delivery_status": delivery_status,
                "notification_type": "slack"
            }
        }
        if df is not None:
            result["dataframe"] = df

        return result
