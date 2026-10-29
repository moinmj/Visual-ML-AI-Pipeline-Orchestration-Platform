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


class PushNotificationSenderRecipe(BaseRecipe):
    recipe_id = "push_notification_sender"
    name = "Push Notification Sender"
    version = "1.0.0"
    category = "integrations"
    description = "Sends mobile (iOS/Android via FCM/APNs) and Web Push notifications to user device tokens, topics, or webhooks."
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
                "push_service": {
                    "type": "string",
                    "title": "Push Delivery Service",
                    "input_type": "dropdown",
                    "enum": [
                        "Firebase Cloud Messaging (FCM)",
                        "Apple Push Notification service (APNs)",
                        "Web Push / Pushover",
                        "Generic Push Webhook"
                    ],
                    "default": "Firebase Cloud Messaging (FCM)",
                    "description": "Underlying push notification delivery gateway."
                },
                "credential_id": {
                    "type": "string",
                    "title": "Push Service Credential",
                    "input_type": "credential_dropdown",
                    "provider": "push",
                    "description": "Stored service account key, APNs cert, or FCM bearer token."
                },
                "audience_type": {
                    "type": "string",
                    "title": "Audience Delivery Type",
                    "input_type": "dropdown",
                    "enum": ["Broadcast Topic", "Specific Device Tokens", "Webhook Endpoint"],
                    "default": "Broadcast Topic",
                    "description": "How recipients are addressed."
                },
                "topic_or_tokens": {
                    "type": "string",
                    "title": "Topic Name or Device Tokens",
                    "input_type": "text",
                    "default": "pipeline-alerts",
                    "description": "Topic name (e.g. 'pipeline-alerts') or comma-separated device registration tokens."
                },
                "webhook_url": {
                    "type": "string",
                    "title": "Push Webhook URL (For Generic Webhook Mode)",
                    "input_type": "text",
                    "description": "Optional direct Web Push / Pushover / Slack compatible push webhook URL."
                },
                "title": {
                    "type": "string",
                    "title": "Notification Title",
                    "input_type": "text",
                    "default": "🚀 ML Pipeline Execution Finished",
                    "description": "Bold banner title shown on device lock screen / status bar."
                },
                "body": {
                    "type": "string",
                    "title": "Notification Body",
                    "input_type": "textarea",
                    "default": "Model training and evaluation completed successfully.",
                    "description": "Detailed text content of the notification."
                },
                "priority": {
                    "type": "string",
                    "title": "Delivery Priority",
                    "input_type": "dropdown",
                    "enum": ["high", "normal"],
                    "default": "high",
                    "description": "High priority wakes sleeping mobile devices immediately."
                },
                "click_action_url": {
                    "type": "string",
                    "title": "Deep Link / Click Action URL",
                    "input_type": "text",
                    "default": "https://ml.datalect.app/pipelines",
                    "description": "URL or app deep-link opened when user taps the notification."
                },
                "badge_count": {
                    "type": "number",
                    "title": "App Badge Count",
                    "input_type": "number",
                    "default": 1,
                    "description": "Number displayed on the mobile app icon badge."
                }
            },
            "required": ["title", "body"]
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        push_service = str(config.get("push_service", "Firebase Cloud Messaging (FCM)")).strip()
        audience_type = str(config.get("audience_type") or config.get("target_type") or "Broadcast Topic").strip()
        target = str(config.get("topic_or_tokens", "pipeline-alerts")).strip()
        webhook_url = str(config.get("webhook_url", "")).strip()
        title = str(config.get("title", "🚀 Pipeline Finished")).strip()
        body = str(config.get("body", "Execution complete.")).strip()
        priority = str(config.get("priority", "high")).lower().strip()
        action_url = str(config.get("click_action_url", "https://ml.datalect.app/pipelines")).strip()
        badge = int(config.get("badge_count", 1))

        df = inputs.get("dataframe")
        if df is None and context and isinstance(context, dict):
            df = context.get("dataframe")

        metrics = inputs.get("metrics") or (context.get("metrics") if context else {})

        # Formulate payload
        payload = {
            "notification": {
                "title": title,
                "body": body,
                "badge": badge,
                "sound": "default",
                "click_action": action_url
            },
            "data": {
                "url": action_url,
                "priority": priority,
                "rows_processed": str(len(df)) if df is not None and (pd is None or isinstance(df, pd.DataFrame)) else "0"
            }
        }
        if audience_type == "Broadcast Topic":
            payload["to"] = f"/topics/{target}"
        else:
            payload["registration_ids"] = [t.strip() for t in target.split(",") if t.strip()]

        delivery_status = "SIMULATED"
        status_msg = f"Push notification simulated to '{target}' via {push_service}."

        # If a direct webhook URL is supplied (e.g. Pushover / custom gateway), dispatch HTTP POST
        if webhook_url.startswith("https://") or webhook_url.startswith("http://"):
            try:
                data_bytes = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    webhook_url,
                    data=data_bytes,
                    headers={"Content-Type": "application/json", "User-Agent": "MLPipeline-Push/1.0"}
                )
                with urllib.request.urlopen(req, timeout=8) as resp:
                    if resp.status in (200, 201, 202, 204):
                        delivery_status = "SENT"
                        status_msg = f"Push notification successfully dispatched to {webhook_url}."
            except Exception as ex:
                logger.debug(f"Direct push webhook dispatch failed: {ex}")
                delivery_status = "FAILED"
                status_msg = f"Push webhook failed: {str(ex)}"

        result = {
            "output_summary": {
                "title": "Push Notification Dispatch",
                "message": status_msg,
                "status": delivery_status,
                "service": push_service,
                "audience_type": audience_type,
                "target": target,
                "priority": priority,
                "click_action": action_url
            },
            "metrics": {
                "delivery_status": delivery_status,
                "priority": priority,
                "badge_count": badge
            }
        }

        if df is not None:
            result["dataframe"] = df

        return result
