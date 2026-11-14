from typing import Dict, Any, List, Optional
import pandas as pd
from backend.app.recipes.base.recipe import BaseRecipe


class WebhookTriggerRecipe(BaseRecipe):
    """
    Webhook Inbound Trigger Recipe (n8n / Boomi style).
    Listens for inbound HTTP POST JSON payloads to trigger pipeline execution.
    """

    recipe_id = "webhook_trigger"
    name = "Webhook Inbound Trigger"
    version = "1.0.0"
    category = "triggers"
    description = "Listens for external HTTP POST webhooks to ingest real-time JSON payloads into the pipeline."
    input_types = []  # Root trigger node, has 0 incoming connections
    output_types = ["dataframe"]
    inputs = []
    outputs = []

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "sections": [
                {
                    "id": "endpoint",
                    "title": "Endpoint Configuration",
                    "description": "Webhook URL identifier and routing options",
                },
                {
                    "id": "security",
                    "title": "Authentication & Security",
                    "description": "Configure inbound header authentication requirements",
                },
                {
                    "id": "payload",
                    "title": "Payload Format",
                    "description": "Define expected payload format and structure",
                },
            ],
            "properties": {
                "webhook_path": {
                    "type": "string",
                    "title": "Webhook URL Identifier",
                    "input_type": "text",
                    "section": "endpoint",
                    "default": "ml_inbound_stream",
                    "description": "Unique URL path endpoint: /api/v1/workflows/trigger/{webhook_path}",
                },
                "auth_header_required": {
                    "type": "string",
                    "title": "Authentication Mode",
                    "input_type": "dropdown",
                    "section": "security",
                    "enum": ["None (Public)", "Bearer Token", "API Key"],
                    "default": "None (Public)",
                    "description": "Require a valid Bearer or custom authorization header in incoming webhook calls.",
                },
                "auth_token": {
                    "type": "string",
                    "title": "Bearer Token Secret",
                    "input_type": "password",
                    "section": "security",
                    "default": "",
                    "description": "Expected Bearer authorization token header value.",
                    "visible_when": {
                        "field": "auth_header_required",
                        "equals": "Bearer Token",
                    },
                },
                "api_key": {
                    "type": "string",
                    "title": "API Key Secret",
                    "input_type": "password",
                    "section": "security",
                    "default": "",
                    "description": "Expected X-API-Key header value.",
                    "visible_when": {
                        "field": "auth_header_required",
                        "equals": "API Key",
                    },
                },
                "payload_format": {
                    "type": "string",
                    "title": "Expected Inbound Payload Format",
                    "input_type": "dropdown",
                    "section": "payload",
                    "enum": [
                        "JSON Array of Objects",
                        "Single JSON Object (Single Row)",
                        "Key-Value Form Data",
                    ],
                    "default": "JSON Array of Objects",
                    "description": "Expected incoming payload content format ('json' or 'form-data').",
                },
            },
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        # If payload was delivered via webhook execution context
        df = inputs.get("dataframe")
        if df is None:
            if context and isinstance(context, dict) and "dataframe" in context:
                df = context["dataframe"]
            elif context and isinstance(context, dict) and "raw_payload" in context:
                raw = context["raw_payload"]
                if isinstance(raw, list):
                    df = pd.json_normalize(raw)
                else:
                    df = pd.json_normalize([raw])
            else:
                # Default empty starter DataFrame for manual canvas test runs
                df = pd.DataFrame([{"event": "test_webhook_ping", "status": "active", "value": 1.0}])

        return {
            "dataframe": df,
            "trigger_metadata": {
                "type": "webhook",
                "webhook_path": config.get("webhook_path", "ml_inbound_stream"),
                "rows_received": len(df)
            }
        }

    def to_code(self, config: Dict[str, Any]) -> str:
        path = config.get("webhook_path", "ml_inbound_stream")
        return f"# Webhook Trigger Endpoint\n# POST http://localhost:8000/api/v1/workflows/trigger/{path}\n# Ingests JSON payload into 'dataframe'"
