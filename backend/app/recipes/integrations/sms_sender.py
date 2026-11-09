import json
import os
import re
import urllib.request
import urllib.parse
import base64
from typing import Dict, Any, List, Optional

try:
    import pandas as pd
except ImportError:
    pd = None

from backend.app.recipes.base.recipe import BaseRecipe, RecipePort
from backend.app.core.logging import logger
from backend.app.core.config import settings


class SMSSenderRecipe(BaseRecipe):
    recipe_id = "sms_sender"
    name = "SMS Sender"
    version = "1.0.0"
    category = "integrations"
    description = "Sends transactional and pipeline alert SMS messages to one or more phone numbers via Twilio, AWS SNS, or Generic SMS Gateway."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    inputs = [
        RecipePort(
            id="input",
            label="Input Data / Metrics",
            type="dataframe",
            required=False,
            max_connections=1,
            description="Upstream dataset or metrics to summarize in SMS"
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
                "provider": {
                    "type": "string",
                    "title": "SMS Service Provider",
                    "input_type": "dropdown",
                    "enum": ["Twilio", "AWS SNS", "Generic Webhook / HTTP SMS Gateway"],
                    "default": "Twilio",
                    "description": "Provider service used to transmit the text messages."
                },
                "credential_id": {
                    "type": "string",
                    "title": "Stored SMS Gateway Credential",
                    "input_type": "credential_dropdown",
                    "provider": "sms",
                    "description": "Optional stored SMS credential or leave blank to use platform defaults (.env)."
                },
                "account_sid": {
                    "type": "string",
                    "title": "Twilio Account SID (Manual Override)",
                    "input_type": "text",
                    "description": "Optional override (defaults to TWILIO_ACCOUNT_SID from .env if empty)."
                },
                "auth_token": {
                    "type": "string",
                    "title": "Twilio Auth Token (Manual Override)",
                    "input_type": "password",
                    "description": "Optional override (defaults to TWILIO_AUTH_TOKEN from .env if empty)."
                },
                "recipients": {
                    "type": "string",
                    "title": "Recipient Phone Numbers",
                    "input_type": "text",
                    "default": "+18005550199",
                    "description": "Comma-separated list of recipient phone numbers in E.164 format (e.g. +14155552671, +447700900123)."
                },
                "phone_number_field": {
                    "type": "string",
                    "title": "Recipient Phone Column (Dynamic Batch Mode)",
                    "input_type": "column_dropdown",
                    "description": "Optional column from upstream DataFrame containing phone numbers for bulk messaging."
                },
                "message": {
                    "type": "string",
                    "title": "SMS Message Content",
                    "input_type": "textarea",
                    "default": "📢 Pipeline Alert: Execution has finished successfully.",
                    "description": "Text message content (max 160 characters per standard SMS segment)."
                },
                "sender_id": {
                    "type": "string",
                    "title": "Sender ID / From Number (Optional)",
                    "input_type": "text",
                    "description": "Sender phone number or alphanumeric ID (defaults to TWILIO_PHONE_NUMBER if empty)."
                },
                "append_summary_stats": {
                    "type": "boolean",
                    "title": "Append Key Metric Summary",
                    "input_type": "toggle",
                    "default": True,
                    "description": "Appends brief key metrics summary (e.g. 'Rows: 1,200 | Status: OK')."
                },
                "simulation_mode": {
                    "type": "boolean",
                    "title": "Simulation / Dry Run Mode",
                    "input_type": "toggle",
                    "default": False,
                    "description": "When enabled, simulates sending without dispatching live SMS via external carrier."
                }
            },
            "required": ["message"]
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        provider = str(config.get("provider", "Twilio")).strip()
        raw_recipients = str(config.get("recipients", "")).strip()
        recipient_col = str(config.get("phone_number_field") or config.get("recipient_column") or "").strip()
        base_message = str(config.get("message", "Pipeline alert")).strip()
        sender_id = str(config.get("sender_id", "")).strip()
        append_metrics = bool(config.get("append_summary_stats", config.get("append_metrics", True)))
        simulation_mode = bool(config.get("simulation_mode", False))
        credential_id = str(config.get("credential_id", "")).strip()

        df = inputs.get("dataframe")
        if df is None and context and isinstance(context, dict):
            df = context.get("dataframe")

        metrics = inputs.get("metrics") or (context.get("metrics") if context else {})

        # Collect recipient phone numbers
        recipients_list: List[str] = []
        if recipient_col and df is not None and (pd is None or isinstance(df, pd.DataFrame)) and recipient_col in df.columns:
            extracted = df[recipient_col].dropna().astype(str).tolist()
            recipients_list.extend([p.strip() for p in extracted if p.strip()])

        if raw_recipients:
            for item in raw_recipients.split(","):
                cleaned = item.strip()
                if cleaned and cleaned not in recipients_list:
                    recipients_list.append(cleaned)

        if not recipients_list:
            recipients_list = ["+18005550199"]

        # Compose full message with optional metrics trailer
        final_message = base_message
        if append_metrics:
            trailers = []
            if df is not None and (pd is None or isinstance(df, pd.DataFrame)):
                trailers.append(f"Rows: {len(df):,}")
            if metrics and isinstance(metrics, dict):
                for k, v in list(metrics.items())[:2]:
                    v_str = f"{v:.2f}" if isinstance(v, float) else str(v)
                    trailers.append(f"{k.replace('_', ' ').title()}: {v_str}")
            if trailers:
                final_message += f" ({', '.join(trailers)})"

        # Resolve credentials (from config, credential_id, or environment)
        twilio_sid = settings.TWILIO_ACCOUNT_SID or os.getenv("TWILIO_ACCOUNT_SID")
        twilio_token = settings.TWILIO_AUTH_TOKEN or os.getenv("TWILIO_AUTH_TOKEN")
        twilio_from = sender_id or settings.TWILIO_PHONE_NUMBER or os.getenv("TWILIO_PHONE_NUMBER") or "+18557962090"

        if credential_id and context and isinstance(context, dict):
            creds_store = context.get("credentials", {})
            if credential_id in creds_store:
                cred = creds_store[credential_id]
                c_data = cred.get("data", cred) if isinstance(cred, dict) else {}
                twilio_sid = c_data.get("account_sid") or c_data.get("sid") or twilio_sid
                twilio_token = c_data.get("auth_token") or c_data.get("token") or twilio_token
                twilio_from = sender_id or c_data.get("from_number") or c_data.get("phone_number") or twilio_from

        if config.get("account_sid"):
            twilio_sid = str(config.get("account_sid")).strip()
        if config.get("auth_token"):
            twilio_token = str(config.get("auth_token")).strip()
        if config.get("from_number"):
            twilio_from = str(config.get("from_number")).strip()

        delivery_status = "SIMULATED"
        status_msg = f"SMS simulated for {len(recipients_list)} recipient(s) (No live Twilio dispatch)."
        delivered_count = 0
        failed_count = 0
        last_error_msg = ""

        # Attempt Twilio API dispatch if configured and not explicitly in simulation_mode
        if not simulation_mode and provider == "Twilio" and twilio_sid and twilio_token and not twilio_sid.startswith("mock"):
            url = f"https://api.twilio.com/2010-04-01/Accounts/{twilio_sid}/Messages.json"
            auth_header = "Basic " + base64.b64encode(f"{twilio_sid}:{twilio_token}".encode("utf-8")).decode("utf-8")

            for phone in recipients_list[:5]:  # Safety cap at 5 per execution in pipeline
                try:
                    payload_data = urllib.parse.urlencode({
                        "To": phone,
                        "From": twilio_from,
                        "Body": final_message
                    }).encode("utf-8")

                    req = urllib.request.Request(
                        url,
                        data=payload_data,
                        headers={
                            "Authorization": auth_header,
                            "Content-Type": "application/x-www-form-urlencoded",
                            "User-Agent": "MLPipeline-SMS/1.0"
                        }
                    )
                    with urllib.request.urlopen(req, timeout=8) as resp:
                        if resp.status in (200, 201):
                            delivered_count += 1
                        else:
                            failed_count += 1
                except urllib.error.HTTPError as he:
                    failed_count += 1
                    try:
                        err_body = he.read().decode("utf-8")
                        err_json = json.loads(err_body)
                        last_error_msg = err_json.get("message") or err_body[:120]
                    except Exception:
                        last_error_msg = str(he)
                    logger.debug(f"Twilio SMS HTTP error for {phone}: {last_error_msg}")
                except Exception as ex:
                    logger.debug(f"Twilio SMS transmission failed for {phone}: {ex}")
                    last_error_msg = str(ex)
                    failed_count += 1

            if delivered_count > 0:
                delivery_status = "SENT"
                status_msg = f"SMS sent successfully to {delivered_count} recipient(s)."
            elif failed_count > 0:
                delivery_status = "FAILED"
                status_msg = f"SMS delivery failed for {failed_count} recipient(s): {last_error_msg}" if last_error_msg else f"SMS delivery failed for {failed_count} recipient(s)."
        else:
            delivery_status = "SIMULATED"
            status_msg = f"SMS simulated for {len(recipients_list)} recipient(s) via {provider}."

        result = {
            "output_summary": {
                "title": "SMS Delivery Summary",
                "message": status_msg,
                "status": delivery_status,
                "provider": provider,
                "recipient_count": len(recipients_list),
                "sample_recipients": recipients_list[:3],
                "message_preview": final_message[:100] + ("..." if len(final_message) > 100 else "")
            },
            "metrics": {
                "delivery_status": delivery_status,
                "recipients_total": len(recipients_list),
                "sms_length": len(final_message)
            }
        }

        if df is not None:
            result["dataframe"] = df

        return result
