import json
import os
import base64
import urllib.request
import urllib.parse
from typing import Dict, Any, List, Optional

try:
    import pandas as pd
except ImportError:
    pd = None

from backend.app.recipes.base.recipe import BaseRecipe, RecipePort
from backend.app.core.logging import logger
from backend.app.core.config import settings


class TwilioConnectorRecipe(BaseRecipe):
    recipe_id = "twilio_connector"
    name = "Twilio Voice/SMS Connector"
    version = "1.0.0"
    category = "integrations"
    description = "Sends SMS, WhatsApp messages, or initiates automated Text-to-Speech (TTS) voice calls through Twilio API."
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
                "channel": {
                    "type": "string",
                    "title": "Communication Channel",
                    "input_type": "dropdown",
                    "enum": [
                        "SMS Message",
                        "Voice Call (Text-to-Speech)",
                        "WhatsApp Message"
                    ],
                    "default": "SMS Message",
                    "description": "Delivery channel for the Twilio transmission."
                },
                "credential_id": {
                    "type": "string",
                    "title": "Stored Twilio Credential",
                    "input_type": "credential_dropdown",
                    "provider": "twilio",
                    "description": "Optional stored Twilio credential or leave blank to use platform defaults from .env."
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
                "from_number": {
                    "type": "string",
                    "title": "Twilio Phone / WhatsApp Number",
                    "input_type": "text",
                    "description": "Twilio source phone number in E.164 format (e.g. +18557962090 or whatsapp:+14155238886)."
                },
                "to_numbers": {
                    "type": "string",
                    "title": "Recipient Phone Numbers",
                    "input_type": "text",
                    "default": "+18005550199",
                    "description": "Comma-separated list of recipient phone numbers."
                },
                "message_or_script": {
                    "type": "string",
                    "title": "Message Content / Voice Script",
                    "input_type": "textarea",
                    "default": "Alert from ML Pipeline: Critical model drift or job completion detected. Immediate review required.",
                    "description": "Text message body or spoken script for Voice Calls (read via Polly/Twilio TTS engine)."
                },
                "voice_engine": {
                    "type": "string",
                    "title": "TTS Voice Accent (For Voice Calls)",
                    "input_type": "dropdown",
                    "enum": [
                        "Polly.Joanna (US Female)",
                        "Polly.Matthew (US Male)",
                        "Polly.Amy (UK Female)",
                        "Polly.Brian (UK Male)",
                        "Alice (Standard Female)"
                    ],
                    "default": "Polly.Joanna (US Female)",
                    "description": "Amazon Polly neural voice used when channel is set to Voice Call."
                },
                "call_timeout_seconds": {
                    "type": "number",
                    "title": "Call Ring Timeout (Seconds)",
                    "input_type": "number",
                    "default": 30,
                    "description": "Ring duration limit before disconnecting if unanswered."
                },
                "simulation_mode": {
                    "type": "boolean",
                    "title": "Simulation / Dry Run Mode",
                    "input_type": "toggle",
                    "default": False,
                    "description": "When enabled, simulates sending without dispatching live voice calls or SMS via Twilio."
                }
            },
            "required": ["message_or_script"]
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        channel = str(config.get("channel", "SMS Message")).strip()
        raw_to = str(config.get("to_numbers", "+18005550199")).strip()
        message = str(config.get("message_or_script", "ML Pipeline alert.")).strip()
        voice_accent = str(config.get("voice_engine", "Polly.Joanna (US Female)")).split(" ")[0]
        timeout_sec = int(config.get("call_timeout_seconds", 30))
        simulation_mode = bool(config.get("simulation_mode", False))

        # Credentials resolution
        sid = str(config.get("account_sid", "")).strip() or settings.TWILIO_ACCOUNT_SID or os.getenv("TWILIO_ACCOUNT_SID", "")
        token = str(config.get("auth_token", "")).strip() or settings.TWILIO_AUTH_TOKEN or os.getenv("TWILIO_AUTH_TOKEN", "")
        from_num = str(config.get("from_number", "")).strip() or settings.TWILIO_PHONE_NUMBER or os.getenv("TWILIO_PHONE_NUMBER", "+18557962090")

        df = inputs.get("dataframe")
        if df is None and context and isinstance(context, dict):
            df = context.get("dataframe")

        recipients = [p.strip() for p in raw_to.split(",") if p.strip()] or ["+18005550199"]

        delivery_status = "SIMULATED"
        status_msg = f"Twilio {channel} simulated for {len(recipients)} recipient(s) (No live dispatch)."
        delivered_count = 0
        failed_count = 0
        last_error_msg = ""

        # Attempt live dispatch if SID and token are configured, not mock, and not in simulation_mode
        if not simulation_mode and sid and token and not sid.startswith("mock") and len(sid) >= 10:
            auth_header = "Basic " + base64.b64encode(f"{sid}:{token}".encode("utf-8")).decode("utf-8")

            if channel in ("SMS Message", "WhatsApp Message"):
                url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
                is_wa = channel == "WhatsApp Message"
                src_num = f"whatsapp:{from_num.replace('whatsapp:', '')}" if is_wa else from_num

                for target in recipients[:3]:
                    dst_num = f"whatsapp:{target.replace('whatsapp:', '')}" if is_wa else target
                    try:
                        data = urllib.parse.urlencode({
                            "To": dst_num,
                            "From": src_num,
                            "Body": message
                        }).encode("utf-8")
                        req = urllib.request.Request(
                            url,
                            data=data,
                            headers={"Authorization": auth_header, "Content-Type": "application/x-www-form-urlencoded"}
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
                        logger.debug(f"Twilio message HTTP error for {target}: {last_error_msg}")
                    except Exception as ex:
                        logger.debug(f"Twilio message delivery failed to {target}: {ex}")
                        last_error_msg = str(ex)
                        failed_count += 1

            elif channel == "Voice Call (Text-to-Speech)":
                url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Calls.json"
                twiml = f"<Response><Say voice='{voice_accent}'>{message}</Say></Response>"

                for target in recipients[:2]:
                    try:
                        data = urllib.parse.urlencode({
                            "To": target,
                            "From": from_num,
                            "Twiml": twiml,
                            "Timeout": str(timeout_sec)
                        }).encode("utf-8")
                        req = urllib.request.Request(
                            url,
                            data=data,
                            headers={"Authorization": auth_header, "Content-Type": "application/x-www-form-urlencoded"}
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
                        logger.debug(f"Twilio voice call HTTP error for {target}: {last_error_msg}")
                    except Exception as ex:
                        logger.debug(f"Twilio voice call initiation failed to {target}: {ex}")
                        last_error_msg = str(ex)
                        failed_count += 1

            if delivered_count > 0:
                delivery_status = "SENT"
                status_msg = f"Twilio {channel} dispatched successfully to {delivered_count} destination(s)."
            elif failed_count > 0:
                delivery_status = "FAILED"
                status_msg = f"Twilio {channel} failed for {failed_count} destination(s): {last_error_msg}" if last_error_msg else f"Twilio {channel} failed for {failed_count} destination(s)."
        else:
            delivery_status = "SIMULATED"
            status_msg = f"Twilio {channel} simulated for {len(recipients)} recipient(s) (Dry Run Mode)."

        result = {
            "output_summary": {
                "title": f"Twilio {channel} Summary",
                "message": status_msg,
                "status": delivery_status,
                "channel": channel,
                "from_number": from_num,
                "recipients_count": len(recipients),
                "recipients": recipients
            },
            "metrics": {
                "delivery_status": delivery_status,
                "channel": channel,
                "recipients_total": len(recipients)
            }
        }

        if df is not None:
            result["dataframe"] = df

        return result
