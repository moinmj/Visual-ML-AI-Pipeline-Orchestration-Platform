import datetime
import json
import smtplib
import ssl
import urllib.error
import urllib.parse
import urllib.request
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from backend.app.core.exceptions import ExecutionException, ValidationException
from backend.app.core.logging import logger
from backend.app.recipes.base.recipe import BaseRecipe, RecipePort


class NotificationsRecipe(BaseRecipe):
    recipe_id = "notifications"
    name = "Notifications"
    version = "1.0.0"
    category = "ingestion"
    description = (
        "Sends a generic pipeline notification to a configured channel "
        "(email, chat, or webhook) with optional upstream data summary."
    )
    input_types = ["dataframe"]
    output_types = ["dataframe"]
    inputs = [
        RecipePort(
            id="input",
            label="Input Data",
            type="dataframe",
            required=False,
            max_connections=1,
            description="Upstream dataset or metrics to summarize and notify on",
        )
    ]
    outputs = [
        RecipePort(
            id="output",
            label="Passthrough Output",
            type="dataframe",
            description="Passthrough dataset for downstream nodes",
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "sections": [
                {
                    "id": "channel",
                    "title": "Channel & Endpoint",
                    "description": "Configure notification destination protocol and URL/recipient parameters",
                },
                {
                    "id": "message_content",
                    "title": "Message Content",
                    "description": "Configure title, message template, severity, and context summaries",
                },
                {
                    "id": "email_settings",
                    "title": "SMTP & Email Credentials",
                    "description": "Configure SMTP host, port, authentication, and encryption",
                },
                {
                    "id": "runtime_limits",
                    "title": "Timeouts & Error Handling",
                    "description": "HTTP headers, network timeout, and failure handling",
                },
            ],
            "properties": {
                "channel_type": {
                    "type": "string",
                    "title": "Channel Type",
                    "enum": ["webhook", "slack", "teams", "discord", "email"],
                    "default": "webhook",
                    "input_type": "dropdown",
                    "section": "channel",
                    "description": (
                        "Target notification protocol: 'webhook' (generic HTTP POST), "
                        "'slack', 'teams', 'discord', or 'email'."
                    ),
                },
                "target_url": {
                    "type": "string",
                    "title": "Webhook / Endpoint URL",
                    "default": "",
                    "input_type": "text",
                    "section": "channel",
                    "description": (
                        "Destination webhook or HTTP endpoint URL "
                        "(for webhook, Slack, Teams, or Discord)."
                    ),
                },
                "title": {
                    "type": "string",
                    "title": "Notification Title / Subject",
                    "default": "Pipeline Notification",
                    "input_type": "text",
                    "section": "message_content",
                    "description": "Headline or subject line for the notification.",
                },
                "message": {
                    "type": "string",
                    "title": "Message Body",
                    "default": "Pipeline execution completed successfully.",
                    "input_type": "textarea",
                    "section": "message_content",
                    "description": (
                        "Message body content. Supports template tokens: "
                        "{title}, {severity}, {row_count}, {col_count}."
                    ),
                },
                "severity": {
                    "type": "string",
                    "title": "Severity Level",
                    "enum": ["info", "success", "warning", "error"],
                    "default": "info",
                    "input_type": "dropdown",
                    "section": "message_content",
                    "description": "Severity level of the notification (affects formatting and badges).",
                },
                "include_dataset_summary": {
                    "type": "boolean",
                    "title": "Include Data Summary",
                    "default": True,
                    "input_type": "checkbox",
                    "section": "message_content",
                    "description": "Automatically append row count, column count, and column names.",
                },
                "include_metrics": {
                    "type": "boolean",
                    "title": "Include Upstream Metrics",
                    "default": True,
                    "input_type": "checkbox",
                    "section": "message_content",
                    "description": "Automatically append model or step metrics from upstream context.",
                },
                "email_recipients": {
                    "type": "string",
                    "title": "Email Recipients",
                    "default": "",
                    "input_type": "text",
                    "section": "email_settings",
                    "description": "Comma-separated list of recipient email addresses when channel_type is 'email'.",
                    "visible_when": {
                        "field": "channel_type",
                        "equals": "email",
                    },
                },
                "email_sender": {
                    "type": "string",
                    "title": "Email Sender Address",
                    "default": "no-reply@pipeline-platform.local",
                    "input_type": "text",
                    "section": "email_settings",
                    "description": "From address for outbound email notifications.",
                    "visible_when": {
                        "field": "channel_type",
                        "equals": "email",
                    },
                },
                "smtp_host": {
                    "type": "string",
                    "title": "SMTP Host",
                    "default": "",
                    "input_type": "text",
                    "section": "email_settings",
                    "description": "SMTP server hostname for email delivery (e.g. smtp.gmail.com).",
                    "visible_when": {
                        "field": "channel_type",
                        "equals": "email",
                    },
                },
                "smtp_port": {
                    "type": "integer",
                    "title": "SMTP Port",
                    "default": 587,
                    "input_type": "number",
                    "section": "email_settings",
                    "description": "SMTP port (e.g. 587 for STARTTLS, 465 for SSL, 25 for plain).",
                    "visible_when": {
                        "field": "channel_type",
                        "equals": "email",
                    },
                },
                "smtp_username": {
                    "type": "string",
                    "title": "SMTP Username",
                    "default": "",
                    "input_type": "text",
                    "section": "email_settings",
                    "description": "Username / email for SMTP authentication.",
                    "visible_when": {
                        "field": "channel_type",
                        "equals": "email",
                    },
                },
                "smtp_password": {
                    "type": "string",
                    "title": "SMTP Password",
                    "default": "",
                    "input_type": "password",
                    "section": "email_settings",
                    "description": "Password or application token for SMTP authentication.",
                    "visible_when": {
                        "field": "channel_type",
                        "equals": "email",
                    },
                },
                "smtp_use_tls": {
                    "type": "boolean",
                    "title": "Use STARTTLS",
                    "default": True,
                    "input_type": "checkbox",
                    "section": "email_settings",
                    "description": "Whether to initiate STARTTLS encryption on port 587.",
                    "visible_when": {
                        "field": "channel_type",
                        "equals": "email",
                    },
                },
                "custom_headers": {
                    "type": "object",
                    "title": "Custom HTTP Headers",
                    "default": {},
                    "input_type": "key_value",
                    "section": "runtime_limits",
                    "description": "Optional dictionary of custom HTTP headers for generic webhook POST requests.",
                },
                "timeout_seconds": {
                    "type": "number",
                    "title": "Request Timeout (Seconds)",
                    "default": 10.0,
                    "input_type": "number",
                    "section": "runtime_limits",
                    "description": "Maximum seconds to wait for network responses.",
                },
                "fail_on_error": {
                    "type": "boolean",
                    "title": "Fail Pipeline on Error",
                    "default": False,
                    "input_type": "checkbox",
                    "section": "runtime_limits",
                    "description": "If True, delivery failures raise an exception; if False, errors are logged and execution continues.",
                },
            },
            "required": ["channel_type"],
        }

    def execute(
        self,
        inputs: Dict[str, Any],
        config: Dict[str, Any],
        context: Optional[Any] = None,
    ) -> Dict[str, Any]:
        # 1. Resolve upstream DataFrame
        df: Optional[pd.DataFrame] = None
        if "dataframe" in inputs and isinstance(inputs["dataframe"], pd.DataFrame):
            df = inputs["dataframe"]
        elif "input" in inputs and isinstance(inputs["input"], pd.DataFrame):
            df = inputs["input"]
        elif "dataframe" in config and isinstance(config["dataframe"], pd.DataFrame):
            df = config["dataframe"]
        elif context and isinstance(context, dict) and "dataframe" in context:
            if isinstance(context["dataframe"], pd.DataFrame):
                df = context["dataframe"]

        # 2. Resolve metrics
        metrics: Dict[str, Any] = {}
        if "metrics" in inputs and isinstance(inputs["metrics"], dict):
            metrics = inputs["metrics"]
        elif context and isinstance(context, dict) and "metrics" in context:
            if isinstance(context["metrics"], dict):
                metrics = context["metrics"]

        # 3. Read configuration parameters
        channel_type = str(config.get("channel_type", "webhook")).strip().lower()
        target_url = str(config.get("target_url", "")).strip()
        raw_title = str(config.get("title", "Pipeline Notification")).strip()
        raw_message = str(config.get("message", "Pipeline execution completed successfully."))
        severity = str(config.get("severity", "info")).strip().lower()
        include_summary = bool(config.get("include_dataset_summary", True))
        include_metrics = bool(config.get("include_metrics", True))
        fail_on_error = bool(config.get("fail_on_error", False))
        timeout = float(config.get("timeout_seconds", 10.0) or 10.0)

        # 4. Interpolate template tokens
        row_count = len(df) if df is not None else 0
        col_count = len(df.columns) if df is not None else 0
        cols_preview = ", ".join(list(df.columns)[:5]) if df is not None and len(df.columns) > 0 else "None"
        if df is not None and len(df.columns) > 5:
            cols_preview += f" (+{len(df.columns) - 5} more)"

        tokens = {
            "title": raw_title,
            "severity": severity.upper(),
            "row_count": f"{row_count:,}",
            "col_count": str(col_count),
            "columns": cols_preview,
        }
        resolved_title = raw_title
        resolved_message = raw_message
        for k, v in tokens.items():
            resolved_title = resolved_title.replace(f"{{{k}}}", str(v))
            resolved_message = resolved_message.replace(f"{{{k}}}", str(v))

        # 5. Build structured payload based on channel
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        dataset_info: Dict[str, Any] = {}
        if include_summary and df is not None:
            dataset_info = {
                "row_count": row_count,
                "column_count": col_count,
                "columns": list(df.columns),
            }

        outbound_payload: Dict[str, Any] = {}
        delivery_status = "SIMULATED"
        error_msg: Optional[str] = None

        if channel_type == "email":
            delivery_status, outbound_payload, error_msg = self._handle_email_notification(
                config=config,
                title=resolved_title,
                message=resolved_message,
                severity=severity,
                dataset_info=dataset_info if include_summary else {},
                metrics=metrics if include_metrics else {},
                timeout=timeout,
            )
        else:
            # Webhook / Chat Channels
            outbound_payload = self._build_chat_or_webhook_payload(
                channel=channel_type,
                title=resolved_title,
                message=resolved_message,
                severity=severity,
                timestamp=now_iso,
                dataset_info=dataset_info if include_summary else {},
                metrics=metrics if include_metrics else {},
            )

            # If no target URL, simulate gracefully
            if not target_url or not target_url.startswith(("http://", "https://")):
                delivery_status = "SIMULATED"
                logger.info(f"Notifications recipe '{self.recipe_id}': No valid target_url provided. Notification simulated.")
            else:
                custom_headers = config.get("custom_headers") or {}
                delivery_status, error_msg = self._dispatch_http_request(
                    url=target_url,
                    payload=outbound_payload,
                    headers=custom_headers if isinstance(custom_headers, dict) else {},
                    timeout=timeout,
                )

        if delivery_status == "FAILED" and fail_on_error:
            raise ExecutionException(
                f"Notification dispatch failed for channel '{channel_type}': {error_msg}"
            )

        # 6. Return passthrough DataFrame along with execution details
        passthrough_df = df if df is not None else pd.DataFrame()
        return {
            "output": passthrough_df,
            "dataframe": passthrough_df,
            "status": delivery_status,
            "channel": channel_type,
            "severity": severity,
            "title": resolved_title,
            "message": resolved_message,
            "delivered_at": now_iso,
            "payload": outbound_payload,
            "error": error_msg,
        }

    def _build_chat_or_webhook_payload(
        self,
        channel: str,
        title: str,
        message: str,
        severity: str,
        timestamp: str,
        dataset_info: Dict[str, Any],
        metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Builds formatted payload for webhook, Slack, Teams, or Discord."""
        color_map = {
            "info": "#2196F3",
            "success": "#4CAF50",
            "warning": "#FF9800",
            "error": "#F44336",
        }
        color_hex = color_map.get(severity, "#2196F3")

        if channel == "slack":
            blocks: List[Dict[str, Any]] = [
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": f"[{severity.upper()}] {title}"},
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": message},
                },
            ]
            fields: List[Dict[str, Any]] = []
            if dataset_info:
                fields.append({"type": "mrkdwn", "text": f"*Rows:* {dataset_info.get('row_count', 0):,}"})
                fields.append({"type": "mrkdwn", "text": f"*Columns:* {dataset_info.get('column_count', 0)}"})
            if metrics:
                for k, v in list(metrics.items())[:4]:
                    val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
                    fields.append({"type": "mrkdwn", "text": f"*{k}:* {val_str}"})
            if fields:
                blocks.append({"type": "section", "fields": fields})

            return {
                "text": f"[{severity.upper()}] {title}: {message}",
                "attachments": [{"color": color_hex, "blocks": blocks}],
            }

        elif channel == "discord":
            embed_fields: List[Dict[str, Any]] = []
            if dataset_info:
                embed_fields.append({"name": "Rows", "value": f"{dataset_info.get('row_count', 0):,}", "inline": True})
                embed_fields.append({"name": "Columns", "value": str(dataset_info.get('column_count', 0)), "inline": True})
            if metrics:
                for k, v in list(metrics.items())[:4]:
                    val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
                    embed_fields.append({"name": str(k), "value": val_str, "inline": True})

            # Decimal color code for discord
            dec_color = int(color_hex.lstrip("#"), 16)
            return {
                "content": f"**[{severity.upper()}]** {title}",
                "embeds": [
                    {
                        "title": title,
                        "description": message,
                        "color": dec_color,
                        "timestamp": timestamp,
                        "fields": embed_fields,
                    }
                ],
            }

        elif channel == "teams":
            facts: List[Dict[str, Any]] = [
                {"name": "Severity", "value": severity.upper()},
            ]
            if dataset_info:
                facts.append({"name": "Rows Processed", "value": f"{dataset_info.get('row_count', 0):,}"})
                facts.append({"name": "Columns", "value": str(dataset_info.get('column_count', 0))})
            if metrics:
                for k, v in list(metrics.items())[:4]:
                    val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
                    facts.append({"name": str(k), "value": val_str})

            return {
                "@type": "MessageCard",
                "@context": "https://schema.org/extensions",
                "themeColor": color_hex.lstrip("#"),
                "summary": title,
                "title": f"[{severity.upper()}] {title}",
                "text": message,
                "sections": [{"facts": facts}],
            }

        else:
            # Generic Webhook POST
            return {
                "event": "pipeline_notification",
                "severity": severity,
                "title": title,
                "message": message,
                "timestamp": timestamp,
                "dataset_summary": dataset_info,
                "metrics": metrics,
            }

    def _dispatch_http_request(
        self,
        url: str,
        payload: Dict[str, Any],
        headers: Dict[str, Any],
        timeout: float,
    ) -> (str, Optional[str]):
        """Dispatches outbound HTTP POST request via standard library."""
        try:
            req_headers = {
                "Content-Type": "application/json",
                "User-Agent": "Visual-MLAI-Pipeline/1.0",
            }
            req_headers.update({str(k): str(v) for k, v in headers.items()})

            body = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=body, headers=req_headers, method="POST")

            # Non-blocking context creation
            ctx = ssl.create_default_context()
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as response:
                status_code = getattr(response, "status", 200)
                if 200 <= status_code < 300:
                    return "DELIVERED", None
                return "FAILED", f"HTTP {status_code}"
        except urllib.error.HTTPError as he:
            err = f"HTTP Error {he.code}: {he.reason}"
            logger.warning(f"Notification request failed: {err}")
            return "FAILED", err
        except Exception as e:
            err = f"Network Error: {str(e)}"
            logger.warning(f"Notification request failed: {err}")
            return "FAILED", err

    def _handle_email_notification(
        self,
        config: Dict[str, Any],
        title: str,
        message: str,
        severity: str,
        dataset_info: Dict[str, Any],
        metrics: Dict[str, Any],
        timeout: float,
    ) -> (str, Dict[str, Any], Optional[str]):
        """Constructs and delivers or simulates email notifications."""
        recipients_str = str(config.get("email_recipients", "")).strip()
        recipients = [r.strip() for r in recipients_str.split(",") if r.strip()]
        sender = str(config.get("email_sender", "no-reply@pipeline-platform.local")).strip()
        smtp_host = str(config.get("smtp_host", "")).strip()
        smtp_port = int(config.get("smtp_port", 587) or 587)
        smtp_username = str(config.get("smtp_username", "")).strip()
        smtp_password = str(config.get("smtp_password", "")).strip()
        use_tls = bool(config.get("smtp_use_tls", True))

        # Build payload record
        email_record = {
            "from": sender,
            "to": recipients,
            "subject": f"[{severity.upper()}] {title}",
            "body": message,
            "dataset_info": dataset_info,
            "metrics": metrics,
        }

        if not recipients or not smtp_host:
            logger.info("Email notification simulated: Missing recipients or SMTP host.")
            return "SIMULATED", email_record, None

        # Build MIME Message
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"[{severity.upper()}] {title}"
            msg["From"] = sender
            msg["To"] = ", ".join(recipients)

            plain_text = f"{title}\nSeverity: {severity.upper()}\n\n{message}\n"
            if dataset_info:
                plain_text += f"\nDataset: {dataset_info.get('row_count', 0)} rows, {dataset_info.get('column_count', 0)} cols"
            if metrics:
                plain_text += f"\nMetrics: {json.dumps(metrics)}"

            msg.attach(MIMEText(plain_text, "plain"))

            # Dispatch via smtplib
            if smtp_port == 465:
                server = smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=timeout)
            else:
                server = smtplib.SMTP(smtp_host, smtp_port, timeout=timeout)
                if use_tls:
                    server.starttls()

            if smtp_username and smtp_password:
                server.login(smtp_username, smtp_password)

            server.sendmail(sender, recipients, msg.as_string())
            server.quit()
            return "DELIVERED", email_record, None
        except Exception as e:
            err = f"SMTP Error: {str(e)}"
            logger.warning(f"Email delivery failed: {err}")
            return "FAILED", email_record, err

