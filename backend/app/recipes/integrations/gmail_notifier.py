import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import io
import pandas as pd
from typing import Dict, Any, List, Optional
from backend.app.recipes.base.recipe import BaseRecipe, RecipePort
from backend.app.core.logging import logger


class GmailRecipe(BaseRecipe):
    recipe_id = "gmail"
    name = "Gmail"
    version = "1.0.0"
    category = "integrations"
    description = "Sends automated email notifications, HTML performance reports, or exported CSV attachments via Gmail / SMTP."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    inputs = [
        RecipePort(
            id="input",
            label="Input Data / Metrics",
            type="dataframe",
            required=False,
            max_connections=1,
            description="Upstream dataset or metrics to email"
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
                "to_email": {
                    "type": "string",
                    "title": "Recipient Email Address(es)",
                    "description": "Recipient email(s) separated by commas (e.g. data-team@company.com)."
                },
                "subject": {
                    "type": "string",
                    "title": "Email Subject",
                    "default": "Pipeline Execution Report - Success",
                    "description": "Email subject line."
                },
                "body": {
                    "type": "string",
                    "title": "Email Body Text",
                    "default": "Your visual ML pipeline execution has finished successfully.",
                    "description": "Plain text or HTML body message."
                },
                "attach_csv": {
                    "type": "boolean",
                    "title": "Attach Data as CSV",
                    "default": False,
                    "description": "If enabled, attaches incoming rows as a CSV file attachment."
                },
                "smtp_host": {
                    "type": "string",
                    "title": "SMTP Host (Optional)",
                    "default": "smtp.gmail.com"
                },
                "smtp_port": {
                    "type": "integer",
                    "title": "SMTP Port",
                    "default": 587
                },
                "smtp_user": {
                    "type": "string",
                    "title": "Sender Email / Username"
                },
                "smtp_password": {
                    "type": "string",
                    "title": "App Password (SMTP)",
                    "description": "Google App Password generated from your Google Account settings."
                }
            },
            "required": ["to_email", "subject", "body"]
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        to_email = str(config.get("to_email", "")).strip()
        subject = config.get("subject", "Pipeline Update")
        body = config.get("body", "Execution finished.")
        attach_csv = bool(config.get("attach_csv", False))
        smtp_user = str(config.get("smtp_user", "")).strip()
        smtp_pass = str(config.get("smtp_password", "")).strip()
        smtp_host = config.get("smtp_host", "smtp.gmail.com")
        smtp_port = int(config.get("smtp_port", 587))

        df = inputs.get("dataframe")
        if df is None and context and isinstance(context, dict):
            df = context.get("dataframe")

        delivery_status = "SIMULATED"
        status_msg = f"Email simulated to {to_email} (No SMTP credentials configured)."

        if smtp_user and smtp_pass and to_email:
            try:
                msg = MIMEMultipart()
                msg["From"] = smtp_user
                msg["To"] = to_email
                msg["Subject"] = subject
                msg.attach(MIMEText(body, "html" if "<" in body and ">" in body else "plain"))

                if attach_csv and df is not None:
                    csv_buffer = io.StringIO()
                    df.to_csv(csv_buffer, index=False)
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(csv_buffer.getvalue().encode("utf-8"))
                    encoders.encode_base64(part)
                    part.add_header("Content-Disposition", "attachment; filename=\"pipeline_data.csv\"")
                    msg.attach(part)

                with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as server:
                    server.starttls()
                    server.login(smtp_user, smtp_pass)
                    server.send_message(msg)

                delivery_status = "SENT"
                status_msg = f"Email successfully sent to {to_email}."
            except Exception as e:
                logger.warning(f"Gmail send failed: {e}")
                delivery_status = "FAILED"
                status_msg = f"SMTP error: {str(e)}"

        result = {
            "output_summary": {
                "title": "Gmail Notification",
                "message": status_msg,
                "status": delivery_status
            },
            "metrics": {
                "recipient": to_email,
                "delivery_status": delivery_status,
                "notification_type": "gmail"
            }
        }
        if df is not None:
            result["dataframe"] = df

        return result
