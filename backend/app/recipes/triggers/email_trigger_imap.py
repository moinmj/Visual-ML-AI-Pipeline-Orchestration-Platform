import imaplib
import email
from email.header import decode_header
import ssl
import io
import datetime
from typing import Dict, Any, List, Optional
import pandas as pd
from backend.app.recipes.base.recipe import BaseRecipe, RecipePort
from backend.app.core.logging import logger


def _decode_mime_header(header_value: Optional[str]) -> str:
    """Decodes MIME encoded header strings (e.g. =?utf-8?b?...?=)."""
    if not header_value:
        return ""
    try:
        decoded_parts = decode_header(header_value)
        result = []
        for content, encoding in decoded_parts:
            if isinstance(content, bytes):
                encoding = encoding or "utf-8"
                try:
                    result.append(content.decode(encoding, errors="replace"))
                except Exception:
                    result.append(content.decode("utf-8", errors="replace"))
            else:
                result.append(str(content))
        return "".join(result)
    except Exception:
        return str(header_value)


class EmailTriggerIMAPRecipe(BaseRecipe):
    """
    Email Trigger (IMAP) Recipe — Built in n8n style.
    Features separate Connection, Parameters, and Settings tabs.
    Retrieves incoming emails, parses headers/bodies, and downloads CSV/Excel attachments into DataFrames.
    """

    recipe_id = "email_trigger_imap"
    name = "Email Trigger (IMAP)"
    version = "1.0.0"
    category = "triggers"
    description = (
        "Triggers pipelines from incoming emails via IMAP. Connects to any standard mailbox "
        "(Gmail, Outlook, custom mail server), retrieves messages with unread/subject/sender filters, "
        "and extracts emails or attachments directly into a DataFrame."
    )
    input_types = []  # Root trigger node
    output_types = ["dataframe"]

    inputs: List[RecipePort] = []
    outputs: List[RecipePort] = [
        RecipePort(
            id="output",
            label="Output Table",
            type="dataframe",
            description="Retrieved email records or parsed attachments"
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "tabs": ["Connection", "Parameters", "Settings"],
            "properties": {
                # ─── 1. Connection Tab (Credentials) ─────────────────────────
                "host": {
                    "type": "text",
                    "title": "Host",
                    "default": "imap.gmail.com",
                    "tab": "Connection",
                    "description": "IMAP server hostname (e.g. imap.gmail.com, outlook.office365.com)."
                },
                "port": {
                    "type": "number",
                    "title": "Port",
                    "default": 993,
                    "tab": "Connection",
                    "description": "IMAP port (typically 993 for SSL/TLS, 143 for STARTTLS)."
                },
                "user": {
                    "type": "text",
                    "title": "User",
                    "default": "",
                    "tab": "Connection",
                    "description": "Email address or username."
                },
                "password": {
                    "type": "password",
                    "title": "Password",
                    "default": "",
                    "tab": "Connection",
                    "description": "Email password or App Password (recommended for Gmail / Outlook)."
                },
                "ssl_tls": {
                    "type": "boolean",
                    "title": "SSL/TLS",
                    "default": True,
                    "tab": "Connection",
                    "description": "Enable SSL/TLS encrypted connection."
                },
                "allow_self_signed": {
                    "type": "boolean",
                    "title": "Allow Self-Signed Certificates",
                    "default": False,
                    "tab": "Connection",
                    "description": "Accept self-signed or untrusted SSL/TLS certificates."
                },

                # ─── 2. Parameters Tab ───────────────────────────────────────
                "mailbox_name": {
                    "type": "text",
                    "title": "Mailbox Name",
                    "default": "INBOX",
                    "tab": "Parameters",
                    "description": "Mailbox folder to monitor (e.g. INBOX, Archive, Sent)."
                },
                "action": {
                    "type": "select",
                    "title": "Action",
                    "enum": ["Mark as Read", "None", "Delete"],
                    "default": "Mark as Read",
                    "tab": "Parameters",
                    "description": "Action to perform on processed emails."
                },
                "format": {
                    "type": "select",
                    "title": "Format",
                    "enum": ["Simple", "Resolved", "RAW"],
                    "default": "Simple",
                    "tab": "Parameters",
                    "description": "Simple (structured headers + text), Resolved (attachments parsed to dataset), RAW (raw MIME)."
                },
                "download_attachments": {
                    "type": "boolean",
                    "title": "Download Attachments",
                    "default": False,
                    "tab": "Parameters",
                    "description": "If enabled, extracts CSV or Excel attachments and outputs their contents as the pipeline table."
                },
                "unread_only": {
                    "type": "boolean",
                    "title": "Unread Only",
                    "default": True,
                    "tab": "Parameters",
                    "description": "Retrieve only unread (UNSEEN) emails."
                },
                "limit": {
                    "type": "number",
                    "title": "Max Emails to Fetch",
                    "default": 10,
                    "tab": "Parameters",
                    "description": "Maximum number of recent messages to process per execution."
                },
                "sender_filter": {
                    "type": "text",
                    "title": "Sender Filter (Optional)",
                    "default": "",
                    "tab": "Parameters",
                    "description": "Filter messages from specific sender or domain (e.g. reports@company.com)."
                },
                "subject_filter": {
                    "type": "text",
                    "title": "Subject Filter (Optional)",
                    "default": "",
                    "tab": "Parameters",
                    "description": "Filter messages with matching keywords in the subject line."
                },

                # ─── 3. Settings Tab (n8n Node Controls) ────────────────────
                "always_output_data": {
                    "type": "boolean",
                    "title": "Always Output Data",
                    "default": False,
                    "tab": "Settings",
                    "description": "Outputs an empty record if no emails are found instead of halting execution."
                },
                "execute_once": {
                    "type": "boolean",
                    "title": "Execute Once",
                    "default": False,
                    "tab": "Settings",
                    "description": "Execute only once even if triggered multiple times."
                },
                "retry_on_fail": {
                    "type": "boolean",
                    "title": "Retry On Fail",
                    "default": False,
                    "tab": "Settings",
                    "description": "Retry connection on transient connection errors."
                },
                "on_error": {
                    "type": "select",
                    "title": "On Error",
                    "enum": ["Stop Workflow", "Continue", "Continue (using error output)"],
                    "default": "Stop Workflow",
                    "tab": "Settings",
                    "description": "Error handling strategy if connection or mailbox access fails."
                }
            },
            "required": ["host", "mailbox_name"]
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        host = str(config.get("host") or "").strip() or "imap.gmail.com"
        port = int(config.get("port") or 993)
        user = str(config.get("user") or "").strip()
        password = str(config.get("password") or "").strip()
        ssl_tls = bool(config.get("ssl_tls", True))
        allow_self_signed = bool(config.get("allow_self_signed", False))

        mailbox_name = str(config.get("mailbox_name") or "INBOX").strip()
        action = config.get("action", "Mark as Read")
        download_attachments = bool(config.get("download_attachments", False))
        fmt = config.get("format", "Simple")
        unread_only = bool(config.get("unread_only", True))
        limit = max(1, int(config.get("limit") or 10))
        sender_filter = str(config.get("sender_filter") or "").strip()
        subject_filter = str(config.get("subject_filter") or "").strip()

        always_output_data = bool(config.get("always_output_data", False))
        on_error = config.get("on_error", "Stop Workflow")

        # ── 1. Live IMAP Fetch if Credentials Provided ─────────────────────────
        if user and password:
            try:
                ssl_context = None
                if ssl_tls:
                    if allow_self_signed:
                        ssl_context = ssl._create_unverified_context()
                    else:
                        ssl_context = ssl.create_default_context()
                    mail = imaplib.IMAP4_SSL(host, port, ssl_context=ssl_context)
                else:
                    mail = imaplib.IMAP4(host, port)

                mail.login(user, password)
                mail.select(mailbox_name)

                # Construct search criteria
                criteria = []
                if unread_only:
                    criteria.append("UNSEEN")
                else:
                    criteria.append("ALL")
                if sender_filter:
                    criteria.append(f'FROM "{sender_filter}"')
                if subject_filter:
                    criteria.append(f'SUBJECT "{subject_filter}"')

                search_query = " ".join(criteria)
                typ, data = mail.search(None, search_query)

                msg_nums = data[0].split() if data and data[0] else []
                # Process latest messages up to limit
                selected_nums = msg_nums[-limit:]
                selected_nums.reverse()

                email_records = []
                attachment_dfs = []

                for num in selected_nums:
                    typ, msg_data = mail.fetch(num, "(RFC822)")
                    if typ != "OK" or not msg_data or not msg_data[0]:
                        continue

                    raw_email = msg_data[0][1]
                    msg = email.message_from_bytes(raw_email)

                    subject = _decode_mime_header(msg.get("Subject", "(No Subject)"))
                    from_addr = _decode_mime_header(msg.get("From", ""))
                    to_addr = _decode_mime_header(msg.get("To", ""))
                    date_str = msg.get("Date", "")

                    body_text = ""
                    attachments_found = []

                    if msg.is_multipart():
                        for part in msg.walk():
                            content_type = part.get_content_type()
                            disposition = str(part.get("Content-Disposition") or "")
                            filename = part.get_filename()

                            if filename:
                                filename = _decode_mime_header(filename)
                                attachments_found.append(filename)
                                if download_attachments:
                                    payload = part.get_payload(decode=True)
                                    if payload:
                                        if filename.endswith(".csv"):
                                            try:
                                                att_df = pd.read_csv(io.BytesIO(payload))
                                                att_df["_source_email_subject"] = subject
                                                attachment_dfs.append(att_df)
                                            except Exception as pe:
                                                logger.warning(f"Failed parsing CSV attachment {filename}: {pe}")
                                        elif filename.endswith((".xlsx", ".xls")):
                                            try:
                                                att_df = pd.read_excel(io.BytesIO(payload))
                                                att_df["_source_email_subject"] = subject
                                                attachment_dfs.append(att_df)
                                            except Exception as pe:
                                                logger.warning(f"Failed parsing Excel attachment {filename}: {pe}")

                            elif content_type == "text/plain" and not body_text and "attachment" not in disposition:
                                try:
                                    body_text = part.get_payload(decode=True).decode("utf-8", errors="replace")
                                except Exception:
                                    body_text = str(part.get_payload())
                    else:
                        try:
                            body_text = msg.get_payload(decode=True).decode("utf-8", errors="replace")
                        except Exception:
                            body_text = str(msg.get_payload())

                    record = {
                        "id": num.decode("utf-8", errors="ignore"),
                        "subject": subject,
                        "from": from_addr,
                        "to": to_addr,
                        "date": date_str,
                        "body": body_text[:1000] if fmt == "Simple" else body_text,
                        "has_attachments": len(attachments_found) > 0,
                        "attachments": ", ".join(attachments_found),
                        "mailbox": mailbox_name
                    }
                    if fmt == "RAW":
                        record["raw"] = raw_email.decode("utf-8", errors="ignore")[:2000]

                    email_records.append(record)

                    # Apply Action
                    if action == "Mark as Read":
                        mail.store(num, "+FLAGS", "\\Seen")
                    elif action == "Delete":
                        mail.store(num, "+FLAGS", "\\Deleted")

                if action == "Delete":
                    mail.expunge()

                mail.close()
                mail.logout()

                # If attachments were downloaded and parsed into tabular data, use that
                if download_attachments and attachment_dfs:
                    df_out = pd.concat(attachment_dfs, ignore_index=True)
                elif email_records:
                    df_out = pd.DataFrame(email_records)
                else:
                    df_out = pd.DataFrame(columns=["id", "subject", "from", "to", "date", "body", "has_attachments", "attachments"])

                if df_out.empty and always_output_data:
                    df_out = pd.DataFrame([{
                        "id": None, "subject": "No new emails", "from": None, "to": None,
                        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "body": None, "has_attachments": False, "attachments": None
                    }])

                return {
                    "dataframe": df_out,
                    "metrics": {
                        "emails_fetched": len(email_records),
                        "attachments_parsed": len(attachment_dfs),
                        "mailbox": mailbox_name,
                        "action_applied": action
                    }
                }

            except Exception as e:
                logger.error(f"IMAP connection or fetch failed: {e}")
                if on_error == "Continue":
                    err_df = pd.DataFrame([{
                        "error": str(e),
                        "status": "FAILED",
                        "host": host,
                        "mailbox": mailbox_name,
                        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
                    }])
                    return {"dataframe": err_df, "metrics": {"error": str(e), "status": "CONTINUED_ON_ERROR"}}
                raise ValueError(f"IMAP Connection Error ({host}:{port}): {str(e)}")

        # ── 2. Simulated n8n-Style Test Mode (No Live Credentials Provided) ─────
        # Enables seamless local canvas testing, auto-wire verification, and mock runs
        now_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        simulated_records = [
            {
                "id": "msg_101",
                "subject": "Q3 Sales Pipeline & Operations Data Export",
                "from": "analytics-ops@enterprise.internal",
                "to": user or "pipeline-inbox@company.com",
                "date": now_iso,
                "body": "Automated pipeline trigger: Q3 sales transaction data is attached for ML forecasting.",
                "has_attachments": True,
                "attachments": "q3_sales_metrics.csv",
                "mailbox": mailbox_name,
                "status": "SIMULATED"
            },
            {
                "id": "msg_102",
                "subject": "Monthly Customer Churn Telemetry",
                "from": "crm-automated@enterprise.internal",
                "to": user or "pipeline-inbox@company.com",
                "date": now_iso,
                "body": "Latest monthly churn records extracted from CRM system.",
                "has_attachments": False,
                "attachments": "",
                "mailbox": mailbox_name,
                "status": "SIMULATED"
            }
        ]

        if download_attachments:
            # Provide mock dataset simulating extracted CSV attachment
            df_out = pd.DataFrame({
                "transaction_id": ["TXN-901", "TXN-902", "TXN-903", "TXN-904"],
                "store": [1, 2, 1, 3],
                "revenue": [1420.50, 2350.00, 980.20, 3100.75],
                "date": ["2026-10-01", "2026-10-02", "2026-10-03", "2026-10-04"],
                "_email_source": ["q3_sales_metrics.csv"] * 4
            })
        else:
            df_out = pd.DataFrame(simulated_records[:limit])

        return {
            "dataframe": df_out,
            "metrics": {
                "emails_fetched": len(df_out),
                "mailbox": mailbox_name,
                "status": "SIMULATED (Configure Connection credentials to connect live IMAP server)",
                "action_applied": action
            }
        }
