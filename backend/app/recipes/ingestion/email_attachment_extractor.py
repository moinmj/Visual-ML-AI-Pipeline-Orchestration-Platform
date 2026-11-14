"""
Email Attachment Extractor Recipe (Ingestion & Orchestration -> File & Web Ingestion).

Scans an IMAP mailbox for emails matching specific senders, subjects, or dates, extracts analytical
attachments (CSV, Excel .xlsx/.xls, Parquet, JSON), and parses them into a pipeline DataFrame or
outputs an attachment manifest.
Includes simulation fallback when mail server is offline, mock email injection for automated tests,
configurable attachment pattern filters, and metadata enrichment (_email_sender, _email_subject, _email_date).
"""

import email
from email import policy
import fnmatch
import io
import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from backend.app.core.exceptions import ExecutionException, ValidationException
from backend.app.core.logging import logger
from backend.app.recipes.base.recipe import BaseRecipe, RecipePort


class EmailAttachmentExtractorRecipe(BaseRecipe):
    recipe_id = "email_attachment_extractor"
    name = "Email Attachment Extractor"
    version = "1.0.0"
    category = "ingestion"
    group = "Ingestion & Orchestration"
    subgroup = "File & Web Ingestion"
    description = "Reads a mailbox and extracts CSV/Excel attachments as datasets into a DataFrame."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    inputs: List[RecipePort] = []

    outputs = [
        RecipePort(
            id="output",
            label="Output Table",
            type="dataframe",
            description="Parsed email attachments dataset or attachment manifest",
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "sections": [
                {
                    "id": "connection",
                    "title": "Mail Server & Credentials",
                    "description": "Configure IMAP server address, port, and authentication credentials",
                },
                {
                    "id": "filters",
                    "title": "Mailbox & Message Filters",
                    "description": "Folder selection, read status, sender, and subject pattern filters",
                },
                {
                    "id": "parsing",
                    "title": "Attachment Matching & Parsing",
                    "description": "Attachment filename glob pattern, format parsing, and Excel sheet options",
                },
                {
                    "id": "runtime_limits",
                    "title": "Execution & Simulation",
                    "description": "Batch email limits, message flags, and simulation fallback",
                },
            ],
            "properties": {
                "imap_server": {
                    "type": "string",
                    "title": "IMAP Server",
                    "default": "imap.gmail.com",
                    "input_type": "text",
                    "section": "connection",
                    "description": "Host address of the secure IMAP mail server.",
                },
                "port": {
                    "type": "integer",
                    "title": "IMAP Port",
                    "default": 993,
                    "minimum": 1,
                    "maximum": 65535,
                    "input_type": "number",
                    "section": "connection",
                    "description": "Network port for secure IMAP SSL/TLS connection (usually 993).",
                },
                "username": {
                    "type": "string",
                    "title": "Email / Username",
                    "default": "data_feed@company.com",
                    "input_type": "text",
                    "section": "connection",
                    "description": "Email address or username used to authenticate to the mailbox.",
                },
                "password": {
                    "type": "string",
                    "title": "Password / App Password",
                    "default": "",
                    "input_type": "password",
                    "section": "connection",
                    "description": "Account password or app-specific password for IMAP authentication.",
                },
                "folder": {
                    "type": "string",
                    "title": "Mailbox Folder",
                    "default": "INBOX",
                    "input_type": "text",
                    "section": "filters",
                    "description": "Mailbox folder to inspect (e.g. 'INBOX', 'Feeds', 'Reports').",
                },
                "search_filter": {
                    "type": "string",
                    "title": "Search Filter",
                    "enum": ["UNSEEN", "ALL"],
                    "default": "UNSEEN",
                    "input_type": "dropdown",
                    "section": "filters",
                    "description": "Message search scope in the target folder.",
                },
                "sender_filter": {
                    "type": "string",
                    "title": "Filter By Sender",
                    "default": "",
                    "input_type": "text",
                    "section": "filters",
                    "description": "Optional sender email address substring or regex pattern.",
                },
                "subject_filter": {
                    "type": "string",
                    "title": "Filter By Subject",
                    "default": "",
                    "input_type": "text",
                    "section": "filters",
                    "description": "Optional email subject line keyword or pattern.",
                },
                "attachment_pattern": {
                    "type": "string",
                    "title": "Attachment Filename Pattern",
                    "default": "*.csv;*.xlsx;*.parquet;*.json",
                    "input_type": "text",
                    "section": "parsing",
                    "description": "Glob patterns separated by semicolon for matching attachment files.",
                },
                "output_mode": {
                    "type": "string",
                    "title": "Output Mode",
                    "enum": ["parse_data", "attachment_manifest"],
                    "default": "parse_data",
                    "input_type": "dropdown",
                    "section": "parsing",
                    "description": "'parse_data' parses attachment rows; 'attachment_manifest' outputs discovered files listing.",
                },
                "excel_sheet_name": {
                    "type": "string",
                    "title": "Excel Sheet Name / Index",
                    "default": "0",
                    "input_type": "text",
                    "section": "parsing",
                    "description": "Sheet name or numeric index (e.g. '0' or 'Sheet1') when parsing Excel attachments.",
                },
                "delimiter": {
                    "type": "string",
                    "title": "CSV Delimiter",
                    "enum": [",", "\t", ";", "|"],
                    "default": ",",
                    "input_type": "dropdown",
                    "section": "parsing",
                    "description": "Field delimiter character for CSV attachments.",
                },
                "max_emails": {
                    "type": "integer",
                    "title": "Max Emails to Scan",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 100,
                    "input_type": "number",
                    "section": "runtime_limits",
                    "description": "Maximum number of candidate emails to inspect per run.",
                },
                "mark_as_read": {
                    "type": "boolean",
                    "title": "Mark Emails as Read",
                    "default": False,
                    "input_type": "checkbox",
                    "section": "runtime_limits",
                    "description": "Flag scanned emails as read (\\Seen) on the mail server.",
                },
                "include_metadata_columns": {
                    "type": "boolean",
                    "title": "Include Email Metadata Columns",
                    "default": True,
                    "input_type": "checkbox",
                    "section": "runtime_limits",
                    "description": "Append '_email_sender', '_email_subject', '_attachment_name', and '_email_date' columns.",
                },
                "simulation_fallback": {
                    "type": "boolean",
                    "title": "Simulation Fallback",
                    "default": True,
                    "input_type": "checkbox",
                    "section": "runtime_limits",
                    "description": "Return synthetic email attachment data if server connection fails or inbox is empty.",
                },
                "mock_emails": {
                    "type": "string",
                    "title": "Mock Emails (JSON)",
                    "default": "",
                    "input_type": "textarea",
                    "section": "runtime_limits",
                    "description": "Optional JSON list of mock email objects with attachments for automated test injection.",
                },
            },
            "required": ["imap_server", "username"],
        }

    def _matches_pattern(self, filename: str, patterns_str: str) -> bool:
        if not patterns_str or patterns_str == "*":
            return True
        patterns = [p.strip() for p in patterns_str.split(";") if p.strip()]
        for pat in patterns:
            if fnmatch.fnmatch(filename, pat):
                return True
            try:
                if re.search(pat, filename, re.IGNORECASE):
                    return True
            except Exception:
                pass
        return False

    def _fetch_real_emails(
        self,
        server: str,
        port: int,
        username: str,
        password: str,
        folder: str,
        search_filter: str,
        sender_filter: str,
        subject_filter: str,
        attachment_pattern: str,
        max_emails: int,
        mark_as_read: bool,
    ) -> List[Dict[str, Any]]:
        import imaplib

        attachments: List[Dict[str, Any]] = []
        mail = imaplib.IMAP4_SSL(server, port, timeout=4.0)
        try:
            mail.login(username, password)
            mail.select(folder)
            typ, data = mail.search(None, search_filter)
            if typ != "OK":
                return []

            mail_ids = data[0].split()
            # Process newest first
            mail_ids.reverse()
            scanned_count = 0

            for mid in mail_ids:
                if scanned_count >= max_emails:
                    break
                typ, msg_data = mail.fetch(mid, "(RFC822)")
                if typ != "OK" or not msg_data or not msg_data[0]:
                    continue

                raw_email = msg_data[0][1]
                msg = email.message_from_bytes(raw_email, policy=policy.default)
                scanned_count += 1

                sender = str(msg.get("From", ""))
                subject = str(msg.get("Subject", ""))
                date_str = str(msg.get("Date", ""))

                if sender_filter and sender_filter.lower() not in sender.lower():
                    continue
                if subject_filter and subject_filter.lower() not in subject.lower():
                    continue

                found_for_msg = False
                for part in msg.walk():
                    if part.get_content_maintype() == "multipart":
                        continue
                    filename = part.get_filename()
                    if not filename:
                        continue
                    if self._matches_pattern(filename, attachment_pattern):
                        payload = part.get_payload(decode=True)
                        if payload:
                            attachments.append({
                                "email_id": mid.decode("utf-8", errors="ignore"),
                                "sender": sender,
                                "subject": subject,
                                "date": date_str,
                                "filename": filename,
                                "bytes": payload,
                                "size_bytes": len(payload),
                                "content_type": part.get_content_type(),
                            })
                            found_for_msg = True

                if found_for_msg and mark_as_read:
                    mail.store(mid, "+FLAGS", "\\Seen")

        finally:
            try:
                mail.close()
            except Exception:
                pass
            mail.logout()

        return attachments

    def _generate_synthetic_attachments(
        self,
        attachment_pattern: str,
        max_emails: int,
    ) -> List[Dict[str, Any]]:
        now_ts = datetime.now(timezone.utc)
        count = min(max(1, max_emails), 2)
        simulated: List[Dict[str, Any]] = []

        is_excel = "xlsx" in attachment_pattern.lower() or "xls" in attachment_pattern.lower()

        for i in range(1, count + 1):
            rows = [
                {"invoice_id": f"INV-{5000 * i + j}", "client": f"Enterprise-{j}", "amount_usd": round(1200.0 * j + i, 2), "paid": True}
                for j in range(1, 5)
            ]
            df = pd.DataFrame(rows)

            bio = io.BytesIO()
            if is_excel:
                fname = f"invoice_batch_{now_ts.strftime('%Y%m%d')}_{i}.xlsx"
                try:
                    df.to_excel(bio, index=False)
                except Exception:
                    # Fallback to csv if openpyxl not installed
                    fname = f"invoice_batch_{now_ts.strftime('%Y%m%d')}_{i}.csv"
                    df.to_csv(bio, index=False)
            else:
                fname = f"invoice_batch_{now_ts.strftime('%Y%m%d')}_{i}.csv"
                df.to_csv(bio, index=False)

            content = bio.getvalue()
            simulated.append({
                "email_id": f"SIM-{i:03d}",
                "sender": f"billing-partner{i}@enterprise-vendor.com",
                "subject": f"Automated Daily Invoice Feed - Part {i}",
                "date": now_ts.isoformat(),
                "filename": fname,
                "bytes": content,
                "size_bytes": len(content),
                "content_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" if fname.endswith(".xlsx") else "text/csv",
            })

        return simulated

    def _parse_attachment_bytes(
        self,
        raw_bytes: bytes,
        filename: str,
        excel_sheet_name: str,
        delimiter: str,
    ) -> pd.DataFrame:
        lower = filename.lower()
        bio = io.BytesIO(raw_bytes)

        if lower.endswith(".xlsx") or lower.endswith(".xls"):
            sheet: Any = 0
            if excel_sheet_name.isdigit():
                sheet = int(excel_sheet_name)
            elif excel_sheet_name.strip():
                sheet = excel_sheet_name.strip()
            return pd.read_excel(bio, sheet_name=sheet)

        if lower.endswith(".parquet") or lower.endswith(".pq"):
            return pd.read_parquet(bio)

        if lower.endswith(".jsonl"):
            return pd.read_json(bio, lines=True)

        if lower.endswith(".json"):
            return pd.read_json(bio)

        # Default CSV
        try:
            return pd.read_csv(bio, sep=delimiter)
        except Exception:
            bio.seek(0)
            return pd.read_csv(bio, sep=None, engine="python")

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.time()
        imap_server = str(config.get("imap_server", "")).strip()
        if not imap_server:
            raise ValidationException("Parameter 'imap_server' is required.")

        port = int(config.get("port", 993))
        if port < 1 or port > 65535:
            raise ValidationException(f"Port must be between 1 and 65535, got {port}.")

        username = str(config.get("username", "")).strip()
        if not username:
            raise ValidationException("Parameter 'username' is required.")

        password = str(config.get("password", "") or "")
        folder = str(config.get("folder", "INBOX")).strip()
        search_filter = str(config.get("search_filter", "UNSEEN")).strip().upper()
        sender_filter = str(config.get("sender_filter", "")).strip()
        subject_filter = str(config.get("subject_filter", "")).strip()
        attachment_pattern = str(config.get("attachment_pattern", "*.csv;*.xlsx")).strip()
        output_mode = str(config.get("output_mode", "parse_data")).strip().lower()
        excel_sheet_name = str(config.get("excel_sheet_name", "0")).strip()
        delimiter = str(config.get("delimiter", ","))
        max_emails = max(1, min(int(config.get("max_emails", 10)), 100))
        mark_as_read = bool(config.get("mark_as_read", False))
        include_meta = bool(config.get("include_metadata_columns", True))
        simulation_fallback = bool(config.get("simulation_fallback", True))
        mock_emails_raw = config.get("mock_emails", "")

        attachments: List[Dict[str, Any]] = []
        is_simulation = False

        # 1. Custom mock emails injected via config (ideal for unit testing)
        if mock_emails_raw:
            try:
                parsed = json.loads(mock_emails_raw) if isinstance(mock_emails_raw, str) else mock_emails_raw
                if isinstance(parsed, list):
                    for idx, item in enumerate(parsed):
                        fn = item.get("filename", f"attachment_{idx}.csv")
                        content = item.get("content", "")
                        b_content = content.encode("utf-8") if isinstance(content, str) else bytes(content)
                        attachments.append({
                            "email_id": item.get("email_id", f"MSG-{idx}"),
                            "sender": item.get("sender", "mock-sender@test.internal"),
                            "subject": item.get("subject", "Test Email with Attachment"),
                            "date": item.get("date", datetime.now(timezone.utc).isoformat()),
                            "filename": fn,
                            "bytes": b_content,
                            "size_bytes": len(b_content),
                            "content_type": item.get("content_type", "text/csv"),
                        })
                    is_simulation = True
            except Exception as e:
                raise ValidationException(f"Invalid JSON provided in 'mock_emails': {e}")

        # 2. Attempt real IMAP fetch if no mock provided
        if not attachments:
            try:
                attachments = self._fetch_real_emails(
                    server=imap_server,
                    port=port,
                    username=username,
                    password=password,
                    folder=folder,
                    search_filter=search_filter,
                    sender_filter=sender_filter,
                    subject_filter=subject_filter,
                    attachment_pattern=attachment_pattern,
                    max_emails=max_emails,
                    mark_as_read=mark_as_read,
                )
            except Exception as e:
                logger.warning(f"IMAP connection to {imap_server}:{port} failed: {e}")
                if not simulation_fallback:
                    raise ExecutionException(f"Failed connecting to IMAP server '{imap_server}:{port}': {e}")
                attachments = self._generate_synthetic_attachments(attachment_pattern, max_emails)
                is_simulation = True

        if not attachments and simulation_fallback:
            attachments = self._generate_synthetic_attachments(attachment_pattern, max_emails)
            is_simulation = True

        total_bytes = sum(a.get("size_bytes", 0) for a in attachments)

        # Branch A: Output mode is attachment_manifest
        if output_mode == "attachment_manifest":
            rows = []
            for a in attachments:
                rows.append({
                    "email_id": a["email_id"],
                    "sender": a["sender"],
                    "subject": a["subject"],
                    "date": a["date"],
                    "filename": a["filename"],
                    "size_bytes": a["size_bytes"],
                    "content_type": a["content_type"],
                })
            manifest_df = pd.DataFrame(rows)
            if manifest_df.empty:
                manifest_df = pd.DataFrame(columns=["email_id", "sender", "subject", "date", "filename", "size_bytes", "content_type"])

            elapsed = round((time.time() - t0) * 1000, 2)
            metrics = {
                "execution_time_ms": elapsed,
                "imap_server": imap_server,
                "folder": folder,
                "attachments_found": len(attachments),
                "total_bytes": total_bytes,
                "is_simulation": is_simulation,
                "output_mode": "attachment_manifest",
            }
            return {"dataframe": manifest_df, "metrics": metrics}

        # Branch B: Output mode is parse_data
        parsed_dfs: List[pd.DataFrame] = []
        for a in attachments:
            try:
                df_part = self._parse_attachment_bytes(
                    raw_bytes=a["bytes"],
                    filename=a["filename"],
                    excel_sheet_name=excel_sheet_name,
                    delimiter=delimiter,
                )
            except Exception as e:
                logger.warning(f"Error parsing attachment '{a['filename']}': {e}")
                continue

            if not df_part.empty:
                if include_meta:
                    df_part["_email_sender"] = a["sender"]
                    df_part["_email_subject"] = a["subject"]
                    df_part["_attachment_name"] = a["filename"]
                    df_part["_email_date"] = a["date"]
                parsed_dfs.append(df_part)

        combined_df = pd.concat(parsed_dfs, ignore_index=True) if parsed_dfs else pd.DataFrame()

        elapsed = round((time.time() - t0) * 1000, 2)
        metrics = {
            "execution_time_ms": elapsed,
            "imap_server": imap_server,
            "folder": folder,
            "attachments_ingested": len(attachments),
            "total_rows": len(combined_df),
            "total_bytes": total_bytes,
            "is_simulation": is_simulation,
            "output_mode": "parse_data",
        }

        return {"dataframe": combined_df, "metrics": metrics}
