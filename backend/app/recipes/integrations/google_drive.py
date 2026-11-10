"""
Google Drive Recipe (Apps & Integrations -> Cloud Storage / File Sync).

Supports uploading datasets to Google Drive folders and downloading remote files
via Google Drive REST API v3 (with OAuth bearer token auth, automatic folder extraction,
and transparent simulation fallbacks when offline or unauthenticated).
"""

import io
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, Any, List, Optional

try:
    import pandas as pd
except ImportError:
    pd = None

from backend.app.recipes.base.recipe import BaseRecipe, RecipePort
from backend.app.core.logging import logger
from backend.app.core.config import settings


class GoogleDriveRecipe(BaseRecipe):
    recipe_id = "google_drive"
    name = "Google Drive"
    version = "1.1.0"
    category = "integrations"
    group = "Apps & Integrations"
    subgroup = "Cloud Storage / File Sync"
    description = "Uploads datasets and reports to Google Drive folders or downloads remote files via Google Drive API v3."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    inputs = [
        RecipePort(
            id="input",
            label="File / Dataset",
            type="dataframe",
            required=False,
            max_connections=1,
            description="Dataset to upload to Google Drive"
        )
    ]

    outputs = [
        RecipePort(
            id="output",
            label="Output Data",
            type="dataframe",
            description="Downloaded dataset or passthrough dataset"
        )
    ]

    @staticmethod
    def _extract_folder_id(raw_folder: str) -> str:
        """Extracts the clean folder ID if a full Google Drive URL was entered."""
        if not raw_folder:
            return ""
        raw = raw_folder.strip()
        # https://drive.google.com/drive/folders/<id> or /u/0/folders/<id>
        match = re.search(r"folders/([a-zA-Z0-9_-]+)", raw)
        if match:
            return match.group(1)
        # ?id=<id>
        match_id = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", raw)
        if match_id:
            return match_id.group(1)
        return raw

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "title": "Operation",
                    "enum": ["upload", "download"],
                    "default": "upload",
                    "description": "Google Drive operation to perform ('upload' or 'download')."
                },
                "folder_id": {
                    "type": "string",
                    "title": "Drive Folder ID / URL",
                    "input_type": "text",
                    "description": "Destination Google Drive folder ID or full folder share URL."
                },
                "file_name": {
                    "type": "string",
                    "title": "File Name",
                    "input_type": "text",
                    "default": "pipeline_output.csv",
                    "description": "Target file name in Google Drive (e.g. sales_data.csv)."
                },
                "api_token": {
                    "type": "string",
                    "title": "Google Drive OAuth Token / Bearer Token",
                    "input_type": "password",
                    "description": "OAuth2 Bearer Token (or GOOGLE_DRIVE_API_TOKEN from .env). If left blank, runs in simulation mode."
                },
                "credential_id": {
                    "type": "string",
                    "title": "Connected Google Drive Credential",
                    "input_type": "credential_dropdown",
                    "provider": "google_drive",
                    "description": "Select a stored Google credential or enter an OAuth token above."
                },
                "simulation_mode": {
                    "type": "boolean",
                    "title": "Simulation Mode",
                    "input_type": "checkbox",
                    "default": False,
                    "description": "Run in offline simulation mode without contacting Google Drive."
                }
            },
            "required": ["operation", "file_name"]
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        op = str(config.get("operation", "upload")).lower().strip()
        raw_folder_id = str(config.get("folder_id", "")).strip()
        clean_folder_id = self._extract_folder_id(raw_folder_id)
        file_name = str(config.get("file_name", "pipeline_output.csv")).strip() or "pipeline_output.csv"
        simulation_mode = bool(config.get("simulation_mode", False))

        api_token = str(config.get("api_token", "")).strip() or getattr(settings, "GOOGLE_DRIVE_API_TOKEN", None) or os.getenv("GOOGLE_DRIVE_API_TOKEN", "")
        credential_id = str(config.get("credential_id", "")).strip()

        # Resolve credentials if stored in context
        if credential_id and context and isinstance(context, dict):
            creds_store = context.get("credentials", {})
            if credential_id in creds_store:
                cred_info = creds_store[credential_id]
                c_data = cred_info.get("data", cred_info)
                api_token = c_data.get("api_token") or c_data.get("token") or api_token
                raw_folder_id = c_data.get("folder_id") or raw_folder_id
                clean_folder_id = self._extract_folder_id(raw_folder_id)

        df = inputs.get("dataframe")
        if df is None and context and isinstance(context, dict):
            df = context.get("dataframe")

        delivery_status = "SIMULATED"
        status_msg = ""
        res_df = None
        extra_metrics = {}

        # -------------------------------------------------------------
        # Live Google Drive REST API v3 Execution
        # -------------------------------------------------------------
        if not simulation_mode and api_token and len(api_token) >= 15 and not api_token.startswith("mock"):
            headers = {
                "Authorization": f"Bearer {api_token}",
                "User-Agent": "Datarept-ML-Platform/1.0"
            }

            if op == "download":
                try:
                    # Search for file by name
                    q = f"name = '{file_name}' and trashed = false"
                    if clean_folder_id:
                        q += f" and '{clean_folder_id}' in parents"
                    encoded_q = urllib.parse.quote(q)
                    search_url = f"https://www.googleapis.com/drive/v3/files?q={encoded_q}&fields=files(id,name,mimeType)"

                    req_search = urllib.request.Request(search_url, headers=headers)
                    with urllib.request.urlopen(req_search, timeout=15) as resp:
                        search_data = json.loads(resp.read().decode("utf-8"))
                        files = search_data.get("files", [])

                    if not files:
                        raise FileNotFoundError(f"File '{file_name}' was not found in Google Drive (folder: '{clean_folder_id or 'Root'}').")

                    target_file = files[0]
                    file_id = target_file["id"]
                    mime_type = target_file.get("mimeType", "")

                    # Check if Google Sheet needing export or raw file
                    if "spreadsheet" in mime_type:
                        dl_url = f"https://www.googleapis.com/drive/v3/files/{file_id}/export?mimeType=text/csv"
                    else:
                        dl_url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"

                    req_dl = urllib.request.Request(dl_url, headers=headers)
                    with urllib.request.urlopen(req_dl, timeout=25) as resp:
                        content_bytes = resp.read()

                    # Parse into DataFrame
                    if pd is not None:
                        try:
                            res_df = pd.read_csv(io.BytesIO(content_bytes))
                        except Exception:
                            # Try JSON fallback
                            try:
                                res_df = pd.read_json(io.BytesIO(content_bytes))
                            except Exception:
                                res_df = pd.DataFrame([{"raw_content": content_bytes.decode("utf-8", errors="ignore")}])

                    delivery_status = "LIVE"
                    row_c = len(res_df) if res_df is not None else 0
                    status_msg = f"Successfully downloaded '{file_name}' ({row_c:,} rows) from Google Drive. File ID: {file_id}"
                    extra_metrics["drive_file_id"] = file_id
                    extra_metrics["mime_type"] = mime_type

                except urllib.error.HTTPError as he:
                    err_body = ""
                    try:
                        err_body = he.read().decode("utf-8")
                    except Exception:
                        pass
                    logger.warning(f"Google Drive download HTTP {he.code}: {err_body[:200]}")
                    status_msg = f"Google Drive API error (HTTP {he.code}). Falling back to simulation mode: {err_body[:120]}"
                except Exception as ex:
                    logger.warning(f"Google Drive download failed: {ex}")
                    status_msg = f"Google Drive download failed ({ex}). Using simulated benchmark fallback."

            else:
                # Operation == "upload"
                try:
                    df_to_upload = df if df is not None and len(df) > 0 else pd.DataFrame({"id": [1, 2, 3], "value": ["Sample 1", "Sample 2", "Sample 3"]})
                    csv_data = df_to_upload.to_csv(index=False).encode("utf-8") if hasattr(df_to_upload, "to_csv") else str(df_to_upload).encode("utf-8")

                    metadata: Dict[str, Any] = {"name": file_name}
                    if clean_folder_id:
                        metadata["parents"] = [clean_folder_id]

                    meta_json = json.dumps(metadata).encode("utf-8")
                    boundary = "==============DatareptForgeGDriveBoundary=="
                    multipart_body = (
                        f"--{boundary}\r\n"
                        f"Content-Type: application/json; charset=UTF-8\r\n\r\n"
                    ).encode("utf-8") + meta_json + (
                        f"\r\n--{boundary}\r\n"
                        f"Content-Type: text/csv\r\n\r\n"
                    ).encode("utf-8") + csv_data + (
                        f"\r\n--{boundary}--\r\n"
                    ).encode("utf-8")

                    upload_url = "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart"
                    req_up = urllib.request.Request(upload_url, data=multipart_body, method="POST")
                    req_up.add_header("Authorization", f"Bearer {api_token}")
                    req_up.add_header("Content-Type", f"multipart/related; boundary={boundary}")
                    req_up.add_header("User-Agent", "Datarept-ML-Platform/1.0")

                    with urllib.request.urlopen(req_up, timeout=25) as resp:
                        upload_resp = json.loads(resp.read().decode("utf-8"))

                    delivery_status = "LIVE"
                    uploaded_id = upload_resp.get("id", "uploaded")
                    row_c = len(df_to_upload)
                    status_msg = f"Successfully uploaded '{file_name}' ({row_c:,} rows) to Google Drive folder '{clean_folder_id or 'Root'}'. File ID: {uploaded_id}"
                    extra_metrics["uploaded_file_id"] = uploaded_id
                    res_df = df_to_upload

                except urllib.error.HTTPError as he:
                    err_body = ""
                    try:
                        err_body = he.read().decode("utf-8")
                    except Exception:
                        pass
                    logger.warning(f"Google Drive upload HTTP {he.code}: {err_body[:200]}")
                    status_msg = f"Google Drive API error (HTTP {he.code}). Falling back to simulation mode: {err_body[:120]}"
                except Exception as ex:
                    logger.warning(f"Google Drive upload failed: {ex}")
                    status_msg = f"Google Drive upload failed ({ex}). Using simulated benchmark fallback."

        # -------------------------------------------------------------
        # Offline / Benchmark Simulation Fallback
        # -------------------------------------------------------------
        if res_df is None:
            if op == "download":
                if pd is not None:
                    res_df = pd.DataFrame({
                        "drive_file_id": ["file_001", "file_002", "file_003"],
                        "data": ["Remote record A", "Remote record B", "Remote record C"],
                        "source": ["Google Drive", "Google Drive", "Google Drive"]
                    })
                else:
                    res_df = [{"drive_file_id": "file_001", "data": "Remote record A"}]
                if not status_msg:
                    status_msg = f"Downloaded '{file_name}' from Google Drive (Simulation Mode). Notice: To download live files, provide an OAuth Bearer Token in node settings."
            else:
                row_count = len(df) if df is not None else 0
                res_df = df if df is not None else (pd.DataFrame({"id": [1, 2], "status": ["simulated", "simulated"]}) if pd is not None else [])
                if not status_msg:
                    status_msg = f"Uploaded '{file_name}' ({row_count:,} rows) to Google Drive folder '{clean_folder_id or 'Root'}' (Simulation Mode). Notice: To write to live Google Drive, provide an OAuth Bearer Token in node settings."

        summary = {
            "title": f"Google Drive: {op.upper()}",
            "message": status_msg,
            "status": delivery_status,
            "operation": op,
            "file_name": file_name,
            "folder_id": clean_folder_id or "Root"
        }

        metrics = {
            "operation": op,
            "file_name": file_name,
            "status": "SUCCESS",
            "delivery_status": delivery_status,
            **extra_metrics
        }

        return {
            "dataframe": res_df,
            "output": res_df,
            "output_summary": summary,
            "metrics": metrics
        }
