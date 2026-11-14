"""
Google BigQuery Connector Recipe (Ingestion & Orchestration -> Database & Warehouse Connectors).

Reads analytical tables or Standard SQL queries from Google Cloud BigQuery into the pipeline as a DataFrame.
Supports service account JSON strings, key files, Application Default Credentials (ADC), stored credentials,
maximum bytes billed guardrails, benchmark simulation fallback, and code generation.
"""

import re
import os
import json
import time
from typing import Dict, Any, List, Optional, Tuple

try:
    import pandas as pd
except ImportError:
    pd = None

from backend.app.recipes.base.recipe import BaseRecipe, RecipePort
from backend.app.core.logging import logger
from backend.app.core.config import settings
from backend.app.core.exceptions import ValidationException


class BigQueryConnectorRecipe(BaseRecipe):
    recipe_id = "bigquery_connector"
    name = "Google BigQuery Connector"
    version = "1.0.0"
    category = "ingestion"
    group = "Ingestion & Orchestration"
    subgroup = "Database & Warehouse Connectors"
    description = "Reads analytical tables or Standard SQL queries from Google BigQuery."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    inputs: List[RecipePort] = []

    outputs = [
        RecipePort(
            id="output",
            label="BigQuery Dataset",
            type="dataframe",
            description="DataFrame containing retrieved BigQuery analytical records"
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "sections": [
                {"id": "connection", "title": "Connection Configuration", "description": "Configure GCP authentication and project"},
                {"id": "query_selection", "title": "Query & Table Selection", "description": "Choose datasets, tables, or Standard SQL"},
                {"id": "runtime_limits", "title": "Limits & Performance", "description": "Row limits and safety billing guardrails"}
            ],
            "properties": {
                # ── 1. Auth Mode & GCP Credentials ──
                "connection_mode": {
                    "type": "string",
                    "title": "Auth Mode",
                    "input_type": "dropdown",
                    "enum": ["service_account_json", "credentials_file", "adc", "sso", "credential"],
                    "default": "service_account_json",
                    "section": "connection",
                    "description": "Method of authentication: Service Account JSON key string, credentials file path, Application Default Credentials (ADC), Single Sign-On (Google SSO / WIF), or stored credential."
                },
                "credential_id": {
                    "type": "string",
                    "title": "Connected GCP Credential",
                    "input_type": "credential_dropdown",
                    "provider": "bigquery",
                    "section": "connection",
                    "description": "Select a stored GCP/BigQuery credential from platform settings.",
                    "visible_when": {
                        "field": "connection_mode",
                        "equals": "credential"
                    }
                },
                "sso_provider": {
                    "type": "string",
                    "title": "SSO Provider",
                    "input_type": "dropdown",
                    "enum": ["google_sso", "workforce_identity_federation"],
                    "default": "google_sso",
                    "section": "connection",
                    "description": "Google Cloud SSO mechanism: Google Workspace OAuth SSO or Workforce Identity Federation (WIF).",
                    "visible_when": {
                        "field": "connection_mode",
                        "equals": "sso"
                    }
                },
                "sso_access_token": {
                    "type": "string",
                    "title": "OAuth / SSO Access Token",
                    "input_type": "password",
                    "default": "",
                    "placeholder": "Enter Google OAuth 2.0 access token",
                    "section": "connection",
                    "description": "OAuth 2.0 Bearer access token for BigQuery API authorization.",
                    "visible_when": {
                        "field": "connection_mode",
                        "equals": "sso"
                    }
                },
                "workforce_pool_provider": {
                    "type": "string",
                    "title": "Workforce Pool Provider Resource",
                    "input_type": "text",
                    "default": "",
                    "placeholder": "locations/global/workforcePools/my-pool/providers/my-idp",
                    "section": "connection",
                    "description": "GCP Workforce Identity Federation pool provider resource identifier (used when SSO Provider is WIF).",
                    "visible_when": {
                        "field": "connection_mode",
                        "equals": "sso"
                    }
                },
                "project_id": {
                    "type": "string",
                    "title": "GCP Project ID",
                    "input_type": "text",
                    "default": "",
                    "placeholder": "e.g. my-gcp-project-123",
                    "section": "connection",
                    "description": "Google Cloud Project ID."
                },
                "credentials_json": {
                    "type": "string",
                    "title": "Service Account JSON Key",
                    "input_type": "textarea",
                    "default": "",
                    "placeholder": "{\"type\": \"service_account\", \"project_id\": \"...\"}",
                    "section": "connection",
                    "description": "Raw JSON string content of the Google Cloud Service Account key file.",
                    "visible_when": {
                        "field": "connection_mode",
                        "equals": "service_account_json"
                    }
                },
                "credentials_file_path": {
                    "type": "string",
                    "title": "Key File Path",
                    "input_type": "text",
                    "default": "",
                    "placeholder": "e.g. /etc/secrets/gcp-sa-key.json",
                    "section": "connection",
                    "description": "Local server file path to the service account JSON key (or set GOOGLE_APPLICATION_CREDENTIALS).",
                    "visible_when": {
                        "field": "connection_mode",
                        "equals": "credentials_file"
                    }
                },
                "location": {
                    "type": "string",
                    "title": "Dataset Location",
                    "input_type": "text",
                    "default": "US",
                    "placeholder": "US",
                    "section": "connection",
                    "description": "Geographic location of the dataset (e.g. 'US', 'EU', 'asia-south1')."
                },
                "timeout": {
                    "type": "integer",
                    "title": "Timeout (seconds)",
                    "input_type": "number",
                    "default": 30,
                    "section": "connection",
                    "description": "Query execution timeout in seconds."
                },

                # ── 2. Read Mode & Target Selection ──
                "read_mode": {
                    "type": "string",
                    "title": "Read Mode",
                    "input_type": "dropdown",
                    "enum": ["table", "query"],
                    "default": "table",
                    "section": "query_selection",
                    "description": "Read an entire table/view or run a custom read-only Standard SQL query."
                },
                "dataset_id": {
                    "type": "string",
                    "title": "Dataset ID",
                    "input_type": "text",
                    "default": "",
                    "placeholder": "e.g. analytics_data or production",
                    "section": "query_selection",
                    "description": "Target BigQuery dataset name."
                },
                "table_id": {
                    "type": "string",
                    "title": "Table Name",
                    "input_type": "text",
                    "default": "",
                    "placeholder": "e.g. events_daily",
                    "section": "query_selection",
                    "description": "Target BigQuery table or view name (used in 'table' mode).",
                    "visible_when": {
                        "field": "read_mode",
                        "equals": "table"
                    }
                },
                "columns": {
                    "type": "string",
                    "title": "Selected Columns",
                    "input_type": "text",
                    "default": "*",
                    "placeholder": "e.g. event_date, user_id, event_name or *",
                    "section": "query_selection",
                    "description": "Comma-separated columns to select (e.g. 'event_date, user_id, event_name'). Leave '*' for all.",
                    "visible_when": {
                        "field": "read_mode",
                        "equals": "table"
                    }
                },
                "query": {
                    "type": "string",
                    "title": "Custom SQL Query",
                    "input_type": "textarea",
                    "default": "",
                    "placeholder": "e.g. SELECT * FROM `my-gcp-project.analytics_data.events_daily` LIMIT 100",
                    "section": "query_selection",
                    "description": "Custom BigQuery Standard SQL query. Only used when Read Mode is 'query'. Destructive DDL/DML statements are prohibited.",
                    "visible_when": {
                        "field": "read_mode",
                        "equals": "query"
                    }
                },

                # ── 3. Limits & Guardrails ──
                "limit": {
                    "type": "integer",
                    "title": "Row Limit",
                    "input_type": "number",
                    "default": 1000,
                    "section": "runtime_limits",
                    "description": "Maximum number of rows to retrieve (1 to 1,000,000)."
                },
                "maximum_bytes_billed": {
                    "type": "integer",
                    "title": "Maximum Bytes Billed (Guardrail)",
                    "input_type": "number",
                    "default": 1073741824,
                    "section": "runtime_limits",
                    "description": "Maximum bytes scanned allowed for query safety (default 1 GB = 1073741824 bytes)."
                },
                "simulation_mode": {
                    "type": "boolean",
                    "title": "Simulation / Demo Mode",
                    "input_type": "checkbox",
                    "default": False,
                    "section": "runtime_limits",
                    "description": "When enabled or when google-cloud-bigquery is offline, generates realistic BigQuery analytical benchmark data."
                }
            },
            "required": ["connection_mode", "project_id", "read_mode"]
        }

    _DISALLOWED_SQL = re.compile(
        r"\b(INSERT\s+INTO|UPDATE\s+|DELETE\s+FROM|DROP\s+(TABLE|VIEW|DATASET|SCHEMA)|"
        r"TRUNCATE\s+TABLE|ALTER\s+(TABLE|VIEW|DATASET)|CREATE\s+(TABLE|VIEW|DATASET|SCHEMA)|"
        r"MERGE\s+INTO|EXPORT\s+DATA)\b",
        re.IGNORECASE
    )

    def _validate_sql_safety(self, sql: str) -> Tuple[bool, Optional[str]]:
        cleaned = re.sub(r"--.*?$|/\*.*?\*/|#.*?$", "", sql, flags=re.MULTILINE).strip()
        if not cleaned:
            return False, "Query cannot be empty."
        if not cleaned.upper().startswith(("SELECT", "WITH", "SHOW", "DESCRIBE", "EXPLAIN")):
            return False, "Only read-only queries (SELECT, WITH, SHOW, DESCRIBE, EXPLAIN) are permitted."
        if self._DISALLOWED_SQL.search(cleaned):
            return False, "Data-modifying or DDL statements (INSERT, UPDATE, DELETE, DROP, TRUNCATE, ALTER, CREATE, MERGE) are strictly prohibited."
        return True, None

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        errors: List[str] = []
        project_id = str(config.get("project_id", "")).strip()
        if not project_id and not config.get("simulation_mode"):
            errors.append("GCP Project ID is required.")

        conn_mode = str(config.get("connection_mode", "service_account_json")).strip().lower()
        if not config.get("simulation_mode"):
            if conn_mode == "service_account_json":
                raw_json = str(config.get("credentials_json", "")).strip()
                if raw_json:
                    try:
                        parsed = json.loads(raw_json)
                        if not isinstance(parsed, dict) or "type" not in parsed:
                            errors.append("Service Account JSON must be a valid Google Service Account dictionary.")
                    except json.JSONDecodeError as jde:
                        errors.append(f"Invalid Service Account JSON format: {jde}")
            elif conn_mode == "credentials_file":
                fpath = str(config.get("credentials_file_path", "")).strip()
                if not fpath:
                    errors.append("Key file path is required in 'credentials_file' mode.")
            elif conn_mode == "sso":
                sso_provider = str(config.get("sso_provider", "google_sso")).strip().lower()
                if sso_provider == "workforce_identity_federation" and not str(config.get("workforce_pool_provider", "")).strip():
                    errors.append("Workforce Pool Provider Resource is required when SSO Provider is WIF.")
            elif conn_mode == "credential":
                if not config.get("credential_id"):
                    errors.append("Credential ID is required in 'credential' mode.")

        read_mode = str(config.get("read_mode", "table")).strip().lower()
        if read_mode == "table":
            if not str(config.get("table_id", "")).strip():
                errors.append("Table name is required when read_mode is 'table'.")
        elif read_mode == "query":
            q = str(config.get("query", "")).strip()
            if not q:
                errors.append("SQL query is required when read_mode is 'query'.")
            else:
                safe, err_msg = self._validate_sql_safety(q)
                if not safe and err_msg:
                    errors.append(err_msg)

        limit = config.get("limit")
        if limit is not None:
            try:
                l_val = int(limit)
                if l_val <= 0:
                    errors.append("Limit must be greater than 0.")
            except (ValueError, TypeError):
                errors.append("Limit must be an integer.")

        return errors

    def _generate_benchmark_data(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        limit = max(1, min(int(config.get("limit", 100)), 5000))
        project_id = str(config.get("project_id", "my-gcp-project"))
        dataset_id = str(config.get("dataset_id", "analytics_data"))
        table_id = str(config.get("table_id", "events_daily"))

        event_names = ["page_view", "scroll", "click", "view_item", "add_to_cart", "purchase"]
        devices = ["mobile", "desktop", "tablet"]
        countries = ["US", "GB", "DE", "SA", "AE", "IN", "JP", "FR"]
        channels = ["Organic Search", "Direct", "Paid Search", "Social", "Referral"]

        records: List[Dict[str, Any]] = []
        for i in range(limit):
            ev = event_names[i % len(event_names)]
            records.append({
                "event_date": f"2026-03-{((i % 28) + 1):02d}",
                "event_timestamp": 1774000000000000 + (i * 1234567),
                "event_name": ev,
                "user_pseudo_id": f"ga4_{100000 + (i % 75):08d}.{800000 + i}",
                "device_category": devices[i % len(devices)],
                "country": countries[i % len(countries)],
                "traffic_source": channels[i % len(channels)],
                "session_engaged": 1 if i % 4 != 0 else 0,
                "engagement_time_sec": round(4.5 + ((i * 3.2) % 180.0), 1),
                "purchase_revenue_usd": round(15.99 + ((i * 7.5) % 350.0), 2) if ev == "purchase" else 0.0,
                "project_source": f"{project_id}.{dataset_id}.{table_id}"
            })
        return records

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        start_time = time.time()
        limit = max(1, min(int(config.get("limit", 1000)), 1000000))
        read_mode = str(config.get("read_mode", "table")).strip().lower()
        simulation_mode = bool(config.get("simulation_mode", False))

        project_id = str(config.get("project_id", "my-gcp-project")).strip()
        dataset_id = str(config.get("dataset_id", "analytics_data")).strip()
        table_id = str(config.get("table_id", "events_daily")).strip()
        conn_mode = str(config.get("connection_mode", "service_account_json")).strip().lower()

        records: List[Dict[str, Any]] = []
        delivery_status = "SIMULATED"
        status_msg = ""

        # Live Execution Attempt
        if not simulation_mode and project_id:
            try:
                from google.cloud import bigquery
                from google.oauth2 import service_account

                client = None
                raw_json = str(config.get("credentials_json", "")).strip()
                fpath = str(config.get("credentials_file_path", "")).strip()

                if conn_mode == "service_account_json" and raw_json:
                    info = json.loads(raw_json)
                    creds = service_account.Credentials.from_service_account_info(info)
                    client = bigquery.Client(project=project_id, credentials=creds)
                elif conn_mode == "credentials_file" and fpath and os.path.exists(fpath):
                    creds = service_account.Credentials.from_service_account_file(fpath)
                    client = bigquery.Client(project=project_id, credentials=creds)
                elif conn_mode == "sso":
                    sso_token = str(config.get("sso_access_token", "")).strip()
                    if sso_token:
                        from google.oauth2.credentials import Credentials
                        creds = Credentials(token=sso_token)
                        client = bigquery.Client(project=project_id, credentials=creds)
                    else:
                        client = bigquery.Client(project=project_id)
                else:
                    client = bigquery.Client(project=project_id)

                if read_mode == "table":
                    cols = str(config.get("columns", "*")).strip() or "*"
                    query_sql = f"SELECT {cols} FROM `{project_id}.{dataset_id}.{table_id}` LIMIT {limit}"
                else:
                    query_sql = str(config.get("query", "")).strip()
                    if not query_sql:
                        raise ValidationException("SQL Query is required when Read Mode is set to 'query'.")
                    if limit and "limit" not in query_sql.lower():
                        query_sql = f"{query_sql} LIMIT {limit}"

                job_config = bigquery.QueryJobConfig()
                max_bytes = config.get("maximum_bytes_billed")
                if max_bytes:
                    job_config.maximum_bytes_billed = int(max_bytes)

                query_job = client.query(query_sql, job_config=job_config)
                df_res = query_job.to_dataframe()
                records = df_res.to_dict(orient="records")

                delivery_status = "LIVE"
                status_msg = f"Retrieved {len(records)} live rows from Google BigQuery ({project_id}.{dataset_id}.{table_id})."
            except Exception as e:
                logger.warning(f"BigQuery connection failed: {e}")
                last_error = e

            if last_error is not None:
                raise ValidationException(f"Google BigQuery execution failed: {str(last_error)}")

        # Simulation Mode (ONLY when simulation_mode is True)
        if simulation_mode:
            records = self._generate_benchmark_data(config)
            delivery_status = "SIMULATED"
            status_msg = f"Loaded {len(records)} benchmark records for BigQuery dataset '{project_id}.{dataset_id}'."

        df_out = pd.DataFrame(records) if pd is not None else records
        duration_ms = int((time.time() - start_time) * 1000)
        col_count = len(df_out.columns) if hasattr(df_out, "columns") else (len(records[0].keys()) if records else 0)

        summary = {
            "title": "Google BigQuery Connector Summary",
            "message": status_msg,
            "status": delivery_status,
            "project_id": project_id,
            "dataset_id": dataset_id,
            "table_id": table_id,
            "rows_retrieved": len(records),
            "columns_count": col_count,
            "read_mode": read_mode,
            "duration_ms": duration_ms
        }

        metrics = {
            "rows_retrieved": len(records),
            "columns_count": col_count,
            "duration_ms": duration_ms,
            "delivery_status": delivery_status
        }

        return {
            "dataframe": df_out,
            "output": df_out,
            "output_summary": summary,
            "metrics": metrics
        }

    def to_code(self, config: Dict[str, Any]) -> str:
        project_id = config.get("project_id", "my-gcp-project")
        dataset_id = config.get("dataset_id", "analytics_data")
        table_id = config.get("table_id", "events_daily")
        read_mode = config.get("read_mode", "table")
        limit = config.get("limit", 1000)

        if read_mode == "table":
            cols = config.get("columns", "*") or "*"
            sql = f"SELECT {cols} FROM `{project_id}.{dataset_id}.{table_id}` LIMIT {limit}"
        else:
            sql = config.get("query", f"SELECT * FROM `{project_id}.{dataset_id}.{table_id}` LIMIT 100")

        return (
            f"# Generated Google BigQuery Extraction Script\n"
            f"from google.cloud import bigquery\n"
            f"import pandas as pd\n\n"
            f"client = bigquery.Client(project='{project_id}')\n"
            f"sql = \"\"\"{sql}\"\"\"\n"
            f"df = client.query(sql).to_dataframe()\n"
            f"print(f'Retrieved {{len(df)}} rows from BigQuery dataset {dataset_id}.')\n"
        )
