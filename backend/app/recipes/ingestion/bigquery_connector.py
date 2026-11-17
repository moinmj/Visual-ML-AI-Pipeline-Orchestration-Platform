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
    icon = "google-bigquery"
    group = "Ingestion & Orchestration"
    subgroup = "Database & Warehouse Connectors"
    description = "Reads analytical tables or Standard SQL queries from Google BigQuery."
    input_types: List[str] = []
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
                    "placeholder": "e.g. analytics_data or bigquery-public-data:samples",
                    "section": "query_selection",
                    "description": "Target BigQuery dataset name. Enter your local dataset or any cross-project dataset path (e.g. 'bigquery-public-data:samples' or 'bigquery-public-data.samples')."
                },
                "table_id": {
                    "type": "string",
                    "title": "Table Name",
                    "input_type": "text",
                    "default": "",
                    "placeholder": "e.g. events_daily or shakespeare",
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

    @staticmethod
    def _build_table_ref(project_id: str, dataset_id: str, table_id: str) -> str:
        """
        Constructs a standard BigQuery backticked table reference.
        Supports:
          - Standard: project.dataset.table
          - Cross-project dataset: bigquery-public-data:samples -> bigquery-public-data.samples.table
          - Dotted dataset: bigquery-public-data.samples -> bigquery-public-data.samples.table
          - Full table_id: bigquery-public-data.samples.table
        """
        clean_table = table_id.strip().strip("`")
        clean_dataset = dataset_id.strip().strip("`").replace(":", ".")
        clean_project = project_id.strip().strip("`")

        # If table_id is already fully qualified (3 parts: proj.ds.tbl)
        if clean_table.count(".") == 2:
            return f"`{clean_table}`"

        # If dataset_id has a project prefix (e.g. bigquery-public-data.samples)
        if "." in clean_dataset:
            return f"`{clean_dataset}.{clean_table}`"

        # Standard: project_id.dataset_id.table_id
        if clean_project and clean_dataset:
            return f"`{clean_project}.{clean_dataset}.{clean_table}`"
        elif clean_dataset:
            return f"`{clean_dataset}.{clean_table}`"
        return f"`{clean_table}`"

    @staticmethod
    def _clean_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Cleans records for JSON serialization safety (Decimals, dates, timestamps, bytes)."""
        import decimal
        import datetime

        cleaned_records = []
        for row in records:
            cleaned_row = {}
            for k, v in row.items():
                if isinstance(v, decimal.Decimal):
                    cleaned_row[k] = float(v) if (v % 1 != 0) else int(v)
                elif isinstance(v, (datetime.date, datetime.datetime, datetime.time)):
                    cleaned_row[k] = v.isoformat()
                elif isinstance(v, bytes):
                    try:
                        cleaned_row[k] = v.decode("utf-8")
                    except Exception:
                        cleaned_row[k] = str(v)
                elif pd is not None and pd.isna(v):
                    cleaned_row[k] = None
                else:
                    cleaned_row[k] = v
            cleaned_records.append(cleaned_row)
        return cleaned_records

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        start_time = time.time()
        limit = max(1, min(int(config.get("limit", 1000)), 1000000))
        read_mode = str(config.get("read_mode", "table")).strip().lower()
        simulation_mode = bool(config.get("simulation_mode", False))

        project_id = str(config.get("project_id", "my-gcp-project")).strip()
        dataset_id = str(config.get("dataset_id", "analytics_data")).strip()
        table_id = str(config.get("table_id", "events_daily")).strip()
        conn_mode = str(config.get("connection_mode", "service_account_json")).strip().lower()

        # Resolve credential from context if credential_id is provided
        credential_id = str(config.get("credential_id", "")).strip()
        if credential_id and context and isinstance(context, dict):
            creds = context.get("credentials", {})
            if credential_id in creds:
                c_data = creds[credential_id]
                if isinstance(c_data, dict):
                    c_data = c_data.get("data", c_data)
                    project_id = c_data.get("project_id") or project_id
                    raw_json = c_data.get("credentials_json") or c_data.get("service_account_json") or config.get("credentials_json", "")
                    fpath = c_data.get("credentials_file_path") or config.get("credentials_file_path", "")
                    config = {**config, "credentials_json": raw_json, "credentials_file_path": fpath, "project_id": project_id}

        # Environment variable overrides
        if not project_id or project_id == "my-gcp-project":
            project_id = os.getenv("GOOGLE_CLOUD_PROJECT", os.getenv("GCP_PROJECT", project_id))

        records: List[Dict[str, Any]] = []
        delivery_status = "SIMULATED"
        status_msg = ""
        last_error = None
        target_ref = ""

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
                    target_ref = self._build_table_ref(project_id, dataset_id, table_id)
                    query_sql = f"SELECT {cols} FROM {target_ref} LIMIT {limit}"
                else:
                    query_sql = str(config.get("query", "")).strip()
                    if not query_sql:
                        raise ValidationException("SQL Query is required when Read Mode is set to 'query'.")
                    safe, err_msg = self._validate_sql_safety(query_sql)
                    if not safe and err_msg:
                        raise ValidationException(err_msg)
                    clean_q = query_sql.rstrip(";").strip()
                    if limit and "limit" not in clean_q.lower():
                        query_sql = f"{clean_q} LIMIT {limit}"
                    else:
                        query_sql = clean_q

                job_config = bigquery.QueryJobConfig()
                max_bytes = config.get("maximum_bytes_billed")
                if max_bytes:
                    job_config.maximum_bytes_billed = int(max_bytes)

                query_job = client.query(query_sql, job_config=job_config)

                # Resilient DataFrame conversion with graceful native fallback if db-dtypes or pyarrow has issues
                try:
                    df_res = query_job.to_dataframe()
                    records = df_res.to_dict(orient="records")
                except Exception as conv_err:
                    logger.info(f"BigQuery to_dataframe conversion failed ({conv_err}); falling back to native row mapping.")
                    query_result = query_job.result()
                    records = [dict(row.items()) for row in query_result]

                records = self._clean_records(records)
                delivery_status = "LIVE"
                target_desc = target_ref if read_mode == "table" else "custom SQL"
                status_msg = f"Retrieved {len(records)} live rows from Google BigQuery ({target_desc})."
            except ImportError as ie:
                logger.warning(f"BigQuery driver error: {ie}")
                last_error = f"Missing Google Cloud dependencies: {ie}. Ensure 'google-cloud-bigquery' and 'db-dtypes' are installed."
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
            "title": "Data Warehouse Query Metrics",
            "message": status_msg,
            "status": delivery_status,
            "project_id": project_id,
            "dataset_id": dataset_id,
            "table_id": table_id,
            "rows_retrieved": int(len(records)),
            "columns_count": int(col_count),
            "read_mode": read_mode,
            "duration_ms": int(duration_ms)
        }

        metrics = {
            "rows_retrieved": int(len(records)),
            "columns_count": int(col_count),
            "duration_ms": int(duration_ms),
            "delivery_status": delivery_status
        }

        return {
            "dataframe": df_out,
            "output": df_out,
            "output_summary": summary,
            "metrics": metrics
        }

    def inspect(
        self,
        action: str,
        config: Dict[str, Any],
        context: Optional[Any] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Introspects Google BigQuery datasets, tables, and schemas in real-time."""
        action = str(action or "test_connection").strip().lower()
        simulation_mode = bool(config.get("simulation_mode", False))
        project_id = str(config.get("project_id", "")).strip()

        is_live = False
        client = None
        inspect_err = ""

        if not simulation_mode and project_id:
            try:
                from google.cloud import bigquery
                from google.oauth2 import service_account

                raw_json = str(config.get("credentials_json", "")).strip()
                fpath = str(config.get("credentials_file_path", "")).strip()
                conn_mode = str(config.get("connection_mode", "service_account_json")).strip().lower()

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

                client.get_project(project_id)
                is_live = True
            except Exception as ex:
                inspect_err = str(ex)
                logger.debug(f"BigQuery inspect connection failed: {ex}")

        # 1. Test Connection
        if action == "test_connection":
            if is_live:
                return {
                    "success": True,
                    "connected": True,
                    "status": "LIVE",
                    "project_id": project_id,
                    "message": f"Successfully connected to Google BigQuery project '{project_id}'."
                }
            elif simulation_mode:
                return {
                    "success": True,
                    "connected": True,
                    "status": "SIMULATED",
                    "project_id": project_id or "demo-bigquery-project",
                    "message": "Google BigQuery simulation mode active."
                }
            else:
                return {
                    "success": False,
                    "connected": False,
                    "status": "UNREACHABLE",
                    "error": inspect_err,
                    "message": f"Could not connect to Google BigQuery: {inspect_err or 'Project unreachable or credentials invalid.'}"
                }

        # 2. Datasets / Schemas / Databases
        if action in ("datasets", "get_datasets", "schemas", "get_schemas", "databases", "get_databases"):
            if not is_live and not simulation_mode:
                return {"success": False, "datasets": [], "schemas": [], "error": inspect_err or "Connection could not be established."}

            if simulation_mode:
                sim_datasets = ["analytics_data", "marketing_dw", "events_prod"]
                return {
                    "success": True,
                    "datasets": sim_datasets,
                    "schemas": sim_datasets,
                    "databases": sim_datasets,
                    "count": len(sim_datasets),
                    "source": "SIMULATED",
                    "message": "Simulation mode active."
                }

            # LIVE: ONLY return datasets that actually exist in the GCP project
            datasets = []
            if is_live and client:
                try:
                    for ds in client.list_datasets(project=project_id):
                        datasets.append(ds.dataset_id)
                except Exception as e:
                    logger.warning(f"BigQuery list_datasets failed for {project_id}: {e}")
                    return {"success": False, "datasets": [], "error": str(e), "message": f"Failed to list datasets in project '{project_id}': {e}"}

            return {
                "success": True,
                "datasets": datasets,
                "schemas": datasets,
                "databases": datasets,
                "count": int(len(datasets)),
                "source": "LIVE",
                "message": f"Found {len(datasets)} dataset(s) in project '{project_id}'." if datasets else f"No datasets found in project '{project_id}'."
            }

        # 3. Tables
        if action in ("tables", "get_tables"):
            if not is_live and not simulation_mode:
                return {"success": False, "tables": [], "error": inspect_err or "Connection could not be established."}

            target_dataset = schema_name or config.get("dataset_id") or ""
            if not target_dataset:
                return {
                    "success": False,
                    "tables": [],
                    "count": 0,
                    "error": "Dataset ID is required to list tables.",
                    "message": "Please specify a Dataset ID."
                }

            if simulation_mode:
                sim_tables = ["events_daily", "users_dimension", "transactions_fact", "pageviews"]
                return {
                    "success": True,
                    "tables": sim_tables,
                    "count": len(sim_tables),
                    "dataset_id": target_dataset,
                    "schema": target_dataset,
                    "source": "SIMULATED"
                }

            # LIVE: ONLY return real tables from BigQuery
            clean_dataset = target_dataset.replace(":", ".")
            tables = []
            if is_live and client:
                try:
                    from google.cloud import bigquery
                    if "." in clean_dataset:
                        parts = clean_dataset.split(".", 1)
                        ds_proj, ds_name = parts[0], parts[1]
                        ds_ref = bigquery.DatasetReference(ds_proj, ds_name)
                    else:
                        ds_ref = bigquery.DatasetReference(project_id, clean_dataset)
                    for tbl in client.list_tables(ds_ref):
                        tables.append(tbl.table_id)
                except Exception as e:
                    logger.warning(f"BigQuery list_tables failed for {clean_dataset}: {e}")
                    return {"success": False, "tables": [], "error": str(e), "message": f"Failed to list tables in dataset '{target_dataset}': {e}"}

            return {
                "success": True,
                "tables": tables,
                "count": int(len(tables)),
                "dataset_id": target_dataset,
                "schema": target_dataset,
                "source": "LIVE",
                "message": f"Found {len(tables)} table(s) in dataset '{target_dataset}'." if tables else f"No tables found in dataset '{target_dataset}'."
            }

        # 4. Columns
        if action in ("columns", "get_columns"):
            if not is_live and not simulation_mode:
                return {"success": False, "columns": [], "error": inspect_err or "Connection could not be established."}

            target_dataset = schema_name or config.get("dataset_id") or ""
            target_table = table_name or config.get("table_id") or ""
            if not target_table or not target_dataset:
                return {
                    "success": False,
                    "columns": [],
                    "count": 0,
                    "error": "Both Dataset ID and Table Name are required to inspect columns.",
                    "message": "Please specify both Dataset ID and Table Name."
                }

            if simulation_mode:
                sim_columns = [
                    {"name": "event_date", "type": "STRING", "nullable": False, "mode": "REQUIRED"},
                    {"name": "event_timestamp", "type": "INTEGER", "nullable": False, "mode": "REQUIRED"},
                    {"name": "event_name", "type": "STRING", "nullable": False, "mode": "REQUIRED"},
                    {"name": "user_pseudo_id", "type": "STRING", "nullable": True, "mode": "NULLABLE"},
                    {"name": "country", "type": "STRING", "nullable": True, "mode": "NULLABLE"},
                    {"name": "purchase_revenue_usd", "type": "FLOAT", "nullable": True, "mode": "NULLABLE"},
                ]
                return {
                    "success": True,
                    "columns": sim_columns,
                    "count": len(sim_columns),
                    "table_name": target_table,
                    "dataset_id": target_dataset,
                    "source": "SIMULATED"
                }

            # LIVE: ONLY return real columns from BigQuery
            columns = []
            if is_live and client:
                try:
                    from google.cloud import bigquery
                    clean_ds = target_dataset.replace(":", ".")
                    if "." in clean_ds:
                        parts = clean_ds.split(".", 1)
                        tbl_ref = bigquery.TableReference(bigquery.DatasetReference(parts[0], parts[1]), target_table)
                    else:
                        tbl_ref = bigquery.TableReference(bigquery.DatasetReference(project_id, clean_ds), target_table)
                    tbl_obj = client.get_table(tbl_ref)
                    for field in tbl_obj.schema:
                        columns.append({
                            "name": field.name,
                            "type": field.field_type,
                            "nullable": field.is_nullable,
                            "mode": field.mode
                        })
                except Exception as e:
                    logger.warning(f"BigQuery get_table schema failed for {target_table}: {e}")
                    return {"success": False, "columns": [], "error": str(e), "message": f"Failed to inspect columns for table '{target_table}': {e}"}

            return {
                "success": True,
                "columns": columns,
                "count": int(len(columns)),
                "table_name": target_table,
                "dataset_id": target_dataset,
                "source": "LIVE",
                "message": f"Found {len(columns)} column(s) in table '{target_table}'." if columns else f"No columns found in table '{target_table}'."
            }

        return {
            "success": False,
            "error": f"Unknown inspection action '{action}'. Supported: 'test_connection', 'datasets', 'schemas', 'databases', 'tables', 'columns'.",
            "message": f"Unknown inspection action '{action}'."
        }

    def to_code(self, config: Dict[str, Any]) -> str:
        project_id = config.get("project_id", "my-gcp-project")
        dataset_id = config.get("dataset_id", "analytics_data")
        if config.get("is_public_dataset") and config.get("public_dataset"):
            dataset_id = config.get("public_dataset")

        table_id = config.get("table_id", "events_daily")
        read_mode = config.get("read_mode", "table")
        limit = config.get("limit", 1000)

        if read_mode == "table":
            cols = config.get("columns", "*") or "*"
            target_table = self._build_table_ref(project_id, dataset_id, table_id)
            sql = f"SELECT {cols} FROM {target_table} LIMIT {limit}"
        else:
            sql = config.get("query", f"SELECT * FROM `{project_id}.{dataset_id}.{table_id}` LIMIT 100")

        return (
            f"# Generated Google BigQuery Extraction Script\n"
            f"from google.cloud import bigquery\n"
            f"import pandas as pd\n\n"
            f"client = bigquery.Client(project='{project_id}')\n"
            f"sql = \"\"\"{sql}\"\"\"\n"
            f"df = client.query(sql).to_dataframe()\n"
            f"print(f'Retrieved {{len(df)}} rows from BigQuery.')\n"
        )
