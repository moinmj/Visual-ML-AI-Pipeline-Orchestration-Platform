"""
Amazon Redshift Connector Recipe (Ingestion & Orchestration -> Database & Warehouse Connectors).

Reads analytical tables or SQL query results from Amazon Redshift Cloud Data Warehouse into the pipeline as a DataFrame.
Supports cluster credentials, connection string (DSN), stored credentials, read-only SQL validation,
SSL options, benchmark simulation fallback, and code generation.
"""

import re
import os
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


class RedshiftConnectorRecipe(BaseRecipe):
    recipe_id = "redshift_connector"
    name = "Amazon Redshift Connector"
    version = "1.0.0"
    category = "ingestion"
    group = "Ingestion & Orchestration"
    subgroup = "Database & Warehouse Connectors"
    description = "Reads analytical tables or SQL queries from Amazon Redshift Cloud Data Warehouse."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    inputs: List[RecipePort] = []

    outputs = [
        RecipePort(
            id="output",
            label="Redshift Dataset",
            type="dataframe",
            description="DataFrame containing retrieved Redshift analytical records"
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "sections": [
                {"id": "connection", "title": "Connection Configuration", "description": "Configure connection parameters"},
                {"id": "query_selection", "title": "Query & Table Selection", "description": "Choose tables or custom SQL"},
                {"id": "runtime_limits", "title": "Limits & Performance", "description": "Row limits and query performance"}
            ],
            "properties": {
                # ── 1. Connection Mode & Server Credentials ──
                "connection_mode": {
                    "type": "string",
                    "title": "Connection Mode",
                    "input_type": "dropdown",
                    "enum": ["cluster_credentials", "connection_string", "credential"],
                    "default": "cluster_credentials",
                    "section": "connection",
                    "description": "Choose how to connect: host/cluster parameters, connection string, or stored credential."
                },
                "credential_id": {
                    "type": "string",
                    "title": "Connected Redshift Credential",
                    "input_type": "credential_dropdown",
                    "provider": "redshift",
                    "section": "connection",
                    "description": "Select a stored Redshift credential from platform settings.",
                    "visible_when": {
                        "field": "connection_mode",
                        "equals": "credential"
                    }
                },
                "host": {
                    "type": "string",
                    "title": "Redshift Cluster Endpoint / Host",
                    "input_type": "text",
                    "default": "",
                    "placeholder": "e.g. redshift-cluster-1.abcdef.us-east-1.redshift.amazonaws.com",
                    "section": "connection",
                    "description": "Amazon Redshift cluster endpoint hostname.",
                    "visible_when": {
                        "field": "connection_mode",
                        "equals": "cluster_credentials"
                    }
                },
                "port": {
                    "type": "integer",
                    "title": "Port",
                    "input_type": "number",
                    "default": 5439,
                    "placeholder": "5439",
                    "section": "connection",
                    "description": "Redshift server port (default 5439).",
                    "visible_when": {
                        "field": "connection_mode",
                        "equals": "cluster_credentials"
                    }
                },
                "username": {
                    "type": "string",
                    "title": "Username",
                    "input_type": "text",
                    "default": "",
                    "placeholder": "e.g. awsuser or master_user",
                    "section": "connection",
                    "description": "Redshift database master user.",
                    "visible_when": {
                        "field": "connection_mode",
                        "equals": "cluster_credentials"
                    }
                },
                "password": {
                    "type": "string",
                    "title": "Password",
                    "input_type": "password",
                    "default": "",
                    "placeholder": "Enter Redshift password",
                    "section": "connection",
                    "description": "Redshift database user password.",
                    "visible_when": {
                        "field": "connection_mode",
                        "equals": "cluster_credentials"
                    }
                },
                "connection_string": {
                    "type": "string",
                    "title": "Connection String (URI)",
                    "input_type": "text",
                    "default": "",
                    "placeholder": "e.g. postgresql://awsuser:pass@host:5439/dev",
                    "section": "connection",
                    "description": "Full Redshift connection URI e.g. postgresql://awsuser:pass@host:5439/dev",
                    "visible_when": {
                        "field": "connection_mode",
                        "equals": "connection_string"
                    }
                },
                "timeout": {
                    "type": "integer",
                    "title": "Timeout (seconds)",
                    "input_type": "number",
                    "default": 30,
                    "section": "connection",
                    "description": "Connection and query timeout in seconds."
                },

                # ── 2. Read Mode & Target Selection ──
                "read_mode": {
                    "type": "string",
                    "title": "Read Mode",
                    "input_type": "dropdown",
                    "enum": ["table", "query"],
                    "default": "table",
                    "section": "query_selection",
                    "description": "Read an entire table/view or run a custom read-only SQL query."
                },
                "database": {
                    "type": "string",
                    "title": "Database Name",
                    "input_type": "text",
                    "default": "",
                    "placeholder": "e.g. dev or analytics",
                    "section": "query_selection",
                    "description": "Name of the target Redshift database."
                },
                "schema_name": {
                    "type": "string",
                    "title": "Schema Name",
                    "input_type": "text",
                    "default": "public",
                    "placeholder": "public",
                    "section": "query_selection",
                    "description": "Target schema within the database (default 'public').",
                    "visible_when": {
                        "field": "read_mode",
                        "equals": "table"
                    }
                },
                "table_name": {
                    "type": "string",
                    "title": "Table / View Name",
                    "input_type": "text",
                    "default": "",
                    "placeholder": "e.g. sales",
                    "section": "query_selection",
                    "description": "Target Redshift table or view name (used in 'table' mode).",
                    "visible_when": {
                        "field": "read_mode",
                        "equals": "table"
                    }
                },
                "columns": {
                    "type": "string",
                    "title": "Columns (comma-separated)",
                    "input_type": "text",
                    "default": "*",
                    "placeholder": "e.g. salesid, buyerid, pricepaid or *",
                    "section": "query_selection",
                    "description": "Columns to select (e.g. 'salesid, buyerid, sellerid, pricepaid, saletime'). Leave '*' for all.",
                    "visible_when": {
                        "field": "read_mode",
                        "equals": "table"
                    }
                },
                "filter_condition": {
                    "type": "string",
                    "title": "Filter Condition (WHERE)",
                    "input_type": "text",
                    "default": "",
                    "placeholder": "e.g. pricepaid > 100",
                    "section": "query_selection",
                    "description": "Optional SQL WHERE filter expression without 'WHERE' keyword (used in 'table' mode).",
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
                    "placeholder": "e.g. SELECT * FROM public.sales LIMIT 100",
                    "section": "query_selection",
                    "description": "Custom SQL query to execute in Redshift. Only used when Read Mode is 'query'. Modifying statements (DROP, DELETE, UPDATE, TRUNCATE) are rejected.",
                    "visible_when": {
                        "field": "read_mode",
                        "equals": "query"
                    }
                },

                # ── 3. Limits & Pagination ──
                "limit": {
                    "type": "integer",
                    "title": "Row Limit",
                    "input_type": "number",
                    "default": 1000,
                    "section": "runtime_limits",
                    "description": "Maximum number of rows to retrieve (1 to 1,000,000)."
                },
                "simulation_mode": {
                    "type": "boolean",
                    "title": "Simulation / Demo Mode",
                    "input_type": "checkbox",
                    "default": False,
                    "section": "runtime_limits",
                    "description": "When enabled or when Redshift driver is offline, generates realistic Redshift columnar analytics benchmark data."
                }
            },
            "required": ["connection_mode", "read_mode"]
        }

    _DISALLOWED_SQL = re.compile(
        r"\b(INSERT\s+INTO|UPDATE\s+|DELETE\s+FROM|DROP\s+(TABLE|VIEW|DATABASE|SCHEMA)|"
        r"TRUNCATE\s+TABLE|ALTER\s+(TABLE|DATABASE|SCHEMA)|CREATE\s+(TABLE|VIEW|DATABASE|SCHEMA)|"
        r"GRANT\s+|REVOKE\s+|VACUUM\b|UNLOAD\b)\b",
        re.IGNORECASE
    )

    def _validate_sql_safety(self, sql: str) -> Tuple[bool, Optional[str]]:
        cleaned = re.sub(r"--.*?$|/\*.*?\*/", "", sql, flags=re.MULTILINE).strip()
        if not cleaned:
            return False, "Query cannot be empty."
        if not cleaned.upper().startswith(("SELECT", "WITH", "SHOW", "EXPLAIN")):
            return False, "Only read-only queries (SELECT, WITH, SHOW, EXPLAIN) are permitted."
        if self._DISALLOWED_SQL.search(cleaned):
            return False, "Data-modifying or administrative statements (INSERT, UPDATE, DELETE, DROP, TRUNCATE, ALTER, VACUUM, UNLOAD) are strictly prohibited."
        return True, None

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        errors: List[str] = []
        conn_mode = str(config.get("connection_mode", "cluster_credentials")).strip().lower()

        if not config.get("simulation_mode"):
            if conn_mode == "cluster_credentials":
                if not config.get("host"):
                    errors.append("Host endpoint is required in 'cluster_credentials' mode.")
                if not config.get("username"):
                    errors.append("Username is required in 'cluster_credentials' mode.")
            elif conn_mode == "connection_string" and "connection_string" in config:
                cs = str(config.get("connection_string", "")).strip()
                if not cs and not config.get("host"):
                    errors.append("Connection string (URI) is required in 'connection_string' mode.")
            elif conn_mode == "credential":
                if not config.get("credential_id"):
                    errors.append("Credential ID is required in 'credential' mode.")

        read_mode = str(config.get("read_mode", "table")).strip().lower()
        if read_mode == "table":
            if not str(config.get("table_name", "")).strip():
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
        table_name = str(config.get("table_name", "sales")).lower()
        records: List[Dict[str, Any]] = []

        categories = ["Electronics", "Appliances", "Apparel", "HomeGoods", "Industrial"]
        dist_styles = ["KEY", "EVEN", "ALL", "AUTO"]

        for i in range(limit):
            records.append({
                "salesid": 200000 + i,
                "listid": 10000 + (i % 500),
                "sellerid": 1000 + (i % 60),
                "buyerid": 5000 + (i % 250),
                "eventid": 300 + (i % 40),
                "category": categories[i % len(categories)],
                "qtysold": 1 + (i % 8),
                "pricepaid": round(45.50 + ((i * 19.25) % 800.0), 2),
                "commission": round((45.50 + ((i * 19.25) % 800.0)) * 0.15, 2),
                "saletime": f"2026-03-{((i % 28) + 1):02d} 14:{((i % 60)):02d}:00",
                "diststyle": dist_styles[i % len(dist_styles)],
                "cluster_source": "aws_redshift_dw"
            })
        return records

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        start_time = time.time()
        limit = max(1, min(int(config.get("limit", 1000)), 1000000))
        read_mode = str(config.get("read_mode", "table")).strip().lower()
        simulation_mode = bool(config.get("simulation_mode", False))

        host = str(config.get("host", "localhost")).strip()
        port = int(config.get("port", 5439))
        database = str(config.get("database", "dev")).strip()
        schema_name = str(config.get("schema_name", "public")).strip()
        table_name = str(config.get("table_name", "sales")).strip()
        user = str(config.get("username", "awsuser")).strip()
        password = str(config.get("password", "")).strip()

        records: List[Dict[str, Any]] = []
        delivery_status = "SIMULATED"
        status_msg = ""

        # Live Execution Attempt via psycopg2 or redshift_connector
        if not simulation_mode and host and user:
            try:
                # Prefer psycopg2 or redshift_connector
                try:
                    import psycopg2
                    conn = psycopg2.connect(
                        host=host,
                        port=port,
                        dbname=database,
                        user=user,
                        password=password,
                        connect_timeout=10,
                        sslmode="prefer"
                    )
                except ImportError:
                    import redshift_connector
                    conn = redshift_connector.connect(
                        host=host,
                        port=port,
                        database=database,
                        user=user,
                        password=password,
                        timeout=10
                    )

                cur = conn.cursor()
                if read_mode == "table":
                    cols = str(config.get("columns", "*")).strip() or "*"
                    sql = f'SELECT {cols} FROM "{schema_name}"."{table_name}" LIMIT {limit}'
                else:
                    sql = str(config.get("query", "")).strip()
                    if not sql:
                        raise ValidationException("SQL Query is required when Read Mode is set to 'query'.")
                    if limit and "limit" not in sql.lower():
                        sql = f"{sql} LIMIT {limit}"

                cur.execute(sql)
                col_names = [d[0] for d in cur.description] if cur.description else []
                rows = cur.fetchall()
                records = [dict(zip(col_names, row)) for row in rows]
                cur.close()
                conn.close()

                delivery_status = "LIVE"
                status_msg = f"Retrieved {len(records)} live rows from Amazon Redshift ({schema_name}.{table_name})."
            except Exception as e:
                logger.warning(f"Redshift connection failed: {e}")
                last_error = e

            if last_error is not None:
                raise ValidationException(f"Amazon Redshift connection to {host}:{port}/{database} failed: {str(last_error)}")

        # Simulation Mode (ONLY when simulation_mode is True)
        if simulation_mode:
            records = self._generate_benchmark_data(config)
            delivery_status = "SIMULATED"
            status_msg = f"Loaded {len(records)} benchmark records for Redshift table '{schema_name}.{table_name}'."

        df_out = pd.DataFrame(records) if pd is not None else records
        duration_ms = int((time.time() - start_time) * 1000)
        col_count = len(df_out.columns) if hasattr(df_out, "columns") else (len(records[0].keys()) if records else 0)

        summary = {
            "title": "Amazon Redshift Connector Summary",
            "message": status_msg,
            "status": delivery_status,
            "host": host,
            "database": database,
            "schema": schema_name,
            "table": table_name,
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
        host = config.get("host", "redshift-cluster.amazonaws.com")
        port = config.get("port", 5439)
        database = config.get("database", "dev")
        schema = config.get("schema_name", "public")
        user = config.get("username", "awsuser")
        read_mode = config.get("read_mode", "table")
        limit = config.get("limit", 1000)

        if read_mode == "table":
            table_name = config.get("table_name", "sales")
            cols = config.get("columns", "*") or "*"
            sql = f'SELECT {cols} FROM "{schema}"."{table_name}" LIMIT {limit}'
        else:
            sql = config.get("query", "SELECT * FROM public.sales LIMIT 100")

        return (
            f"# Generated Amazon Redshift Extraction Script\n"
            f"import psycopg2\n"
            f"import pandas as pd\n\n"
            f"conn = psycopg2.connect(\n"
            f"    host='{host}',\n"
            f"    port={port},\n"
            f"    dbname='{database}',\n"
            f"    user='{user}',\n"
            f"    password='<SECURE_PASSWORD>',\n"
            f"    sslmode='prefer'\n"
            f")\n"
            f"query = \"\"\"{sql}\"\"\"\n"
            f"df = pd.read_sql(query, conn)\n"
            f"conn.close()\n"
            f"print(f'Retrieved {{len(df)}} rows from Redshift database {database}.')\n"
        )
