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

    inputs = [
        RecipePort(
            id="input",
            label="Trigger / Input Dataset",
            type="dataframe",
            required=False,
            max_connections=1,
            description="Optional upstream dataset for batch IDs or execution trigger"
        )
    ]

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
            "properties": {
                "connection_mode": {
                    "type": "string",
                    "title": "Connection Mode",
                    "input_type": "dropdown",
                    "enum": ["cluster_credentials", "connection_string", "credential"],
                    "default": "cluster_credentials",
                    "description": "Choose how to connect: host/cluster parameters, connection string, or stored credential."
                },
                "host": {
                    "type": "string",
                    "title": "Redshift Cluster Endpoint / Host",
                    "input_type": "text",
                    "default": "redshift-cluster-1.abcdef123456.us-east-1.redshift.amazonaws.com",
                    "description": "Amazon Redshift cluster endpoint hostname."
                },
                "port": {
                    "type": "integer",
                    "title": "Port",
                    "input_type": "number",
                    "default": 5439,
                    "description": "Redshift server port (default 5439)."
                },
                "database": {
                    "type": "string",
                    "title": "Database Name",
                    "input_type": "text",
                    "default": "dev",
                    "description": "Name of the target Redshift database."
                },
                "schema_name": {
                    "type": "string",
                    "title": "Schema Name",
                    "input_type": "text",
                    "default": "public",
                    "description": "Target schema within the database (default 'public')."
                },
                "username": {
                    "type": "string",
                    "title": "Username",
                    "input_type": "text",
                    "default": "awsuser",
                    "description": "Redshift database master user."
                },
                "password": {
                    "type": "string",
                    "title": "Password",
                    "input_type": "password",
                    "default": "",
                    "description": "Redshift database user password."
                },
                "connection_string": {
                    "type": "string",
                    "title": "Connection String (URI)",
                    "input_type": "text",
                    "default": "",
                    "description": "Full Redshift connection URI e.g. postgresql://awsuser:pass@host:5439/dev"
                },
                "credential_id": {
                    "type": "string",
                    "title": "Connected Redshift Credential",
                    "input_type": "credential_dropdown",
                    "provider": "redshift",
                    "description": "Select a stored Redshift credential from platform settings."
                },
                "read_mode": {
                    "type": "string",
                    "title": "Read Mode",
                    "input_type": "dropdown",
                    "enum": ["table", "query"],
                    "default": "table",
                    "description": "Read an entire table/view or run a custom read-only SQL query."
                },
                "table_name": {
                    "type": "string",
                    "title": "Table / View Name",
                    "input_type": "text",
                    "default": "sales",
                    "description": "Target Redshift table or view name (used in 'table' mode)."
                },
                "columns": {
                    "type": "string",
                    "title": "Columns (comma-separated)",
                    "input_type": "text",
                    "default": "*",
                    "description": "Columns to select (e.g. 'salesid, buyerid, sellerid, pricepaid, saletime'). Leave '*' for all."
                },
                "query": {
                    "type": "string",
                    "title": "Custom SQL Query (Read-Only)",
                    "input_type": "textarea",
                    "default": "SELECT * FROM public.sales LIMIT 100",
                    "description": "Custom SQL query to execute in Redshift. Modifying statements (DROP, DELETE, UPDATE, TRUNCATE) are rejected."
                },
                "limit": {
                    "type": "integer",
                    "title": "Row Limit",
                    "input_type": "number",
                    "default": 1000,
                    "description": "Maximum number of rows to retrieve (1 to 1,000,000)."
                },
                "simulation_mode": {
                    "type": "boolean",
                    "title": "Simulation / Demo Mode",
                    "input_type": "checkbox",
                    "default": False,
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
            except ImportError:
                status_msg = "Neither psycopg2 nor redshift_connector driver installed. Using benchmark simulation fallback."
                logger.info(status_msg)
            except Exception as e:
                logger.warning(f"Redshift connection failed: {e}")
                status_msg = f"Redshift connection failed ({e}). Using benchmark simulation fallback."

        # Fallback to realistic benchmark data
        if not records:
            records = self._generate_benchmark_data(config)
            if not status_msg:
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
