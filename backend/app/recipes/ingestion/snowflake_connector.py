"""
Snowflake Connector Recipe (Ingestion & Orchestration -> Database & Warehouse Connectors).

Reads tables or analytical SQL queries from Snowflake Cloud Data Warehouse into the pipeline as a DataFrame.
Supports account/user/password/warehouse/database/schema credentials, connection strings,
stored credentials, read-only SQL validation, benchmark simulation fallback, and code generation.
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


class SnowflakeConnectorRecipe(BaseRecipe):
    recipe_id = "snowflake_connector"
    name = "Snowflake Connector"
    version = "1.0.0"
    category = "ingestion"
    group = "Ingestion & Orchestration"
    subgroup = "Database & Warehouse Connectors"
    description = "Reads tables or analytical query results from Snowflake Cloud Data Warehouse."
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
            label="Snowflake Dataset",
            type="dataframe",
            description="DataFrame containing retrieved Snowflake analytical records"
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
                    "enum": ["account_credentials", "connection_string", "credential"],
                    "default": "account_credentials",
                    "description": "Choose how to connect: Snowflake account credentials, connection string (SQLAlchemy URI), or stored credential."
                },
                "account": {
                    "type": "string",
                    "title": "Snowflake Account Identifier",
                    "input_type": "text",
                    "default": "xy12345.us-east-1",
                    "description": "Snowflake account identifier (e.g. 'xy12345.us-east-1' or '<orgname>-<accountname>')."
                },
                "user": {
                    "type": "string",
                    "title": "Username",
                    "input_type": "text",
                    "default": "analyst",
                    "description": "Snowflake user login name."
                },
                "password": {
                    "type": "string",
                    "title": "Password",
                    "input_type": "password",
                    "default": "",
                    "description": "Snowflake user password."
                },
                "warehouse": {
                    "type": "string",
                    "title": "Warehouse",
                    "input_type": "text",
                    "default": "COMPUTE_WH",
                    "description": "Virtual warehouse used to execute queries (e.g. 'COMPUTE_WH', 'ANALYTICS_WH')."
                },
                "database": {
                    "type": "string",
                    "title": "Database Name",
                    "input_type": "text",
                    "default": "ANALYTICS_DB",
                    "description": "Name of the target Snowflake database."
                },
                "schema_name": {
                    "type": "string",
                    "title": "Schema Name",
                    "input_type": "text",
                    "default": "PUBLIC",
                    "description": "Target schema within the database (default 'PUBLIC')."
                },
                "role": {
                    "type": "string",
                    "title": "Role",
                    "input_type": "text",
                    "default": "PUBLIC",
                    "description": "Security role to assume (e.g. 'ACCOUNTADMIN', 'SYSADMIN', 'DATA_ANALYST')."
                },
                "connection_string": {
                    "type": "string",
                    "title": "Connection String (URI)",
                    "input_type": "text",
                    "default": "",
                    "description": "Full Snowflake SQLAlchemy URI e.g. snowflake://user:pass@account/db/schema?warehouse=wh&role=role"
                },
                "credential_id": {
                    "type": "string",
                    "title": "Connected Snowflake Credential",
                    "input_type": "credential_dropdown",
                    "provider": "snowflake",
                    "description": "Select a stored credential from platform settings."
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
                    "default": "FCT_ORDERS",
                    "description": "Target Snowflake table or view name (used in 'table' mode)."
                },
                "columns": {
                    "type": "string",
                    "title": "Columns (comma-separated)",
                    "input_type": "text",
                    "default": "*",
                    "description": "Columns to select (e.g. 'ORDER_ID, CUSTOMER_ID, TOTAL_AMOUNT, CREATED_AT'). Leave '*' for all."
                },
                "query": {
                    "type": "string",
                    "title": "Custom SQL Query (Read-Only)",
                    "input_type": "textarea",
                    "default": "SELECT * FROM PUBLIC.FCT_ORDERS LIMIT 100",
                    "description": "Custom SQL query to execute in Snowflake. Modifying statements (DROP, DELETE, UPDATE, TRUNCATE) are rejected."
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
                    "description": "When enabled or when Snowflake driver is offline, generates realistic Snowflake warehouse benchmark data."
                }
            },
            "required": ["connection_mode", "read_mode"]
        }

    # Strict query guardrails
    _DISALLOWED_SQL = re.compile(
        r"\b(INSERT\s+INTO|UPDATE\s+|DELETE\s+FROM|DROP\s+(TABLE|VIEW|DATABASE|SCHEMA)|"
        r"TRUNCATE\s+TABLE|ALTER\s+(TABLE|DATABASE|SCHEMA)|CREATE\s+(TABLE|VIEW|DATABASE|SCHEMA)|"
        r"GRANT\s+|REVOKE\s+|MERGE\s+INTO|COPY\s+INTO)\b",
        re.IGNORECASE
    )

    def _validate_sql_safety(self, sql: str) -> Tuple[bool, Optional[str]]:
        cleaned = re.sub(r"--.*?$|/\*.*?\*/", "", sql, flags=re.MULTILINE).strip()
        if not cleaned:
            return False, "Query cannot be empty."
        if not cleaned.upper().startswith(("SELECT", "WITH", "SHOW", "DESCRIBE", "EXPLAIN")):
            return False, "Only read-only queries (SELECT, WITH, SHOW, DESCRIBE, EXPLAIN) are permitted."
        if self._DISALLOWED_SQL.search(cleaned):
            return False, "Data-modifying or DDL statements (INSERT, UPDATE, DELETE, DROP, TRUNCATE, ALTER, MERGE, COPY) are strictly prohibited."
        return True, None

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        errors: List[str] = []
        conn_mode = str(config.get("connection_mode", "account_credentials")).strip().lower()

        if not config.get("simulation_mode"):
            if conn_mode == "account_credentials":
                if not config.get("account"):
                    errors.append("Snowflake Account identifier is required in 'account_credentials' mode.")
                if not config.get("user"):
                    errors.append("Username is required in 'account_credentials' mode.")
            elif conn_mode == "connection_string" and "connection_string" in config:
                cs = str(config.get("connection_string", "")).strip()
                if not cs and not config.get("account"):
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

    def _resolve_connection_params(self, config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        conn_mode = str(config.get("connection_mode", "account_credentials")).strip().lower()
        account = str(config.get("account", "xy12345.us-east-1")).strip()
        user = str(config.get("user", "analyst")).strip()
        password = str(config.get("password", "")).strip()
        warehouse = str(config.get("warehouse", "COMPUTE_WH")).strip()
        database = str(config.get("database", "ANALYTICS_DB")).strip()
        schema_name = str(config.get("schema_name", "PUBLIC")).strip()
        role = str(config.get("role", "PUBLIC")).strip()
        cs = str(config.get("connection_string", "")).strip()

        if conn_mode == "credential" and config.get("credential_id") and context:
            cred_id = config.get("credential_id")
            if hasattr(context, "get_credential"):
                cred = context.get_credential(cred_id)
                if cred and isinstance(cred, dict):
                    account = cred.get("account", account)
                    user = cred.get("user", cred.get("username", user))
                    password = cred.get("password", password)
                    warehouse = cred.get("warehouse", warehouse)
                    database = cred.get("database", database)
                    schema_name = cred.get("schema", schema_name)
                    role = cred.get("role", role)
                    cs = cred.get("connection_string", cs)

        return {
            "account": account,
            "user": user,
            "password": password,
            "warehouse": warehouse,
            "database": database,
            "schema": schema_name,
            "role": role,
            "connection_string": cs,
            "mode": conn_mode
        }

    def _generate_benchmark_data(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        limit = max(1, min(int(config.get("limit", 100)), 5000))
        table_name = str(config.get("table_name", "FCT_ORDERS")).upper()
        records: List[Dict[str, Any]] = []

        warehouses = ["COMPUTE_WH", "TRANSFORM_WH", "REPORTING_WH", "ML_TRAINING_WH"]
        tiers = ["ENTERPRISE", "BUSINESS_CRITICAL", "STANDARD"]
        regions = ["AWS_US_EAST_1", "AWS_EU_WEST_1", "AZURE_EAST_US2", "GCP_US_CENTRAL1"]

        for i in range(limit):
            records.append({
                "QUERY_ID": f"01a{i:04d}b-0001-2345-0000-{100000000000 + i}",
                "TRANSACTION_ID": 100000 + i,
                "CUSTOMER_ID": f"CUST_{5000 + (i % 80):05d}",
                "WAREHOUSE_NAME": warehouses[i % len(warehouses)],
                "WAREHOUSE_SIZE": "X-Small" if i % 4 == 0 else "Medium",
                "CREDITS_USED": round(0.015 + ((i * 0.04) % 2.5), 4),
                "EXECUTION_TIME_MS": 120 + ((i * 17) % 850),
                "ROWS_PRODUCED": 50 + ((i * 35) % 10000),
                "ACCOUNT_TIER": tiers[i % len(tiers)],
                "CLOUD_REGION": regions[i % len(regions)],
                "TOTAL_AMOUNT_USD": round(25.0 + ((i * 13.75) % 950.0), 2),
                "STATUS": "COMPLETED" if i % 10 != 0 else "QUEUED",
                "RECORD_TIMESTAMP": f"2026-03-{((i % 28) + 1):02d}T10:15:{((i % 60)):02d}Z"
            })
        return records

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        start_time = time.time()
        limit = max(1, min(int(config.get("limit", 1000)), 1000000))
        read_mode = str(config.get("read_mode", "table")).strip().lower()
        simulation_mode = bool(config.get("simulation_mode", False))

        params = self._resolve_connection_params(config, context)
        account = params["account"]
        database = params["database"]
        schema = params["schema"]
        warehouse = params["warehouse"]
        table_name = str(config.get("table_name", "FCT_ORDERS")).strip()

        records: List[Dict[str, Any]] = []
        delivery_status = "SIMULATED"
        status_msg = ""

        # Live Execution Attempt
        if not simulation_mode and account and params["user"]:
            try:
                import snowflake.connector
                conn_kwargs = {
                    "account": account,
                    "user": params["user"],
                    "password": params["password"],
                    "warehouse": warehouse,
                    "database": database,
                    "schema": schema,
                    "role": params["role"]
                }
                conn = snowflake.connector.connect(**conn_kwargs)
                cur = conn.cursor()

                if read_mode == "table":
                    cols = str(config.get("columns", "*")).strip() or "*"
                    sql = f'SELECT {cols} FROM "{database}"."{schema}"."{table_name}" LIMIT {limit}'
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
                status_msg = f"Retrieved {len(records)} live rows from Snowflake {database}.{schema}.{table_name}."
            except ImportError:
                status_msg = "snowflake-connector-python driver not installed. Using benchmark simulation fallback."
                logger.info(status_msg)
            except Exception as e:
                logger.warning(f"Snowflake connection failed: {e}")
                status_msg = f"Snowflake connection failed ({e}). Using benchmark simulation fallback."

        # Fallback to realistic benchmark data
        if not records:
            records = self._generate_benchmark_data(config)
            if not status_msg:
                status_msg = f"Loaded {len(records)} benchmark records for Snowflake warehouse '{warehouse}' ({database}.{schema})."

        df_out = pd.DataFrame(records) if pd is not None else records
        duration_ms = int((time.time() - start_time) * 1000)
        col_count = len(df_out.columns) if hasattr(df_out, "columns") else (len(records[0].keys()) if records else 0)

        summary = {
            "title": "Snowflake Connector Summary",
            "message": status_msg,
            "status": delivery_status,
            "warehouse": warehouse,
            "database": database,
            "schema": schema,
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
        account = config.get("account", "xy12345.us-east-1")
        user = config.get("user", "analyst")
        warehouse = config.get("warehouse", "COMPUTE_WH")
        database = config.get("database", "ANALYTICS_DB")
        schema = config.get("schema_name", "PUBLIC")
        role = config.get("role", "PUBLIC")
        read_mode = config.get("read_mode", "table")
        limit = config.get("limit", 1000)

        if read_mode == "table":
            table_name = config.get("table_name", "FCT_ORDERS")
            cols = config.get("columns", "*") or "*"
            sql = f'SELECT {cols} FROM "{database}"."{schema}"."{table_name}" LIMIT {limit}'
        else:
            sql = config.get("query", "SELECT * FROM FCT_ORDERS LIMIT 100")

        return (
            f"# Generated Snowflake Data Extraction Script\n"
            f"import snowflake.connector\n"
            f"import pandas as pd\n\n"
            f"conn = snowflake.connector.connect(\n"
            f"    account='{account}',\n"
            f"    user='{user}',\n"
            f"    password='<SECURE_PASSWORD>',\n"
            f"    warehouse='{warehouse}',\n"
            f"    database='{database}',\n"
            f"    schema='{schema}',\n"
            f"    role='{role}'\n"
            f")\n"
            f"query = \"\"\"{sql}\"\"\"\n"
            f"df = pd.read_sql(query, conn)\n"
            f"conn.close()\n"
            f"print(f'Retrieved {{len(df)}} rows from Snowflake warehouse {warehouse}.')\n"
        )
