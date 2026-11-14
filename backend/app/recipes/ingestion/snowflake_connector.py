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
from backend.app.core.exceptions import ValidationException


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

    inputs: List[RecipePort] = []

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
            "sections": [
                {"id": "connection", "title": "Connection Configuration", "description": "Configure connection parameters"},
                {"id": "query_selection", "title": "Query & Table Selection", "description": "Choose tables or custom SQL"},
                {"id": "runtime_limits", "title": "Limits & Performance", "description": "Row limits and warehouse performance"}
            ],
            "properties": {
                # ── 1. Connection Mode & Server Credentials ──
                "connection_mode": {
                    "type": "string",
                    "title": "Connection Mode",
                    "input_type": "dropdown",
                    "enum": ["account_credentials", "sso", "connection_string", "credential"],
                    "default": "account_credentials",
                    "section": "connection",
                    "description": "Choose how to connect: Snowflake account credentials, Single Sign-On (SSO / OAuth / Okta), connection string (SQLAlchemy URI), or stored credential."
                },
                "credential_id": {
                    "type": "string",
                    "title": "Connected Snowflake Credential",
                    "input_type": "credential_dropdown",
                    "provider": "snowflake",
                    "section": "connection",
                    "description": "Select a stored credential from platform settings.",
                    "visible_when": {
                        "field": "connection_mode",
                        "equals": "credential"
                    }
                },
                "sso_authenticator": {
                    "type": "string",
                    "title": "SSO Authenticator",
                    "input_type": "dropdown",
                    "enum": ["externalbrowser", "oauth", "okta"],
                    "default": "externalbrowser",
                    "section": "connection",
                    "description": "Snowflake SSO authenticator method: 'externalbrowser' (browser-based IdP prompt), 'oauth' (OAuth 2.0 access token), or 'okta' (Okta SAML/SSO URL).",
                    "visible_when": {
                        "field": "connection_mode",
                        "equals": "sso"
                    }
                },
                "sso_token": {
                    "type": "string",
                    "title": "OAuth Access Token",
                    "input_type": "password",
                    "default": "",
                    "placeholder": "Enter OAuth Bearer token",
                    "section": "connection",
                    "description": "OAuth 2.0 access token used when SSO Authenticator is set to 'oauth'.",
                    "visible_when": {
                        "field": "connection_mode",
                        "equals": "sso"
                    }
                },
                "idp_url": {
                    "type": "string",
                    "title": "Okta / IdP URL",
                    "input_type": "text",
                    "default": "",
                    "placeholder": "e.g. https://mycompany.okta.com",
                    "section": "connection",
                    "description": "Identity provider endpoint URL (e.g. https://<org>.okta.com) used when SSO Authenticator is set to 'okta'.",
                    "visible_when": {
                        "field": "connection_mode",
                        "equals": "sso"
                    }
                },
                "account": {
                    "type": "string",
                    "title": "Snowflake Account Identifier",
                    "input_type": "text",
                    "default": "",
                    "placeholder": "e.g. xy12345.us-east-1 or myorg-myaccount",
                    "section": "connection",
                    "description": "Snowflake account identifier (e.g. 'xy12345.us-east-1' or '<orgname>-<accountname>')."
                },
                "user": {
                    "type": "string",
                    "title": "Username / SSO Email",
                    "input_type": "text",
                    "default": "",
                    "placeholder": "e.g. user@company.com or analyst",
                    "section": "connection",
                    "description": "Snowflake username or SSO identity email."
                },
                "password": {
                    "type": "string",
                    "title": "Password",
                    "input_type": "password",
                    "default": "",
                    "placeholder": "Enter Snowflake password",
                    "section": "connection",
                    "description": "Snowflake user password.",
                    "visible_when": {
                        "field": "connection_mode",
                        "equals": "account_credentials"
                    }
                },
                "role": {
                    "type": "string",
                    "title": "Role",
                    "input_type": "text",
                    "default": "PUBLIC",
                    "placeholder": "PUBLIC",
                    "section": "connection",
                    "description": "Snowflake security role to assume (e.g. 'ACCOUNTADMIN', 'SYSADMIN', 'DATA_ANALYST')."
                },
                "warehouse": {
                    "type": "string",
                    "title": "Warehouse",
                    "input_type": "text",
                    "default": "",
                    "placeholder": "e.g. COMPUTE_WH or ANALYTICS_WH",
                    "section": "connection",
                    "description": "Virtual warehouse used to execute queries (e.g. 'COMPUTE_WH', 'ANALYTICS_WH')."
                },
                "connection_string": {
                    "type": "string",
                    "title": "Connection String (URI)",
                    "input_type": "text",
                    "default": "",
                    "placeholder": "e.g. snowflake://user:pass@account/db/schema?warehouse=wh&role=role",
                    "section": "connection",
                    "description": "Full Snowflake SQLAlchemy URI e.g. snowflake://user:pass@account/db/schema?warehouse=wh&role=role",
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
                    "placeholder": "e.g. ANALYTICS_DB or PROD_DB",
                    "section": "query_selection",
                    "description": "Name of the target Snowflake database."
                },
                "schema_name": {
                    "type": "string",
                    "title": "Schema Name",
                    "input_type": "text",
                    "default": "PUBLIC",
                    "placeholder": "PUBLIC",
                    "section": "query_selection",
                    "description": "Target schema within the database (default 'PUBLIC').",
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
                    "placeholder": "e.g. FCT_ORDERS",
                    "section": "query_selection",
                    "description": "Target Snowflake table or view name (used in 'table' mode).",
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
                    "placeholder": "e.g. ORDER_ID, CUSTOMER_ID, TOTAL_AMOUNT or *",
                    "section": "query_selection",
                    "description": "Columns to select (e.g. 'ORDER_ID, CUSTOMER_ID, TOTAL_AMOUNT, CREATED_AT'). Leave '*' for all.",
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
                    "placeholder": "e.g. STATUS = 'COMPLETED' AND AMOUNT > 0",
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
                    "placeholder": "e.g. SELECT * FROM PUBLIC.FCT_ORDERS LIMIT 100",
                    "section": "query_selection",
                    "description": "Custom SQL query to execute in Snowflake. Only used when Read Mode is 'query'. Modifying statements (DROP, DELETE, UPDATE, TRUNCATE) are rejected.",
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
            elif conn_mode == "sso":
                acc = str(config.get("account") or config.get("sso_account", "")).strip()
                usr = str(config.get("user") or config.get("sso_user", "")).strip()
                if not acc:
                    errors.append("Snowflake Account identifier is required in 'sso' mode.")
                if not usr:
                    errors.append("Username / SSO Email is required in 'sso' mode.")
                auth_method = str(config.get("sso_authenticator", "externalbrowser")).strip().lower()
                if auth_method == "oauth" and not str(config.get("sso_token", "")).strip():
                    errors.append("OAuth Access Token is required when SSO Authenticator is 'oauth'.")
                elif auth_method == "okta" and not str(config.get("idp_url", "")).strip():
                    errors.append("Okta / IdP URL is required when SSO Authenticator is 'okta'.")
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
        account = str(config.get("account") or config.get("sso_account", "xy12345.us-east-1")).strip()
        user = str(config.get("user") or config.get("sso_user", "analyst")).strip()
        password = str(config.get("password", "")).strip()
        warehouse = str(config.get("warehouse") or config.get("sso_warehouse", "COMPUTE_WH")).strip()
        database = str(config.get("database", "ANALYTICS_DB")).strip()
        schema_name = str(config.get("schema_name", "PUBLIC")).strip()
        role = str(config.get("role") or config.get("sso_role", "PUBLIC")).strip()
        cs = str(config.get("connection_string", "")).strip()
        sso_authenticator = str(config.get("sso_authenticator", "externalbrowser")).strip().lower()
        sso_token = str(config.get("sso_token", "")).strip()
        idp_url = str(config.get("idp_url", "")).strip()

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
                    sso_authenticator = cred.get("sso_authenticator", sso_authenticator)
                    sso_token = cred.get("sso_token", sso_token)
                    idp_url = cred.get("idp_url", idp_url)

        return {
            "account": account,
            "user": user,
            "password": password,
            "warehouse": warehouse,
            "database": database,
            "schema": schema_name,
            "role": role,
            "connection_string": cs,
            "mode": conn_mode,
            "sso_authenticator": sso_authenticator,
            "sso_token": sso_token,
            "idp_url": idp_url
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
        if not simulation_mode:
            last_error = None
            try:
                import snowflake.connector
                conn_kwargs = {
                    "account": account,
                    "user": params["user"],
                    "warehouse": warehouse,
                    "database": database,
                    "schema": schema,
                    "role": params["role"]
                }
                if params.get("mode") == "sso":
                    auth = params.get("sso_authenticator", "externalbrowser")
                    if auth == "oauth":
                        conn_kwargs["authenticator"] = "oauth"
                        conn_kwargs["token"] = params.get("sso_token", "")
                    elif auth == "okta":
                        conn_kwargs["authenticator"] = params.get("idp_url", "")
                    else:
                        conn_kwargs["authenticator"] = "externalbrowser"
                else:
                    conn_kwargs["password"] = params["password"]
                conn = snowflake.connector.connect(**conn_kwargs)
                cur = conn.cursor()

                if read_mode == "table":
                    cols = str(config.get("columns", "*")).strip() or "*"
                    filter_cond = str(config.get("filter_condition", "")).strip()
                    where_clause = f" WHERE {filter_cond}" if filter_cond else ""
                    db_id = database if database.startswith('"') and database.endswith('"') else f'"{database.upper()}"'
                    schema_id = schema if schema.startswith('"') and schema.endswith('"') else f'"{schema.upper()}"'
                    tbl_id = table_name if table_name.startswith('"') and table_name.endswith('"') else f'"{table_name.upper()}"'
                    sql = f'SELECT {cols} FROM {db_id}.{schema_id}.{tbl_id}{where_clause} LIMIT {limit}'
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
            except Exception as e:
                logger.warning(f"Snowflake connection failed: {e}")
                last_error = e

            if last_error is not None:
                raise ValidationException(f"Snowflake connection to {account}/{database}.{schema} failed: {str(last_error)}")

        # Simulation Mode (ONLY when simulation_mode is True)
        if simulation_mode:
            records = self._generate_benchmark_data(config)
            delivery_status = "SIMULATED"
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
            filter_cond = str(config.get("filter_condition", "")).strip()
            where_clause = f" WHERE {filter_cond}" if filter_cond else ""
            db_id = database if database.startswith('"') and database.endswith('"') else f'"{database.upper()}"'
            schema_id = schema if schema.startswith('"') and schema.endswith('"') else f'"{schema.upper()}"'
            tbl_id = table_name if table_name.startswith('"') and table_name.endswith('"') else f'"{table_name.upper()}"'
            sql = f'SELECT {cols} FROM {db_id}.{schema_id}.{tbl_id}{where_clause} LIMIT {limit}'
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

    def inspect(
        self,
        action: str,
        config: Dict[str, Any],
        context: Optional[Any] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Introspects connected Snowflake warehouse in real-time."""
        action = str(action or "test_connection").strip().lower()
        params = self._resolve_connection_params(config, context)
        simulation_mode = bool(config.get("simulation_mode", False))

        is_live = False
        server_info = ""
        inspect_err = ""
        conn = None

        if not simulation_mode and params.get("account") and params.get("user"):
            try:
                import snowflake.connector
                inspect_conn_kwargs = {
                    "account": params["account"],
                    "user": params["user"],
                    "warehouse": params["warehouse"],
                    "database": params["database"],
                    "schema": params["schema"],
                    "role": params["role"],
                    "login_timeout": int(config.get("timeout", 10)),
                }
                if params.get("mode") == "sso":
                    auth = params.get("sso_authenticator", "externalbrowser")
                    if auth == "oauth":
                        inspect_conn_kwargs["authenticator"] = "oauth"
                        inspect_conn_kwargs["token"] = params.get("sso_token", "")
                    elif auth == "okta":
                        inspect_conn_kwargs["authenticator"] = params.get("idp_url", "")
                    else:
                        inspect_conn_kwargs["authenticator"] = "externalbrowser"
                else:
                    inspect_conn_kwargs["password"] = params["password"]

                conn = snowflake.connector.connect(**inspect_conn_kwargs)
                cur = conn.cursor()
                cur.execute("SELECT CURRENT_VERSION();")
                row = cur.fetchone()
                if row:
                    server_info = str(row[0])
                    is_live = True
                cur.close()
            except Exception as ex:
                inspect_err = str(ex)
                logger.debug(f"Snowflake inspect connection failed: {ex}")

        if action == "test_connection":
            if is_live:
                if conn: conn.close()
                return {"success": True, "connected": True, "status": "LIVE", "server_version": server_info, "database": params.get("database"), "message": "Successfully connected to Snowflake."}
            elif simulation_mode:
                return {"success": True, "connected": True, "status": "SIMULATED", "server_version": "Snowflake 8.12.0 [SIMULATION]", "database": params.get("database"), "message": "Snowflake simulation mode active."}
            else:
                return {"success": False, "connected": False, "status": "UNREACHABLE", "error": inspect_err, "message": f"Could not connect to Snowflake: {inspect_err or 'Unreachable'}"}

        if action in ("databases", "get_databases"):
            if not is_live and not simulation_mode:
                return {"success": False, "databases": [], "error": inspect_err or "Connection could not be established."}
            dbs = []
            if is_live and conn:
                try:
                    cur = conn.cursor()
                    cur.execute("SHOW DATABASES;")
                    dbs = [row[1] for row in cur.fetchall()]
                    cur.close()
                    conn.close()
                except Exception as e:
                    logger.warning(f"Snowflake SHOW DATABASES failed: {e}")
            if not dbs and simulation_mode:
                dbs = ["ANALYTICS", "PRODUCTION", "STAGING", "DEMO_DB"]
            return {"success": True, "databases": dbs, "count": len(dbs), "source": "LIVE" if is_live else "SIMULATED"}

        if action in ("tables", "get_tables"):
            if not is_live and not simulation_mode:
                return {"success": False, "tables": [], "error": inspect_err or "Connection could not be established."}
            target_schema = schema_name or params.get("schema") or "PUBLIC"
            tbls = []
            if is_live and conn:
                try:
                    cur = conn.cursor()
                    cur.execute(f"SHOW TABLES IN SCHEMA {target_schema};")
                    tbls = [row[1] for row in cur.fetchall()]
                    cur.close()
                    conn.close()
                except Exception as e:
                    logger.warning(f"Snowflake SHOW TABLES failed: {e}")
            if not tbls and simulation_mode:
                tbls = ["FCT_ORDERS", "DIM_CUSTOMERS", "FCT_SALES", "DIM_PRODUCTS"]
            return {"success": True, "tables": tbls, "count": len(tbls), "schema": target_schema, "source": "LIVE" if is_live else "SIMULATED"}

        if action in ("columns", "get_columns"):
            if not is_live and not simulation_mode:
                return {"success": False, "columns": [], "error": inspect_err or "Connection could not be established."}
            target_tbl = table_name or config.get("table_name", "FCT_ORDERS")
            cols = []
            if is_live and conn:
                try:
                    cur = conn.cursor()
                    cur.execute(f"SHOW COLUMNS IN TABLE {target_tbl};")
                    for row in cur.fetchall():
                        cols.append({"name": row[2], "type": row[3], "nullable": True, "primary_key": False})
                    cur.close()
                    conn.close()
                except Exception as e:
                    logger.warning(f"Snowflake SHOW COLUMNS failed: {e}")
            if not cols and simulation_mode:
                cols = [
                    {"name": "ORDER_ID", "type": "NUMBER", "nullable": False, "primary_key": True},
                    {"name": "CUSTOMER_ID", "type": "VARCHAR", "nullable": False, "primary_key": False},
                    {"name": "NET_AMOUNT", "type": "FLOAT", "nullable": False, "primary_key": False},
                    {"name": "CREATED_AT", "type": "TIMESTAMP_NTZ", "nullable": True, "primary_key": False},
                ]
            return {"success": True, "columns": cols, "count": len(cols), "table": target_tbl, "source": "LIVE" if is_live else "SIMULATED"}

        return {"success": False, "error": f"Unknown action '{action}'."}
