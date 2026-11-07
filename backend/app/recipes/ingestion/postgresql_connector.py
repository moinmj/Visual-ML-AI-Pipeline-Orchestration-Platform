"""
PostgreSQL Connector Recipe (Ingestion & Orchestration -> Database & Warehouse Connectors).

Reads tables or query results from a PostgreSQL database into the pipeline as a DataFrame.
Supports connection strings (URI), host/port, or saved credentials, schema selection,
table selection, column selection ('all' or specific subset), WHERE filters, ORDER BY,
SSL modes, dynamic database inspection (test connection, schemas, tables, columns),
benchmark simulation fallback, and Python code generation.
"""

import re
import os
import time
from typing import Dict, Any, List, Optional, Tuple, Union
from urllib.parse import urlparse, unquote, parse_qs, quote_plus

try:
    import pandas as pd
except ImportError:
    pd = None

from backend.app.recipes.base.recipe import BaseRecipe, RecipePort
from backend.app.core.logging import logger
from backend.app.core.config import settings
from backend.app.core.exceptions import ValidationException


class PostgreSQLConnectorRecipe(BaseRecipe):
    recipe_id = "postgresql_connector"
    name = "PostgreSQL Connector"
    version = "1.1.0"
    category = "ingestion"
    group = "Ingestion & Orchestration"
    subgroup = "Database & Warehouse Connectors"
    description = "Reads tables or query results from a PostgreSQL database."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    inputs = [
        RecipePort(
            id="input",
            label="Trigger / Input Dataset",
            type="dataframe",
            required=False,
            max_connections=1,
            description="Optional upstream dataset for batch IDs or execution trigger",
        )
    ]

    outputs = [
        RecipePort(
            id="output",
            label="PostgreSQL Dataset",
            type="dataframe",
            description="DataFrame containing retrieved PostgreSQL records",
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
                    "enum": ["host_port", "connection_string", "credential"],
                    "default": "host_port",
                    "description": "Choose how to connect: standard host/port parameters, full connection string (URI), or stored credential.",
                },
                "host": {
                    "type": "string",
                    "title": "Host",
                    "input_type": "text",
                    "default": "localhost",
                    "description": "PostgreSQL server hostname or IP address (used in 'host_port' mode).",
                },
                "port": {
                    "type": "integer",
                    "title": "Port",
                    "input_type": "number",
                    "default": 5432,
                    "description": "PostgreSQL server port (default 5432).",
                },
                "database": {
                    "type": "string",
                    "title": "Database Name",
                    "input_type": "text",
                    "default": "postgres",
                    "description": "Name of the PostgreSQL database.",
                },
                "username": {
                    "type": "string",
                    "title": "Username",
                    "input_type": "text",
                    "default": "postgres",
                    "description": "PostgreSQL user role.",
                },
                "password": {
                    "type": "string",
                    "title": "Password",
                    "input_type": "password",
                    "default": "",
                    "description": "PostgreSQL user password.",
                },
                "connection_string": {
                    "type": "string",
                    "title": "Connection String (URI)",
                    "input_type": "text",
                    "default": "",
                    "description": "Full PostgreSQL connection URI e.g. postgresql://user:password@localhost:5432/dbname",
                },
                "credential_id": {
                    "type": "string",
                    "title": "Connected PostgreSQL Credential",
                    "input_type": "credential_dropdown",
                    "provider": "postgresql",
                    "description": "Select a stored credential from platform settings.",
                },
                "ssl_mode": {
                    "type": "string",
                    "title": "SSL Mode",
                    "input_type": "dropdown",
                    "enum": ["prefer", "require", "disable", "verify-ca", "verify-full"],
                    "default": "prefer",
                    "description": "PostgreSQL SSL connection mode.",
                },
                "timeout": {
                    "type": "integer",
                    "title": "Timeout (seconds)",
                    "input_type": "number",
                    "default": 15,
                    "description": "Connection and query execution timeout in seconds.",
                },
                "read_mode": {
                    "type": "string",
                    "title": "Read Mode",
                    "input_type": "dropdown",
                    "enum": ["table", "query"],
                    "default": "table",
                    "description": "Read from a table (specifying schema, table, columns) or execute a custom SQL query.",
                },
                "schema_name": {
                    "type": "string",
                    "title": "Schema Name",
                    "input_type": "dropdown",
                    "enum": [
                        "public",
                        "analytics",
                        "staging",
                        "reporting",
                        "sales",
                        "retail",
                        "ecommerce",
                        "raw_data",
                        "information_schema"
                    ],
                    "default": "public",
                    "description": "PostgreSQL schema name (e.g. 'public', 'analytics', 'staging').",
                },
                "table_name": {
                    "type": "string",
                    "title": "Table Name",
                    "input_type": "dropdown",
                    "enum": [
                        "sales",
                        "retail_sales",
                        "customers",
                        "orders",
                        "transactions",
                        "products",
                        "users",
                        "inventory",
                        "invoices",
                        "payments",
                        "dim_customers",
                        "fct_sales"
                    ],
                    "default": "sales",
                    "description": "Table to read from the selected schema (e.g. 'sales', 'retail_sales', 'customers', 'orders').",
                },
                "select_all_columns": {
                    "type": "boolean",
                    "title": "Select All Columns (*)",
                    "input_type": "checkbox",
                    "default": True,
                    "description": "If enabled, retrieves all columns ('*'). Disable to specify individual columns in 'Columns to Select'.",
                },
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "title": "Columns to Select",
                    "input_type": "multi_select",
                    "enum": [
                        "invoice_id",
                        "branch",
                        "city",
                        "customer_type",
                        "gender",
                        "product_line",
                        "unit_price",
                        "quantity",
                        "tax_5",
                        "total",
                        "date",
                        "time",
                        "payment",
                        "cogs",
                        "gross_margin_percentage",
                        "gross_income",
                        "rating",
                        "id",
                        "customer_id",
                        "amount",
                        "currency",
                        "status",
                        "region",
                        "created_at",
                        "order_id",
                        "user_id"
                    ],
                    "description": "Select columns to retrieve from the dropdown when 'Select All Columns' is disabled (leave empty or keep 'Select All Columns' enabled for all columns '*').",
                },
                "where_clause": {
                    "type": "string",
                    "title": "Filter Condition (WHERE)",
                    "input_type": "text",
                    "default": "",
                    "description": "Optional SQL WHERE filter expression without 'WHERE' keyword (e.g. status = 'ACTIVE' AND amount > 0).",
                },
                "order_by": {
                    "type": "string",
                    "title": "Sort Order (ORDER BY)",
                    "input_type": "text",
                    "default": "",
                    "description": "Optional SQL sort expression without 'ORDER BY' keyword (e.g. created_at DESC).",
                },
                "limit": {
                    "type": "integer",
                    "title": "Row Limit",
                    "input_type": "number",
                    "default": 1000,
                    "description": "Maximum number of rows to retrieve (1 to 100000).",
                },
                "offset": {
                    "type": "integer",
                    "title": "Row Offset",
                    "input_type": "number",
                    "default": 0,
                    "description": "Number of initial rows to skip for pagination.",
                },
                "query": {
                    "type": "string",
                    "title": "SQL Query",
                    "input_type": "textarea",
                    "default": "SELECT * FROM public.sales LIMIT 100",
                    "description": "Read-only SQL query to execute when Read Mode is 'query' (SELECT, WITH, SHOW, etc.).",
                },
                "simulation_mode": {
                    "type": "boolean",
                    "title": "Simulation / Demo Mode",
                    "input_type": "checkbox",
                    "default": False,
                    "description": "Generate realistic benchmark dataset without connecting to an external database.",
                },
            },
            "required": ["read_mode"],
            "templates": [
                {
                    "label": "Read Table (Public Schema - All Columns)",
                    "config": {
                        "read_mode": "table",
                        "schema_name": "public",
                        "table_name": "users",
                        "select_all_columns": True,
                        "limit": 500,
                    },
                },
                {
                    "label": "Read Selected Columns with Filter",
                    "config": {
                        "read_mode": "table",
                        "schema_name": "public",
                        "table_name": "sales",
                        "select_all_columns": False,
                        "columns": "transaction_id, customer_id, amount, status",
                        "where_clause": "status = 'COMPLETED'",
                        "limit": 1000,
                    },
                },
                {
                    "label": "Custom Filter Query",
                    "config": {
                        "read_mode": "query",
                        "query": "SELECT id, customer_name, total_amount, created_at FROM public.sales WHERE total_amount > 100.0 ORDER BY created_at DESC",
                        "limit": 1000,
                    },
                },
            ],
        }

    def _strip_comments(self, sql: str) -> str:
        """Strip SQL line and block comments."""
        cleaned = re.sub(r"--.*?(\n|$)", " ", sql or "")
        cleaned = re.sub(r"/\*.*?\*/", " ", cleaned, flags=re.DOTALL)
        return cleaned.strip()

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        errors = []
        read_mode = str(config.get("read_mode", "table")).strip().lower()

        # Validate Port if provided
        port = config.get("port")
        if port is not None:
            try:
                p = int(port)
                if p < 1 or p > 65535:
                    errors.append(f"Invalid PostgreSQL port '{port}'. Must be between 1 and 65535.")
            except (ValueError, TypeError):
                errors.append(f"Port '{port}' must be a valid integer.")

        # Validate Read Mode Requirements
        if read_mode == "table":
            table_name = str(config.get("table_name", "")).strip()
            if not table_name:
                errors.append("Table name is required when read_mode is 'table'.")
            schema_name = str(config.get("schema_name", "public")).strip()
            if not schema_name:
                errors.append("Schema name cannot be empty when read_mode is 'table'.")

            # Check if select_all_columns is explicitly False and no columns provided
            if config.get("select_all_columns") is False:
                raw_cols = config.get("columns")
                has_cols = False
                if isinstance(raw_cols, list) and len(raw_cols) > 0:
                    has_cols = True
                elif isinstance(raw_cols, str) and raw_cols.strip():
                    has_cols = True
                if not has_cols:
                    errors.append(
                        "When 'Select All Columns' is disabled, at least one column must be specified in 'Columns to Select'."
                    )

            # Validate WHERE clause if present
            where_clause = str(config.get("where_clause", "")).strip()
            if where_clause:
                cleaned_where = self._strip_comments(where_clause).lower()
                disallowed = ["drop", "truncate", "delete", "insert", "update", "alter", "create", "grant", "revoke"]
                for kw in disallowed:
                    if re.search(rf"\b{kw}\b", cleaned_where):
                        errors.append(f"Disallowed DDL/DML keyword '{kw.upper()}' detected in WHERE clause.")
                        break

            # Validate ORDER BY if present
            order_by = str(config.get("order_by", "")).strip()
            if order_by:
                cleaned_order = self._strip_comments(order_by).lower()
                disallowed = ["drop", "truncate", "delete", "insert", "update", "alter", "create", "grant", "revoke"]
                for kw in disallowed:
                    if re.search(rf"\b{kw}\b", cleaned_order):
                        errors.append(f"Disallowed DDL/DML keyword '{kw.upper()}' detected in ORDER BY clause.")
                        break

        elif read_mode == "query":
            q = (config.get("query") or "").strip()
            if not q:
                errors.append("SQL query is required when read_mode is 'query'.")
            else:
                cleaned = self._strip_comments(q).lower()
                first_word = cleaned.split()[0] if cleaned.split() else ""
                if first_word not in ("select", "with", "show", "explain"):
                    errors.append(
                        f"Only read-only queries (SELECT, WITH, SHOW, EXPLAIN) are permitted. Found '{first_word}'."
                    )

                disallowed = ["drop", "truncate", "delete", "insert", "update", "alter", "create", "grant", "revoke"]
                for kw in disallowed:
                    if re.search(rf"\b{kw}\b", cleaned):
                        errors.append(f"Disallowed DDL/DML keyword '{kw.upper()}' detected in PostgreSQL query.")
                        break

        # Validate Connection mode requirements if not simulated
        conn_mode = str(
            config.get("connection_mode", "connection_string" if config.get("connection_string") else "host_port")
        ).strip().lower()

        if not config.get("simulation_mode"):
            if conn_mode == "connection_string":
                cs = str(config.get("connection_string", "")).strip()
                if not cs:
                    errors.append("Connection string (URI) is required in 'connection_string' mode.")
            elif conn_mode == "credential":
                cred_id = str(config.get("credential_id", "")).strip()
                if not cred_id:
                    errors.append("Credential selection is required in 'credential' mode.")

        return errors

    def _resolve_connection_params(self, config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        """Resolves database credentials, connection strings, and parameters from config or context."""
        mode = str(
            config.get("connection_mode", "host_port" if (config.get("host") and not config.get("connection_string")) else "connection_string" if config.get("connection_string") else "host_port")
        ).strip().lower()
        credential_id = str(config.get("credential_id", "")).strip()

        host = str(config.get("host", "localhost")).strip()
        try:
            port = int(config.get("port", 5432))
        except (ValueError, TypeError):
            port = 5432
        database = str(config.get("database", "postgres")).strip()
        username = str(config.get("username", "postgres")).strip()
        password = str(config.get("password", "")).strip()
        conn_str = str(config.get("connection_string", "")).strip()
        ssl_mode = str(config.get("ssl_mode", "prefer")).strip()

        # Context credential lookup
        if credential_id and context and isinstance(context, dict):
            creds = context.get("credentials", {})
            if credential_id in creds:
                c_data = creds[credential_id]
                c_data = c_data.get("data", c_data)
                host = c_data.get("host") or host
                port = int(c_data.get("port") or port)
                database = c_data.get("database") or database
                username = c_data.get("username") or c_data.get("user") or username
                password = c_data.get("password") or password
                conn_str = c_data.get("connection_string") or c_data.get("url") or conn_str

        # Environment variable overrides if empty
        if not password:
            password = os.getenv("PGPASSWORD", os.getenv("POSTGRES_PASSWORD", ""))
        if not username:
            username = os.getenv("PGUSER", os.getenv("POSTGRES_USER", "postgres"))

        # Synchronize connection parameters
        if mode == "host_port":
            # Direct parameters mode: build connection string from explicit fields with URL-encoded password
            quoted_user = quote_plus(str(username)) if username else ""
            quoted_pass = quote_plus(str(password)) if password else ""
            auth_part = f"{quoted_user}:{quoted_pass}@" if (quoted_user or quoted_pass) else ""
            ssl_param = f"?sslmode={ssl_mode}" if ssl_mode else ""
            conn_str = f"postgresql://{auth_part}{host}:{port}/{database}{ssl_param}"
        elif mode == "connection_string":
            if conn_str:
                if conn_str.startswith("postgres://"):
                    conn_str = "postgresql://" + conn_str.split("://", 1)[1]
                try:
                    parsed = urlparse(conn_str)
                    # Check if conn_str hostname is localhost/127.0.0.1 but user specifically provided a remote host
                    user_host = str(config.get("host", "")).strip()
                    if user_host and user_host not in ("localhost", "127.0.0.1", "") and parsed.hostname in ("localhost", "127.0.0.1", None):
                        host = user_host
                        if config.get("database"):
                            database = str(config.get("database")).strip()
                        if config.get("username"):
                            username = str(config.get("username")).strip()
                        if config.get("password"):
                            password = str(config.get("password")).strip()
                        if config.get("port"):
                            port = int(config.get("port", 5432))
                        quoted_user = quote_plus(str(username)) if username else ""
                        quoted_pass = quote_plus(str(password)) if password else ""
                        auth_part = f"{quoted_user}:{quoted_pass}@" if (quoted_user or quoted_pass) else ""
                        ssl_param = f"?sslmode={ssl_mode}" if ssl_mode else ""
                        conn_str = f"postgresql://{auth_part}{host}:{port}/{database}{ssl_param}"
                    else:
                        if parsed.hostname:
                            host = parsed.hostname
                        if parsed.port:
                            port = parsed.port
                        if parsed.path and len(parsed.path) > 1:
                            clean_path = parsed.path.lstrip("/").split("?")[0]
                            if clean_path:
                                database = clean_path
                        if parsed.username:
                            username = unquote(parsed.username)
                        if parsed.password:
                            password = unquote(parsed.password)
                        if parsed.query:
                            qs = parse_qs(parsed.query)
                            if "sslmode" in qs and qs["sslmode"]:
                                ssl_mode = qs["sslmode"][0]
                except Exception as parse_e:
                    logger.warning(f"Failed to parse connection_string URL: {parse_e}")
            elif host and database:
                quoted_user = quote_plus(str(username)) if username else ""
                quoted_pass = quote_plus(str(password)) if password else ""
                auth_part = f"{quoted_user}:{quoted_pass}@" if (quoted_user or quoted_pass) else ""
                ssl_param = f"?sslmode={ssl_mode}" if ssl_mode else ""
                conn_str = f"postgresql://{auth_part}{host}:{port}/{database}{ssl_param}"
        else:
            # Stored credential or fallback
            if not conn_str and host and database:
                quoted_user = quote_plus(str(username)) if username else ""
                quoted_pass = quote_plus(str(password)) if password else ""
                auth_part = f"{quoted_user}:{quoted_pass}@" if (quoted_user or quoted_pass) else ""
                ssl_param = f"?sslmode={ssl_mode}" if ssl_mode else ""
                conn_str = f"postgresql://{auth_part}{host}:{port}/{database}{ssl_param}"

        return {
            "mode": mode,
            "host": host,
            "port": port,
            "database": database,
            "username": username,
            "password": password,
            "connection_string": conn_str,
            "ssl_mode": ssl_mode,
        }

    def _generate_benchmark_data(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generates realistic enterprise records for benchmark and testing fallback."""
        table_name = str(config.get("table_name", "customers")).lower().split(".")[-1]
        limit = max(1, min(int(config.get("limit", 100)), 500))

        records: List[Dict[str, Any]] = []
        statuses = ["ACTIVE", "PENDING", "COMPLETED", "SUSPENDED", "ARCHIVED"]
        regions = ["NA-East", "NA-West", "EMEA-Central", "APAC-South", "LATAM-North"]

        for i in range(1, limit + 1):
            if "sale" in table_name or "order" in table_name or "transaction" in table_name:
                records.append({
                    "transaction_id": 10000 + i,
                    "customer_id": 500 + (i % 50),
                    "amount": round(25.50 + (i * 3.75) % 850.0, 2),
                    "currency": "USD",
                    "status": statuses[i % len(statuses)],
                    "region": regions[i % len(regions)],
                    "created_at": f"2026-03-{((i % 28) + 1):02d}T10:15:30Z",
                })
            elif "user" in table_name or "customer" in table_name or "account" in table_name:
                records.append({
                    "id": i,
                    "account_code": f"ACC-{2000 + i}",
                    "name": f"Enterprise Client {i}",
                    "email": f"client{i}@example-enterprise.com",
                    "tier": "Enterprise" if i % 3 == 0 else "Standard",
                    "is_active": (i % 7 != 0),
                    "created_at": f"2026-01-{((i % 28) + 1):02d}T08:00:00Z",
                })
            else:
                records.append({
                    "id": i,
                    "name": f"Record {i}",
                    "category": f"Category-{(i % 5) + 1}",
                    "value": round(10.0 + (i * 1.5), 2),
                    "status": statuses[i % len(statuses)],
                    "updated_at": f"2026-04-{((i % 28) + 1):02d}T12:00:00Z",
                })
        return records

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        start_time = time.time()
        read_mode = str(config.get("read_mode", "table")).strip().lower()
        limit = max(1, min(int(config.get("limit", 1000)), 100000))
        simulation_mode = bool(config.get("simulation_mode", False))
        params = self._resolve_connection_params(config, context)

        # Build execution SQL
        select_all_columns = config.get("select_all_columns")
        cols_raw = config.get("columns", "")
        cols_list: List[str] = []

        if isinstance(cols_raw, list):
            cols_list = [str(c).strip() for c in cols_raw if str(c).strip() and str(c).strip() != "*"]
        elif isinstance(cols_raw, str) and cols_raw.strip():
            cols_list = [c.strip() for c in cols_raw.split(",") if c.strip() and c.strip() != "*"]

        if select_all_columns is None:
            select_all_columns = False if cols_list else True

        if read_mode == "table":
            table_name = str(config.get("table_name", "customers")).strip()
            schema_name = str(config.get("schema_name", "public")).strip() or "public"

            # Parse dot notation e.g. "analytics.orders"
            if "." in table_name:
                parts = table_name.split(".", 1)
                schema_name = parts[0].strip() or schema_name
                table_name = parts[1].strip()

            if select_all_columns is False and cols_list:
                cols_str = ", ".join(f'"{c}"' if not c.startswith('"') else c for c in cols_list)
            else:
                cols_str = "*"

            full_table = f'"{schema_name}"."{table_name}"'
            target_desc = f"{schema_name}.{table_name}"

            # WHERE clause
            where_clause = str(config.get("where_clause", "")).strip()
            where_sql = ""
            if where_clause:
                if where_clause.lower().startswith("where "):
                    where_clause = where_clause[6:].strip()
                where_sql = f" WHERE {where_clause}"

            # ORDER BY clause
            order_by = str(config.get("order_by", "")).strip()
            order_sql = ""
            if order_by:
                if order_by.lower().startswith("order by "):
                    order_by = order_by[9:].strip()
                order_sql = f" ORDER BY {order_by}"

            # OFFSET
            offset = int(config.get("offset", 0))
            offset_sql = f" OFFSET {offset}" if offset > 0 else ""

            query = f"SELECT {cols_str} FROM {full_table}{where_sql}{order_sql} LIMIT {limit}{offset_sql}"
        else:
            query = str(config.get("query", "SELECT * FROM public.customers")).strip()
            target_desc = "Custom Query"
            if "limit" not in query.lower():
                query = f"{query} LIMIT {limit}"

        records: List[Dict[str, Any]] = []
        delivery_status = "SIMULATED"
        status_msg = ""
        columns_retrieved: List[str] = []

        # Live Execution Attempt
        if not simulation_mode:
            if not params.get("database"):
                raise ValidationException("PostgreSQL 'Database Name' is required for live execution.")
            if not params.get("host") and not params.get("connection_string"):
                raise ValidationException("PostgreSQL 'Host' or 'Connection String' is required for live execution.")

            last_error = None
            try:
                import sqlalchemy
                engine = sqlalchemy.create_engine(
                    params["connection_string"],
                    connect_args={"connect_timeout": int(config.get("timeout", 15))},
                )
                with engine.connect() as conn:
                    result = conn.execute(sqlalchemy.text(query))
                    columns_retrieved = list(result.keys())
                    rows = result.fetchall()
                    records = [dict(zip(columns_retrieved, row)) for row in rows]
                    delivery_status = "LIVE"
                    status_msg = f"Retrieved {len(records)} live rows from PostgreSQL ({target_desc})."
            except Exception as sql_e:
                logger.warning(f"PostgreSQL SQLAlchemy live connection attempt failed: {sql_e}")
                last_error = sql_e
                # Fallback to direct psycopg2 kwargs connection (immune to URI parsing bugs)
                try:
                    import psycopg2
                    import psycopg2.extras
                    conn = psycopg2.connect(
                        host=params.get("host"),
                        port=params.get("port", 5432),
                        dbname=params.get("database"),
                        user=params.get("username"),
                        password=params.get("password"),
                        sslmode=params.get("ssl_mode", "prefer"),
                        connect_timeout=int(config.get("timeout", 15)),
                    )
                    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                        cur.execute(query)
                        rows = cur.fetchall()
                        records = [dict(r) for r in rows]
                        if records:
                            columns_retrieved = list(records[0].keys())
                        delivery_status = "LIVE"
                        status_msg = f"Retrieved {len(records)} live rows from PostgreSQL via psycopg2 ({target_desc})."
                    conn.close()
                    last_error = None
                except Exception as raw_e:
                    logger.warning(f"PostgreSQL psycopg2 direct connection attempt failed: {raw_e}")
                    last_error = raw_e

            # If live execution failed, raise ValidationException so user sees the real database error
            if last_error is not None:
                raise ValidationException(
                    f"PostgreSQL connection to {params.get('host')}:{params.get('port')}/{params.get('database')} failed: {str(last_error)}"
                )

        # Simulation / Benchmark Data (ONLY when simulation_mode is True)
        if simulation_mode:
            records = self._generate_benchmark_data(config)
            # If specific columns were requested, filter simulated records
            if select_all_columns is False and cols_list:
                filtered = []
                for r in records:
                    item = {}
                    for col in cols_list:
                        clean_col = col.strip('"')
                        if clean_col in r:
                            item[clean_col] = r[clean_col]
                        else:
                            item[clean_col] = f"{clean_col}_val"
                    filtered.append(item)
                records = filtered
            if records:
                columns_retrieved = list(records[0].keys())
            delivery_status = "SIMULATED"
            status_msg = f"PostgreSQL benchmark simulation mode active. Generated {len(records)} test rows for table '{target_desc}'."

        df_out = pd.DataFrame(records) if pd is not None else records
        duration_ms = int((time.time() - start_time) * 1000)

        summary = {
            "title": "PostgreSQL Connector Summary",
            "message": status_msg,
            "status": delivery_status,
            "target": target_desc,
            "records_retrieved": len(records),
            "columns_count": len(columns_retrieved),
            "columns": columns_retrieved,
            "duration_ms": duration_ms,
        }

        metrics = {
            "records_retrieved": len(records),
            "columns_count": len(columns_retrieved),
            "duration_ms": duration_ms,
            "delivery_status": delivery_status,
        }

        return {
            "dataframe": df_out,
            "output": df_out,
            "output_summary": summary,
            "metrics": metrics,
        }

    def inspect(
        self,
        action: str,
        config: Dict[str, Any],
        context: Optional[Any] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Introspects the connected PostgreSQL database to discover:
        - 'test_connection': verifies connectivity and returns database & engine info
        - 'schemas': list of available database schemas
        - 'tables': list of tables/views in a given schema
        - 'columns': list of columns and types in a given table
        Falls back gracefully to realistic simulated metadata when offline.
        """
        action = str(action or "test_connection").strip().lower()
        params = self._resolve_connection_params(config, context)
        timeout = int(config.get("timeout", 10))

        engine = None
        conn = None
        is_live = False
        server_info = ""

        inspect_err = ""
        if not config.get("simulation_mode") and (params.get("connection_string") or params.get("host")):
            try:
                import sqlalchemy
                engine = sqlalchemy.create_engine(
                    params["connection_string"],
                    connect_args={"connect_timeout": timeout},
                )
                with engine.connect() as test_conn:
                    res = test_conn.execute(sqlalchemy.text("SELECT version();")).fetchone()
                    if res:
                        server_info = str(res[0])
                        is_live = True
            except Exception as e:
                logger.debug(f"PostgreSQL live inspection via SQLAlchemy failed: {e}")
                inspect_err = str(e)
                try:
                    import psycopg2
                    conn = psycopg2.connect(
                        host=params.get("host"),
                        port=params.get("port", 5432),
                        dbname=params.get("database"),
                        user=params.get("username"),
                        password=params.get("password"),
                        sslmode=params.get("ssl_mode", "prefer"),
                        connect_timeout=timeout,
                    )
                    with conn.cursor() as cur:
                        cur.execute("SELECT version();")
                        res = cur.fetchone()
                        if res:
                            server_info = str(res[0])
                            is_live = True
                            inspect_err = ""
                except Exception as p_err:
                    logger.debug(f"PostgreSQL live inspection via psycopg2 failed: {p_err}")
                    inspect_err = str(p_err)

        # 1. Action: test_connection
        if action == "test_connection":
            if is_live:
                if conn:
                    conn.close()
                return {
                    "success": True,
                    "connected": True,
                    "status": "LIVE",
                    "server_version": server_info,
                    "database": params["database"],
                    "message": f"Successfully connected to PostgreSQL database '{params['database']}'.",
                }
            else:
                return {
                    "success": False,
                    "connected": False,
                    "status": "UNREACHABLE",
                    "database": params["database"],
                    "message": f"Could not establish connection to PostgreSQL server ({params.get('host')}:{params.get('port')}/{params.get('database')}). Error: {inspect_err or 'Connection timed out or host unreachable'}",
                    "error": inspect_err,
                }

        # 2. Action: schemas
        if action in ("schemas", "get_schemas", "list_schemas"):
            schemas: List[str] = []
            if is_live and engine:
                try:
                    import sqlalchemy
                    with engine.connect() as c:
                        res = c.execute(sqlalchemy.text(
                            "SELECT schema_name FROM information_schema.schemata "
                            "WHERE schema_name NOT IN ('pg_catalog', 'information_schema', 'pg_toast') "
                            "ORDER BY schema_name;"
                        ))
                        schemas = [row[0] for row in res.fetchall()]
                except Exception as ex:
                    logger.warning(f"Failed to query schemas from live DB: {ex}")
            elif is_live and conn:
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT schema_name FROM information_schema.schemata "
                            "WHERE schema_name NOT IN ('pg_catalog', 'information_schema', 'pg_toast') "
                            "ORDER BY schema_name;"
                        )
                        schemas = [row[0] for row in cur.fetchall()]
                except Exception as ex:
                    logger.warning(f"Failed to query schemas from live DB: {ex}")

            if conn:
                conn.close()

            if not schemas:
                schemas = ["public", "analytics", "staging", "reporting", "sales", "retail", "ecommerce", "raw_data", "information_schema"]

            return {
                "success": True,
                "schemas": schemas,
                "count": len(schemas),
                "source": "LIVE" if is_live else "SIMULATED",
            }

        # 3. Action: tables
        if action in ("tables", "get_tables", "list_tables"):
            target_schema = schema_name or config.get("schema_name") or "public"
            tables: List[str] = []
            if is_live and engine:
                try:
                    import sqlalchemy
                    with engine.connect() as c:
                        res = c.execute(
                            sqlalchemy.text(
                                "SELECT table_name FROM information_schema.tables "
                                "WHERE table_schema = :schema AND table_type IN ('BASE TABLE', 'VIEW') "
                                "ORDER BY table_name;"
                            ),
                            {"schema": target_schema},
                        )
                        tables = [row[0] for row in res.fetchall()]
                except Exception as ex:
                    logger.warning(f"Failed to query tables from live DB: {ex}")
            elif is_live and conn:
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT table_name FROM information_schema.tables "
                            "WHERE table_schema = %s AND table_type IN ('BASE TABLE', 'VIEW') "
                            "ORDER BY table_name;",
                            (target_schema,),
                        )
                        tables = [row[0] for row in cur.fetchall()]
                except Exception as ex:
                    logger.warning(f"Failed to query tables from live DB: {ex}")

            if conn:
                conn.close()

            if not tables:
                tables = ["sales", "retail_sales", "customers", "orders", "transactions", "products", "users", "inventory", "invoices", "payments"]

            return {
                "success": True,
                "schema": target_schema,
                "tables": tables,
                "count": len(tables),
                "source": "LIVE" if is_live else "SIMULATED",
            }

        # 4. Action: columns
        if action in ("columns", "get_columns", "list_columns"):
            target_schema = schema_name or config.get("schema_name") or "public"
            target_table = table_name or config.get("table_name") or "customers"
            if "." in target_table:
                parts = target_table.split(".", 1)
                target_schema = parts[0].strip() or target_schema
                target_table = parts[1].strip()

            columns: List[Dict[str, Any]] = []
            if is_live and engine:
                try:
                    import sqlalchemy
                    with engine.connect() as c:
                        res = c.execute(
                            sqlalchemy.text(
                                "SELECT column_name, data_type, is_nullable "
                                "FROM information_schema.columns "
                                "WHERE table_schema = :schema AND table_name = :table "
                                "ORDER BY ordinal_position;"
                            ),
                            {"schema": target_schema, "table": target_table},
                        )
                        columns = [
                            {"name": row[0], "type": row[1], "nullable": row[2] == "YES"}
                            for row in res.fetchall()
                        ]
                except Exception as ex:
                    logger.warning(f"Failed to query columns from live DB: {ex}")
            elif is_live and conn:
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT column_name, data_type, is_nullable "
                            "FROM information_schema.columns "
                            "WHERE table_schema = %s AND table_name = %s "
                            "ORDER BY ordinal_position;",
                            (target_schema, target_table),
                        )
                        columns = [
                            {"name": row[0], "type": row[1], "nullable": row[2] == "YES"}
                            for row in cur.fetchall()
                        ]
                except Exception as ex:
                    logger.warning(f"Failed to query columns from live DB: {ex}")

            if conn:
                conn.close()

            if not columns:
                if target_table.lower() in ("sales", "retail_sales"):
                    columns = [
                        {"name": "invoice_id", "type": "character varying(30)", "nullable": False},
                        {"name": "branch", "type": "character varying(5)", "nullable": True},
                        {"name": "city", "type": "character varying(30)", "nullable": True},
                        {"name": "customer_type", "type": "character varying(30)", "nullable": True},
                        {"name": "gender", "type": "character varying(10)", "nullable": True},
                        {"name": "product_line", "type": "character varying(100)", "nullable": True},
                        {"name": "unit_price", "type": "numeric(10,2)", "nullable": True},
                        {"name": "quantity", "type": "integer", "nullable": True},
                        {"name": "tax_5", "type": "numeric(10,4)", "nullable": True},
                        {"name": "total", "type": "numeric(10,4)", "nullable": True},
                        {"name": "date", "type": "date", "nullable": True},
                        {"name": "time", "type": "time without time zone", "nullable": True},
                        {"name": "payment", "type": "character varying(30)", "nullable": True},
                        {"name": "cogs", "type": "numeric(10,2)", "nullable": True},
                        {"name": "gross_margin_percentage", "type": "numeric(11,9)", "nullable": True},
                        {"name": "gross_income", "type": "numeric(10,4)", "nullable": True},
                        {"name": "rating", "type": "numeric(3,1)", "nullable": True},
                    ]
                else:
                    mock_config = {**config, "table_name": target_table}
                    mock_recs = self._generate_benchmark_data(mock_config)
                    if mock_recs:
                        first = mock_recs[0]
                        for k, v in first.items():
                            dtype = "numeric" if isinstance(v, (int, float)) else "text"
                            columns.append({"name": k, "type": dtype, "nullable": True})
                    else:
                        columns = [
                            {"name": "id", "type": "integer", "nullable": False},
                            {"name": "name", "type": "character varying", "nullable": True},
                            {"name": "status", "type": "character varying", "nullable": True},
                            {"name": "created_at", "type": "timestamp with time zone", "nullable": True},
                        ]

            return {
                "success": True,
                "schema": target_schema,
                "table": target_table,
                "columns": columns,
                "count": len(columns),
                "source": "LIVE" if is_live else "SIMULATED",
            }

        if conn:
            conn.close()

        return {
            "success": False,
            "error": f"Unknown inspect action '{action}'. Supported actions: 'test_connection', 'schemas', 'tables', 'columns'.",
        }

    def to_code(self, config: Dict[str, Any]) -> str:
        read_mode = config.get("read_mode", "table")
        params = self._resolve_connection_params(config)
        host = params.get("host", "localhost")
        port = params.get("port", 5432)
        database = params.get("database", "postgres")
        username = params.get("username", "postgres")
        conn_str = params.get("connection_string", "")
        limit = config.get("limit", 1000)

        if read_mode == "table":
            schema_name = config.get("schema_name", "public") or "public"
            table_name = config.get("table_name", "customers")
            if "." in table_name:
                parts = table_name.split(".", 1)
                schema_name = parts[0].strip() or schema_name
                table_name = parts[1].strip()

            select_all = config.get("select_all_columns", True)
            raw_cols = config.get("columns", "")
            cols_list = []
            if isinstance(raw_cols, list):
                cols_list = [str(c).strip() for c in raw_cols if str(c).strip() and str(c).strip() != "*"]
            elif isinstance(raw_cols, str) and raw_cols.strip():
                cols_list = [c.strip() for c in raw_cols.split(",") if c.strip() and c.strip() != "*"]

            if select_all is False and cols_list:
                cols_str = ", ".join(f'"{c}"' for c in cols_list)
            else:
                cols_str = "*"

            where_clause = str(config.get("where_clause", "")).strip()
            where_sql = f" WHERE {where_clause}" if where_clause else ""
            sql_stmt = f'SELECT {cols_str} FROM "{schema_name}"."{table_name}"{where_sql} LIMIT {limit}'
        else:
            sql_stmt = config.get("query", f"SELECT * FROM public.customers LIMIT {limit}")

        return (
            f"# Recipe: PostgreSQL Connector\n"
            f"# ID: {self.recipe_id}\n"
            f"import os\n"
            f"import pandas as pd\n"
            f"import psycopg2\n\n"
            f"# Connect using DSN or connection parameters\n"
            f"conn = psycopg2.connect(\n"
            f"    dsn=os.getenv('PG_CONNECTION_STRING', '{conn_str}')\n"
            f") if os.getenv('PG_CONNECTION_STRING', '{conn_str}') else psycopg2.connect(\n"
            f"    host=os.getenv('PGHOST', '{host}'),\n"
            f"    port=int(os.getenv('PGPORT', {port})),\n"
            f"    dbname=os.getenv('PGDATABASE', '{database}'),\n"
            f"    user=os.getenv('PGUSER', '{username}'),\n"
            f"    password=os.getenv('PGPASSWORD', '')\n"
            f")\n\n"
            f"query = '''{sql_stmt}'''\n"
            f"df = pd.read_sql_query(query, conn)\n"
            f"conn.close()\n"
            f"print(f'Retrieved {{len(df)}} rows from PostgreSQL.')\n"
        )
