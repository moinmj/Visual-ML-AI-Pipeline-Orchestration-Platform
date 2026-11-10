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
                # ── 1. Connection Configuration ──
                "connection_mode": {
                    "type": "string",
                    "title": "Connection Mode",
                    "input_type": "dropdown",
                    "enum": ["host_port", "connection_string", "credential"],
                    "default": "host_port",
                    "description": "Choose how to connect: standard host/port parameters, full connection string (URI), or stored credential.",
                },
                "credential_id": {
                    "type": "string",
                    "title": "Connected PostgreSQL Credential",
                    "input_type": "credential_dropdown",
                    "provider": "postgresql",
                    "description": "Select a stored credential from platform settings.",
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
                    "description": "Name of the target PostgreSQL database.",
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

                # ── 2. Read Mode & Target Selection ──
                "read_mode": {
                    "type": "string",
                    "title": "Read Mode",
                    "input_type": "dropdown",
                    "enum": ["table", "query"],
                    "default": "table",
                    "description": "Read from tables (specifying schema, tables, columns) or execute a custom SQL query.",
                },
                # Table Mode Fields
                "schema_name": {
                    "type": "string",
                    "title": "Schema Name",
                    "input_type": "text",
                    "default": "public",
                    "description": "PostgreSQL schema name (default 'public'). Discovered dynamically from the database upon connection.",
                },
                "select_all_tables": {
                    "type": "boolean",
                    "title": "Select All Tables in Schema (*)",
                    "input_type": "checkbox",
                    "default": False,
                    "description": "If enabled or if no table is specified, retrieves all tables in the selected schema.",
                },
                "tables": {
                    "type": "array",
                    "items": {"type": "string"},
                    "title": "Table(s) to Read",
                    "input_type": "tags",
                    "description": "Specify one or multiple tables to read (e.g. ['sales', 'retail_sales']). Leave empty or enable 'Select All Tables' to read all tables.",
                },
                "table_name": {
                    "type": "string",
                    "title": "Table Name",
                    "input_type": "text",
                    "default": "sales",
                    "description": "Single table to read (e.g. 'sales'). Ignored if 'Table(s) to Read' or 'Select All Tables' is specified.",
                },
                "select_all_columns": {
                    "type": "boolean",
                    "title": "Select All Columns (*)",
                    "input_type": "checkbox",
                    "default": True,
                    "description": "If enabled and no specific columns are entered in 'Columns to Select', retrieves all columns ('*').",
                },
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "title": "Columns to Select",
                    "input_type": "tags",
                    "description": "Specify one or more column names to select (e.g. ['invoice_id', 'branch']). When specified, ONLY these columns are retrieved.",
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
                # Query Mode Field
                "query": {
                    "type": "string",
                    "title": "SQL Query",
                    "input_type": "textarea",
                    "default": "SELECT * FROM public.sales LIMIT 100",
                    "description": "Read-only SQL query to execute when Read Mode is 'query' (SELECT, WITH, SHOW, etc.).",
                },

                # ── 3. Limits & Pagination ──
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

                # ── 4. Demo / Benchmark ──
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
            select_all_tables = bool(config.get("select_all_tables", False))
            table_name = str(config.get("table_name", "")).strip()
            tables = config.get("tables", [])
            has_tables = bool(
                (table_name and table_name != "*")
                or (isinstance(tables, list) and len(tables) > 0)
                or (isinstance(tables, str) and tables.strip() and tables.strip() != "*")
            )
            if not select_all_tables and not has_tables:
                errors.append("Table name is required when read_mode is 'table' (or enable 'Select All Tables').")
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

        # 1. Resolve columns
        cols_raw = config.get("columns", [])
        cols_list: List[str] = []
        if isinstance(cols_raw, list):
            cols_list = [str(c).strip().strip('"').strip("'") for c in cols_raw if str(c).strip() and str(c).strip() != "*"]
        elif isinstance(cols_raw, str) and cols_raw.strip():
            cols_list = [c.strip().strip('"').strip("'") for c in cols_raw.split(",") if c.strip() and c.strip() != "*"]

        # If user explicitly specified one or more columns, ALWAYS query ONLY those columns!
        if cols_list:
            cols_str = ", ".join(f'"{c}"' if not c.startswith('"') else c for c in cols_list)
            is_select_all_cols = False
        else:
            cols_str = "*"
            is_select_all_cols = True

        # 2. Resolve tables
        schema_name = str(config.get("schema_name", "public")).strip() or "public"
        select_all_tables = bool(config.get("select_all_tables", False))

        target_tables: List[str] = []
        raw_tables = config.get("tables")
        if isinstance(raw_tables, list):
            target_tables = [str(t).strip().strip('"').strip("'") for t in raw_tables if str(t).strip() and str(t).strip() != "*"]
        elif isinstance(raw_tables, str) and raw_tables.strip():
            target_tables = [t.strip().strip('"').strip("'") for t in raw_tables.split(",") if t.strip() and t.strip() != "*"]

        single_tbl = str(config.get("table_name", "")).strip().strip('"').strip("'")
        if single_tbl and single_tbl != "*":
            if "," in single_tbl:
                for t in single_tbl.split(","):
                    clean_t = t.strip().strip('"').strip("'")
                    if clean_t and clean_t not in target_tables:
                        target_tables.append(clean_t)
            elif single_tbl not in target_tables:
                target_tables.append(single_tbl)

        # Handle dot-delimited schema and table (e.g. "finance.invoices")
        if len(target_tables) == 1 and "." in target_tables[0]:
            parts = target_tables[0].split(".", 1)
            schema_name = parts[0].strip() or schema_name
            target_tables[0] = parts[1].strip()

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

        records: List[Dict[str, Any]] = []
        delivery_status = "SIMULATED"
        status_msg = ""
        columns_retrieved: List[str] = []
        per_table_data: Dict[str, List[Dict[str, Any]]] = {}

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
                    if read_mode == "table":
                        # If select_all_tables or no table was provided, discover all tables in schema
                        if select_all_tables or not target_tables:
                            res_tables = conn.execute(
                                sqlalchemy.text(
                                    "SELECT table_name FROM information_schema.tables "
                                    "WHERE table_schema = :schema AND table_type IN ('BASE TABLE', 'VIEW') "
                                    "ORDER BY table_name;"
                                ),
                                {"schema": schema_name},
                            )
                            target_tables = [r[0] for r in res_tables.fetchall()]
                            if not target_tables:
                                raise ValidationException(
                                    f"No tables found in schema '{schema_name}' for database '{params.get('database')}'."
                                )

                        is_multi = len(target_tables) > 1
                        for tbl in target_tables:
                            t_schema = schema_name
                            t_name = tbl
                            if "." in t_name:
                                p = t_name.split(".", 1)
                                t_schema = p[0].strip() or t_schema
                                t_name = p[1].strip()

                            q = f'SELECT {cols_str} FROM "{t_schema}"."{t_name}"{where_sql}{order_sql} LIMIT {limit}{offset_sql}'
                            res = conn.execute(sqlalchemy.text(q))
                            t_cols = list(res.keys())
                            if not columns_retrieved:
                                columns_retrieved = list(t_cols)
                            rows = [dict(zip(t_cols, r)) for r in res.fetchall()]
                            if is_multi:
                                for r in rows:
                                    r["_table"] = t_name
                            per_table_data[t_name] = rows
                            records.extend(rows)

                        delivery_status = "LIVE"
                        if is_multi:
                            if "_table" not in columns_retrieved:
                                columns_retrieved.insert(0, "_table")
                            status_msg = f"Retrieved {len(records)} live rows across {len(target_tables)} tables ({', '.join(target_tables)}) from schema '{schema_name}'."
                        else:
                            status_msg = f"Retrieved {len(records)} live rows from PostgreSQL ({schema_name}.{target_tables[0]})."
                    else:
                        query = str(config.get("query", "SELECT * FROM public.sales")).strip()
                        if "limit" not in query.lower():
                            query = f"{query} LIMIT {limit}"
                        res = conn.execute(sqlalchemy.text(query))
                        columns_retrieved = list(res.keys())
                        records = [dict(zip(columns_retrieved, r)) for r in res.fetchall()]
                        delivery_status = "LIVE"
                        status_msg = f"Retrieved {len(records)} live rows from custom query."
            except Exception as sql_e:
                logger.warning(f"PostgreSQL SQLAlchemy live execution failed: {sql_e}")
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
                    records = []
                    per_table_data = {}
                    columns_retrieved = []
                    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                        if read_mode == "table":
                            if select_all_tables or not target_tables:
                                cur.execute(
                                    "SELECT table_name FROM information_schema.tables "
                                    "WHERE table_schema = %s AND table_type IN ('BASE TABLE', 'VIEW') "
                                    "ORDER BY table_name;",
                                    (schema_name,),
                                )
                                target_tables = [r["table_name"] for r in cur.fetchall()]
                                if not target_tables:
                                    raise ValidationException(
                                        f"No tables found in schema '{schema_name}' for database '{params.get('database')}'."
                                    )

                            is_multi = len(target_tables) > 1
                            for tbl in target_tables:
                                t_schema = schema_name
                                t_name = tbl
                                if "." in t_name:
                                    p = t_name.split(".", 1)
                                    t_schema = p[0].strip() or t_schema
                                    t_name = p[1].strip()

                                q = f'SELECT {cols_str} FROM "{t_schema}"."{t_name}"{where_sql}{order_sql} LIMIT {limit}{offset_sql}'
                                cur.execute(q)
                                rows = [dict(r) for r in cur.fetchall()]
                                if rows and not columns_retrieved:
                                    columns_retrieved = list(rows[0].keys())
                                if is_multi:
                                    for r in rows:
                                        r["_table"] = t_name
                                per_table_data[t_name] = rows
                                records.extend(rows)

                            delivery_status = "LIVE"
                            if is_multi:
                                if "_table" not in columns_retrieved:
                                    columns_retrieved.insert(0, "_table")
                                status_msg = f"Retrieved {len(records)} live rows across {len(target_tables)} tables via psycopg2 ({', '.join(target_tables)})."
                            else:
                                status_msg = f"Retrieved {len(records)} live rows from PostgreSQL via psycopg2 ({schema_name}.{target_tables[0]})."
                        else:
                            query = str(config.get("query", "SELECT * FROM public.sales")).strip()
                            if "limit" not in query.lower():
                                query = f"{query} LIMIT {limit}"
                            cur.execute(query)
                            records = [dict(r) for r in cur.fetchall()]
                            if records:
                                columns_retrieved = list(records[0].keys())
                            delivery_status = "LIVE"
                            status_msg = f"Retrieved {len(records)} live rows via psycopg2."
                    conn.close()
                    last_error = None
                except Exception as raw_e:
                    logger.warning(f"PostgreSQL psycopg2 direct execution failed: {raw_e}")
                    last_error = raw_e

            if last_error is not None:
                raise ValidationException(
                    f"PostgreSQL connection to {params.get('host')}:{params.get('port')}/{params.get('database')} failed: {str(last_error)}"
                )

        # Simulation / Benchmark Data (ONLY when simulation_mode is True)
        if simulation_mode:
            if not target_tables:
                target_tables = ["sales"]
            is_multi = len(target_tables) > 1
            for tbl in target_tables:
                sim_recs = self._generate_benchmark_data(config)
                if cols_list:
                    filtered = []
                    for r in sim_recs:
                        item = {}
                        for c in cols_list:
                            clean_c = c.strip('"')
                            item[clean_c] = r.get(clean_c, f"{clean_c}_val")
                        filtered.append(item)
                    sim_recs = filtered
                if is_multi:
                    for r in sim_recs:
                        r["_table"] = tbl
                per_table_data[tbl] = sim_recs
                records.extend(sim_recs)

            if is_multi:
                columns_retrieved = ["_table"] + (cols_list if cols_list else (list(records[0].keys()) if records else []))
                target_desc = f"{schema_name} ({', '.join(target_tables)})"
            else:
                target_desc = f"{schema_name}.{target_tables[0]}"
                columns_retrieved = cols_list if cols_list else (list(records[0].keys()) if records else [])

            delivery_status = "SIMULATED"
            status_msg = f"PostgreSQL benchmark simulation mode active. Generated {len(records)} test rows for table '{target_desc}'."

        df_out = pd.DataFrame(records, columns=columns_retrieved if (records and columns_retrieved) else None) if pd is not None else records
        duration_ms = int((time.time() - start_time) * 1000)

        per_table_dfs = {k: pd.DataFrame(v) for k, v in per_table_data.items()} if pd is not None else per_table_data
        target_name = target_desc if 'target_desc' in locals() else (f"{schema_name}.{target_tables[0]}" if target_tables else "PostgreSQL")

        summary = {
            "title": "PostgreSQL Connector Summary",
            "message": status_msg,
            "status": delivery_status,
            "target": target_name,
            "database": params.get("database"),
            "schema": schema_name,
            "tables_retrieved": target_tables,
            "tables_count": len(target_tables),
            "records_retrieved": len(records),
            "columns_count": len(columns_retrieved),
            "columns": columns_retrieved,
            "duration_ms": duration_ms,
        }

        metrics = {
            "records_retrieved": len(records),
            "tables_count": len(target_tables),
            "columns_count": len(columns_retrieved),
            "duration_ms": duration_ms,
            "delivery_status": delivery_status,
        }

        return {
            "dataframe": df_out,
            "output": df_out,
            "tables": per_table_dfs,
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
        Introspects the connected PostgreSQL database in real-time:
        - 'test_connection': verifies connectivity and returns database & server engine info
        - 'databases': list of all available databases on the host
        - 'schemas': list of available schemas in the target database
        - 'tables': list of tables/views in a given schema
        - 'columns': list of columns, data types, nullability, and primary key status
        """
        action = str(action or "test_connection").strip().lower()
        params = self._resolve_connection_params(config, context)
        timeout = int(config.get("timeout", 10))
        simulation_mode = bool(config.get("simulation_mode", False))

        engine = None
        conn = None
        is_live = False
        server_info = ""
        inspect_err = ""

        if not simulation_mode and (params.get("connection_string") or params.get("host")):
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
            elif simulation_mode:
                return {
                    "success": True,
                    "connected": True,
                    "status": "SIMULATED",
                    "server_version": "PostgreSQL 16.2 (Debian 16.2-1.pgdg120+2) on x86_64 [SIMULATION]",
                    "database": params["database"],
                    "message": "PostgreSQL simulation mode connection active.",
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

        # 2. Action: databases
        if action in ("databases", "get_databases", "list_databases"):
            if not is_live and not simulation_mode:
                return {
                    "success": False,
                    "databases": [],
                    "count": 0,
                    "error": inspect_err or "Connection could not be established.",
                    "message": f"Failed to connect to PostgreSQL server: {inspect_err or 'Host unreachable'}",
                    "source": "ERROR",
                }

            databases: List[str] = []
            if is_live and engine:
                try:
                    import sqlalchemy
                    with engine.connect() as c:
                        res = c.execute(sqlalchemy.text(
                            "SELECT datname FROM pg_database WHERE datistemplate = false ORDER BY datname;"
                        ))
                        databases = [row[0] for row in res.fetchall()]
                except Exception as ex:
                    logger.warning(f"Failed to query databases via SQLAlchemy: {ex}")
            elif is_live and conn:
                try:
                    with conn.cursor() as cur:
                        cur.execute("SELECT datname FROM pg_database WHERE datistemplate = false ORDER BY datname;")
                        databases = [row[0] for row in cur.fetchall()]
                except Exception as ex:
                    logger.warning(f"Failed to query databases via psycopg2: {ex}")

            if conn:
                conn.close()

            if not databases and simulation_mode:
                databases = ["walmart", "postgres", "analytics_db", "staging_db"]

            return {
                "success": True,
                "databases": databases,
                "count": len(databases),
                "source": "LIVE" if is_live else "SIMULATED",
            }

        # 3. Action: schemas
        if action in ("schemas", "get_schemas", "list_schemas"):
            if not is_live and not simulation_mode:
                return {
                    "success": False,
                    "schemas": [],
                    "count": 0,
                    "error": inspect_err or "Connection could not be established.",
                    "message": f"Failed to connect to PostgreSQL server: {inspect_err or 'Host unreachable'}",
                    "source": "ERROR",
                }

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

            if not schemas and simulation_mode:
                schemas = ["public", "analytics", "staging", "reporting", "sales", "retail"]

            return {
                "success": True,
                "schemas": schemas,
                "count": len(schemas),
                "database": params.get("database"),
                "source": "LIVE" if is_live else "SIMULATED",
            }

        # 4. Action: tables
        if action in ("tables", "get_tables", "list_tables"):
            if not is_live and not simulation_mode:
                return {
                    "success": False,
                    "tables": [],
                    "count": 0,
                    "error": inspect_err or "Connection could not be established.",
                    "message": f"Failed to connect to PostgreSQL server: {inspect_err or 'Host unreachable'}",
                    "source": "ERROR",
                }

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

            if not tables and simulation_mode:
                tables = ["sales", "retail_sales", "customers", "orders", "transactions", "products", "users"]

            return {
                "success": True,
                "schema": target_schema,
                "tables": tables,
                "count": len(tables),
                "database": params.get("database"),
                "source": "LIVE" if is_live else "SIMULATED",
            }

        # 5. Action: columns
        if action in ("columns", "get_columns", "list_columns"):
            if not is_live and not simulation_mode:
                return {
                    "success": False,
                    "columns": [],
                    "count": 0,
                    "error": inspect_err or "Connection could not be established.",
                    "message": f"Failed to connect to PostgreSQL server: {inspect_err or 'Host unreachable'}",
                    "source": "ERROR",
                }

            target_schema = schema_name or config.get("schema_name") or "public"
            target_table = table_name or config.get("table_name") or "sales"
            if "." in target_table:
                parts = target_table.split(".", 1)
                target_schema = parts[0].strip() or target_schema
                target_table = parts[1].strip()

            col_query = """
                SELECT c.column_name, c.data_type, c.is_nullable,
                       CASE WHEN pk.column_name IS NOT NULL THEN true ELSE false END AS is_pk
                FROM information_schema.columns c
                LEFT JOIN (
                    SELECT kcu.column_name, kcu.table_schema, kcu.table_name
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.key_column_usage kcu
                      ON tc.constraint_name = kcu.constraint_name
                     AND tc.table_schema = kcu.table_schema
                    WHERE tc.constraint_type = 'PRIMARY KEY'
                ) pk ON c.column_name = pk.column_name AND c.table_schema = pk.table_schema AND c.table_name = pk.table_name
                WHERE c.table_schema = :schema AND c.table_name = :table
                ORDER BY c.ordinal_position;
            """
            col_query_psycopg = col_query.replace(":schema", "%s").replace(":table", "%s")

            columns: List[Dict[str, Any]] = []
            if is_live and engine:
                try:
                    import sqlalchemy
                    with engine.connect() as c:
                        res = c.execute(sqlalchemy.text(col_query), {"schema": target_schema, "table": target_table})
                        columns = [
                            {"name": row[0], "type": row[1], "nullable": row[2] == "YES", "primary_key": bool(row[3])}
                            for row in res.fetchall()
                        ]
                except Exception as ex:
                    logger.warning(f"Failed to query columns with PK from live DB: {ex}")
                    # Fallback to simple query
                    try:
                        with engine.connect() as c:
                            res = c.execute(
                                sqlalchemy.text(
                                    "SELECT column_name, data_type, is_nullable FROM information_schema.columns "
                                    "WHERE table_schema = :schema AND table_name = :table ORDER BY ordinal_position;"
                                ),
                                {"schema": target_schema, "table": target_table},
                            )
                            columns = [
                                {"name": row[0], "type": row[1], "nullable": row[2] == "YES", "primary_key": False}
                                for row in res.fetchall()
                            ]
                    except Exception as ex2:
                        logger.warning(f"Fallback columns query also failed: {ex2}")
            elif is_live and conn:
                try:
                    with conn.cursor() as cur:
                        cur.execute(col_query_psycopg, (target_schema, target_table))
                        columns = [
                            {"name": row[0], "type": row[1], "nullable": row[2] == "YES", "primary_key": bool(row[3])}
                            for row in cur.fetchall()
                        ]
                except Exception as ex:
                    logger.warning(f"Failed to query columns with PK via psycopg2: {ex}")
                    try:
                        with conn.cursor() as cur:
                            cur.execute(
                                "SELECT column_name, data_type, is_nullable FROM information_schema.columns "
                                "WHERE table_schema = %s AND table_name = %s ORDER BY ordinal_position;",
                                (target_schema, target_table),
                            )
                            columns = [
                                {"name": row[0], "type": row[1], "nullable": row[2] == "YES", "primary_key": False}
                                for row in cur.fetchall()
                            ]
                    except Exception as ex2:
                        logger.warning(f"Fallback columns query also failed: {ex2}")

            if conn:
                conn.close()

            if not columns and simulation_mode:
                if "customer" in target_table.lower() or "user" in target_table.lower():
                    columns = [
                        {"name": "id", "type": "integer", "nullable": False, "primary_key": True},
                        {"name": "name", "type": "character varying(255)", "nullable": False, "primary_key": False},
                        {"name": "email", "type": "character varying(255)", "nullable": True, "primary_key": False},
                        {"name": "status", "type": "character varying(50)", "nullable": True, "primary_key": False},
                        {"name": "created_at", "type": "timestamp with time zone", "nullable": True, "primary_key": False},
                    ]
                else:
                    columns = [
                        {"name": "invoice_id", "type": "character varying(30)", "nullable": False, "primary_key": True},
                        {"name": "branch", "type": "character varying(5)", "nullable": True, "primary_key": False},
                        {"name": "city", "type": "character varying(30)", "nullable": True, "primary_key": False},
                        {"name": "customer_type", "type": "character varying(30)", "nullable": True, "primary_key": False},
                        {"name": "gender", "type": "character varying(10)", "nullable": True, "primary_key": False},
                        {"name": "product_line", "type": "character varying(100)", "nullable": True, "primary_key": False},
                        {"name": "unit_price", "type": "numeric(10,2)", "nullable": True, "primary_key": False},
                        {"name": "quantity", "type": "integer", "nullable": True, "primary_key": False},
                        {"name": "tax_5", "type": "numeric(10,4)", "nullable": True, "primary_key": False},
                        {"name": "total", "type": "numeric(10,4)", "nullable": True, "primary_key": False},
                        {"name": "date", "type": "date", "nullable": True, "primary_key": False},
                    ]

            return {
                "success": True,
                "schema": target_schema,
                "table": target_table,
                "columns": columns,
                "count": len(columns),
                "database": params.get("database"),
                "source": "LIVE" if is_live else "SIMULATED",
            }

        return {
            "success": False,
            "error": f"Unknown inspection action '{action}'. Supported: 'test_connection', 'databases', 'schemas', 'tables', 'columns'.",
            "message": f"Unknown inspection action '{action}'.",
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
