"""
MySQL Connector Recipe (Ingestion & Orchestration -> Database & Warehouse Connectors).

Reads tables or query results from a MySQL / MariaDB database into the pipeline as a DataFrame.
Supports host/port, connection string, or saved credentials, database selection,
multi-table and single table selection, column subset selection, WHERE filters, ORDER BY,
charset encoding, real-time dynamic database inspection (test connection, databases, tables, columns),
benchmark simulation fallback, and code generation.
"""

import re
import os
import time
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse, unquote, parse_qs, quote_plus

try:
    import pandas as pd
except ImportError:
    pd = None

from backend.app.recipes.base.recipe import BaseRecipe, RecipePort
from backend.app.core.logging import logger
from backend.app.core.config import settings
from backend.app.core.exceptions import ValidationException


class MySQLConnectorRecipe(BaseRecipe):
    recipe_id = "mysql_connector"
    name = "MySQL Connector"
    version = "1.1.0"
    category = "ingestion"
    group = "Ingestion & Orchestration"
    subgroup = "Database & Warehouse Connectors"
    description = "Reads tables or query results from a MySQL / MariaDB database."
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
            label="MySQL Dataset",
            type="dataframe",
            description="DataFrame containing retrieved MySQL records"
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                # ── 1. Connection Mode & Server Credentials ──
                "connection_mode": {
                    "type": "string",
                    "title": "Connection Mode",
                    "input_type": "dropdown",
                    "enum": ["host_port", "connection_string", "credential"],
                    "default": "host_port",
                    "description": "Choose connection method: standard host/port parameters, connection URI, or stored credential."
                },
                "credential_id": {
                    "type": "string",
                    "title": "Connected MySQL Credential",
                    "input_type": "credential_dropdown",
                    "provider": "mysql",
                    "description": "Select a stored MySQL credential from platform settings."
                },
                "host": {
                    "type": "string",
                    "title": "Host",
                    "input_type": "text",
                    "default": "localhost",
                    "description": "MySQL / MariaDB server hostname or IP address (used in 'host_port' mode)."
                },
                "port": {
                    "type": "integer",
                    "title": "Port",
                    "input_type": "number",
                    "default": 3306,
                    "description": "MySQL server port (default 3306)."
                },
                "username": {
                    "type": "string",
                    "title": "Username",
                    "input_type": "text",
                    "default": "root",
                    "description": "MySQL user role."
                },
                "password": {
                    "type": "string",
                    "title": "Password",
                    "input_type": "password",
                    "default": "",
                    "description": "MySQL user password."
                },
                "connection_string": {
                    "type": "string",
                    "title": "Connection String (URI)",
                    "input_type": "text",
                    "default": "",
                    "description": "Full MySQL connection URI e.g. mysql+pymysql://user:password@localhost:3306/dbname"
                },
                "charset": {
                    "type": "string",
                    "title": "Character Set",
                    "input_type": "dropdown",
                    "enum": ["utf8mb4", "utf8", "latin1"],
                    "default": "utf8mb4",
                    "description": "Connection character set (default 'utf8mb4')."
                },
                "timeout": {
                    "type": "integer",
                    "title": "Timeout (seconds)",
                    "input_type": "number",
                    "default": 15,
                    "description": "Connection and query timeout in seconds."
                },

                # ── 2. Read Mode & Cascading Data Exploration ──
                "read_mode": {
                    "type": "string",
                    "title": "Read Mode",
                    "input_type": "dropdown",
                    "enum": ["table", "query"],
                    "default": "table",
                    "description": "Read from tables (specifying database, tables, columns) or execute a custom SQL query."
                },
                "database": {
                    "type": "string",
                    "title": "Database Name",
                    "input_type": "text",
                    "default": "shop_db",
                    "description": "Name of the target MySQL database / schema. Discovered dynamically from the server or entered manually."
                },
                "select_all_tables": {
                    "type": "boolean",
                    "title": "Select All Tables in Database (*)",
                    "input_type": "checkbox",
                    "default": False,
                    "description": "If enabled or if no table is specified, retrieves all tables in the selected database."
                },
                "tables": {
                    "type": "array",
                    "title": "Table(s) to Read",
                    "input_type": "tag_input",
                    "items": {"type": "string"},
                    "default": [],
                    "description": "Specify one or multiple tables to read (e.g. ['Businesses', 'BookingServices']). Leave empty or enable 'Select All Tables' to read all tables."
                },
                "table_name": {
                    "type": "string",
                    "title": "Table Name",
                    "input_type": "text",
                    "default": "orders",
                    "description": "Single table to read (e.g. 'Businesses', 'orders'). Ignored if 'Table(s) to Read' or 'Select All Tables' is specified."
                },
                "select_all_columns": {
                    "type": "boolean",
                    "title": "Select All Columns (*)",
                    "input_type": "checkbox",
                    "default": True,
                    "description": "If enabled and no specific columns are entered in 'Columns to Select', retrieves all columns ('*')."
                },
                "columns": {
                    "type": "array",
                    "title": "Columns to Select",
                    "input_type": "tag_input",
                    "items": {"type": "string"},
                    "default": [],
                    "description": "Specify one or more column names to select (e.g. ['Id', 'Name']). When specified, ONLY these columns are retrieved."
                },
                "where_clause": {
                    "type": "string",
                    "title": "Filter Condition (WHERE)",
                    "input_type": "text",
                    "default": "",
                    "description": "Optional SQL WHERE filter expression without 'WHERE' keyword (e.g. status = 'COMPLETED' AND amount > 0)."
                },
                "order_by": {
                    "type": "string",
                    "title": "Sort Order (ORDER BY)",
                    "input_type": "text",
                    "default": "",
                    "description": "Optional SQL sort expression without 'ORDER BY' keyword (e.g. created_at DESC)."
                },
                "query": {
                    "type": "string",
                    "title": "SQL Query",
                    "input_type": "textarea",
                    "default": "SELECT * FROM orders WHERE status = 'COMPLETED' LIMIT 100",
                    "description": "Read-only SQL query to execute (SELECT, WITH, SHOW, etc.)."
                },
                "limit": {
                    "type": "integer",
                    "title": "Row Limit",
                    "input_type": "number",
                    "default": 1000,
                    "description": "Maximum number of rows to retrieve (1 to 100000)."
                },
                "offset": {
                    "type": "integer",
                    "title": "Row Offset",
                    "input_type": "number",
                    "default": 0,
                    "description": "Number of initial rows to skip for pagination."
                },
                "simulation_mode": {
                    "type": "boolean",
                    "title": "Simulation / Demo Mode",
                    "input_type": "checkbox",
                    "default": False,
                    "description": "Generate realistic benchmark dataset without connecting to an external database."
                }
            },
            "required": ["read_mode"],
            "templates": [
                {
                    "label": "Read Orders Table",
                    "config": {
                        "read_mode": "table",
                        "table_name": "orders",
                        "limit": 500
                    }
                },
                {
                    "label": "Custom Aggregation Query",
                    "config": {
                        "read_mode": "query",
                        "query": "SELECT category, COUNT(*) AS item_count, AVG(price) AS avg_price FROM products GROUP BY category ORDER BY item_count DESC"
                    }
                }
            ]
        }

    def _strip_comments(self, sql: str) -> str:
        cleaned = re.sub(r"--.*?(\n|$)", " ", sql or "")
        cleaned = re.sub(r"/\*.*?\*/", " ", cleaned, flags=re.DOTALL)
        return cleaned.strip()

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        errors = []
        read_mode = str(config.get("read_mode", "table")).strip().lower()

        # Validate Port
        port = config.get("port")
        if port is not None:
            try:
                p = int(port)
                if p < 1 or p > 65535:
                    errors.append(f"Invalid MySQL port '{port}'. Must be between 1 and 65535.")
            except (ValueError, TypeError):
                errors.append(f"Port '{port}' must be a valid integer.")

        # Validate Read Mode Requirements
        if read_mode == "table":
            table_name = str(config.get("table_name", "")).strip()
            tables = config.get("tables", [])
            select_all_tables = bool(config.get("select_all_tables", False))
            if not table_name and not tables and not select_all_tables:
                errors.append("Table name is required when read_mode is 'table' (or specify 'Table(s) to Read' or enable 'Select All Tables').")
        elif read_mode == "query":
            q = (config.get("query") or "").strip()
            if not q:
                errors.append("SQL query is required when read_mode is 'query'.")
            else:
                cleaned = self._strip_comments(q).lower()
                first_word = cleaned.split()[0] if cleaned.split() else ""
                if first_word not in ("select", "with", "show", "describe", "explain"):
                    errors.append(f"Only read-only queries (SELECT, WITH, SHOW, DESCRIBE, EXPLAIN) are permitted. Found '{first_word}'.")

                # Disallow destructive keywords
                disallowed = ["drop", "truncate", "delete", "insert", "update", "alter", "create", "grant", "revoke"]
                for kw in disallowed:
                    if re.search(rf"\b{kw}\b", cleaned):
                        errors.append(f"Disallowed DDL/DML keyword '{kw.upper()}' detected in MySQL query.")
                        break

        # Validate Connection mode requirements if not simulated
        conn_mode = str(config.get("connection_mode", "host_port")).strip().lower()
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
        mode = str(config.get("connection_mode", "host_port")).strip().lower()
        credential_id = str(config.get("credential_id", "")).strip()

        host = str(config.get("host", "localhost")).strip()
        port = int(config.get("port", 3306))
        database = str(config.get("database", "shop_db")).strip()
        username = str(config.get("username", "root")).strip()
        password = str(config.get("password", "")).strip()
        conn_str = str(config.get("connection_string", "")).strip()
        charset = str(config.get("charset", "utf8mb4")).strip()

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

        # Environment variable overrides if not set
        if not password:
            password = os.getenv("MYSQL_PASSWORD", os.getenv("MYSQL_ROOT_PASSWORD", ""))
        if not username:
            username = os.getenv("MYSQL_USER", "root")

        # Parse connection string if supplied
        if mode == "connection_string" or (conn_str and not host):
            try:
                parsed = urlparse(conn_str)
                if parsed.hostname:
                    host = parsed.hostname
                if parsed.port:
                    port = parsed.port
                if parsed.path:
                    clean_path = parsed.path.lstrip("/").split("?")[0]
                    if clean_path:
                        database = clean_path
                if parsed.username:
                    username = unquote(parsed.username)
                if parsed.password:
                    password = unquote(parsed.password)
                if parsed.query:
                    qs = parse_qs(parsed.query)
                    if "charset" in qs and qs["charset"]:
                        charset = qs["charset"][0]
            except Exception as parse_e:
                logger.warning(f"Failed to parse MySQL connection_string: {parse_e}")
        elif host and database:
            quoted_user = quote_plus(str(username)) if username else ""
            quoted_pass = quote_plus(str(password)) if password else ""
            auth_part = f"{quoted_user}:{quoted_pass}@" if (quoted_user or quoted_pass) else ""
            conn_str = f"mysql+pymysql://{auth_part}{host}:{port}/{database}?charset={charset}"

        return {
            "mode": mode,
            "host": host,
            "port": port,
            "database": database,
            "username": username,
            "password": password,
            "connection_string": conn_str,
            "charset": charset
        }

    def _generate_benchmark_data(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        table_name = str(config.get("table_name", "orders")).lower()
        limit = max(1, min(int(config.get("limit", 100)), 500))

        records: List[Dict[str, Any]] = []
        statuses = ["COMPLETED", "PROCESSING", "SHIPPED", "CANCELLED", "REFUNDED"]
        payment_methods = ["CREDIT_CARD", "PAYPAL", "WIRE", "APPLE_PAY", "INVOICE"]

        for i in range(1, limit + 1):
            if "business" in table_name or "company" in table_name:
                records.append({
                    "id": i,
                    "owner_id": 250 + i,
                    "name": f"Enterprise Business #{i}",
                    "description": f"Verified business operations entity {i}",
                    "business_type_id": 1,
                    "address": f"{100 + i} Commercial Ave, Suite {i}"
                })
            elif "prod" in table_name or "item" in table_name or "invent" in table_name:
                records.append({
                    "sku": f"SKU-{1000 + i}",
                    "product_name": f"Enterprise Gear Part #{i}",
                    "category": f"Hardware-{(i % 4) + 1}",
                    "unit_price": round(15.99 + (i * 2.50) % 350.0, 2),
                    "stock_quantity": (i * 13) % 250,
                    "is_active": 1 if i % 9 != 0 else 0,
                    "last_restocked": f"2026-02-{((i % 28) + 1):02d} 14:00:00"
                })
            else:
                records.append({
                    "order_id": 50000 + i,
                    "customer_id": f"CUST-{(i % 60) + 100}",
                    "order_total": round(45.00 + (i * 5.25) % 1200.0, 2),
                    "tax_amount": round((45.00 + (i * 5.25) % 1200.0) * 0.08, 2),
                    "status": statuses[i % len(statuses)],
                    "payment_method": payment_methods[i % len(payment_methods)],
                    "order_date": f"2026-03-{((i % 28) + 1):02d} 09:30:15"
                })
        return records

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        start_time = time.time()
        read_mode = str(config.get("read_mode", "table")).strip().lower()
        limit = max(1, min(int(config.get("limit", 1000)), 100000))
        offset_val = max(0, int(config.get("offset", 0)))
        simulation_mode = bool(config.get("simulation_mode", False))
        params = self._resolve_connection_params(config, context)

        # 1. Resolve columns
        cols_raw = config.get("columns", [])
        cols_list: List[str] = []
        if isinstance(cols_raw, list):
            cols_list = [str(c).strip().strip('`') for c in cols_raw if str(c).strip() and str(c).strip() != "*"]
        elif isinstance(cols_raw, str) and cols_raw.strip():
            cols_list = [c.strip().strip('`') for c in cols_raw.split(",") if c.strip() and c.strip() != "*"]

        if cols_list:
            cols_str = ", ".join(f"`{c}`" if not c.startswith("`") else c for c in cols_list)
        else:
            cols_str = "*"

        # 2. Resolve tables
        database = str(config.get("database", "shop_db")).strip() or "shop_db"
        select_all_tables = bool(config.get("select_all_tables", False))

        target_tables: List[str] = []
        raw_tables = config.get("tables")
        if isinstance(raw_tables, list):
            target_tables = [str(t).strip().strip('`') for t in raw_tables if str(t).strip() and str(t).strip() != "*"]
        elif isinstance(raw_tables, str) and raw_tables.strip():
            target_tables = [t.strip().strip('`') for t in raw_tables.split(",") if t.strip() and t.strip() != "*"]

        single_tbl = str(config.get("table_name", "")).strip().strip('`')
        if single_tbl and single_tbl != "*":
            if "," in single_tbl:
                for t in single_tbl.split(","):
                    clean_t = t.strip().strip('`')
                    if clean_t and clean_t not in target_tables:
                        target_tables.append(clean_t)
            elif single_tbl not in target_tables:
                target_tables.append(single_tbl)

        # 3. Filters, Sorting, and Pagination
        where_raw = str(config.get("where_clause", "")).strip()
        where_sql = f" WHERE {where_raw}" if where_raw else ""
        order_raw = str(config.get("order_by", "")).strip()
        order_sql = f" ORDER BY {order_raw}" if order_raw else ""
        offset_sql = f" OFFSET {offset_val}" if offset_val > 0 else ""

        records: List[Dict[str, Any]] = []
        delivery_status = "SIMULATED"
        status_msg = ""
        columns_retrieved: List[str] = []
        target_desc = ""

        # Live Execution Attempt
        if not simulation_mode:
            last_error = None
            try:
                import pymysql
                import pymysql.cursors
                conn = pymysql.connect(
                    host=params["host"],
                    port=int(params["port"]),
                    user=params["username"],
                    password=params["password"],
                    database=params["database"],
                    charset=params["charset"],
                    cursorclass=pymysql.cursors.DictCursor,
                    connect_timeout=int(config.get("timeout", 15))
                )
                with conn.cursor() as cur:
                    if read_mode == "table":
                        if select_all_tables or not target_tables:
                            cur.execute(
                                "SELECT table_name FROM information_schema.tables "
                                "WHERE table_schema = %s AND table_type IN ('BASE TABLE', 'VIEW') "
                                "ORDER BY table_name;",
                                (database,)
                            )
                            rows = cur.fetchall()
                            target_tables = [list(r.values())[0] if isinstance(r, dict) else r[0] for r in rows]
                            if not target_tables:
                                raise ValidationException(f"No tables found in MySQL database '{database}'.")

                        is_multi = len(target_tables) > 1
                        for tbl in target_tables:
                            q = f"SELECT {cols_str} FROM `{tbl}`{where_sql}{order_sql} LIMIT {limit}{offset_sql}"
                            cur.execute(q)
                            rows = cur.fetchall()
                            if rows and not columns_retrieved:
                                columns_retrieved = list(rows[0].keys())
                            if is_multi:
                                for r in rows:
                                    r["_table"] = tbl
                            records.extend(rows)

                        delivery_status = "LIVE"
                        if is_multi:
                            if "_table" not in columns_retrieved:
                                columns_retrieved.insert(0, "_table")
                            target_desc = f"{database} ({', '.join(target_tables)})"
                            status_msg = f"Retrieved {len(records)} live rows across {len(target_tables)} tables from MySQL ({target_desc})."
                        else:
                            target_desc = f"{database}.{target_tables[0]}"
                            status_msg = f"Retrieved {len(records)} live rows from MySQL ({target_desc})."
                    else:
                        query = str(config.get("query", f"SELECT * FROM `{target_tables[0] if target_tables else 'orders'}`")).strip()
                        target_desc = "Custom Query"
                        if "limit" not in query.lower():
                            query = f"{query} LIMIT {limit}"
                        cur.execute(query)
                        rows = cur.fetchall()
                        if rows:
                            columns_retrieved = list(rows[0].keys())
                        records = rows
                        delivery_status = "LIVE"
                        status_msg = f"Retrieved {len(records)} live rows from MySQL custom query."
                conn.close()
                last_error = None
            except Exception as ex:
                logger.warning(f"MySQL live execution failed: {ex}")
                last_error = ex

            # STRICT ERROR ENFORCEMENT: Never show fake data when live execution is requested!
            if last_error is not None:
                raise ValidationException(
                    f"MySQL connection to {params.get('host')}:{params.get('port')}/{params.get('database')} failed: {str(last_error)}"
                )

        # Simulation Mode (ONLY when explicitly enabled)
        if simulation_mode:
            if not target_tables:
                target_tables = ["orders"]
            is_multi = len(target_tables) > 1
            for tbl in target_tables:
                sim_recs = self._generate_benchmark_data(config)
                if cols_list:
                    filtered = []
                    for r in sim_recs:
                        item = {c: r.get(c, f"{c}_val") for c in cols_list}
                        filtered.append(item)
                    sim_recs = filtered
                if is_multi:
                    for r in sim_recs:
                        r["_table"] = tbl
                records.extend(sim_recs)

            if is_multi:
                columns_retrieved = ["_table"] + (cols_list if cols_list else (list(records[0].keys()) if records else []))
                target_desc = f"{database} ({', '.join(target_tables)})"
            else:
                target_desc = f"{database}.{target_tables[0]}"
                columns_retrieved = cols_list if cols_list else (list(records[0].keys()) if records else [])

            delivery_status = "SIMULATED"
            status_msg = f"MySQL benchmark simulation mode active. Generated {len(records)} test rows for table '{target_desc}'."

        df_out = pd.DataFrame(records, columns=columns_retrieved if (records and columns_retrieved) else None) if pd is not None else records
        duration_ms = int((time.time() - start_time) * 1000)

        summary = {
            "title": "MySQL Connector Summary",
            "message": status_msg,
            "status": delivery_status,
            "target": target_desc,
            "records_retrieved": len(records),
            "columns_count": len(columns_retrieved),
            "duration_ms": duration_ms
        }

        metrics = {
            "records_retrieved": len(records),
            "columns_count": len(columns_retrieved),
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
        read_mode = config.get("read_mode", "table")
        host = config.get("host", "localhost")
        port = config.get("port", 3306)
        database = config.get("database", "shop_db")
        username = config.get("username", "root")
        table_name = config.get("table_name", "orders")
        query = config.get("query", f"SELECT * FROM `{table_name}`")
        limit = config.get("limit", 1000)

        if read_mode == "table":
            sql_stmt = f"SELECT * FROM `{table_name}` LIMIT {limit}"
        else:
            sql_stmt = query

        return (
            f"# MySQL Ingestion Pipeline Code\n"
            f"import pymysql\n"
            f"import pandas as pd\n"
            f"import os\n\n"
            f"conn = pymysql.connect(\n"
            f"    host=os.getenv('MYSQL_HOST', '{host}'),\n"
            f"    port=int(os.getenv('MYSQL_PORT', {port})),\n"
            f"    database=os.getenv('MYSQL_DATABASE', '{database}'),\n"
            f"    user=os.getenv('MYSQL_USER', '{username}'),\n"
            f"    password=os.getenv('MYSQL_PASSWORD', ''),\n"
            f"    charset='utf8mb4'\n"
            f")\n\n"
            f"query = '''{sql_stmt}'''\n"
            f"df = pd.read_sql_query(query, conn)\n"
            f"conn.close()\n"
            f"print(f'Retrieved {{len(df)}} rows from MySQL.')\n"
        )

    def inspect(
        self,
        action: str,
        config: Dict[str, Any],
        context: Optional[Any] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Introspects the connected MySQL / MariaDB database in real-time:
        - 'test_connection': verifies connectivity and server version
        - 'databases': list of accessible databases (schemas)
        - 'tables': list of tables/views in a database
        - 'columns': list of columns with types, nullability, and primary keys
        """
        action = str(action or "test_connection").strip().lower()
        params = self._resolve_connection_params(config, context)
        timeout = int(config.get("timeout", 10))
        simulation_mode = bool(config.get("simulation_mode", False))

        is_live = False
        server_info = ""
        inspect_err = ""
        conn = None

        if not simulation_mode and (params.get("connection_string") or params.get("host")):
            try:
                import pymysql
                import pymysql.cursors
                target_db = params.get("database") or None
                conn = pymysql.connect(
                    host=params.get("host", "localhost"),
                    port=int(params.get("port", 3306)),
                    database=target_db,
                    user=params.get("username", "root"),
                    password=params.get("password", ""),
                    charset=params.get("charset", "utf8mb4"),
                    connect_timeout=timeout,
                    cursorclass=pymysql.cursors.DictCursor,
                )
                with conn.cursor() as cur:
                    cur.execute("SELECT VERSION();")
                    row = cur.fetchone()
                    if row:
                        server_info = list(row.values())[0] if isinstance(row, dict) else str(row[0])
                        is_live = True
            except Exception as e:
                inspect_err = str(e)
                logger.debug(f"MySQL inspect connection failed: {e}")

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
                    "database": params.get("database"),
                    "message": f"Successfully connected to MySQL server '{params.get('host')}:{params.get('port')}'.",
                }
            elif simulation_mode:
                return {
                    "success": True,
                    "connected": True,
                    "status": "SIMULATED",
                    "server_version": "8.0.35-MySQL Community Server [SIMULATION]",
                    "database": params.get("database"),
                    "message": "MySQL simulation mode connection active.",
                }
            else:
                return {
                    "success": False,
                    "connected": False,
                    "status": "UNREACHABLE",
                    "database": params.get("database"),
                    "message": f"Could not establish connection to MySQL server ({params.get('host')}:{params.get('port')}). Error: {inspect_err or 'Server unreachable'}",
                    "error": inspect_err,
                }

        # 2. Action: databases / schemas
        if action in ("databases", "get_databases", "list_databases", "schemas", "get_schemas"):
            if not is_live and not simulation_mode:
                return {
                    "success": False,
                    "databases": [],
                    "count": 0,
                    "error": inspect_err or "Connection could not be established.",
                    "message": f"Failed to connect to MySQL: {inspect_err or 'Host unreachable'}",
                    "source": "ERROR",
                }

            databases = []
            if is_live and conn:
                try:
                    with conn.cursor() as cur:
                        cur.execute("SHOW DATABASES;")
                        rows = cur.fetchall()
                        databases = [list(r.values())[0] if isinstance(r, dict) else r[0] for r in rows]
                        databases = [db for db in databases if db not in ('information_schema', 'mysql', 'performance_schema', 'sys')]
                except Exception as ex:
                    logger.warning(f"Failed to query MySQL databases: {ex}")

            if conn:
                conn.close()

            if not databases and simulation_mode:
                databases = ["ProjectX", "shop_db", "inventory_db", "users_db"]

            return {
                "success": True,
                "databases": databases,
                "schemas": databases,
                "count": len(databases),
                "source": "LIVE" if is_live else "SIMULATED",
            }

        # 3. Action: tables
        if action in ("tables", "get_tables", "list_tables"):
            if not is_live and not simulation_mode:
                return {
                    "success": False,
                    "tables": [],
                    "count": 0,
                    "error": inspect_err or "Connection could not be established.",
                    "message": f"Failed to connect to MySQL: {inspect_err or 'Host unreachable'}",
                    "source": "ERROR",
                }

            target_db = schema_name or config.get("database") or "shop_db"
            tables = []
            if is_live and conn:
                try:
                    with conn.cursor() as cur:
                        cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = %s AND table_type IN ('BASE TABLE', 'VIEW') ORDER BY table_name;", (target_db,))
                        rows = cur.fetchall()
                        tables = [list(r.values())[0] if isinstance(r, dict) else r[0] for r in rows]
                except Exception as ex:
                    logger.warning(f"Failed to query MySQL tables: {ex}")

            if conn:
                conn.close()

            if not tables and simulation_mode:
                tables = ["Businesses", "BookingServices", "orders", "customers", "inventory"]

            return {
                "success": True,
                "database": target_db,
                "tables": tables,
                "count": len(tables),
                "source": "LIVE" if is_live else "SIMULATED",
            }

        # 4. Action: columns
        if action in ("columns", "get_columns", "list_columns"):
            if not is_live and not simulation_mode:
                return {
                    "success": False,
                    "columns": [],
                    "count": 0,
                    "error": inspect_err or "Connection could not be established.",
                    "message": f"Failed to connect to MySQL: {inspect_err or 'Host unreachable'}",
                    "source": "ERROR",
                }

            target_db = schema_name or config.get("database") or "shop_db"
            target_table = table_name or config.get("table_name") or "orders"
            columns = []
            if is_live and conn:
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT column_name, data_type, is_nullable, column_key "
                            "FROM information_schema.columns "
                            "WHERE table_schema = %s AND table_name = %s "
                            "ORDER BY ordinal_position;",
                            (target_db, target_table)
                        )
                        rows = cur.fetchall()
                        for r in rows:
                            vals = list(r.values()) if isinstance(r, dict) else r
                            columns.append({
                                "name": vals[0],
                                "type": vals[1],
                                "nullable": (vals[2] == "YES"),
                                "primary_key": (vals[3] == "PRI")
                            })
                except Exception as ex:
                    logger.warning(f"Failed to query MySQL columns: {ex}")

            if conn:
                conn.close()

            if not columns and simulation_mode:
                if "business" in target_table.lower():
                    columns = [
                        {"name": "Id", "type": "int", "nullable": False, "primary_key": True},
                        {"name": "Owner_Id", "type": "int", "nullable": False, "primary_key": False},
                        {"name": "Name", "type": "varchar(255)", "nullable": False, "primary_key": False},
                        {"name": "Description", "type": "text", "nullable": True, "primary_key": False},
                        {"name": "Business_typeId", "type": "int", "nullable": False, "primary_key": False},
                        {"name": "Address", "type": "varchar(255)", "nullable": True, "primary_key": False},
                    ]
                else:
                    columns = [
                        {"name": "order_id", "type": "int", "nullable": False, "primary_key": True},
                        {"name": "customer_id", "type": "int", "nullable": False, "primary_key": False},
                        {"name": "amount", "type": "decimal(10,2)", "nullable": False, "primary_key": False},
                        {"name": "status", "type": "varchar(50)", "nullable": False, "primary_key": False},
                        {"name": "created_at", "type": "datetime", "nullable": False, "primary_key": False},
                    ]

            return {
                "success": True,
                "database": target_db,
                "table": target_table,
                "columns": columns,
                "count": len(columns),
                "source": "LIVE" if is_live else "SIMULATED",
            }

        return {
            "success": False,
            "error": f"Unknown inspection action '{action}'. Supported: 'test_connection', 'databases', 'schemas', 'tables', 'columns'.",
            "message": f"Unknown inspection action '{action}'.",
        }
