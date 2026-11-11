"""
MySQL Connector Recipe (Ingestion & Orchestration -> Database & Warehouse Connectors).

Reads tables or query results from a MySQL / MariaDB database into the pipeline as a DataFrame.
Supports host/port, connection string, or saved credentials, read-only query validation,
charset encoding, benchmark simulation fallback, and code generation.
"""

import re
import os
import time
from typing import Dict, Any, List, Optional

try:
    import pandas as pd
except ImportError:
    pd = None

from backend.app.recipes.base.recipe import BaseRecipe, RecipePort
from backend.app.core.logging import logger
from backend.app.core.config import settings


class MySQLConnectorRecipe(BaseRecipe):
    recipe_id = "mysql_connector"
    name = "MySQL Connector"
    version = "1.0.0"
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
                "connection_mode": {
                    "type": "string",
                    "title": "Connection Mode",
                    "input_type": "dropdown",
                    "enum": ["host_port", "connection_string", "credential"],
                    "default": "host_port",
                    "description": "Choose connection method: standard host/port parameters, connection URI, or stored credential."
                },
                "host": {
                    "type": "string",
                    "title": "Host",
                    "input_type": "text",
                    "default": "localhost",
                    "description": "MySQL / MariaDB server hostname or IP address."
                },
                "port": {
                    "type": "integer",
                    "title": "Port",
                    "input_type": "number",
                    "default": 3306,
                    "description": "MySQL server port (default 3306)."
                },
                "database": {
                    "type": "string",
                    "title": "Database Name",
                    "input_type": "text",
                    "default": "shop_db",
                    "description": "Name of the MySQL schema / database."
                },
                "username": {
                    "type": "string",
                    "title": "Username",
                    "input_type": "text",
                    "default": "root",
                    "description": "MySQL username."
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
                    "description": "Full MySQL connection URI e.g. mysql+pymysql://user:pass@localhost:3306/dbname"
                },
                "credential_id": {
                    "type": "string",
                    "title": "Connected MySQL Credential",
                    "input_type": "credential_dropdown",
                    "provider": "mysql",
                    "description": "Select a stored MySQL credential from platform settings."
                },
                "read_mode": {
                    "type": "string",
                    "title": "Read Mode",
                    "input_type": "dropdown",
                    "enum": ["table", "query"],
                    "default": "table",
                    "description": "Read an entire table or execute a custom SQL query."
                },
                "table_name": {
                    "type": "string",
                    "title": "Table Name",
                    "input_type": "text",
                    "default": "orders",
                    "description": "Table to read (e.g. 'orders', 'inventory', 'customers')."
                },
                "columns": {
                    "type": "string",
                    "title": "Columns to Select",
                    "input_type": "text",
                    "default": "",
                    "description": "Comma-separated column names to select (leave blank for all columns '*')."
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
            if not table_name:
                errors.append("Table name is required when read_mode is 'table'.")
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

        # Environment variable overrides
        if not password:
            password = os.getenv("MYSQL_PASSWORD", os.getenv("MYSQL_ROOT_PASSWORD", ""))
        if not username:
            username = os.getenv("MYSQL_USER", "root")

        if not conn_str and host and database:
            auth_part = f"{username}:{password}@" if username else ""
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
            if "prod" in table_name or "item" in table_name or "invent" in table_name:
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
        simulation_mode = bool(config.get("simulation_mode", False))
        params = self._resolve_connection_params(config, context)

        if read_mode == "table":
            table_name = str(config.get("table_name", "orders")).strip()
            cols_raw = config.get("columns", "")
            if isinstance(cols_raw, list):
                cols = ", ".join(cols_raw) if cols_raw else "*"
            elif isinstance(cols_raw, str) and cols_raw.strip():
                cols = cols_raw.strip()
            else:
                cols = "*"
            query = f"SELECT {cols} FROM `{table_name}` LIMIT {limit}"
            target_desc = table_name
        else:
            query = str(config.get("query", "SELECT * FROM orders")).strip()
            target_desc = "Custom Query"
            if "limit" not in query.lower():
                query = f"{query} LIMIT {limit}"

        records: List[Dict[str, Any]] = []
        delivery_status = "SIMULATED"
        status_msg = ""
        columns_retrieved: List[str] = []

        # Live Execution Attempt
        if not simulation_mode:
            # 1. Try PyMySQL
            try:
                import pymysql
                import pymysql.cursors
                conn = pymysql.connect(
                    host=params["host"],
                    port=params["port"],
                    user=params["username"],
                    password=params["password"],
                    database=params["database"],
                    charset=params["charset"],
                    cursorclass=pymysql.cursors.DictCursor,
                    connect_timeout=int(config.get("timeout", 15))
                )
                with conn.cursor() as cur:
                    cur.execute(query)
                    records = cur.fetchall()
                    if records:
                        columns_retrieved = list(records[0].keys())
                    delivery_status = "LIVE"
                    status_msg = f"Retrieved {len(records)} live rows from MySQL (`{target_desc}`)."
                conn.close()
            except ImportError:
                # 2. Try SQLAlchemy
                try:
                    import sqlalchemy
                    engine = sqlalchemy.create_engine(
                        params["connection_string"],
                        connect_args={"connect_timeout": int(config.get("timeout", 15))}
                    )
                    with engine.connect() as conn:
                        res = conn.execute(sqlalchemy.text(query))
                        columns_retrieved = list(res.keys())
                        rows = res.fetchall()
                        records = [dict(zip(columns_retrieved, row)) for row in rows]
                        delivery_status = "LIVE"
                        status_msg = f"Retrieved {len(records)} live rows from MySQL via SQLAlchemy."
                except Exception as sqla_e:
                    logger.warning(f"MySQL SQLAlchemy attempt failed: {sqla_e}")
                    status_msg = f"MySQL connection failed ({sqla_e}). Using benchmark simulation fallback."
            except Exception as ex:
                logger.warning(f"MySQL connection failed: {ex}")
                status_msg = f"MySQL connection failed ({ex}). Using benchmark simulation fallback."

        # Fallback to benchmark data
        if not records:
            records = self._generate_benchmark_data(config)
            if records:
                columns_retrieved = list(records[0].keys())
            if not status_msg:
                status_msg = f"Loaded {len(records)} benchmark records for MySQL table '{target_desc}'."

        df_out = pd.DataFrame(records) if pd is not None else records
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
            f"# Recipe: MySQL Connector\n"
            f"# ID: {self.recipe_id}\n"
            f"import os\n"
            f"import pandas as pd\n"
            f"import pymysql\n\n"
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
                conn = pymysql.connect(
                    host=params.get("host", "localhost"),
                    port=int(params.get("port", 3306)),
                    database=params.get("database", "shop_db"),
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
                    "message": f"Could not establish connection to MySQL ({params.get('host')}:{params.get('port')}). Error: {inspect_err or 'Server unreachable'}",
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
                databases = ["shop_db", "inventory_db", "users_db"]

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
                tables = ["orders", "order_items", "products", "customers", "inventory"]

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
