"""
Unit & Integration Tests for Database & Warehouse Connectors:
- PostgreSQL Connector (postgresql_connector)
- MySQL Connector (mysql_connector)
- MongoDB Connector (mongodb_connector)

Tests schema generation, pre-flight validation, DDL/DML injection protection,
credential resolution, BSON serialization, document flattening, simulation modes,
code generation, and end-to-end DAG execution.
"""

import pytest
import pandas as pd
from typing import Dict, Any

from backend.app.recipes.base.registry import recipe_registry
from backend.app.recipes.ingestion.postgresql_connector import PostgreSQLConnectorRecipe
from backend.app.recipes.ingestion.mysql_connector import MySQLConnectorRecipe
from backend.app.recipes.ingestion.mongodb_connector import MongoDBConnectorRecipe
from backend.app.engine.dag.graph import WorkflowGraph, WorkflowNode, WorkflowEdge
from backend.app.engine.execution.executor import DAGExecutor


# =====================================================================
# 1. PostgreSQL Connector Tests
# =====================================================================

def test_postgresql_connector_metadata_and_schema():
    recipe = PostgreSQLConnectorRecipe()
    assert recipe.recipe_id == "postgresql_connector"
    assert recipe.category == "ingestion"
    assert recipe.group == "Ingestion & Orchestration"
    assert recipe.subgroup == "Database & Warehouse Connectors"
    assert "dataframe" in recipe.output_types

    schema = recipe.get_schema()
    assert "properties" in schema
    props = schema["properties"]
    assert "connection_mode" in props
    assert "host" in props
    assert "port" in props
    assert "database" in props
    assert "read_mode" in props
    assert "table_name" in props
    assert "query" in props
    assert "limit" in props
    assert "simulation_mode" in props


def test_postgresql_connector_validation_edge_cases():
    recipe = PostgreSQLConnectorRecipe()

    # 1. Valid Table Config
    val_table = recipe.validate_config({"read_mode": "table", "table_name": "sales", "port": 5432})
    assert len(val_table) == 0

    # 2. Missing Table Name in table mode
    err_table = recipe.validate_config({"read_mode": "table", "table_name": ""})
    assert any("Table name is required" in e for e in err_table)

    # 3. Valid Query Config
    val_query = recipe.validate_config({"read_mode": "query", "query": "SELECT id, name FROM users"})
    assert len(val_query) == 0

    # 4. Missing Query in query mode
    err_query = recipe.validate_config({"read_mode": "query", "query": ""})
    assert any("SQL query is required" in e for e in err_query)

    # 5. Invalid Port (out of bounds)
    err_port_high = recipe.validate_config({"read_mode": "table", "table_name": "users", "port": 70000})
    assert any("Invalid PostgreSQL port" in e for e in err_port_high)

    err_port_str = recipe.validate_config({"read_mode": "table", "table_name": "users", "port": "abc"})
    assert any("valid integer" in e for e in err_port_str)

    # 6. SQL Injection / Destructive DDL/DML keywords blocked
    destructive_queries = [
        "DROP TABLE users;",
        "DELETE FROM sales WHERE id > 0;",
        "TRUNCATE TABLE logs;",
        "ALTER TABLE accounts ADD COLUMN secret TEXT;",
        "UPDATE users SET is_admin = true;",
        "/* comment */ DROP DATABASE production;"
    ]
    for q in destructive_queries:
        errs = recipe.validate_config({"read_mode": "query", "query": q})
        assert len(errs) > 0, f"Query '{q}' should have been flagged as destructive."


def test_postgresql_connector_execution_table_mode():
    recipe = PostgreSQLConnectorRecipe()
    config = {
        "read_mode": "table",
        "table_name": "sales_transactions",
        "limit": 50,
        "simulation_mode": True
    }
    result = recipe.execute(inputs={}, config=config)

    assert "dataframe" in result
    df = result["dataframe"]
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 50
    assert "amount" in df.columns
    assert "transaction_id" in df.columns

    summary = result.get("output_summary", {})
    assert summary.get("title") == "PostgreSQL Connector Summary"
    assert summary.get("records_retrieved") == 50
    assert summary.get("status") == "SIMULATED"


def test_postgresql_connector_credential_context_and_code():
    recipe = PostgreSQLConnectorRecipe()
    context = {
        "credentials": {
            "cred_pg_1": {
                "data": {
                    "host": "db.corp.internal",
                    "port": 5433,
                    "database": "warehouse",
                    "username": "etl_reader"
                }
            }
        }
    }
    config = {
        "connection_mode": "credential",
        "credential_id": "cred_pg_1",
        "read_mode": "table",
        "table_name": "client_accounts",
        "limit": 25,
        "simulation_mode": True
    }
    result = recipe.execute(inputs={}, config=config, context=context)
    df = result["dataframe"]
    assert len(df) == 25
    assert "account_code" in df.columns

    code = recipe.to_code(config)
    assert "psycopg2" in code
    assert "client_accounts" in code


# =====================================================================
# 2. MySQL Connector Tests
# =====================================================================

def test_mysql_connector_metadata_and_schema():
    recipe = MySQLConnectorRecipe()
    assert recipe.recipe_id == "mysql_connector"
    assert recipe.category == "ingestion"
    assert recipe.group == "Ingestion & Orchestration"
    assert recipe.subgroup == "Database & Warehouse Connectors"
    assert "dataframe" in recipe.output_types

    schema = recipe.get_schema()
    props = schema["properties"]
    assert "connection_mode" in props
    assert "host" in props
    assert "port" in props
    assert "database" in props
    assert "charset" in props
    assert "table_name" in props


def test_mysql_connector_validation_edge_cases():
    recipe = MySQLConnectorRecipe()

    # 1. Valid Table Config
    val_table = recipe.validate_config({"read_mode": "table", "table_name": "orders", "port": 3306})
    assert len(val_table) == 0

    # 2. Missing Table Name
    err_table = recipe.validate_config({"read_mode": "table", "table_name": ""})
    assert any("Table name is required" in e for e in err_table)

    # 3. Invalid Port
    err_port = recipe.validate_config({"read_mode": "table", "table_name": "orders", "port": 0})
    assert any("Invalid MySQL port" in e for e in err_port)

    # 4. Destructive Queries Blocked
    destructive_queries = [
        "DROP DATABASE store;",
        "TRUNCATE orders;",
        "DELETE FROM inventory;",
        "UPDATE products SET price = 0.0;",
        "ALTER TABLE customers DROP COLUMN password;"
    ]
    for q in destructive_queries:
        errs = recipe.validate_config({"read_mode": "query", "query": q})
        assert len(errs) > 0, f"Query '{q}' should have been flagged as destructive."


def test_mysql_connector_execution_table_mode():
    recipe = MySQLConnectorRecipe()
    config = {
        "read_mode": "table",
        "table_name": "orders",
        "limit": 40,
        "simulation_mode": True
    }
    result = recipe.execute(inputs={}, config=config)

    assert "dataframe" in result
    df = result["dataframe"]
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 40
    assert "order_id" in df.columns
    assert "order_total" in df.columns
    assert "status" in df.columns

    summary = result.get("output_summary", {})
    assert summary.get("title") == "MySQL Connector Summary"
    assert summary.get("records_retrieved") == 40


def test_mysql_connector_custom_products_simulation_and_code():
    recipe = MySQLConnectorRecipe()
    config = {
        "read_mode": "table",
        "table_name": "products_inventory",
        "limit": 30,
        "simulation_mode": True
    }
    result = recipe.execute(inputs={}, config=config)
    df = result["dataframe"]
    assert len(df) == 30
    assert "sku" in df.columns
    assert "unit_price" in df.columns

    code = recipe.to_code(config)
    assert "pymysql" in code
    assert "products_inventory" in code


# =====================================================================
# 3. MongoDB Connector Tests
# =====================================================================

def test_mongodb_connector_metadata_and_schema():
    recipe = MongoDBConnectorRecipe()
    assert recipe.recipe_id == "mongodb_connector"
    assert recipe.category == "ingestion"
    assert recipe.group == "Ingestion & Orchestration"
    assert recipe.subgroup == "Database & Warehouse Connectors"
    assert "dataframe" in recipe.output_types

    schema = recipe.get_schema()
    props = schema["properties"]
    assert "connection_string" in props
    assert "database" in props
    assert "collection" in props
    assert "filter_query" in props
    assert "projection" in props
    assert "flatten_nested" in props
    assert "convert_objectid" in props


def test_mongodb_connector_validation_edge_cases():
    recipe = MongoDBConnectorRecipe()

    # 1. Valid Config
    val_cfg = recipe.validate_config({
        "database": "production",
        "collection": "events",
        "filter_query": '{"status": "active"}'
    })
    assert len(val_cfg) == 0

    # 2. Missing Database or Collection
    err_db = recipe.validate_config({"database": "", "collection": "events"})
    assert any("database name is required" in e for e in err_db)

    err_coll = recipe.validate_config({"database": "production", "collection": ""})
    assert any("collection name is required" in e for e in err_coll)

    # 3. Malformed JSON in filter_query or projection
    err_json_filter = recipe.validate_config({
        "database": "test",
        "collection": "logs",
        "filter_query": "{invalid_json: 123"
    })
    assert any("Invalid JSON in 'filter_query'" in e for e in err_json_filter)

    err_json_proj = recipe.validate_config({
        "database": "test",
        "collection": "logs",
        "projection": "['unclosed_bracket'"
    })
    assert any("Invalid JSON in 'projection'" in e for e in err_json_proj)


def test_mongodb_connector_document_flattening_and_objectid():
    recipe = MongoDBConnectorRecipe()

    # 1. Flatten nested documents enabled
    config_flat = {
        "database": "analytics",
        "collection": "events",
        "limit": 20,
        "flatten_nested": True,
        "convert_objectid": True,
        "simulation_mode": True
    }
    res_flat = recipe.execute(inputs={}, config=config_flat)
    df_flat = res_flat["dataframe"]
    assert len(df_flat) == 20
    # Nested fields flattened into dot notation
    assert "client.device" in df_flat.columns
    assert "client.os" in df_flat.columns
    assert "metrics.revenue" in df_flat.columns
    # ObjectId converted to string
    assert isinstance(df_flat["_id"].iloc[0], str)

    # 2. Flatten nested disabled
    config_nested = {
        "database": "analytics",
        "collection": "events",
        "limit": 10,
        "flatten_nested": False,
        "simulation_mode": True
    }
    res_nested = recipe.execute(inputs={}, config=config_nested)
    df_nested = res_nested["dataframe"]
    assert len(df_nested) == 10
    assert "client" in df_nested.columns
    assert isinstance(df_nested["client"].iloc[0], dict)

    summary = res_flat.get("output_summary", {})
    assert summary.get("title") == "MongoDB Connector Summary"
    assert summary.get("database") == "analytics"
    assert summary.get("collection") == "events"


def test_mongodb_connector_users_collection_and_code():
    recipe = MongoDBConnectorRecipe()
    config = {
        "database": "production",
        "collection": "user_accounts",
        "limit": 15,
        "flatten_nested": True,
        "simulation_mode": True
    }
    result = recipe.execute(inputs={}, config=config)
    df = result["dataframe"]
    assert len(df) == 15
    assert "profile.full_name" in df.columns
    assert "profile.email" in df.columns

    code = recipe.to_code(config)
    assert "MongoClient" in code
    assert "user_accounts" in code


# =====================================================================
# 4. Registry & DAG Workflow Integration Tests
# =====================================================================

def test_database_connectors_in_recipe_registry():
    # Canonical IDs
    assert recipe_registry.has("postgresql_connector") is True
    assert recipe_registry.has("mysql_connector") is True
    assert recipe_registry.has("mongodb_connector") is True

    # Aliases
    assert recipe_registry.get("postgres").recipe_id == "postgresql_connector"
    assert recipe_registry.get("postgresql").recipe_id == "postgresql_connector"
    assert recipe_registry.get("mysql").recipe_id == "mysql_connector"
    assert recipe_registry.get("mariadb").recipe_id == "mysql_connector"
    assert recipe_registry.get("mongo").recipe_id == "mongodb_connector"
    assert recipe_registry.get("mongodb").recipe_id == "mongodb_connector"


def test_postgresql_connector_in_dag_workflow_end_to_end():
    """Build a multi-node workflow starting with PostgreSQL Connector into Category Sanitizer."""
    nodes = [
        WorkflowNode(
            id="pg_source",
            recipe_id="postgresql_connector",
            config={
                "read_mode": "table",
                "table_name": "orders",
                "limit": 25,
                "simulation_mode": True
            }
        ),
        WorkflowNode(
            id="sanitizer_step",
            recipe_id="category_sanitizer",
            config={
                "columns": ["status", "region"],
                "lowercase": True,
                "trim_whitespace": True
            }
        )
    ]
    edges = [
        WorkflowEdge(id="e1", source="pg_source", target="sanitizer_step")
    ]
    graph = WorkflowGraph(id="wf_pg_test", name="PG DAG Test", nodes=nodes, edges=edges)

    exec_result = DAGExecutor.execute_workflow(
        execution_id="test_pg_exec_100",
        workflow=graph,
        include_node_outputs=True
    )

    assert exec_result.status == "SUCCESS"
    assert len(exec_result.node_results) == 2

    # Check node 1: pg_source
    res_pg = next(r for r in exec_result.node_results if r.node_id == "pg_source")
    assert res_pg.status == "SUCCESS"
    summary_pg = res_pg.output_summary.get("output_summary") or res_pg.output_summary
    assert summary_pg.get("title") == "PostgreSQL Connector Summary"

    # Check node 2: sanitizer_step
    res_sanitizer = next(r for r in exec_result.node_results if r.node_id == "sanitizer_step")
    assert res_sanitizer.status == "SUCCESS"
    summary_sanitizer = res_sanitizer.output_summary.get("output_summary") or res_sanitizer.output_summary
    assert summary_sanitizer.get("title") == "Categorical Sanitizer Summary"


def test_mongodb_connector_in_dag_workflow_end_to_end():
    """Build a multi-node workflow starting with MongoDB Connector into Category Sanitizer."""
    nodes = [
        WorkflowNode(
            id="mongo_source",
            recipe_id="mongodb_connector",
            config={
                "database": "telemetry",
                "collection": "events",
                "limit": 30,
                "flatten_nested": True,
                "simulation_mode": True
            }
        ),
        WorkflowNode(
            id="sanitizer_step",
            recipe_id="category_sanitizer",
            config={
                "columns": ["event_type", "client.device"],
                "lowercase": True
            }
        )
    ]
    edges = [
        WorkflowEdge(id="e2", source="mongo_source", target="sanitizer_step")
    ]
    graph = WorkflowGraph(id="wf_mongo_test", name="Mongo DAG Test", nodes=nodes, edges=edges)

    exec_result = DAGExecutor.execute_workflow(
        execution_id="test_mongo_exec_200",
        workflow=graph,
        include_node_outputs=True
    )

    assert exec_result.status == "SUCCESS"
    assert len(exec_result.node_results) == 2

    res_mongo = next(r for r in exec_result.node_results if r.node_id == "mongo_source")
    assert res_mongo.status == "SUCCESS"
    summary_mongo = res_mongo.output_summary.get("output_summary") or res_mongo.output_summary
    assert summary_mongo.get("title") == "MongoDB Connector Summary"


def test_postgresql_connector_connection_string_parsing():
    """Verify that full connection_string URI extracts host, port, database, user, password, and sslmode."""
    recipe = PostgreSQLConnectorRecipe()
    config = {
        "connection_mode": "connection_string",
        "connection_string": "postgresql://dbuser:SecretPass123@prod-db.internal:5433/sales_warehouse?sslmode=require",
        "read_mode": "table",
        "table_name": "orders",
        "simulation_mode": True,
    }
    params = recipe._resolve_connection_params(config)
    assert params["host"] == "prod-db.internal"
    assert params["port"] == 5433
    assert params["database"] == "sales_warehouse"
    assert params["username"] == "dbuser"
    assert params["password"] == "SecretPass123"
    assert params["ssl_mode"] == "require"


def test_postgresql_connector_schema_and_table_selection():
    """Verify selecting a custom schema and table name formats proper SQL and metadata."""
    recipe = PostgreSQLConnectorRecipe()
    # 1. Explicit schema_name and table_name
    config1 = {
        "read_mode": "table",
        "schema_name": "analytics",
        "table_name": "monthly_revenue",
        "simulation_mode": True,
    }
    res1 = recipe.execute(inputs={}, config=config1)
    assert res1["output_summary"]["target"] == "analytics.monthly_revenue"

    # 2. Dot-delimited table name e.g. "finance.invoices"
    config2 = {
        "read_mode": "table",
        "schema_name": "public",
        "table_name": "finance.invoices",
        "simulation_mode": True,
    }
    res2 = recipe.execute(inputs={}, config=config2)
    assert res2["output_summary"]["target"] == "finance.invoices"


def test_postgresql_connector_select_all_columns_vs_specific_columns():
    """Verify select_all_columns toggle: select all columns or select specific subset."""
    recipe = PostgreSQLConnectorRecipe()

    # 1. select_all_columns = True
    config_all = {
        "read_mode": "table",
        "schema_name": "public",
        "table_name": "sales",
        "select_all_columns": True,
        "limit": 10,
        "simulation_mode": True,
    }
    res_all = recipe.execute(inputs={}, config=config_all)
    df_all = res_all["dataframe"]
    assert "amount" in df_all.columns
    assert "transaction_id" in df_all.columns
    assert "status" in df_all.columns
    assert len(df_all.columns) >= 5

    # 2. select_all_columns = False with list of columns
    config_subset_list = {
        "read_mode": "table",
        "schema_name": "public",
        "table_name": "sales",
        "select_all_columns": False,
        "columns": ["transaction_id", "amount"],
        "limit": 10,
        "simulation_mode": True,
    }
    res_subset = recipe.execute(inputs={}, config=config_subset_list)
    df_subset = res_subset["dataframe"]
    assert list(df_subset.columns) == ["transaction_id", "amount"]
    assert len(df_subset) == 10

    # 3. select_all_columns = False with comma-separated string
    config_subset_str = {
        "read_mode": "table",
        "schema_name": "public",
        "table_name": "sales",
        "select_all_columns": False,
        "columns": "customer_id, currency, status",
        "limit": 5,
        "simulation_mode": True,
    }
    res_str = recipe.execute(inputs={}, config=config_subset_str)
    df_str = res_str["dataframe"]
    assert list(df_str.columns) == ["customer_id", "currency", "status"]

    # 4. Validation: select_all_columns is False but no columns given
    errs = recipe.validate_config({
        "read_mode": "table",
        "table_name": "sales",
        "select_all_columns": False,
        "columns": "",
    })
    assert any("at least one column must be specified" in e for e in errs)


def test_postgresql_connector_where_clause_and_security():
    """Verify WHERE clause filters and SQL injection / destructive keyword protection."""
    recipe = PostgreSQLConnectorRecipe()

    # Valid WHERE and ORDER BY
    val_config = {
        "read_mode": "table",
        "table_name": "sales",
        "where_clause": "amount > 100 AND status = 'COMPLETED'",
        "order_by": "created_at DESC",
    }
    assert len(recipe.validate_config(val_config)) == 0

    # Destructive keyword in WHERE clause blocked
    bad_where = {
        "read_mode": "table",
        "table_name": "sales",
        "where_clause": "amount > 0; DROP TABLE sales;",
    }
    errs = recipe.validate_config(bad_where)
    assert any("Disallowed DDL/DML keyword 'DROP'" in e for e in errs)


def test_postgresql_connector_dynamic_inspection():
    """Verify dynamic database inspection for connection, schemas, tables, and columns."""
    recipe = PostgreSQLConnectorRecipe()
    config = {
        "connection_mode": "connection_string",
        "connection_string": "postgresql://user:pass@localhost:5432/mock_db",
        "simulation_mode": True,
    }

    # 1. Test Connection
    test_res = recipe.inspect("test_connection", config)
    assert "connected" in test_res
    assert test_res["database"] == "mock_db"

    # 2. Schemas inspection
    schemas_res = recipe.inspect("schemas", config)
    assert schemas_res["success"] is True
    assert "public" in schemas_res["schemas"]
    assert len(schemas_res["schemas"]) >= 3

    # 3. Tables inspection in selected schema
    tables_res = recipe.inspect("tables", config, schema_name="analytics")
    assert tables_res["success"] is True
    assert tables_res["schema"] == "analytics"
    assert "customers" in tables_res["tables"] or "orders" in tables_res["tables"]

    # 4. Columns inspection in selected table
    cols_res = recipe.inspect("columns", config, schema_name="public", table_name="customers")
    assert cols_res["success"] is True
    assert cols_res["table"] == "customers"
    col_names = [c["name"] for c in cols_res["columns"]]
    assert "id" in col_names or "name" in col_names or "created_at" in col_names



def test_postgresql_connector_live_mode_failure_raises_validation_exception():
    """Verify that live execution (simulation_mode=False) raises ValidationException on connection error instead of returning fake benchmark data."""
    from backend.app.core.exceptions import ValidationException
    recipe = PostgreSQLConnectorRecipe()
    config = {
        "connection_mode": "host_port",
        "host": "192.0.2.1",  # RFC 5737 TEST-NET unreachable IP
        "port": 5432,
        "database": "walmart",
        "username": "dbuser",
        "password": "Classic@123",
        "read_mode": "table",
        "table_name": "sales",
        "timeout": 1,
        "simulation_mode": False,
    }
    with pytest.raises(ValidationException) as excinfo:
        recipe.execute(inputs={}, config=config)
    assert "PostgreSQL connection to" in str(excinfo.value)
    assert "walmart" in str(excinfo.value)


def test_postgresql_connector_host_override_and_password_encoding():
    """Verify that when host is updated to a remote IP and password contains special chars like '@', connection parameters are properly synchronized and URL-encoded."""
    recipe = PostgreSQLConnectorRecipe()
    # 1. Connection string had 127.0.0.1, but user entered Host: 88.222.215.30
    config_override = {
        "connection_mode": "connection_string",
        "connection_string": "postgresql://dbuser:Classic%40123@127.0.0.1:5432/walmart",
        "host": "88.222.215.30",
        "database": "walmart",
        "username": "dbuser",
        "password": "Classic@123",
    }
    params = recipe._resolve_connection_params(config_override)
    assert params["host"] == "88.222.215.30"
    assert params["database"] == "walmart"
    assert "88.222.215.30" in params["connection_string"]
    assert "Classic%40123" in params["connection_string"]

    # 2. Host port mode with password containing '@'
    config_host_port = {
        "connection_mode": "host_port",
        "host": "88.222.215.30",
        "port": 5432,
        "database": "walmart",
        "username": "dbuser",
        "password": "Classic@123",
        "ssl_mode": "prefer",
    }
    hp_params = recipe._resolve_connection_params(config_host_port)
    assert hp_params["host"] == "88.222.215.30"
    assert hp_params["password"] == "Classic@123"
    assert "Classic%40123@88.222.215.30:5432/walmart" in hp_params["connection_string"]


def test_postgresql_connector_schema_and_multi_table_and_column_isolation():
    """Verify that schema properties have no static pre-populated enums, supports select_all_tables, multi-table lists, and strictly isolates 1 or 2 columns."""
    recipe = PostgreSQLConnectorRecipe()
    schema = recipe.get_schema()
    props = schema["properties"]

    # 1. No static hardcoded enum before connection
    assert "enum" not in props["schema_name"]
    assert "enum" not in props["table_name"]
    assert "enum" not in props["columns"]
    assert props["select_all_tables"]["input_type"] == "checkbox"
    assert props["select_all_columns"]["input_type"] == "checkbox"

    # 2. Strict 1 or 2 columns selection isolation (even if select_all_columns is left at True)
    cfg_isolated_cols = {
        "read_mode": "table",
        "schema_name": "public",
        "table_name": "sales",
        "select_all_columns": True,  # User didn't uncheck checkbox
        "columns": ["transaction_id", "amount"],  # But selected 2 specific columns
        "limit": 5,
        "simulation_mode": True,
    }
    res_iso = recipe.execute(inputs={}, config=cfg_isolated_cols)
    df_iso = res_iso["dataframe"]
    assert list(df_iso.columns) == ["transaction_id", "amount"]
    assert len(df_iso) == 5

    # 3. Multi-table selection
    cfg_multi_tables = {
        "read_mode": "table",
        "schema_name": "public",
        "tables": ["sales", "retail_sales"],
        "limit": 4,
        "simulation_mode": True,
    }
    res_multi = recipe.execute(inputs={}, config=cfg_multi_tables)
    df_multi = res_multi["dataframe"]
    assert "_table" in df_multi.columns
    assert "tables" in res_multi
    assert "sales" in res_multi["tables"]
    assert "retail_sales" in res_multi["tables"]
    assert len(df_multi) == 8

    # 4. Select all tables checkbox enabled
    cfg_all_tables = {
        "read_mode": "table",
        "schema_name": "public",
        "select_all_tables": True,
        "limit": 3,
        "simulation_mode": True,
    }
    res_all_tbls = recipe.execute(inputs={}, config=cfg_all_tables)
    assert res_all_tbls["output_summary"]["tables_count"] >= 1



def test_postgresql_connector_property_hierarchy():
    """Verify that properties in parameters_schema are arranged in a clean, logical hierarchy."""
    recipe = PostgreSQLConnectorRecipe()
    props = list(recipe.get_schema()["properties"].keys())

    # 1. Connection group: credentials establish connection to server
    assert props.index("connection_mode") < props.index("credential_id") < props.index("host")
    assert props.index("host") < props.index("port") < props.index("username") < props.index("password")
    assert props.index("password") < props.index("connection_string") < props.index("ssl_mode") < props.index("timeout")

    # 2. Read mode and cascading discovery hierarchy: Database -> Schema -> Tables -> Columns
    assert props.index("timeout") < props.index("read_mode") < props.index("database") < props.index("schema_name")
    assert props.index("schema_name") < props.index("select_all_tables") < props.index("tables") < props.index("table_name")
    assert props.index("table_name") < props.index("select_all_columns") < props.index("columns")
    assert props.index("columns") < props.index("where_clause") < props.index("order_by") < props.index("query")

    # 3. Limits & pagination at bottom
    assert props.index("query") < props.index("limit") < props.index("offset") < props.index("simulation_mode")


def test_postgresql_connector_inspect_databases_and_columns_metadata():
    """Verify inspection for databases action and rich column metadata including primary_key."""
    recipe = PostgreSQLConnectorRecipe()
    config = {"simulation_mode": True, "database": "walmart"}

    # 1. Databases action
    res_db = recipe.inspect("databases", config)
    assert res_db["success"] is True
    assert "databases" in res_db
    assert len(res_db["databases"]) >= 2
    assert "walmart" in res_db["databases"] or "postgres" in res_db["databases"]

    # 2. Rich columns metadata with primary_key
    res_cols = recipe.inspect("columns", config, schema_name="public", table_name="sales")
    assert res_cols["success"] is True
    assert "columns" in res_cols
    assert len(res_cols["columns"]) > 0
    for col in res_cols["columns"]:
        assert "name" in col
        assert "type" in col
        assert "nullable" in col
        assert "primary_key" in col
        assert isinstance(col["primary_key"], bool)

    pk_cols = [c for c in res_cols["columns"] if c["primary_key"]]
    assert len(pk_cols) >= 1
    assert pk_cols[0]["name"] == "invoice_id"


def test_mysql_connector_inspection():
    """Verify that MySQL connector provides real-time dynamic inspection."""
    from backend.app.recipes.ingestion.mysql_connector import MySQLConnectorRecipe
    recipe = MySQLConnectorRecipe()
    config = {"simulation_mode": True, "database": "shop_db"}

    # 1. Test Connection
    res_conn = recipe.inspect("test_connection", config)
    assert res_conn["success"] is True
    assert "connected" in res_conn

    # 2. Databases / Schemas
    res_db = recipe.inspect("databases", config)
    assert res_db["success"] is True
    assert "shop_db" in res_db["databases"]

    # 3. Tables
    res_tbl = recipe.inspect("tables", config, schema_name="shop_db")
    assert res_tbl["success"] is True
    assert "orders" in res_tbl["tables"] or "customers" in res_tbl["tables"]

    # 4. Columns
    res_col = recipe.inspect("columns", config, schema_name="shop_db", table_name="orders")
    assert res_col["success"] is True
    assert len(res_col["columns"]) > 0
    assert any(c["name"] == "order_id" for c in res_col["columns"])
    pk_col = next(c for c in res_col["columns"] if c["name"] == "order_id")
    assert pk_col["primary_key"] is True


def test_postgresql_connector_error_when_unreachable_and_not_simulated():
    """Verify that when simulation_mode is False and DB is unreachable, inspect returns an explicit error."""
    recipe = PostgreSQLConnectorRecipe()
    config = {
        "simulation_mode": False,
        "host": "192.0.2.1",  # TEST-NET-1 (RFC 5737 unreachable dummy IP)
        "port": 5432,
        "database": "walmart",
        "username": "dbuser",
        "password": "wrongpassword",
        "timeout": 1,
    }

    # Test connection action
    res_test = recipe.inspect("test_connection", config)
    assert res_test["success"] is False
    assert res_test["connected"] is False
    assert res_test["status"] == "UNREACHABLE"
    assert "error" in res_test

    # Databases action
    res_db = recipe.inspect("databases", config)
    assert res_db["success"] is False
    assert res_db["source"] == "ERROR"
    assert res_db["databases"] == []

    # Tables action
    res_tbl = recipe.inspect("tables", config, schema_name="public")
    assert res_tbl["success"] is False
    assert res_tbl["source"] == "ERROR"
    assert res_tbl["tables"] == []

    # Columns action
    res_col = recipe.inspect("columns", config, schema_name="public", table_name="sales")
    assert res_col["success"] is False
    assert res_col["source"] == "ERROR"
    assert res_col["columns"] == []


def test_credential_encryption_and_serialization():
    """Verify credential encryption/decryption roundtrip and Credential model to_dict()."""
    import uuid
    from datetime import datetime, timezone
    from backend.app.credentials.service import encrypt_payload, decrypt_payload
    from backend.app.credentials.models import Credential

    raw_payload = {
        "host": "88.222.215.30",
        "port": 5432,
        "database": "walmart",
        "username": "dbuser",
        "password": "Classic@123",
    }
    encrypted = encrypt_payload(raw_payload)
    assert isinstance(encrypted, str)
    assert encrypted != ""

    decrypted = decrypt_payload(encrypted)
    assert decrypted["host"] == "88.222.215.30"
    assert decrypted["password"] == "Classic@123"

    # Test model
    now = datetime.now(timezone.utc)
    cred = Credential(
        id=str(uuid.uuid4()),
        tenant_id=1,
        user_id="user-123",
        provider="postgresql",
        name="Walmart Production",
        auth_type="database",
        encrypted_data=encrypted,
        metadata_info={"host": "88.222.215.30", "database": "walmart"},
        is_active=True,
        created_at=now,
        updated_at=now,
    )
    data = cred.to_dict()
    assert data["provider"] == "postgresql"
    assert data["name"] == "Walmart Production"
    assert data["metadata_info"]["database"] == "walmart"
    assert "password" not in data["metadata_info"]  # metadata_info must never leak password


def test_request_and_response_pydantic_models():
    """Verify CreateCredentialRequest, RecipeInspectRequest, and RecipeInspectResponse."""
    from backend.app.credentials.router import CreateCredentialRequest
    from backend.app.recipes.router import RecipeInspectRequest, RecipeInspectResponse

    # Credential request
    cred_req = CreateCredentialRequest(
        provider="postgresql",
        name="Test DB",
        data={"host": "88.222.215.30", "database": "walmart"},
        auth_type="database",
    )
    assert cred_req.provider == "postgresql"
    assert cred_req.data["host"] == "88.222.215.30"

    # Recipe inspect request
    inspect_req = RecipeInspectRequest(
        action="tables",
        schema_name="public",
        database_name="walmart",
        save_credential=True,
        credential_name="Saved DB",
        config={"host": "88.222.215.30"},
    )
    assert inspect_req.action == "tables"
    assert inspect_req.save_credential is True

    # Recipe inspect response
    inspect_resp = RecipeInspectResponse(
        recipe_id="postgresql_connector",
        action="tables",
        success=True,
        data={"tables": ["sales", "customers"]},
        saved_credential_id="cred-uuid-123",
    )
    assert inspect_resp.success is True
    assert inspect_resp.saved_credential_id == "cred-uuid-123"


def test_mysql_connector_property_hierarchy():
    """Verify that properties in MySQL parameters_schema are arranged in clean cascading hierarchy."""
    recipe = MySQLConnectorRecipe()
    props = list(recipe.get_schema()["properties"].keys())

    # 1. Connection credentials
    assert props.index("connection_mode") < props.index("credential_id") < props.index("host")
    assert props.index("host") < props.index("port") < props.index("username") < props.index("password")
    assert props.index("password") < props.index("connection_string") < props.index("charset") < props.index("timeout")

    # 2. Cascading data discovery: Database -> Tables -> Columns
    assert props.index("timeout") < props.index("read_mode") < props.index("database")
    assert props.index("database") < props.index("select_all_tables") < props.index("tables") < props.index("table_name")
    assert props.index("table_name") < props.index("select_all_columns") < props.index("columns")
    assert props.index("columns") < props.index("where_clause") < props.index("order_by") < props.index("query")

    # 3. Limits & pagination
    assert props.index("query") < props.index("limit") < props.index("offset") < props.index("simulation_mode")


def test_mysql_connector_error_when_unreachable_and_not_simulated():
    """Verify that when simulation_mode is False and DB is unreachable, execute raises ValidationException and inspect returns error."""
    from backend.app.core.exceptions import ValidationException

    recipe = MySQLConnectorRecipe()
    config = {
        "simulation_mode": False,
        "host": "192.0.2.1",  # Dummy test IP
        "port": 3306,
        "database": "ProjectX",
        "username": "MyUser",
        "password": "wrongpassword",
        "table_name": "Businesses",
        "timeout": 1,
    }

    # 1. Execute must strictly raise ValidationException with server error, NOT fake data!
    with pytest.raises(ValidationException) as exc_info:
        recipe.execute(inputs={}, config=config)
    assert "MySQL connection to" in str(exc_info.value)

    # 2. Inspect test_connection must return UNREACHABLE
    res_test = recipe.inspect("test_connection", config)
    assert res_test["success"] is False
    assert res_test["connected"] is False
    assert res_test["status"] == "UNREACHABLE"
    assert "error" in res_test

    # 3. Inspect databases must return error, NOT fake databases
    res_db = recipe.inspect("databases", config)
    assert res_db["success"] is False
    assert res_db["databases"] == []

    # 4. Inspect tables must return error, NOT fake tables
    res_tbl = recipe.inspect("tables", config, schema_name="ProjectX")
    assert res_tbl["success"] is False
    assert res_tbl["tables"] == []

    # 5. Inspect columns must return error, NOT fake columns
    res_col = recipe.inspect("columns", config, schema_name="ProjectX", table_name="Businesses")
    assert res_col["success"] is False
    assert res_col["columns"] == []


def test_snowflake_and_mongodb_connector_error_when_unreachable():
    """Verify that Snowflake and MongoDB raise ValidationException when live connection fails."""
    from backend.app.recipes.ingestion.snowflake_connector import SnowflakeConnectorRecipe
    from backend.app.recipes.ingestion.mongodb_connector import MongoDBConnectorRecipe
    from backend.app.core.exceptions import ValidationException

    # Snowflake live failure
    sf_recipe = SnowflakeConnectorRecipe()
    with pytest.raises(ValidationException):
        sf_recipe.execute(inputs={}, config={"simulation_mode": False, "account": "invalid", "user": "test", "password": "x"})

    # MongoDB live failure
    mongo_recipe = MongoDBConnectorRecipe()
    with pytest.raises(ValidationException):
        mongo_recipe.execute(inputs={}, config={"simulation_mode": False, "host": "192.0.2.1", "port": 27017, "timeout": 1})


def test_connectors_have_clean_defaults_and_no_dummy_queries():
    """Verify that all connectors have clean empty defaults for queries and tables, with helpful placeholders."""
    from backend.app.recipes.ingestion.mysql_connector import MySQLConnectorRecipe
    from backend.app.recipes.ingestion.postgresql_connector import PostgreSQLConnectorRecipe
    from backend.app.recipes.ingestion.snowflake_connector import SnowflakeConnectorRecipe
    from backend.app.recipes.ingestion.redshift_connector import RedshiftConnectorRecipe
    from backend.app.recipes.ingestion.bigquery_connector import BigQueryConnectorRecipe

    for recipe_cls in [MySQLConnectorRecipe, PostgreSQLConnectorRecipe, SnowflakeConnectorRecipe, RedshiftConnectorRecipe, BigQueryConnectorRecipe]:
        r = recipe_cls()
        props = r.get_schema()["properties"]
        assert "query" in props, f"Missing 'query' in {r.recipe_id}"
        assert props["query"].get("default", "") == "", f"Query default must be empty in {r.recipe_id}"
        assert "placeholder" in props["query"], f"Query must have placeholder in {r.recipe_id}"

        tbl_key = "table_id" if "table_id" in props else "table_name"
        if tbl_key in props:
            assert props[tbl_key].get("default", "") == "", f"{tbl_key} default must be empty in {r.recipe_id}"
            assert "placeholder" in props[tbl_key], f"{tbl_key} must have placeholder in {r.recipe_id}"


def test_node_test_step_response_schema_and_failure_summary():
    """Verify that NodeTestStepResponse holds rich diagnostic output_summary on failure."""
    from backend.app.workflows.schemas import NodeTestStepResponse

    err_summary = {
        "title": "MySQL Connection Failed",
        "status": "FAILED",
        "message": "Can't connect to MySQL server on '88.222.215.30'",
        "error": "Can't connect to MySQL server on '88.222.215.30'",
        "suggestion": "Please verify credentials, firewall, and host/port.",
        "duration_ms": 120.5
    }

    resp = NodeTestStepResponse(
        success=False,
        session_id="test_sess",
        node_id="node_mysql",
        recipe_id="mysql_connector",
        recipe_name="MySQL Connector",
        status="FAILED",
        duration_ms=120.5,
        error_message="Can't connect to MySQL server on '88.222.215.30'",
        error_title="MySQL Connection Failed",
        error_suggestion="Please verify credentials, firewall, and host/port.",
        output_summary=err_summary
    )

    assert resp.status == "FAILED"
    assert resp.success is False
    assert resp.output_summary["status"] == "FAILED"
    assert resp.output_summary["error"] == "Can't connect to MySQL server on '88.222.215.30'"
    assert resp.output_summary["title"] == "MySQL Connection Failed"
    assert "firewall" in resp.output_summary["suggestion"]


def test_mongodb_clean_schema_defaults_and_placeholders():
    """Verify that MongoDB connector has empty defaults and descriptive placeholders."""
    from backend.app.recipes.ingestion.mongodb_connector import MongoDBConnectorRecipe
    r = MongoDBConnectorRecipe()
    props = r.get_schema()["properties"]

    assert props["connection_string"].get("default", "") == ""
    assert "placeholder" in props["connection_string"]

    assert props["host"].get("default", "") == ""
    assert "placeholder" in props["host"]

    assert props["database"].get("default", "") == ""
    assert "placeholder" in props["database"]

    assert props["collection"].get("default", "") == ""
    assert "placeholder" in props["collection"]


def test_mongodb_resolve_connection_params_host_port():
    """Verify host_port mode builds clean URI and does not let leftover connection_string overwrite it."""
    from backend.app.recipes.ingestion.mongodb_connector import MongoDBConnectorRecipe
    r = MongoDBConnectorRecipe()
    cfg = {
        "connection_mode": "host_port",
        "connection_string": "",
        "host": "88.222.215.30",
        "port": 27017,
        "username": "my_user",
        "password": "my_password",
        "database": "production_db",
        "collection": "events",
        "auth_source": "admin"
    }
    params = r._resolve_connection_params(cfg)
    assert params["host"] == "88.222.215.30"
    assert params["port"] == 27017
    assert params["database"] == "production_db"
    assert params["collection"] == "events"
    assert "mongodb://my_user:my_password@88.222.215.30:27017/production_db?authSource=admin" == params["connection_string"]


def test_draft_nodes_do_not_block_graph_validation():
    """Verify that unconfigured draft nodes on canvas with 0 edges do not block validation if another node is configured."""
    nodes = [
        WorkflowNode(
            id="mongo_1",
            recipe_id="mongodb_connector",
            config={"database": "test_db", "collection": "test_coll", "simulation_mode": True}
        ),
        WorkflowNode(
            id="snowflake_draft",
            recipe_id="snowflake_connector",
            config={}  # unconfigured draft node
        ),
        WorkflowNode(
            id="bigquery_draft",
            recipe_id="bigquery_connector",
            config={}  # unconfigured draft node
        ),
        WorkflowNode(
            id="redshift_draft",
            recipe_id="redshift_connector",
            config={}  # unconfigured draft node
        )
    ]
    graph = WorkflowGraph(id="wf_draft_test", name="Draft Canvas Test", nodes=nodes, edges=[])
    diag = graph.get_diagnostics()

    assert diag["is_valid"] is True, f"Graph should be valid despite draft nodes! Errors: {diag['errors']}"
    assert len(diag["errors"]) == 0
    assert len(diag["warnings"]) >= 3
    draft_warnings = [w["message"] for w in diag["warnings"] if "Incomplete Draft Node" in w["message"]]
    assert len(draft_warnings) >= 3


def test_connected_node_configuration_error_blocks_validation():
    """Verify that if a node IS connected to an active pipeline and missing configuration, it blocks validation."""
    nodes = [
        WorkflowNode(
            id="mongo_1",
            recipe_id="mongodb_connector",
            config={"database": "test_db", "collection": "test_coll", "simulation_mode": True}
        ),
        WorkflowNode(
            id="snowflake_connected",
            recipe_id="snowflake_connector",
            config={}  # connected but missing table_name
        )
    ]
    edges = [
        WorkflowEdge(id="e1", source="mongo_1", target="snowflake_connected")
    ]
    graph = WorkflowGraph(id="wf_conn_test", name="Connected Test", nodes=nodes, edges=edges)
    diag = graph.get_diagnostics()

    assert diag["is_valid"] is False
    assert any("Configuration Error" in e["message"] for e in diag["errors"])


def test_executor_skips_unconnected_draft_nodes():
    """Verify DAGExecutor executes configured nodes and gracefully skips unconnected draft nodes."""
    nodes = [
        WorkflowNode(
            id="mongo_configured",
            recipe_id="mongodb_connector",
            config={
                "database": "telemetry",
                "collection": "events",
                "limit": 10,
                "flatten_nested": True,
                "simulation_mode": True
            }
        ),
        WorkflowNode(
            id="snowflake_draft",
            recipe_id="snowflake_connector",
            config={}  # unconfigured draft
        )
    ]
    graph = WorkflowGraph(id="wf_skip_draft", name="Skip Draft DAG Test", nodes=nodes, edges=[])
    exec_result = DAGExecutor.execute_workflow(
        execution_id="test_skip_draft_exec",
        workflow=graph,
        include_node_outputs=True
    )

    assert exec_result.status == "SUCCESS"
    assert len(exec_result.node_results) == 2
    res_mongo = next(r for r in exec_result.node_results if r.node_id == "mongo_configured")
    assert res_mongo.status == "SUCCESS"
    res_draft = next(r for r in exec_result.node_results if r.node_id == "snowflake_draft")
    assert res_draft.status == "SKIPPED"


def test_connector_schema_property_ordering_consistency():
    """Verify that all database and warehouse connectors follow the standardized 3-section ordering and visible_when spec."""
    from backend.app.recipes.ingestion.mysql_connector import MySQLConnectorRecipe
    from backend.app.recipes.ingestion.postgresql_connector import PostgreSQLConnectorRecipe
    from backend.app.recipes.ingestion.mongodb_connector import MongoDBConnectorRecipe
    from backend.app.recipes.ingestion.snowflake_connector import SnowflakeConnectorRecipe
    from backend.app.recipes.ingestion.redshift_connector import RedshiftConnectorRecipe
    from backend.app.recipes.ingestion.bigquery_connector import BigQueryConnectorRecipe

    connectors = [
        MySQLConnectorRecipe(),
        PostgreSQLConnectorRecipe(),
        MongoDBConnectorRecipe(),
        SnowflakeConnectorRecipe(),
        RedshiftConnectorRecipe(),
        BigQueryConnectorRecipe()
    ]

    expected_section_ids = {"connection", "query_selection", "runtime_limits"}

    for conn in connectors:
        # Connectors are ingestion entry points: no upstream required
        assert len(conn.inputs) == 0, f"{conn.recipe_id} inputs should be empty list, got {conn.inputs}"
        assert len(conn.outputs) >= 1, f"{conn.recipe_id} should provide an output port"

        schema = conn.get_schema()
        # 1. Verify standard accordion sections metadata
        assert "sections" in schema, f"{conn.recipe_id} schema missing 'sections'"
        sections = schema["sections"]
        assert len(sections) == 3, f"{conn.recipe_id} sections should contain 3 sections"
        section_ids = {s["id"] for s in sections}
        assert section_ids == expected_section_ids, f"{conn.recipe_id} section IDs mismatch: {section_ids}"

        props = schema["properties"]
        keys = list(props.keys())

        # 2. Connection mode must be the first property
        assert keys[0] == "connection_mode", f"{conn.recipe_id} first property should be connection_mode, got {keys[0]}"

        # 3. credential_id must be in connection section and have visible_when
        assert "credential_id" in keys
        assert keys.index("credential_id") <= 2, f"{conn.recipe_id} credential_id should be near top, got index {keys.index('credential_id')}"
        cred_field = props["credential_id"]
        assert cred_field.get("section") == "connection"
        assert cred_field.get("visible_when") == {"field": "connection_mode", "equals": "credential"}

        # 4. Target data fields (database or dataset_id) must appear after credentials
        target_key = "dataset_id" if "dataset_id" in keys else "database"
        assert keys.index(target_key) > keys.index("credential_id")

        # 5. Limit must appear in runtime_limits
        assert "limit" in keys
        assert keys.index("limit") > keys.index(target_key)
        assert props["limit"].get("section") == "runtime_limits"

        # 6. Every property must specify a valid section
        for prop_name, prop_def in props.items():
            assert "section" in prop_def, f"{conn.recipe_id}.{prop_name} is missing 'section'"
            assert prop_def["section"] in expected_section_ids, f"{conn.recipe_id}.{prop_name} has invalid section: {prop_def['section']}"

        # 7. Relational / warehouse connectors with read_mode have visible_when on query
        if "read_mode" in props and "query" in props:
            assert props["query"].get("visible_when") == {"field": "read_mode", "equals": "query"}


def test_cloud_database_warehouse_sso_schema_support():
    """Verify that all cloud database and warehouse connectors provide SSO in connection_mode with full schema."""
    from backend.app.recipes.ingestion.snowflake_connector import SnowflakeConnectorRecipe
    from backend.app.recipes.ingestion.redshift_connector import RedshiftConnectorRecipe
    from backend.app.recipes.ingestion.bigquery_connector import BigQueryConnectorRecipe
    from backend.app.recipes.ingestion.postgresql_connector import PostgreSQLConnectorRecipe
    from backend.app.recipes.ingestion.mysql_connector import MySQLConnectorRecipe
    from backend.app.recipes.ingestion.mongodb_connector import MongoDBConnectorRecipe
    from backend.app.recipes.ingestion.azure_data_lake_connector import AzureDataLakeConnectorRecipe

    sso_connectors = [
        (SnowflakeConnectorRecipe(), "connection_mode", ["sso_authenticator", "sso_account", "sso_user", "sso_token"]),
        (RedshiftConnectorRecipe(), "connection_mode", ["sso_provider", "sso_host", "sso_username", "sso_token"]),
        (BigQueryConnectorRecipe(), "connection_mode", ["sso_provider", "sso_access_token"]),
        (PostgreSQLConnectorRecipe(), "connection_mode", ["sso_provider", "sso_host", "sso_username", "sso_token"]),
        (MySQLConnectorRecipe(), "connection_mode", ["sso_provider", "sso_host", "sso_username", "sso_token"]),
        (MongoDBConnectorRecipe(), "connection_mode", ["sso_provider", "sso_host", "sso_username", "sso_token"]),
        (AzureDataLakeConnectorRecipe(), "auth_mode", ["tenant_id", "client_id", "sso_token"]),
    ]

    for recipe, mode_field, expected_sso_props in sso_connectors:
        schema = recipe.get_schema()
        props = schema["properties"]
        assert mode_field in props, f"{recipe.recipe_id} missing {mode_field}"
        mode_def = props[mode_field]
        assert "sso" in mode_def["enum"], f"{recipe.recipe_id} {mode_field} does not have 'sso' in enum: {mode_def['enum']}"

        for prop_name in expected_sso_props:
            assert prop_name in props, f"{recipe.recipe_id} missing expected SSO property '{prop_name}'"
            p_def = props[prop_name]
            assert p_def.get("section") == "connection", f"{recipe.recipe_id}.{prop_name} not in 'connection' section"
            assert p_def.get("visible_when") == {"field": mode_field, "equals": "sso"}, (
                f"{recipe.recipe_id}.{prop_name} visible_when mismatch: {p_def.get('visible_when')}"
            )


def test_cloud_database_warehouse_sso_validation_and_resolution():
    """Verify that SSO configurations validate and resolve correctly across all cloud DBs."""
    from backend.app.recipes.ingestion.snowflake_connector import SnowflakeConnectorRecipe
    from backend.app.recipes.ingestion.redshift_connector import RedshiftConnectorRecipe
    from backend.app.recipes.ingestion.bigquery_connector import BigQueryConnectorRecipe
    from backend.app.recipes.ingestion.postgresql_connector import PostgreSQLConnectorRecipe
    from backend.app.recipes.ingestion.mysql_connector import MySQLConnectorRecipe
    from backend.app.recipes.ingestion.mongodb_connector import MongoDBConnectorRecipe
    from backend.app.recipes.ingestion.azure_data_lake_connector import AzureDataLakeConnectorRecipe

    # 1. Snowflake SSO
    sf = SnowflakeConnectorRecipe()
    assert sf.validate_config({
        "connection_mode": "sso",
        "sso_authenticator": "externalbrowser",
        "sso_account": "xy12345.us-east-1",
        "sso_user": "analyst@company.com",
        "read_mode": "table",
        "table_name": "ORDERS"
    }) == []
    # Missing user/account in SSO should produce error
    sf_errs = sf.validate_config({"connection_mode": "sso", "read_mode": "table", "table_name": "ORDERS"})
    assert any("Account" in e for e in sf_errs)
    assert any("Username" in e for e in sf_errs)

    # 2. Redshift SSO
    rs = RedshiftConnectorRecipe()
    assert rs.validate_config({
        "connection_mode": "sso",
        "sso_provider": "aws_iam",
        "sso_host": "redshift-cluster.amazonaws.com",
        "sso_username": "awsuser",
        "read_mode": "table",
        "table_name": "sales"
    }) == []
    rs_errs = rs.validate_config({"connection_mode": "sso", "read_mode": "table", "table_name": "sales"})
    assert any("Host" in e or "host" in e for e in rs_errs)

    # 3. BigQuery SSO
    bq = BigQueryConnectorRecipe()
    assert bq.validate_config({
        "connection_mode": "sso",
        "sso_provider": "google_sso",
        "project_id": "analytics-prod-123",
        "read_mode": "table",
        "table_id": "events"
    }) == []

    # 4. PostgreSQL SSO
    pg = PostgreSQLConnectorRecipe()
    assert pg.validate_config({
        "connection_mode": "sso",
        "sso_provider": "aws_iam_rds",
        "sso_host": "rds.amazonaws.com",
        "sso_username": "dbuser",
        "read_mode": "table",
        "table_name": "users"
    }) == []
    pg_params = pg._resolve_connection_params({
        "connection_mode": "sso",
        "sso_host": "rds.amazonaws.com",
        "sso_port": 5432,
        "sso_username": "dbuser",
        "sso_token": "token123",
        "database": "users_db"
    })
    assert "rds.amazonaws.com" in pg_params["connection_string"]
    assert "token123" in pg_params["connection_string"]

    # 5. MySQL SSO
    my = MySQLConnectorRecipe()
    assert my.validate_config({
        "connection_mode": "sso",
        "sso_provider": "aws_iam_rds",
        "sso_host": "rds-mysql.amazonaws.com",
        "sso_username": "dbuser",
        "read_mode": "table",
        "table_name": "orders"
    }) == []
    my_params = my._resolve_connection_params({
        "connection_mode": "sso",
        "sso_host": "rds-mysql.amazonaws.com",
        "sso_username": "dbuser",
        "sso_token": "mysql_token",
        "database": "shop_db"
    })
    assert "rds-mysql.amazonaws.com" in my_params["connection_string"]

    # 6. MongoDB SSO
    mg = MongoDBConnectorRecipe()
    assert mg.validate_config({
        "connection_mode": "sso",
        "sso_provider": "mongodb_oidc",
        "sso_host": "cluster0.mongodb.net",
        "database": "analytics",
        "collection": "events"
    }) == []
    mg_params = mg._resolve_connection_params({
        "connection_mode": "sso",
        "sso_provider": "mongodb_oidc",
        "sso_host": "cluster0.mongodb.net",
        "sso_username": "analyst@company.com",
        "database": "analytics",
        "collection": "events"
    })
    assert "authMechanism=MONGODB-OIDC" in mg_params["connection_string"]

    # 7. Azure Data Lake SSO
    adls = AzureDataLakeConnectorRecipe()
    assert adls.validate_config({
        "auth_mode": "sso",
        "account_name": "analyticslake",
        "tenant_id": "00000000-0000-0000-0000-000000000000",
        "container_name": "raw",
        "blob_path": "events/data.parquet"
    }) == []

