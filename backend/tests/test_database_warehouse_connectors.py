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
