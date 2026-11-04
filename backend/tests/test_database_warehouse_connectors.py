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
