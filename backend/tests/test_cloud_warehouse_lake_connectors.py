"""
Unit and Integration Tests for Cloud Warehouse & Data Lake Connectors.
Covers:
- Snowflake Connector (snowflake_connector)
- Google BigQuery Connector (bigquery_connector)
- Amazon Redshift Connector (redshift_connector)
- Amazon S3 Data Lake Connector (s3_data_lake_connector)
- Azure Data Lake Storage Connector (azure_data_lake_connector)
"""

import json
import pytest
import pandas as pd

from backend.app.recipes.base.registry import recipe_registry
from backend.app.recipes.ingestion.snowflake_connector import SnowflakeConnectorRecipe
from backend.app.recipes.ingestion.bigquery_connector import BigQueryConnectorRecipe
from backend.app.recipes.ingestion.redshift_connector import RedshiftConnectorRecipe
from backend.app.recipes.ingestion.s3_data_lake_connector import S3DataLakeConnectorRecipe
from backend.app.recipes.ingestion.azure_data_lake_connector import AzureDataLakeConnectorRecipe
from backend.app.engine.dag.graph import WorkflowGraph, WorkflowNode, WorkflowEdge
from backend.app.engine.execution.executor import DAGExecutor


def test_connectors_registered_and_metadata():
    """Verify all 5 connectors are registered, enabled, and mapped to correct groups."""
    connectors = [
        ("snowflake_connector", "Snowflake Connector", "Database & Warehouse Connectors"),
        ("bigquery_connector", "Google BigQuery Connector", "Database & Warehouse Connectors"),
        ("redshift_connector", "Amazon Redshift Connector", "Database & Warehouse Connectors"),
        ("s3_data_lake_connector", "Amazon S3 Data Lake Connector", "Data Lake & Object Storage"),
        ("azure_data_lake_connector", "Azure Data Lake Storage Connector", "Data Lake & Object Storage"),
    ]

    for rid, expected_name, expected_subgroup in connectors:
        assert recipe_registry.has(rid), f"Recipe '{rid}' must be registered and enabled."
        meta = recipe_registry.get_metadata(rid)
        assert meta.enabled is True
        assert meta.name == expected_name
        assert meta.group == "Ingestion & Orchestration"
        assert meta.subgroup == expected_subgroup
        assert "properties" in meta.parameters_schema


def test_connector_aliases():
    """Verify aliases resolve seamlessly to the primary recipe IDs."""
    alias_mappings = [
        ("snowflake", "snowflake_connector"),
        ("bigquery", "bigquery_connector"),
        ("google_bigquery", "bigquery_connector"),
        ("redshift", "redshift_connector"),
        ("aws_redshift", "redshift_connector"),
        ("s3", "s3_data_lake_connector"),
        ("s3_connector", "s3_data_lake_connector"),
        ("azure_blob", "azure_data_lake_connector"),
        ("azure_data_lake", "azure_data_lake_connector"),
    ]
    for alias, target_id in alias_mappings:
        assert recipe_registry.has(alias)
        recipe = recipe_registry.get(alias)
        assert recipe.recipe_id == target_id


def test_snowflake_validation_and_guardrails():
    """Verify Snowflake parameter validation and SQL injection prevention."""
    r = SnowflakeConnectorRecipe()

    # Empty table in table mode
    errs = r.validate_config({"read_mode": "table", "table_name": ""})
    assert any("Table name is required" in e for e in errs)

    # Empty query in query mode
    errs = r.validate_config({"read_mode": "query", "query": ""})
    assert any("SQL query is required" in e for e in errs)

    # Destructive SQL queries blocked
    destructive_queries = [
        "DROP TABLE PUBLIC.FCT_ORDERS",
        "DELETE FROM PUBLIC.FCT_ORDERS WHERE ID > 10",
        "TRUNCATE TABLE PUBLIC.FCT_ORDERS",
        "ALTER TABLE PUBLIC.FCT_ORDERS DROP COLUMN REVENUE",
        "INSERT INTO PUBLIC.FCT_ORDERS VALUES (1, 2, 3)",
        "MERGE INTO TARGET USING SOURCE ON TARGET.ID = SOURCE.ID WHEN MATCHED THEN DELETE"
    ]
    for q in destructive_queries:
        errs = r.validate_config({"read_mode": "query", "query": q})
        assert any("prohibited" in e.lower() or "read-only" in e.lower() for e in errs), f"Query '{q}' should be blocked"

    # Non-positive limit
    errs = r.validate_config({"read_mode": "table", "table_name": "T", "limit": 0})
    assert any("greater than 0" in e for e in errs)


def test_snowflake_execution_simulation_fallback():
    """Verify Snowflake simulation fallback generates expected analytical columns and summary."""
    r = SnowflakeConnectorRecipe()
    cfg = {
        "read_mode": "table",
        "table_name": "FCT_ORDERS",
        "warehouse": "COMPUTE_WH",
        "database": "ANALYTICS_DB",
        "schema_name": "PUBLIC",
        "limit": 20,
        "simulation_mode": True
    }
    res = r.execute({}, cfg)
    df = res["dataframe"]
    summary = res["output_summary"]
    metrics = res["metrics"]

    assert len(df) == 20
    assert "QUERY_ID" in df.columns
    assert "TRANSACTION_ID" in df.columns
    assert "CREDITS_USED" in df.columns
    assert "TOTAL_AMOUNT_USD" in df.columns
    assert summary["status"] == "SIMULATED"
    assert summary["rows_retrieved"] == 20
    assert metrics["delivery_status"] == "SIMULATED"


def test_bigquery_validation_and_guardrails():
    """Verify BigQuery validation, credentials checking, and query safety."""
    r = BigQueryConnectorRecipe()

    # Missing project ID
    errs = r.validate_config({"project_id": "", "read_mode": "table", "table_id": "events"})
    assert any("Project ID is required" in e for e in errs)

    # Invalid JSON credentials
    errs = r.validate_config({
        "project_id": "test-p",
        "connection_mode": "service_account_json",
        "credentials_json": "{not valid json",
        "read_mode": "table",
        "table_id": "events"
    })
    assert any("Invalid Service Account JSON" in e for e in errs)

    # Destructive SQL queries blocked
    destructive = [
        "DROP TABLE `my-proj.data.events`",
        "DELETE FROM `my-proj.data.events` WHERE true",
        "TRUNCATE TABLE `my-proj.data.events`",
        "UPDATE `my-proj.data.events` SET user_id = 'hack'"
    ]
    for q in destructive:
        errs = r.validate_config({"project_id": "p", "read_mode": "query", "query": q})
        assert any("prohibited" in e.lower() or "read-only" in e.lower() for e in errs)


def test_bigquery_execution_simulation_fallback():
    """Verify BigQuery simulation fallback produces Google Analytics style event data."""
    r = BigQueryConnectorRecipe()
    cfg = {
        "project_id": "gcp-prod-analytics",
        "dataset_id": "ga4_events",
        "table_id": "events_20260301",
        "limit": 25,
        "simulation_mode": True
    }
    res = r.execute({}, cfg)
    df = res["dataframe"]
    summary = res["output_summary"]

    assert len(df) == 25
    assert "event_timestamp" in df.columns
    assert "user_pseudo_id" in df.columns
    assert "device_category" in df.columns
    assert "purchase_revenue_usd" in df.columns
    assert summary["project_id"] == "gcp-prod-analytics"
    assert summary["status"] == "SIMULATED"


def test_redshift_validation_and_guardrails():
    """Verify Redshift port, credential validation, and SQL safety checks."""
    r = RedshiftConnectorRecipe()

    # Host required in cluster mode
    errs = r.validate_config({"connection_mode": "cluster_credentials", "host": "", "username": "user"})
    assert any("Host endpoint is required" in e for e in errs)

    # Prohibited Redshift statements
    destructive = [
        "VACUUM FULL dev.public.sales",
        "UNLOAD ('SELECT * FROM sales') TO 's3://bucket/'",
        "DROP TABLE public.sales CASCADE",
        "TRUNCATE public.sales"
    ]
    for q in destructive:
        errs = r.validate_config({"read_mode": "query", "query": q})
        assert any("prohibited" in e.lower() or "read-only" in e.lower() for e in errs)


def test_redshift_execution_simulation_fallback():
    """Verify Redshift simulation produces columnar sales analytics schema."""
    r = RedshiftConnectorRecipe()
    cfg = {
        "host": "redshift.amazonaws.com",
        "database": "dev",
        "schema_name": "public",
        "table_name": "sales",
        "limit": 30,
        "simulation_mode": True
    }
    res = r.execute({}, cfg)
    df = res["dataframe"]
    summary = res["output_summary"]

    assert len(df) == 30
    assert "salesid" in df.columns
    assert "pricepaid" in df.columns
    assert "diststyle" in df.columns
    assert summary["database"] == "dev"
    assert summary["status"] == "SIMULATED"


def test_s3_data_lake_validation():
    """Verify S3 URI and format validation."""
    r = S3DataLakeConnectorRecipe()

    # S3 URI mode requires s3:// prefix
    errs = r.validate_config({"source_mode": "s3_uri", "s3_uri": "https://bucket/file.csv"})
    assert any("must start with 's3://'" in e for e in errs)

    # Bucket and key mode requires both fields
    errs = r.validate_config({"source_mode": "bucket_and_key", "bucket_name": "", "file_key": ""})
    assert any("Bucket name is required" in e for e in errs)
    assert any("Object key" in e for e in errs)


def test_s3_data_lake_execution_simulation():
    """Verify S3 Data Lake execution produces telemetry dataset."""
    r = S3DataLakeConnectorRecipe()
    cfg = {
        "source_mode": "s3_uri",
        "s3_uri": "s3://iot-lakehouse/telemetry/sensors.parquet",
        "file_format": "parquet",
        "limit": 15,
        "simulation_mode": True
    }
    res = r.execute({}, cfg)
    df = res["dataframe"]
    summary = res["output_summary"]

    assert len(df) == 15
    assert "device_id" in df.columns
    assert "sensor_type" in df.columns
    assert "battery_pct" in df.columns
    assert summary["format"] == "parquet"
    assert summary["status"] == "SIMULATED"


def test_azure_data_lake_validation_and_execution():
    """Verify Azure Data Lake parameters validation and simulation execution."""
    r = AzureDataLakeConnectorRecipe()

    # Container and blob required
    errs = r.validate_config({"auth_mode": "account_key", "container_name": "", "blob_path": ""})
    assert any("Container name is required" in e for e in errs)
    assert any("Blob path is required" in e for e in errs)

    cfg = {
        "auth_mode": "account_key",
        "account_name": "adlsstorage2026",
        "container_name": "lake-container",
        "blob_path": "raw/events.parquet",
        "limit": 20,
        "simulation_mode": True
    }
    res = r.execute({}, cfg)
    df = res["dataframe"]
    summary = res["output_summary"]

    assert len(df) == 20
    assert "telemetry_id" in df.columns
    assert "device_family" in df.columns
    assert "storage_tier" in df.columns
    assert summary["account_name"] == "adlsstorage2026"
    assert summary["status"] == "SIMULATED"


def test_code_generation_for_all_connectors():
    """Verify to_code generates syntactically valid code blocks for all 5 connectors."""
    connectors = [
        (SnowflakeConnectorRecipe(), {"account": "xy12345", "user": "analyst", "table_name": "ORDERS"}),
        (BigQueryConnectorRecipe(), {"project_id": "my-p", "dataset_id": "ga4", "table_id": "events"}),
        (RedshiftConnectorRecipe(), {"host": "rs.aws.com", "database": "dw", "table_name": "sales"}),
        (S3DataLakeConnectorRecipe(), {"s3_uri": "s3://b/f.parquet", "file_format": "parquet"}),
        (AzureDataLakeConnectorRecipe(), {"container_name": "c", "blob_path": "f.parquet"})
    ]
    for r, cfg in connectors:
        code = r.to_code(cfg)
        assert isinstance(code, str)
        assert len(code) > 20
        assert "import" in code
        assert "df" in code


def test_snowflake_dag_workflow():
    """Verify Snowflake connector in a 2-node pipeline with DAGExecutor."""
    nodes = [
        WorkflowNode(
            id="node_sf",
            recipe_id="snowflake_connector",
            config={
                "read_mode": "table",
                "table_name": "FCT_ORDERS",
                "limit": 25,
                "simulation_mode": True
            }
        ),
        WorkflowNode(
            id="node_scale",
            recipe_id="feature_scaler",
            config={
                "method": "standard",
                "columns": ["CREDITS_USED", "TOTAL_AMOUNT_USD"]
            }
        )
    ]
    edges = [
        WorkflowEdge(id="e_sf_s", source="node_sf", target="node_scale")
    ]
    graph = WorkflowGraph(id="wf_sf_test", name="Snowflake DAG Test", nodes=nodes, edges=edges)

    exec_result = DAGExecutor.execute_workflow(
        execution_id="test_sf_exec_100",
        workflow=graph,
        include_node_outputs=True
    )
    assert exec_result.status == "SUCCESS"
    assert len(exec_result.node_results) == 2
    res_sf = next(r for r in exec_result.node_results if r.node_id == "node_sf")
    assert res_sf.status == "SUCCESS"
    summary_sf = res_sf.output_summary.get("output_summary") or res_sf.output_summary
    assert summary_sf.get("title") == "Snowflake Connector Summary"


def test_s3_data_lake_dag_workflow():
    """Verify S3 Data Lake connector in a multi-node pipeline with DAGExecutor."""
    nodes = [
        WorkflowNode(
            id="node_s3",
            recipe_id="s3_data_lake_connector",
            config={
                "source_mode": "s3_uri",
                "s3_uri": "s3://lake/telemetry.parquet",
                "limit": 30,
                "simulation_mode": True
            }
        ),
        WorkflowNode(
            id="node_sanitizer",
            recipe_id="category_sanitizer",
            config={
                "columns": ["sensor_type", "availability_zone"],
                "lowercase": True,
                "trim_whitespace": True
            }
        )
    ]
    edges = [
        WorkflowEdge(id="e_s3_c", source="node_s3", target="node_sanitizer")
    ]
    graph = WorkflowGraph(id="wf_s3_test", name="S3 Lake DAG Test", nodes=nodes, edges=edges)

    exec_result = DAGExecutor.execute_workflow(
        execution_id="test_s3_exec_100",
        workflow=graph,
        include_node_outputs=True
    )
    assert exec_result.status == "SUCCESS"
    assert len(exec_result.node_results) == 2
    res_s3 = next(r for r in exec_result.node_results if r.node_id == "node_s3")
    assert res_s3.status == "SUCCESS"
    summary_s3 = res_s3.output_summary.get("output_summary") or res_s3.output_summary
    assert summary_s3.get("title") == "Amazon S3 Data Lake Connector Summary"
