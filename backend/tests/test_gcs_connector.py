import io
import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from backend.app.recipes.base.registry import recipe_registry
from backend.app.recipes.ingestion.gcs_connector import GCSConnectorRecipe


def test_gcs_connector_registration():
    """Verify GCS connector is registered, enabled, and in Ingestion & Orchestration."""
    recipe = recipe_registry.get("gcs_connector")
    assert isinstance(recipe, GCSConnectorRecipe)
    assert recipe.recipe_id == "gcs_connector"
    assert recipe_registry.has("gcs_connector")

    meta = recipe_registry._to_metadata_with_group(recipe)
    assert meta.enabled is True
    assert meta.group == "Ingestion & Orchestration"
    assert meta.subgroup == "Cloud Storage Connectors"


def test_gcs_connector_aliases():
    """Verify registered aliases resolve to gcs_connector."""
    assert recipe_registry.get("gcs").recipe_id == "gcs_connector"
    assert recipe_registry.get("google_cloud_storage").recipe_id == "gcs_connector"
    assert recipe_registry.get("google_storage").recipe_id == "gcs_connector"
    assert recipe_registry.get("gcs_lake").recipe_id == "gcs_connector"


def test_gcs_connector_schema_contract():
    """Verify all schema properties have input_type and description."""
    recipe = GCSConnectorRecipe()
    schema = recipe.get_schema()
    props = schema.get("properties", {})
    assert len(props) > 0
    for name, p_data in props.items():
        assert "input_type" in p_data, f"Property '{name}' is missing 'input_type'"
        assert "description" in p_data, f"Property '{name}' is missing 'description'"


def test_gcs_connector_uri_parsing():
    """Test extracting bucket and blob path from gs:// URIs."""
    recipe = GCSConnectorRecipe()
    b, p = recipe._parse_gcs_uri("gs://my-lakehouse/data/orders_2026.parquet")
    assert b == "my-lakehouse"
    assert p == "data/orders_2026.parquet"

    b2, p2 = recipe._parse_gcs_uri("gs://bucket-only")
    assert b2 == "bucket-only"
    assert p2 == ""

    b3, p3 = recipe._parse_gcs_uri("invalid://something")
    assert b3 == ""
    assert p3 == ""


def test_gcs_connector_format_inference():
    """Test format auto-inference from file extensions."""
    recipe = GCSConnectorRecipe()
    assert recipe._infer_format("gs://b/file.parquet", "auto") == "parquet"
    assert recipe._infer_format("gs://b/file.csv", "auto") == "csv"
    assert recipe._infer_format("gs://b/file.jsonl", "auto") == "jsonl"
    assert recipe._infer_format("gs://b/file.ndjson", "auto") == "jsonl"
    assert recipe._infer_format("gs://b/file.json", "auto") == "json"
    # Explicit format overrides extension
    assert recipe._infer_format("gs://b/file.txt", "csv") == "csv"


def test_gcs_connector_simulation_fallback():
    """Test simulation mode returns realistic benchmark records when offline."""
    recipe = GCSConnectorRecipe()
    config = {
        "source_mode": "gcs_uri",
        "gcs_uri": "gs://telemetry-bucket/raw/iot.parquet",
        "simulation_mode": True,
        "limit": 50,
    }
    result = recipe.execute(inputs={}, config=config)
    df = result["dataframe"]
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 50
    assert "device_id" in df.columns
    assert "metric_value" in df.columns
    assert result["status"] == "SIMULATED"
    assert result["rows_retrieved"] == 50
    assert "output_summary" in result
    assert "metrics" in result


def test_gcs_connector_bucket_and_blob_mode():
    """Test source_mode='bucket_and_blob' generates correct URI and records."""
    recipe = GCSConnectorRecipe()
    config = {
        "source_mode": "bucket_and_blob",
        "bucket_name": "custom-analytics-bucket",
        "blob_path": "finance/transactions_q1.csv",
        "simulation_mode": True,
        "limit": 25,
    }
    result = recipe.execute(inputs={}, config=config)
    df = result["dataframe"]
    assert len(df) == 25
    summary = result["output_summary"]
    assert summary["bucket"] == "custom-analytics-bucket"
    assert summary["blob_path"] == "finance/transactions_q1.csv"
    assert summary["gcs_uri"] == "gs://custom-analytics-bucket/finance/transactions_q1.csv"


def test_gcs_connector_mock_live_csv():
    """Test parsing downloaded CSV bytes using mock GCS client."""
    recipe = GCSConnectorRecipe()
    csv_bytes = b"customer_id,region,spend\nC100,US,250.5\nC101,EU,120.0\nC102,APAC,410.2\n"

    with patch.object(recipe, "_read_blob_bytes", return_value=csv_bytes):
        config = {
            "source_mode": "gcs_uri",
            "gcs_uri": "gs://sales-bucket/customers.csv",
            "file_format": "csv",
            "simulation_mode": False,
        }
        result = recipe.execute(inputs={}, config=config)
        df = result["dataframe"]
        assert len(df) == 3
        assert list(df.columns) == ["customer_id", "region", "spend"]
        assert result["status"] == "LIVE"


def test_gcs_connector_mock_live_parquet():
    """Test parsing downloaded Parquet bytes using mock GCS client."""
    recipe = GCSConnectorRecipe()
    raw_df = pd.DataFrame({"id": [1, 2], "val": ["x", "y"]})
    bio = io.BytesIO()
    raw_df.to_parquet(bio)
    parquet_bytes = bio.getvalue()

    with patch.object(recipe, "_read_blob_bytes", return_value=parquet_bytes):
        config = {
            "source_mode": "gcs_uri",
            "gcs_uri": "gs://parquet-bucket/sample.parquet",
            "file_format": "parquet",
            "simulation_mode": False,
        }
        result = recipe.execute(inputs={}, config=config)
        df = result["dataframe"]
        assert len(df) == 2
        assert list(df.columns) == ["id", "val"]
        assert result["status"] == "LIVE"


def test_gcs_connector_mock_live_json():
    """Test parsing downloaded JSON records bytes using mock GCS client."""
    recipe = GCSConnectorRecipe()
    json_bytes = b'[{"dept": "AI", "count": 15}, {"dept": "DevOps", "count": 8}]'

    with patch.object(recipe, "_read_blob_bytes", return_value=json_bytes):
        config = {
            "source_mode": "gcs_uri",
            "gcs_uri": "gs://json-bucket/depts.json",
            "file_format": "json",
            "simulation_mode": False,
        }
        result = recipe.execute(inputs={}, config=config)
        df = result["dataframe"]
        assert len(df) == 2
        assert "dept" in df.columns
        assert result["status"] == "LIVE"
