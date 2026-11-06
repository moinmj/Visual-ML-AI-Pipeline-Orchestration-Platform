import json
import pandas as pd
import pytest

from backend.app.core.exceptions import ValidationException
from backend.app.recipes.base.registry import recipe_registry
from backend.app.recipes.ingestion.cdc_ingestion import CDCIngestionRecipe


def test_cdc_ingestion_registration():
    """Verify CDC ingestion recipe is registered, enabled, and in Ingestion & Orchestration."""
    recipe = recipe_registry.get("cdc_ingestion")
    assert isinstance(recipe, CDCIngestionRecipe)
    assert recipe.recipe_id == "cdc_ingestion"
    assert recipe_registry.has("cdc_ingestion")

    meta = recipe_registry._to_metadata_with_group(recipe)
    assert meta.enabled is True
    assert meta.group == "Ingestion & Orchestration"
    assert meta.subgroup == "Streaming & Change Data"


def test_cdc_ingestion_aliases():
    """Verify registered aliases resolve to cdc_ingestion."""
    assert recipe_registry.get("cdc").recipe_id == "cdc_ingestion"
    assert recipe_registry.get("change_data_capture").recipe_id == "cdc_ingestion"
    assert recipe_registry.get("debezium").recipe_id == "cdc_ingestion"
    assert recipe_registry.get("debezium_cdc").recipe_id == "cdc_ingestion"
    assert recipe_registry.get("cdc_stream").recipe_id == "cdc_ingestion"


def test_cdc_ingestion_schema_contract():
    """Verify all schema properties have input_type and description."""
    recipe = CDCIngestionRecipe()
    schema = recipe.get_schema()
    props = schema.get("properties", {})
    assert len(props) > 0
    for name, p_data in props.items():
        assert "input_type" in p_data, f"Property '{name}' is missing 'input_type'"
        assert "description" in p_data, f"Property '{name}' is missing 'description'"


def test_cdc_ingestion_synthetic_simulation():
    """Test generating synthetic CDC stream events in simulation mode."""
    recipe = CDCIngestionRecipe()
    config = {
        "source_type": "debezium_kafka",
        "simulation_mode": True,
        "batch_size": 25,
    }
    result = recipe.execute(inputs={}, config=config)
    df = result["dataframe"]
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 25
    assert "_cdc_op" in df.columns
    assert "_cdc_table" in df.columns
    assert "_cdc_timestamp" in df.columns
    assert "_cdc_lsn" in df.columns
    assert result["status"] == "SIMULATED"
    assert result["rows_retrieved"] == 25
    assert "metrics" in result
    assert result["metrics"]["events_captured"] == 25


def test_cdc_ingestion_inline_debezium_events():
    """Test processing raw Debezium envelope JSON string with insert, update, and delete."""
    recipe = CDCIngestionRecipe()
    raw_payload = [
        {
            "op": "c",
            "ts_ms": 1712345678000,
            "source": {"table": "customers", "lsn": 1001},
            "before": None,
            "after": {"id": 1, "name": "Alice", "city": "NYC"},
        },
        {
            "op": "u",
            "ts_ms": 1712345688000,
            "source": {"table": "customers", "lsn": 1002},
            "before": {"id": 1, "name": "Alice", "city": "NYC"},
            "after": {"id": 1, "name": "Alice", "city": "Boston"},
        },
        {
            "op": "d",
            "ts_ms": 1712345698000,
            "source": {"table": "customers", "lsn": 1003},
            "before": {"id": 2, "name": "Bob", "city": "Chicago"},
            "after": None,
        },
    ]

    config = {
        "raw_events": json.dumps(raw_payload),
        "payload_format": "flattened_after",
        "include_before_image": True,
    }
    result = recipe.execute(inputs={}, config=config)
    df = result["dataframe"]
    assert len(df) == 3
    assert list(df["_cdc_op"]) == ["INSERT", "UPDATE", "DELETE"]
    # Check that update has before image
    assert df.iloc[1]["city"] == "Boston"
    assert df.iloc[1]["_before_city"] == "NYC"
    # Check delete retains before payload
    assert df.iloc[2]["name"] == "Bob"


def test_cdc_ingestion_operations_filter_inserts_only():
    """Test operation filtering to include only inserts."""
    recipe = CDCIngestionRecipe()
    events = [
        {"op": "c", "table": "orders", "after": {"order_id": 10, "amt": 50}},
        {"op": "u", "table": "orders", "after": {"order_id": 10, "amt": 60}},
        {"op": "d", "table": "orders", "before": {"order_id": 11, "amt": 30}},
        {"op": "c", "table": "orders", "after": {"order_id": 12, "amt": 80}},
    ]
    config = {
        "raw_events": events,
        "operations": "inserts_only",
    }
    result = recipe.execute(inputs={}, config=config)
    df = result["dataframe"]
    assert len(df) == 2
    assert all(df["_cdc_op"] == "INSERT")


def test_cdc_ingestion_operations_filter_updates_only():
    """Test operation filtering to include only updates."""
    recipe = CDCIngestionRecipe()
    events = [
        {"op": "c", "table": "orders", "after": {"order_id": 10}},
        {"op": "u", "table": "orders", "after": {"order_id": 10}},
        {"op": "d", "table": "orders", "before": {"order_id": 11}},
    ]
    config = {
        "raw_events": events,
        "operations": "updates_only",
    }
    result = recipe.execute(inputs={}, config=config)
    df = result["dataframe"]
    assert len(df) == 1
    assert df.iloc[0]["_cdc_op"] == "UPDATE"


def test_cdc_ingestion_table_filter():
    """Test table whitelist filtering."""
    recipe = CDCIngestionRecipe()
    events = [
        {"op": "c", "table": "orders", "after": {"id": 1}},
        {"op": "c", "table": "audit_logs", "after": {"id": 2}},
        {"op": "c", "table": "users", "after": {"id": 3}},
        {"op": "c", "table": "orders", "after": {"id": 4}},
    ]
    config = {
        "raw_events": events,
        "table_filter": "orders,users",
    }
    result = recipe.execute(inputs={}, config=config)
    df = result["dataframe"]
    assert len(df) == 3
    assert set(df["_cdc_table"]) == {"orders", "users"}


def test_cdc_ingestion_payload_format_full_envelope():
    """Test full_envelope mode preserves serialized before and after JSON."""
    recipe = CDCIngestionRecipe()
    events = [
        {
            "op": "u",
            "table": "products",
            "before": {"sku": "A1", "price": 10.0},
            "after": {"sku": "A1", "price": 12.5},
        }
    ]
    config = {
        "raw_events": events,
        "payload_format": "full_envelope",
    }
    result = recipe.execute(inputs={}, config=config)
    df = result["dataframe"]
    assert len(df) == 1
    assert "_cdc_before_json" in df.columns
    assert "_cdc_after_json" in df.columns
    assert json.loads(df.iloc[0]["_cdc_after_json"])["price"] == 12.5


def test_cdc_ingestion_payload_format_compact_diff():
    """Test compact_diff mode includes only changed fields for updates."""
    recipe = CDCIngestionRecipe()
    events = [
        {
            "op": "u",
            "table": "accounts",
            "before": {"id": 50, "name": "Corp", "tier": "silver", "active": True},
            "after": {"id": 50, "name": "Corp", "tier": "gold", "active": True},
        }
    ]
    config = {
        "raw_events": events,
        "payload_format": "compact_diff",
        "include_before_image": True,
    }
    result = recipe.execute(inputs={}, config=config)
    df = result["dataframe"]
    assert len(df) == 1
    assert "tier" in df.columns
    assert "_before_tier" in df.columns
    assert df.iloc[0]["tier"] == "gold"
    assert df.iloc[0]["_before_tier"] == "silver"
    # Unchanged column 'active' not explicitly duplicated in compact diff
    assert "active" not in df.columns


def test_cdc_ingestion_empty_events():
    """Test empty events input returns schema-safe empty DataFrame."""
    recipe = CDCIngestionRecipe()
    result = recipe.execute(inputs={}, config={"raw_events": []})
    df = result["dataframe"]
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    assert list(df.columns) == ["_cdc_op", "_cdc_table", "_cdc_timestamp", "_cdc_lsn"]


def test_cdc_ingestion_invalid_json_raises():
    """Test malformed JSON string raises ValidationException."""
    recipe = CDCIngestionRecipe()
    with pytest.raises(ValidationException) as exc_info:
        recipe.execute(inputs={}, config={"raw_events": "[[{unclosed json"})
    assert "Invalid JSON in raw_events" in str(exc_info.value)
