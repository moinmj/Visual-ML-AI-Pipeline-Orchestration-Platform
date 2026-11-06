import json
import pandas as pd
import pytest

from backend.app.core.exceptions import ValidationException
from backend.app.recipes.base.registry import recipe_registry
from backend.app.recipes.ingestion.cdc_replicator import CDCReplicatorRecipe


def test_cdc_replicator_registration():
    """Verify CDC replicator is registered, enabled, and in Ingestion & Orchestration."""
    recipe = recipe_registry.get("cdc_replicator")
    assert isinstance(recipe, CDCReplicatorRecipe)
    assert recipe.recipe_id == "cdc_replicator"
    assert recipe_registry.has("cdc_replicator")

    meta = recipe_registry._to_metadata_with_group(recipe)
    assert meta.enabled is True
    assert meta.group == "Ingestion & Orchestration"
    assert meta.subgroup == "Streaming & Change Data"


def test_cdc_replicator_aliases():
    """Verify registered aliases resolve to cdc_replicator."""
    assert recipe_registry.get("replicator").recipe_id == "cdc_replicator"
    assert recipe_registry.get("database_replicator").recipe_id == "cdc_replicator"
    assert recipe_registry.get("cdc_sync").recipe_id == "cdc_replicator"
    assert recipe_registry.get("db_replicator").recipe_id == "cdc_replicator"


def test_cdc_replicator_schema_contract():
    """Verify all schema properties have input_type and description."""
    recipe = CDCReplicatorRecipe()
    schema = recipe.get_schema()
    props = schema.get("properties", {})
    assert len(props) > 0
    for name, p_data in props.items():
        assert "input_type" in p_data, f"Property '{name}' is missing 'input_type'"
        assert "description" in p_data, f"Property '{name}' is missing 'description'"


def test_cdc_replicator_simulation_mode():
    """Test generating synthetic baseline and applying synthetic mutations in demo mode."""
    recipe = CDCReplicatorRecipe()
    config = {
        "simulation_mode": True,
        "batch_size": 15,
        "primary_key": "id",
    }
    result = recipe.execute(inputs={}, config=config)
    df = result["dataframe"]
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "id" in df.columns
    assert result["status"] == "SIMULATED"
    assert "metrics" in result
    assert result["metrics"]["rows_inserted"] > 0


def test_cdc_replicator_apply_insert_and_update():
    """Test applying an insert and an update onto an existing base DataFrame."""
    recipe = CDCReplicatorRecipe()
    base_df = pd.DataFrame([
        {"id": 101, "name": "Alpha Corp", "mrr": 500, "updated_at": "2026-01-01T00:00:00Z"},
        {"id": 102, "name": "Beta LLC", "mrr": 1200, "updated_at": "2026-01-01T00:00:00Z"},
    ])

    mutations = [
        # Update row 101
        {
            "op": "u",
            "lsn": 501,
            "after": {"id": 101, "name": "Alpha Corp", "mrr": 750, "updated_at": "2026-01-02T00:00:00Z"},
        },
        # Insert row 103
        {
            "op": "c",
            "lsn": 502,
            "after": {"id": 103, "name": "Gamma Inc", "mrr": 300, "updated_at": "2026-01-02T00:00:00Z"},
        },
    ]

    config = {
        "primary_key": "id",
        "raw_mutations": mutations,
    }
    result = recipe.execute(inputs={"dataframe": base_df}, config=config)
    df = result["dataframe"]
    assert len(df) == 3
    # Check row 101 was updated in place
    row_101 = df[df["id"] == 101].iloc[0]
    assert row_101["mrr"] == 750
    # Check row 103 was inserted
    row_103 = df[df["id"] == 103].iloc[0]
    assert row_103["name"] == "Gamma Inc"
    assert result["metrics"]["rows_updated"] == 1
    assert result["metrics"]["rows_inserted"] == 1


def test_cdc_replicator_hard_delete():
    """Test hard_delete policy drops row completely from replica table."""
    recipe = CDCReplicatorRecipe()
    base_df = pd.DataFrame([
        {"id": 1, "sku": "A100", "qty": 10},
        {"id": 2, "sku": "B200", "qty": 20},
    ])
    mutations = [
        {"op": "d", "before": {"id": 1, "sku": "A100"}},
    ]
    config = {
        "primary_key": "id",
        "delete_handling": "hard_delete",
        "raw_mutations": mutations,
    }
    result = recipe.execute(inputs={"dataframe": base_df}, config=config)
    df = result["dataframe"]
    assert len(df) == 1
    assert list(df["id"]) == [2]
    assert result["metrics"]["rows_deleted"] == 1


def test_cdc_replicator_soft_delete():
    """Test soft_delete policy retains row and marks _is_deleted=True with tombstone timestamp."""
    recipe = CDCReplicatorRecipe()
    base_df = pd.DataFrame([
        {"id": 1, "sku": "A100", "qty": 10},
    ])
    mutations = [
        {"op": "d", "ts_ms": 1712345678000, "before": {"id": 1}},
    ]
    config = {
        "primary_key": "id",
        "delete_handling": "soft_delete",
        "raw_mutations": mutations,
    }
    result = recipe.execute(inputs={"dataframe": base_df}, config=config)
    df = result["dataframe"]
    assert len(df) == 1
    row = df.iloc[0]
    assert bool(row["_is_deleted"]) is True
    assert "_deleted_at" in row
    assert result["metrics"]["rows_deleted"] == 1


def test_cdc_replicator_composite_primary_key():
    """Test matching rows using a composite primary key (tenant_id + order_id)."""
    recipe = CDCReplicatorRecipe()
    base_df = pd.DataFrame([
        {"tenant_id": "T1", "order_id": 99, "status": "PENDING"},
        {"tenant_id": "T2", "order_id": 99, "status": "PENDING"},
    ])
    mutations = [
        {"op": "u", "after": {"tenant_id": "T1", "order_id": 99, "status": "PAID"}},
    ]
    config = {
        "primary_key": "tenant_id, order_id",
        "raw_mutations": mutations,
    }
    result = recipe.execute(inputs={"dataframe": base_df}, config=config)
    df = result["dataframe"]
    assert len(df) == 2
    row_t1 = df[df["tenant_id"] == "T1"].iloc[0]
    row_t2 = df[df["tenant_id"] == "T2"].iloc[0]
    assert row_t1["status"] == "PAID"
    assert row_t2["status"] == "PENDING"


def test_cdc_replicator_conflict_resolution_latest_timestamp():
    """Test out-of-order update with older timestamp is ignored under latest_timestamp_wins."""
    recipe = CDCReplicatorRecipe()
    base_df = pd.DataFrame([
        {"id": 10, "status": "SHIPPED", "updated_at": "2026-03-10T12:00:00Z"},
    ])
    mutations = [
        # Older mutation arrives late
        {"op": "u", "after": {"id": 10, "status": "PROCESSING", "updated_at": "2026-03-09T10:00:00Z"}},
    ]
    config = {
        "primary_key": "id",
        "conflict_resolution": "latest_timestamp_wins",
        "timestamp_column": "updated_at",
        "raw_mutations": mutations,
    }
    result = recipe.execute(inputs={"dataframe": base_df}, config=config)
    df = result["dataframe"]
    assert df.iloc[0]["status"] == "SHIPPED"
    assert result["metrics"]["rows_updated"] == 0


def test_cdc_replicator_conflict_resolution_highest_lsn():
    """Test mutation with higher LSN is applied under highest_lsn_wins."""
    recipe = CDCReplicatorRecipe()
    base_df = pd.DataFrame([
        {"id": 20, "val": "initial", "_cdc_lsn": 500},
    ])
    mutations = [
        {"op": "u", "lsn": 400, "after": {"id": 20, "val": "old_lsn"}},
        {"op": "u", "lsn": 600, "after": {"id": 20, "val": "new_lsn"}},
    ]
    config = {
        "primary_key": "id",
        "conflict_resolution": "highest_lsn_wins",
        "raw_mutations": mutations,
    }
    result = recipe.execute(inputs={"dataframe": base_df}, config=config)
    df = result["dataframe"]
    assert df.iloc[0]["val"] == "new_lsn"


def test_cdc_replicator_empty_mutations():
    """Test passing empty mutations preserves existing base DataFrame unchanged."""
    recipe = CDCReplicatorRecipe()
    base_df = pd.DataFrame([{"id": 1, "v": "a"}, {"id": 2, "v": "b"}])
    result = recipe.execute(inputs={"dataframe": base_df}, config={"raw_mutations": []})
    df = result["dataframe"]
    assert len(df) == 2
    assert list(df["id"]) == [1, 2]
    assert result["metrics"]["rows_inserted"] == 0
    assert result["metrics"]["rows_updated"] == 0


def test_cdc_replicator_invalid_json():
    """Test malformed JSON string raises ValidationException."""
    recipe = CDCReplicatorRecipe()
    with pytest.raises(ValidationException) as exc_info:
        recipe.execute(inputs={}, config={"raw_mutations": "[[bad json"})
    assert "Invalid JSON in raw_mutations" in str(exc_info.value)
