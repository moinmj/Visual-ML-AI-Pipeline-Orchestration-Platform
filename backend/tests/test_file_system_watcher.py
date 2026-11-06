"""
Unit tests for File System Watcher Recipe.
"""

import os
import shutil
import tempfile
import time
import pytest
import pandas as pd

from backend.app.core.exceptions import ExecutionException, ValidationException
from backend.app.recipes.ingestion.file_system_watcher import FileSystemWatcherRecipe
from backend.app.recipes.base.registry import recipe_registry


def test_file_system_watcher_registration():
    """Verify recipe registration and metadata."""
    recipe = recipe_registry.get("file_system_watcher")
    assert recipe is not None
    assert recipe.recipe_id == "file_system_watcher"
    assert recipe.group == "Ingestion & Orchestration"
    assert recipe.subgroup == "File & Web Ingestion"


def test_file_system_watcher_schema_contract():
    """Verify all schema properties have input_type and description."""
    recipe = FileSystemWatcherRecipe()
    schema = recipe.get_schema()
    assert schema["type"] == "object"
    props = schema["properties"]
    for field_name, spec in props.items():
        assert "input_type" in spec, f"Field '{field_name}' missing input_type"
        assert "description" in spec, f"Field '{field_name}' missing description"


def test_file_system_watcher_validation_errors():
    """Test parameter validation rules."""
    recipe = FileSystemWatcherRecipe()

    # Empty directory path
    with pytest.raises(ValidationException) as exc_1:
        recipe.execute(inputs={}, config={"directory_path": ""})
    assert "Parameter 'directory_path' cannot be empty" in str(exc_1.value)

    # Invalid action_after_read
    with pytest.raises(ValidationException) as exc_2:
        recipe.execute(
            inputs={},
            config={"directory_path": "./data", "action_after_read": "burn_it"},
        )
    assert "Unsupported action_after_read" in str(exc_2.value)


def test_file_system_watcher_simulation_fallback():
    """Test synthetic fallback when directory does not exist."""
    recipe = FileSystemWatcherRecipe()
    config = {
        "directory_path": "/path/to/non_existent_folder_abc123",
        "file_pattern": "*.csv",
        "simulation_fallback": True,
        "include_metadata_columns": True,
    }
    result = recipe.execute(inputs={}, config=config)
    df = result["dataframe"]
    metrics = result["metrics"]

    assert not df.empty
    assert metrics["is_simulation"] is True
    assert metrics["files_processed"] > 0
    assert "_file_path" in df.columns
    assert "_file_name" in df.columns
    assert "record_id" in df.columns


def test_file_system_watcher_simulation_disabled_raises():
    """Test error raised when folder is missing and simulation fallback is false."""
    recipe = FileSystemWatcherRecipe()
    config = {
        "directory_path": "/path/to/non_existent_folder_xyz789",
        "simulation_fallback": False,
    }
    with pytest.raises(ExecutionException) as exc:
        recipe.execute(inputs={}, config=config)
    assert "contains no matching files" in str(exc.value)


def test_file_system_watcher_real_csv_ingestion():
    """Test scanning a real temporary directory containing CSV files."""
    recipe = FileSystemWatcherRecipe()
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create 2 CSV files
        f1 = os.path.join(tmpdir, "orders_part1.csv")
        f2 = os.path.join(tmpdir, "orders_part2.csv")

        pd.DataFrame([{"id": 1, "product": "Widget"}, {"id": 2, "product": "Gadget"}]).to_csv(f1, index=False)
        pd.DataFrame([{"id": 3, "product": "Doohickey"}]).to_csv(f2, index=False)

        config = {
            "directory_path": tmpdir,
            "file_pattern": "orders_*.csv",
            "output_mode": "parse_data",
            "sort_by": "name_asc",
            "include_metadata_columns": True,
        }
        result = recipe.execute(inputs={}, config=config)
        df = result["dataframe"]
        metrics = result["metrics"]

        assert len(df) == 3
        assert list(df["id"]) == [1, 2, 3]
        assert metrics["is_simulation"] is False
        assert metrics["files_processed"] == 2
        assert "_file_name" in df.columns


def test_file_system_watcher_file_manifest_mode():
    """Test generating a metadata manifest from directory files."""
    recipe = FileSystemWatcherRecipe()
    with tempfile.TemporaryDirectory() as tmpdir:
        f1 = os.path.join(tmpdir, "report_a.csv")
        f2 = os.path.join(tmpdir, "report_b.csv")
        with open(f1, "w") as fp:
            fp.write("data\n1\n")
        with open(f2, "w") as fp:
            fp.write("data\n2\n3\n")

        config = {
            "directory_path": tmpdir,
            "file_pattern": "*.csv",
            "output_mode": "file_manifest",
        }
        result = recipe.execute(inputs={}, config=config)
        df = result["dataframe"]
        metrics = result["metrics"]

        assert len(df) == 2
        assert set(df["filename"]) == {"report_a.csv", "report_b.csv"}
        assert "size_bytes" in df.columns
        assert "modified_at" in df.columns
        assert metrics["output_mode"] == "file_manifest"


def test_file_system_watcher_parquet_ingestion():
    """Test reading Parquet files from directory."""
    recipe = FileSystemWatcherRecipe()
    with tempfile.TemporaryDirectory() as tmpdir:
        pq_path = os.path.join(tmpdir, "metrics.parquet")
        sample_df = pd.DataFrame([{"metric": "cpu", "value": 75.5}, {"metric": "mem", "value": 42.1}])
        sample_df.to_parquet(pq_path, index=False)

        config = {
            "directory_path": tmpdir,
            "file_pattern": "*.parquet",
            "file_format": "auto",
        }
        result = recipe.execute(inputs={}, config=config)
        df = result["dataframe"]
        assert len(df) == 2
        assert list(df["metric"]) == ["cpu", "mem"]


def test_file_system_watcher_recursive_scan():
    """Test recursive subdirectory matching."""
    recipe = FileSystemWatcherRecipe()
    with tempfile.TemporaryDirectory() as tmpdir:
        subdir = os.path.join(tmpdir, "nested", "level2")
        os.makedirs(subdir, exist_ok=True)
        f_nested = os.path.join(subdir, "nested_data.csv")
        pd.DataFrame([{"nested_id": 999}]).to_csv(f_nested, index=False)

        # Without recursive, matches 0 files
        res_non_rec = recipe.execute(
            inputs={},
            config={"directory_path": tmpdir, "recursive": False, "simulation_fallback": True},
        )
        assert res_non_rec["metrics"]["is_simulation"] is True

        # With recursive, finds the real nested file
        res_rec = recipe.execute(
            inputs={},
            config={"directory_path": tmpdir, "recursive": True, "simulation_fallback": False},
        )
        df = res_rec["dataframe"]
        assert len(df) == 1
        assert df.iloc[0]["nested_id"] == 999
        assert res_rec["metrics"]["is_simulation"] is False


def test_file_system_watcher_action_after_read_delete():
    """Test post-read deletion of ingested files."""
    recipe = FileSystemWatcherRecipe()
    with tempfile.TemporaryDirectory() as tmpdir:
        f_path = os.path.join(tmpdir, "delete_me.csv")
        pd.DataFrame([{"temp": "val"}]).to_csv(f_path, index=False)
        assert os.path.exists(f_path)

        recipe.execute(
            inputs={},
            config={"directory_path": tmpdir, "action_after_read": "delete"},
        )
        assert not os.path.exists(f_path)


def test_file_system_watcher_action_after_read_move():
    """Test post-read moving of ingested files to archive."""
    recipe = FileSystemWatcherRecipe()
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "move_me.csv")
        arch_dir = os.path.join(tmpdir, "archived")
        pd.DataFrame([{"saved": 1}]).to_csv(src_path, index=False)

        recipe.execute(
            inputs={},
            config={"directory_path": tmpdir, "action_after_read": "move", "archive_directory": arch_dir},
        )
        assert not os.path.exists(src_path)
        assert os.path.exists(os.path.join(arch_dir, "move_me.csv"))
