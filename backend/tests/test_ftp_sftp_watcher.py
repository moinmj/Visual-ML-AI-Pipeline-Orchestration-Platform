"""
Unit tests for FTP/SFTP File Watcher Recipe.
"""

import io
import json
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch

from backend.app.core.exceptions import ExecutionException, ValidationException
from backend.app.recipes.ingestion.ftp_sftp_watcher import FTPSFTPWatcherRecipe
from backend.app.recipes.base.registry import recipe_registry


def test_ftp_sftp_watcher_registration():
    """Verify recipe registration in registry."""
    recipe = recipe_registry.get("ftp_sftp_watcher")
    assert recipe is not None
    assert recipe.recipe_id == "ftp_sftp_watcher"
    assert recipe.group == "Ingestion & Orchestration"
    assert recipe.subgroup == "File & Web Ingestion"


def test_ftp_sftp_watcher_schema_contract():
    """Verify schema properties all have input_type and description."""
    recipe = FTPSFTPWatcherRecipe()
    schema = recipe.get_schema()
    assert schema["type"] == "object"
    props = schema["properties"]
    for field_name, spec in props.items():
        assert "input_type" in spec, f"Field '{field_name}' missing input_type"
        assert "description" in spec, f"Field '{field_name}' missing description"


def test_ftp_sftp_watcher_validation_errors():
    """Test required parameter validation."""
    recipe = FTPSFTPWatcherRecipe()

    # Invalid protocol
    with pytest.raises(ValidationException) as exc_1:
        recipe.execute(inputs={}, config={"protocol": "http", "host": "ftp.example.com"})
    assert "Unsupported protocol" in str(exc_1.value)

    # Missing host
    with pytest.raises(ValidationException) as exc_2:
        recipe.execute(inputs={}, config={"protocol": "sftp", "host": ""})
    assert "Parameter 'host' is required" in str(exc_2.value)

    # Invalid port
    with pytest.raises(ValidationException) as exc_3:
        recipe.execute(inputs={}, config={"protocol": "sftp", "host": "example.com", "port": 99999})
    assert "Port must be between 1 and 65535" in str(exc_3.value)

    # Malformed mock entries JSON
    with pytest.raises(ValidationException) as exc_4:
        recipe.execute(
            inputs={},
            config={"host": "example.com", "mock_file_entries": "{broken json"},
        )
    assert "Invalid JSON provided in 'mock_file_entries'" in str(exc_4.value)


def test_ftp_sftp_watcher_simulation_fallback():
    """Verify simulation fallback generates synthetic data when remote host is unreachable."""
    recipe = FTPSFTPWatcherRecipe()
    config = {
        "protocol": "sftp",
        "host": "unreachable.domain.test",
        "remote_directory": "/incoming/feeds",
        "file_pattern": "*.csv",
        "simulation_fallback": True,
    }
    result = recipe.execute(inputs={}, config=config)
    df = result["dataframe"]
    metrics = result["metrics"]

    assert not df.empty
    assert metrics["is_simulation"] is True
    assert metrics["files_ingested"] > 0
    assert "_source_file" in df.columns
    assert "_ingested_at" in df.columns
    assert "transaction_id" in df.columns


def test_ftp_sftp_watcher_file_manifest_mode():
    """Verify file_manifest output mode returns file listing without parsing rows."""
    recipe = FTPSFTPWatcherRecipe()
    mock_files = [
        {"filename": "orders_20260101.csv", "content": "id,val\n1,100\n"},
        {"filename": "orders_20260102.csv", "content": "id,val\n2,200\n"},
    ]
    config = {
        "protocol": "ftp",
        "host": "ftp.example.com",
        "output_mode": "file_manifest",
        "action_after_download": "move",
        "archive_directory": "/incoming/done",
        "mock_file_entries": json.dumps(mock_files),
    }
    result = recipe.execute(inputs={}, config=config)
    df = result["dataframe"]
    metrics = result["metrics"]

    assert len(df) == 2
    assert "filename" in df.columns
    assert "path" in df.columns
    assert "size_bytes" in df.columns
    assert "protocol" in df.columns
    assert df.iloc[0]["filename"] == "orders_20260101.csv"
    assert metrics["files_found"] == 2
    assert metrics["action_after_download"] == "move"


def test_ftp_sftp_watcher_mock_csv_ingestion():
    """Verify parsing multiple CSV mock files with metadata columns."""
    recipe = FTPSFTPWatcherRecipe()
    mock_files = [
        {"filename": "batch_a.csv", "content": "id,name,score\n1,Alice,95.5\n2,Bob,88.0\n"},
        {"filename": "batch_b.csv", "content": "id,name,score\n3,Charlie,91.2\n"},
    ]
    config = {
        "protocol": "sftp",
        "host": "sftp.example.com",
        "mock_file_entries": json.dumps(mock_files),
        "include_metadata_columns": True,
    }
    result = recipe.execute(inputs={}, config=config)
    df = result["dataframe"]
    assert len(df) == 3
    assert list(df["name"]) == ["Alice", "Bob", "Charlie"]
    assert list(df["_source_file"]) == ["batch_a.csv", "batch_a.csv", "batch_b.csv"]
    assert result["metrics"]["files_ingested"] == 2


def test_ftp_sftp_watcher_mock_json_ingestion():
    """Verify parsing JSON mock files."""
    recipe = FTPSFTPWatcherRecipe()
    json_data = [
        {"sensor_id": "SN-01", "temperature": 23.4},
        {"sensor_id": "SN-02", "temperature": 24.1},
    ]
    mock_files = [
        {"filename": "sensors.json", "content": json.dumps(json_data)},
    ]
    config = {
        "protocol": "sftp",
        "host": "sftp.example.com",
        "file_format": "json",
        "mock_file_entries": json.dumps(mock_files),
        "include_metadata_columns": False,
    }
    result = recipe.execute(inputs={}, config=config)
    df = result["dataframe"]
    assert len(df) == 2
    assert "_source_file" not in df.columns
    assert "sensor_id" in df.columns
    assert list(df["temperature"]) == [23.4, 24.1]


def test_ftp_sftp_watcher_mock_parquet_ingestion():
    """Verify parsing Parquet format."""
    recipe = FTPSFTPWatcherRecipe()
    sample_df = pd.DataFrame({"sku": ["SKU-1", "SKU-2"], "qty": [10, 25]})
    bio = io.BytesIO()
    sample_df.to_parquet(bio, index=False)
    raw_pq_bytes = bio.getvalue()

    # Pass bytes via mock fetch
    with patch.object(recipe, "_fetch_remote_files", return_value=[{
        "filename": "inventory.parquet",
        "path": "/data/inventory.parquet",
        "size": len(raw_pq_bytes),
        "mtime": "2026-11-06T12:00:00Z",
        "bytes": raw_pq_bytes,
    }]):
        result = recipe.execute(inputs={}, config={"host": "sftp.example.com", "file_format": "auto"})
        df = result["dataframe"]
        assert len(df) == 2
        assert "sku" in df.columns
        assert list(df["qty"]) == [10, 25]


def test_ftp_sftp_watcher_simulation_disabled_raises_on_failure():
    """Verify ExecutionException is raised when simulation fallback is false and remote fails."""
    recipe = FTPSFTPWatcherRecipe()
    with patch.object(recipe, "_fetch_remote_files", side_effect=ConnectionRefusedError("Connection refused")):
        with pytest.raises(ExecutionException) as exc:
            recipe.execute(
                inputs={},
                config={"host": "offline.host", "simulation_fallback": False},
            )
        assert "FTP/SFTP connection to offline.host" in str(exc.value)


def test_ftp_sftp_watcher_pattern_matching():
    """Test glob and regex matching helper."""
    recipe = FTPSFTPWatcherRecipe()
    assert recipe._match_pattern("orders_2026.csv", "*.csv") is True
    assert recipe._match_pattern("orders_2026.parquet", "*.csv") is False
    assert recipe._match_pattern("feed_raw.tsv", r"^feed_.*\.tsv$") is True
    assert recipe._match_pattern("anything.dat", "*") is True
