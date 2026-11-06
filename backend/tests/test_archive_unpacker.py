"""
Unit tests for Zip/Archive Unpacker Recipe.
"""

import base64
import io
import os
import tarfile
import tempfile
import zipfile
import pytest
import pandas as pd

from backend.app.core.exceptions import ExecutionException, ValidationException
from backend.app.recipes.ingestion.archive_unpacker import ArchiveUnpackerRecipe
from backend.app.recipes.base.registry import recipe_registry


def test_archive_unpacker_registration():
    """Verify recipe registration and catalog metadata."""
    recipe = recipe_registry.get("archive_unpacker")
    assert recipe is not None
    assert recipe.recipe_id == "archive_unpacker"
    assert recipe.group == "Ingestion & Orchestration"
    assert recipe.subgroup == "File & Web Ingestion"


def test_archive_unpacker_schema_contract():
    """Verify all schema properties have input_type and description."""
    recipe = ArchiveUnpackerRecipe()
    schema = recipe.get_schema()
    assert schema["type"] == "object"
    props = schema["properties"]
    for field_name, spec in props.items():
        assert "input_type" in spec, f"Field '{field_name}' missing input_type"
        assert "description" in spec, f"Field '{field_name}' missing description"


def test_archive_unpacker_validation_errors():
    """Test parameter validation rules."""
    recipe = ArchiveUnpackerRecipe()

    # Empty archive_path
    with pytest.raises(ValidationException) as exc_1:
        recipe.execute(inputs={}, config={"archive_path": ""})
    assert "Parameter 'archive_path' cannot be empty" in str(exc_1.value)

    # Malformed mock base64
    with pytest.raises(ValidationException) as exc_2:
        recipe.execute(
            inputs={},
            config={"archive_path": "test.zip", "mock_archive_base64": "not_valid_b64!!!"},
        )
    assert "Failed decoding or extracting" in str(exc_2.value)


def test_archive_unpacker_simulation_fallback():
    """Verify synthetic sales data generated when archive file is missing."""
    recipe = ArchiveUnpackerRecipe()
    config = {
        "archive_path": "/missing/path/to/archive.zip",
        "simulation_fallback": True,
        "include_metadata_columns": True,
    }
    result = recipe.execute(inputs={}, config=config)
    df = result["dataframe"]
    metrics = result["metrics"]

    assert not df.empty
    assert metrics["is_simulation"] is True
    assert metrics["members_parsed"] == 2
    assert len(df) == 3
    assert "_archive_path" in df.columns
    assert "_member_name" in df.columns
    assert "transaction_id" in df.columns


def test_archive_unpacker_manifest_mode():
    """Verify archive_manifest mode returns member metadata table."""
    recipe = ArchiveUnpackerRecipe()
    # Create an in-memory zip
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w") as zf:
        zf.writestr("table1.csv", "id,val\n1,10\n")
        zf.writestr("table2.csv", "id,val\n2,20\n")
    b64_str = base64.b64encode(bio.getvalue()).decode("ascii")

    config = {
        "archive_path": "mock.zip",
        "mock_archive_base64": b64_str,
        "output_mode": "archive_manifest",
    }
    result = recipe.execute(inputs={}, config=config)
    df = result["dataframe"]
    metrics = result["metrics"]

    assert len(df) == 2
    assert set(df["filename"]) == {"table1.csv", "table2.csv"}
    assert "size_bytes" in df.columns
    assert "modified_at" in df.columns
    assert metrics["output_mode"] == "archive_manifest"


def test_archive_unpacker_real_zip_file():
    """Verify reading real ZIP file on disk."""
    recipe = ArchiveUnpackerRecipe()
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "package.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("customers_a.csv", "cust_id,tier\nC-1,Gold\n")
            zf.writestr("customers_b.csv", "cust_id,tier\nC-2,Silver\n")

        config = {
            "archive_path": zip_path,
            "file_pattern": "*.csv",
            "output_mode": "parse_data",
            "include_metadata_columns": True,
        }
        result = recipe.execute(inputs={}, config=config)
        df = result["dataframe"]
        metrics = result["metrics"]

        assert len(df) == 2
        assert list(df["cust_id"]) == ["C-1", "C-2"]
        assert metrics["is_simulation"] is False
        assert metrics["members_parsed"] == 2


def test_archive_unpacker_real_tar_gz_file():
    """Verify reading real TAR.GZ file on disk."""
    recipe = ArchiveUnpackerRecipe()
    with tempfile.TemporaryDirectory() as tmpdir:
        tar_path = os.path.join(tmpdir, "archive.tar.gz")
        with tarfile.open(tar_path, "w:gz") as tf:
            data = b"product,price\nKeyboard,79.99\nMouse,49.99\n"
            ti = tarfile.TarInfo(name="catalog/items.csv")
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))

        config = {
            "archive_path": tar_path,
            "file_pattern": "*.csv",
        }
        result = recipe.execute(inputs={}, config=config)
        df = result["dataframe"]
        assert len(df) == 2
        assert list(df["product"]) == ["Keyboard", "Mouse"]


def test_archive_unpacker_zip_slip_protection():
    """Verify Zip Slip path traversal attempts are safely ignored."""
    recipe = ArchiveUnpackerRecipe()
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w") as zf:
        zf.writestr("../../etc/passwd", "root:x:0:0::/root:/bin/bash\n")
        zf.writestr("legit_data.csv", "col\n1\n")
    b64_str = base64.b64encode(bio.getvalue()).decode("ascii")

    config = {
        "archive_path": "mock.zip",
        "mock_archive_base64": b64_str,
        "output_mode": "archive_manifest",
    }
    result = recipe.execute(inputs={}, config=config)
    df = result["dataframe"]
    # Only legit_data.csv should be included; dangerous path traversal skipped
    assert len(df) == 1
    assert df.iloc[0]["filename"] == "legit_data.csv"


def test_archive_unpacker_simulation_disabled_raises():
    """Verify error raised when archive file is missing and simulation fallback is false."""
    recipe = ArchiveUnpackerRecipe()
    config = {
        "archive_path": "/nonexistent/test_123.zip",
        "simulation_fallback": False,
    }
    with pytest.raises(ExecutionException) as exc:
        recipe.execute(inputs={}, config=config)
    assert "Archive file not found" in str(exc.value)
