import os
import tempfile
import pandas as pd
import pytest

from backend.app.recipes.security.audit_trail_logger import AuditTrailLoggerRecipe


def test_audit_trail_logger_metadata_and_schema():
    recipe = AuditTrailLoggerRecipe()
    assert recipe.recipe_id == "audit_trail_logger"
    assert recipe.category == "security"
    assert "dataframe" in recipe.output_types

    schema = recipe.get_schema()
    assert schema["type"] == "object"
    props = schema["properties"]
    assert "actor_id" in props
    assert "event_type" in props
    assert "compliance_standard" in props
    assert "tamper_evident_hashing" in props

    for prop_name, prop_data in props.items():
        assert "description" in prop_data, f"Property {prop_name} missing description"
        assert "input_type" in prop_data, f"Property {prop_name} missing input_type"


def test_audit_trail_logger_in_memory_and_tamper_evident_hash():
    recipe = AuditTrailLoggerRecipe()
    result = recipe.execute(
        {},
        {
            "actor_id": "service_account_mlops",
            "event_type": "pipeline_run",
            "compliance_standard": "SOC2_TYPE_II",
            "include_schema_fingerprint": True,
            "tamper_evident_hashing": True,
        },
    )

    assert "audit_entry" in result
    assert "audit_hash" in result
    entry = result["audit_entry"]
    assert entry["actor_id"] == "service_account_mlops"
    assert entry["compliance_standard"] == "SOC2_TYPE_II"
    assert entry["dataset_summary"]["schema_fingerprint"] is not None
    assert len(result["audit_hash"]) == 64  # SHA-256


def test_audit_trail_logger_file_append():
    recipe = AuditTrailLoggerRecipe()
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = os.path.join(tmpdir, "test_audit.jsonl")
        df = pd.DataFrame({"item": ["A", "B"], "qty": [10, 20]})

        result = recipe.execute(
            {"input": df},
            {
                "audit_destination": "append_log_file",
                "log_file_path": log_file,
                "actor_id": "auditor_jane",
            },
        )

        assert os.path.exists(log_file)
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 1
        assert "auditor_jane" in lines[0]
        assert result["dataframe"].equals(df)
