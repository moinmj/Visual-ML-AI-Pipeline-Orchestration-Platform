"""
Unit tests for Dead-Letter Queue (DLQ) Handler Recipe.
"""

import os
import tempfile
import pytest
import pandas as pd

from backend.app.core.exceptions import ExecutionException, ValidationException
from backend.app.recipes.flow_control.dead_letter_queue import DeadLetterQueueRecipe
from backend.app.recipes.base.registry import recipe_registry


def test_dead_letter_queue_registration():
    """Verify recipe registration and catalog metadata."""
    recipe = recipe_registry.get("dead_letter_queue")
    assert recipe is not None
    assert recipe.recipe_id == "dead_letter_queue"
    assert recipe.group == "Ingestion & Orchestration"
    assert recipe.subgroup == "Control Flow & Reliability"


def test_dead_letter_queue_schema_contract():
    """Verify all schema properties have input_type and description."""
    recipe = DeadLetterQueueRecipe()
    schema = recipe.get_schema()
    assert schema["type"] == "object"
    props = schema["properties"]
    for field_name, spec in props.items():
        assert "input_type" in spec, f"Field '{field_name}' missing input_type"
        assert "description" in spec, f"Field '{field_name}' missing description"


def test_dead_letter_queue_simulation_fallback():
    """Verify default simulation fallback creates clean output and DLQ isolation."""
    recipe = DeadLetterQueueRecipe()
    config = {
        "dlq_mode": "trap_and_isolate",
        "simulation_fallback": True,
    }
    result = recipe.execute(inputs={}, config=config)
    clean_df = result["dataframe"]
    dlq_df = result["dlq_output"]
    metrics = result["metrics"]

    assert len(clean_df) == 8
    assert len(dlq_df) == 2
    assert metrics["is_simulation"] is True
    assert metrics["quarantined_records_count"] == 2
    assert "_dlq_id" in dlq_df.columns
    assert "_dlq_timestamp" in dlq_df.columns
    assert "_replay_status" in dlq_df.columns


def test_dead_letter_queue_explicit_failed_input():
    """Verify routing explicit failed_input into the DLQ stream."""
    recipe = DeadLetterQueueRecipe()
    clean_in = pd.DataFrame([{"id": 1, "val": "good"}])
    failed_in = pd.DataFrame([{"id": 99, "val": "bad", "error_message": "Invalid type"}])

    config = {
        "dlq_mode": "trap_and_isolate",
        "error_category_default": "SCHEMA_VIOLATION",
    }
    result = recipe.execute(
        inputs={"dataframe": clean_in, "failed_input": failed_in},
        config=config,
    )
    clean_out = result["dataframe"]
    dlq_out = result["dlq_output"]

    assert len(clean_out) == 1
    assert len(dlq_out) == 1
    assert dlq_out.iloc[0]["id"] == 99
    assert dlq_out.iloc[0]["_error_category"] == "SCHEMA_VIOLATION"
    assert dlq_out.iloc[0]["_replay_status"] == "PENDING"


def test_dead_letter_queue_validation_column_filter():
    """Verify filtering null/empty validation_column from primary data into DLQ."""
    recipe = DeadLetterQueueRecipe()
    primary = pd.DataFrame([
        {"id": 1, "email": "alice@corp.com"},
        {"id": 2, "email": None},  # Corrupt
        {"id": 3, "email": "charlie@corp.com"},
        {"id": 4, "email": "   "},  # Corrupt
    ])
    config = {
        "dlq_mode": "pass_clean_only",
        "validation_column": "email",
    }
    result = recipe.execute(inputs={"dataframe": primary}, config=config)
    clean_out = result["dataframe"]
    dlq_out = result["dlq_output"]

    assert len(clean_out) == 2
    assert set(clean_out["id"]) == {1, 3}
    assert len(dlq_out) == 2
    assert set(dlq_out["id"]) == {2, 4}
    assert dlq_out.iloc[0]["_error_category"] == "VALIDATION_FAILED"


def test_dead_letter_queue_persistence_to_disk():
    """Verify appending quarantined records to disk file."""
    recipe = DeadLetterQueueRecipe()
    failed = pd.DataFrame([{"err_item": "A"}, {"err_item": "B"}])
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_file = os.path.join(tmpdir, "dlq.jsonl")
        config = {
            "persist_to_disk": True,
            "storage_file_path": storage_file,
        }
        recipe.execute(inputs={"failed_input": failed}, config=config)

        assert os.path.exists(storage_file)
        # Read back from file
        reloaded = pd.read_json(storage_file, lines=True)
        assert len(reloaded) == 2
        assert "_dlq_id" in reloaded.columns


def test_dead_letter_queue_replay_from_storage():
    """Verify replay_from_storage mode reads back quarantined file and tags replayed."""
    recipe = DeadLetterQueueRecipe()
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_file = os.path.join(tmpdir, "replay.jsonl")
        pd.DataFrame([{"remedied_id": "REM-1"}]).to_json(storage_file, orient="records", lines=True)

        config = {
            "dlq_mode": "replay_from_storage",
            "storage_file_path": storage_file,
        }
        result = recipe.execute(inputs={}, config=config)
        replayed_df = result["dataframe"]

        assert len(replayed_df) == 1
        assert replayed_df.iloc[0]["remedied_id"] == "REM-1"
        assert replayed_df.iloc[0]["_replay_status"] == "REPLAYED"


def test_dead_letter_queue_simulation_disabled_raises():
    """Verify error raised when inputs empty and simulation fallback is false."""
    recipe = DeadLetterQueueRecipe()
    config = {
        "dlq_mode": "trap_and_isolate",
        "simulation_fallback": False,
    }
    with pytest.raises(ExecutionException) as exc:
        recipe.execute(inputs={}, config=config)
    assert "No input data provided" in str(exc.value)
