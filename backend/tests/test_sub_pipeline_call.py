"""
Unit tests for Sub-Pipeline Call Recipe.
"""

import json
import pytest
import pandas as pd

from backend.app.core.exceptions import ExecutionException, ValidationException
from backend.app.recipes.flow_control.sub_pipeline_call import SubPipelineCallRecipe
from backend.app.recipes.base.registry import recipe_registry


def test_sub_pipeline_call_registration():
    """Verify recipe registration and catalog metadata."""
    recipe = recipe_registry.get("sub_pipeline_call")
    assert recipe is not None
    assert recipe.recipe_id == "sub_pipeline_call"
    assert recipe.group == "Ingestion & Orchestration"
    assert recipe.subgroup == "Control Flow & Reliability"


def test_sub_pipeline_call_schema_contract():
    """Verify all schema properties have input_type and description."""
    recipe = SubPipelineCallRecipe()
    schema = recipe.get_schema()
    assert schema["type"] == "object"
    props = schema["properties"]
    for field_name, spec in props.items():
        assert "input_type" in spec, f"Field '{field_name}' missing input_type"
        assert "description" in spec, f"Field '{field_name}' missing description"


def test_sub_pipeline_call_validation_errors():
    """Test required parameter and JSON formatting validations."""
    recipe = SubPipelineCallRecipe()

    # Missing pipeline_id
    with pytest.raises(ValidationException) as exc_1:
        recipe.execute(inputs={}, config={"pipeline_id": ""})
    assert "Parameter 'pipeline_id' is required" in str(exc_1.value)

    # Malformed parameter overrides JSON
    with pytest.raises(ValidationException) as exc_2:
        recipe.execute(
            inputs={},
            config={"pipeline_id": "test_pipe", "parameter_overrides": "{invalid_json"},
        )
    assert "Invalid JSON provided in 'parameter_overrides'" in str(exc_2.value)

    # Malformed sub_pipeline_steps JSON
    with pytest.raises(ValidationException) as exc_3:
        recipe.execute(
            inputs={},
            config={"pipeline_id": "test_pipe", "sub_pipeline_steps": "{unclosed"},
        )
    assert "Invalid JSON provided in 'sub_pipeline_steps'" in str(exc_3.value)


def test_sub_pipeline_call_simulation_with_input_df():
    """Verify simulation mode enriches upstream input dataframe."""
    recipe = SubPipelineCallRecipe()
    input_df = pd.DataFrame([{"id": 1, "score": 85.0}, {"id": 2, "score": 92.5}])
    config = {
        "pipeline_id": "clean_and_enrich_pipe_v2",
        "simulation_fallback": True,
        "pass_inputs": True,
    }
    result = recipe.execute(inputs={"dataframe": input_df}, config=config)
    df = result["dataframe"]
    metrics = result["metrics"]

    assert len(df) == 2
    assert "_sub_pipeline_applied" in df.columns
    assert df.iloc[0]["_sub_pipeline_applied"] == "clean_and_enrich_pipe_v2"
    assert metrics["is_simulation"] is True
    assert metrics["input_rows"] == 2
    assert metrics["output_rows"] == 2


def test_sub_pipeline_call_simulation_without_input_df():
    """Verify simulation mode generates standalone dataset when no inputs provided."""
    recipe = SubPipelineCallRecipe()
    config = {
        "pipeline_id": "standalone_source_pipe",
        "simulation_fallback": True,
    }
    result = recipe.execute(inputs={}, config=config)
    df = result["dataframe"]
    assert not df.empty
    assert "sub_record_id" in df.columns
    assert "status" in df.columns


def test_sub_pipeline_call_inline_steps_execution():
    """Verify chaining real registered recipes inline inside the sub-pipeline."""
    recipe = SubPipelineCallRecipe()
    input_df = pd.DataFrame([
        {"id": 3, "val": 30},
        {"id": 1, "val": 10},
        {"id": 2, "val": 20},
        {"id": 4, "val": 40},
    ])

    # Step 1: Sort by 'id' ascending
    # Step 2: Limit to 2 rows
    inline_steps = [
        {"recipe_id": "sort", "config": {"columns": ["id"], "ascending": True}},
        {"recipe_id": "limit", "config": {"rows": 2}},
    ]

    config = {
        "pipeline_id": "inline_sort_limit_pipe",
        "sub_pipeline_steps": json.dumps(inline_steps),
        "pass_inputs": True,
    }
    result = recipe.execute(inputs={"dataframe": input_df}, config=config)
    df = result["dataframe"]
    metrics = result["metrics"]

    assert len(df) == 2
    assert list(df["id"]) == [1, 2]
    assert metrics["is_simulation"] is False
    assert len(metrics["step_logs"]) == 2
    assert metrics["step_logs"][0]["status"] == "SUCCESS"
    assert metrics["step_logs"][1]["status"] == "SUCCESS"


def test_sub_pipeline_call_error_propagation_true():
    """Verify ExecutionException raised when inline step fails and propagate_errors is true."""
    recipe = SubPipelineCallRecipe()
    input_df = pd.DataFrame([{"val": 100}])
    # Step refers to non-existent recipe ID
    bad_steps = [{"recipe_id": "completely_invalid_recipe_xyz", "config": {}}]
    config = {
        "pipeline_id": "failing_pipe",
        "sub_pipeline_steps": json.dumps(bad_steps),
        "propagate_errors": True,
    }
    with pytest.raises(ExecutionException) as exc:
        recipe.execute(inputs={"dataframe": input_df}, config=config)
    assert "Sub-pipeline failed at step 1" in str(exc.value)


def test_sub_pipeline_call_error_propagation_false():
    """Verify sub-pipeline stops and returns partial df when propagate_errors is false."""
    recipe = SubPipelineCallRecipe()
    input_df = pd.DataFrame([{"val": 100}])
    bad_steps = [{"recipe_id": "completely_invalid_recipe_xyz", "config": {}}]
    config = {
        "pipeline_id": "failing_pipe_tolerated",
        "sub_pipeline_steps": json.dumps(bad_steps),
        "propagate_errors": False,
    }
    result = recipe.execute(inputs={"dataframe": input_df}, config=config)
    assert len(result["dataframe"]) == 1
    assert result["metrics"]["step_logs"][0]["status"] == "FAILED"


def test_sub_pipeline_call_simulation_disabled_raises():
    """Verify ExecutionException when sub-pipeline not found and simulation fallback is false."""
    recipe = SubPipelineCallRecipe()
    config = {
        "pipeline_id": "non_existent_pipe",
        "simulation_fallback": False,
        "execution_mode": "synchronous",
    }
    with pytest.raises(ExecutionException) as exc:
        recipe.execute(inputs={}, config=config)
    assert "Target sub-pipeline 'non_existent_pipe' not found" in str(exc.value)
