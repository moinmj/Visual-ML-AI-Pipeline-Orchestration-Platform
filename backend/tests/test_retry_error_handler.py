"""
Unit tests for Retry / Error-Handling Wrapper Recipe.
"""

import json
import pytest
import pandas as pd

from backend.app.core.exceptions import ExecutionException, ValidationException
from backend.app.recipes.flow_control.retry_error_handler import RetryErrorHandlerRecipe
from backend.app.recipes.base.registry import recipe_registry


def test_retry_error_handler_registration():
    """Verify recipe registration and catalog metadata."""
    recipe = recipe_registry.get("retry_error_handler")
    assert recipe is not None
    assert recipe.recipe_id == "retry_error_handler"
    assert recipe.group == "Ingestion & Orchestration"
    assert recipe.subgroup == "Control Flow & Reliability"


def test_retry_error_handler_schema_contract():
    """Verify all schema properties have input_type and description."""
    recipe = RetryErrorHandlerRecipe()
    schema = recipe.get_schema()
    assert schema["type"] == "object"
    props = schema["properties"]
    for field_name, spec in props.items():
        assert "input_type" in spec, f"Field '{field_name}' missing input_type"
        assert "description" in spec, f"Field '{field_name}' missing description"


def test_retry_error_handler_validation_errors():
    """Test parameter validation rules."""
    recipe = RetryErrorHandlerRecipe()

    # Missing target_recipe_id
    with pytest.raises(ValidationException) as exc_1:
        recipe.execute(inputs={}, config={"target_recipe_id": ""})
    assert "Parameter 'target_recipe_id' is required" in str(exc_1.value)

    # Malformed target_recipe_config
    with pytest.raises(ValidationException) as exc_2:
        recipe.execute(
            inputs={},
            config={"target_recipe_id": "limit", "target_recipe_config": "{bad json"},
        )
    assert "Invalid JSON in 'target_recipe_config'" in str(exc_2.value)


def test_retry_error_handler_success_first_attempt():
    """Verify immediate success without triggering retries."""
    recipe = RetryErrorHandlerRecipe()
    input_df = pd.DataFrame([{"val": 10}, {"val": 20}, {"val": 30}])
    config = {
        "target_recipe_id": "limit",
        "target_recipe_config": json.dumps({"rows": 2}),
        "max_retries": 3,
    }
    result = recipe.execute(inputs={"dataframe": input_df}, config=config)
    df = result["dataframe"]
    metrics = result["metrics"]

    assert len(df) == 2
    assert metrics["success"] is True
    assert metrics["attempts_made"] == 1
    assert result["error_output"].empty


def test_retry_error_handler_transient_failure_recovered():
    """Verify recovery after 1 transient failure on attempt 2."""
    recipe = RetryErrorHandlerRecipe()
    input_df = pd.DataFrame([{"val": 10}, {"val": 20}, {"val": 30}])
    config = {
        "target_recipe_id": "limit",
        "target_recipe_config": json.dumps({"rows": 1}),
        "max_retries": 3,
        "simulate_failure_attempts": 1,
        "initial_delay_seconds": 0.01,
    }
    result = recipe.execute(inputs={"dataframe": input_df}, config=config)
    df = result["dataframe"]
    metrics = result["metrics"]

    assert len(df) == 1
    assert metrics["success"] is True
    assert metrics["attempts_made"] == 2
    assert metrics["attempt_history"][0]["status"] == "FAILED"
    assert metrics["attempt_history"][1]["status"] == "SUCCESS"


def test_retry_error_handler_exhaustion_return_input():
    """Verify retries exhausted fallback strategy: return_input."""
    recipe = RetryErrorHandlerRecipe()
    input_df = pd.DataFrame([{"val": 100}])
    config = {
        "target_recipe_id": "non_existent_recipe_abc",
        "max_retries": 2,
        "initial_delay_seconds": 0.01,
        "fallback_strategy": "return_input",
    }
    result = recipe.execute(inputs={"dataframe": input_df}, config=config)
    df = result["dataframe"]
    metrics = result["metrics"]
    err_df = result["error_output"]

    assert len(df) == 1
    assert df.iloc[0]["val"] == 100
    assert metrics["success"] is False
    assert metrics["attempts_made"] == 2
    assert metrics["exhausted_retries"] is True
    assert len(err_df) == 1
    assert "error_message" in err_df.columns


def test_retry_error_handler_exhaustion_empty_dataframe():
    """Verify retries exhausted fallback strategy: empty_dataframe."""
    recipe = RetryErrorHandlerRecipe()
    input_df = pd.DataFrame([{"val": 100}])
    config = {
        "target_recipe_id": "non_existent_recipe_abc",
        "max_retries": 2,
        "initial_delay_seconds": 0.01,
        "fallback_strategy": "empty_dataframe",
    }
    result = recipe.execute(inputs={"dataframe": input_df}, config=config)
    assert result["dataframe"].empty
    assert len(result["error_output"]) == 1


def test_retry_error_handler_exhaustion_fallback_recipe():
    """Verify executing fallback_recipe when primary target fails."""
    recipe = RetryErrorHandlerRecipe()
    input_df = pd.DataFrame([{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}])
    config = {
        "target_recipe_id": "non_existent_recipe_abc",
        "max_retries": 2,
        "initial_delay_seconds": 0.01,
        "fallback_strategy": "fallback_recipe",
        "fallback_recipe_id": "limit",
        "fallback_recipe_config": json.dumps({"rows": 2}),
    }
    result = recipe.execute(inputs={"dataframe": input_df}, config=config)
    df = result["dataframe"]
    assert len(df) == 2
    assert result["metrics"]["success"] is False


def test_retry_error_handler_non_retryable_aborts_early():
    """Verify non-retryable error aborts immediately without attempting remaining retries."""
    recipe = RetryErrorHandlerRecipe()
    input_df = pd.DataFrame([{"val": 10}])
    config = {
        "target_recipe_id": "non_existent_recipe_abc",
        "max_retries": 5,
        "retryable_error_patterns": "*OnlySpecificTimeoutError*",
    }
    result = recipe.execute(inputs={"dataframe": input_df}, config=config)
    # Target failed with NotFoundException / RecipeNotFound, which does NOT match *OnlySpecificTimeoutError*
    # So it should abort after attempt 1
    assert result["metrics"]["attempts_made"] == 1
    assert result["metrics"]["success"] is False
