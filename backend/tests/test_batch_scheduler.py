"""
Unit tests for Batch Scheduler with Dependency Graph Recipe.
"""

import json
import pytest
import pandas as pd

from backend.app.core.exceptions import ExecutionException, ValidationException
from backend.app.recipes.flow_control.batch_scheduler import BatchSchedulerRecipe
from backend.app.recipes.base.registry import recipe_registry


def test_batch_scheduler_registration():
    """Verify recipe registration and catalog metadata."""
    recipe = recipe_registry.get("batch_scheduler")
    assert recipe is not None
    assert recipe.recipe_id == "batch_scheduler"
    assert recipe.group == "Ingestion & Orchestration"
    assert recipe.subgroup == "Control Flow & Reliability"


def test_batch_scheduler_schema_contract():
    """Verify all schema properties have input_type and description."""
    recipe = BatchSchedulerRecipe()
    schema = recipe.get_schema()
    assert schema["type"] == "object"
    props = schema["properties"]
    for field_name, spec in props.items():
        assert "input_type" in spec, f"Field '{field_name}' missing input_type"
        assert "description" in spec, f"Field '{field_name}' missing description"


def test_batch_scheduler_validation_errors():
    """Test parameter validation rules."""
    recipe = BatchSchedulerRecipe()

    # Malformed dependency graph JSON
    with pytest.raises(ValidationException) as exc_1:
        recipe.execute(inputs={}, config={"dependency_graph": "{unclosed_json"})
    assert "Invalid JSON in 'dependency_graph'" in str(exc_1.value)

    # Malformed custom job list JSON
    with pytest.raises(ValidationException) as exc_2:
        recipe.execute(inputs={}, config={"custom_job_list": "{not_a_list"})
    assert "Invalid JSON in 'custom_job_list'" in str(exc_2.value)


def test_batch_scheduler_simulation_fallback():
    """Verify simulation fallback creates synthetic DAG stages and wave levels."""
    recipe = BatchSchedulerRecipe()
    config = {
        "batch_strategy": "partition_column",
        "simulation_fallback": True,
        "output_mode": "execute_batches",
    }
    result = recipe.execute(inputs={}, config=config)
    df = result["dataframe"]
    plan = result["batch_plan"]
    metrics = result["metrics"]

    assert not df.empty
    assert metrics["is_simulation"] is True
    assert metrics["waves_executed"] == 3
    assert "_batch_id" in df.columns
    assert "_wave_level" in df.columns
    assert "wave_level" in plan.columns
    assert len(plan) == 3


def test_batch_scheduler_plan_only_mode():
    """Verify plan_only output mode produces DAG execution schedule without data mutations."""
    recipe = BatchSchedulerRecipe()
    input_df = pd.DataFrame([
        {"dept": "Sales", "amount": 100},
        {"dept": "Marketing", "amount": 50},
        {"dept": "Finance", "amount": 200},
    ])
    deps = {
        "Finance": ["Sales"],
        "Marketing": ["Sales"],
    }
    config = {
        "batch_strategy": "partition_column",
        "partition_column": "dept",
        "dependency_graph": json.dumps(deps),
        "output_mode": "plan_only",
    }
    result = recipe.execute(inputs={"dataframe": input_df}, config=config)
    plan = result["batch_plan"]
    metrics = result["metrics"]

    assert len(plan) == 3
    assert metrics["output_mode"] == "plan_only"
    # Wave 1 should be Sales; Wave 2 should contain Finance & Marketing
    sales_wave = plan[plan["batch_id"] == "Sales"]["wave_level"].iloc[0]
    finance_wave = plan[plan["batch_id"] == "Finance"]["wave_level"].iloc[0]
    assert sales_wave == 1
    assert finance_wave == 2


def test_batch_scheduler_fixed_size_chunking():
    """Verify chunking input dataset into fixed row size batches."""
    recipe = BatchSchedulerRecipe()
    input_df = pd.DataFrame([{"row_id": i} for i in range(1, 11)])
    config = {
        "batch_strategy": "fixed_size",
        "batch_size": 4,
        "output_mode": "execute_batches",
    }
    result = recipe.execute(inputs={"dataframe": input_df}, config=config)
    df = result["dataframe"]
    metrics = result["metrics"]

    assert len(df) == 10
    # 10 rows chunked by 4 yields 3 chunks: chunk_1 (4), chunk_2 (4), chunk_3 (2)
    assert metrics["total_batches"] == 3
    assert set(df["_batch_id"]) == {"chunk_1", "chunk_2", "chunk_3"}


def test_batch_scheduler_cyclic_dependency_detection():
    """Verify cycle detection raises ValidationException."""
    recipe = BatchSchedulerRecipe()
    input_df = pd.DataFrame([{"batch_id": "A"}, {"batch_id": "B"}])
    # Cycle: A depends on B, B depends on A
    cyclic_deps = {
        "A": ["B"],
        "B": ["A"],
    }
    config = {
        "batch_strategy": "partition_column",
        "dependency_graph": json.dumps(cyclic_deps),
    }
    with pytest.raises(ValidationException) as exc:
        recipe.execute(inputs={"dataframe": input_df}, config=config)
    assert "Cyclic dependency detected" in str(exc.value)


def test_batch_scheduler_custom_job_dag():
    """Verify custom job DAG definition execution."""
    recipe = BatchSchedulerRecipe()
    custom_jobs = [
        {"job_id": "extract_job", "depends_on": []},
        {"job_id": "transform_job", "depends_on": ["extract_job"]},
        {"job_id": "publish_job", "depends_on": ["transform_job"]},
    ]
    config = {
        "batch_strategy": "custom_job_dag",
        "custom_job_list": json.dumps(custom_jobs),
        "output_mode": "execute_batches",
    }
    result = recipe.execute(inputs={"dataframe": pd.DataFrame([{"val": 1}])}, config=config)
    metrics = result["metrics"]
    assert metrics["total_batches"] == 3
    assert metrics["waves_executed"] == 3
