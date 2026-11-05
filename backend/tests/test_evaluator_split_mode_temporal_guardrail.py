import pytest
import numpy as np
import pandas as pd

from backend.app.recipes.splitting.stratified_split import StratifiedSplitRecipe
from backend.app.recipes.splitting.train_test_split import TrainTestSplitRecipe
from backend.app.recipes.splitting.time_series_split import TimeSeriesSplitRecipe
from backend.app.recipes.splitting.walk_forward_split import WalkForwardSplitRecipe
from backend.app.recipes.evaluation.model_evaluator import ModelEvaluatorRecipe
from backend.app.recipes.training.random_forest_trainer import RandomForestTrainerRecipe
from backend.app.engine.dag.graph import WorkflowGraph, WorkflowNode, WorkflowEdge
from backend.app.engine.execution.executor import DAGExecutor


@pytest.fixture
def sample_dataset_with_timestamp():
    """Synthetic dataset with a timestamp column, numeric features, and a binary target."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2026-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "timestamp": dates.strftime("%Y-%m-%d"),
        "feature_1": np.random.randn(n),
        "feature_2": np.random.randn(n) * 10,
        "target_churn": np.random.choice([0, 1], size=n, p=[0.7, 0.3])
    })
    return df


def test_stratified_split_recipe_sets_is_temporal_false(sample_dataset_with_timestamp):
    """
    Stratified split randomly shuffles rows to balance class distributions.
    Verify that StratifiedSplitRecipe outputs is_temporal=False.
    """
    recipe = StratifiedSplitRecipe()
    out = recipe.execute(
        inputs={"dataframe": sample_dataset_with_timestamp},
        config={"target_column": "target_churn", "test_size": 0.2, "random_state": 42}
    )

    assert out["split_mode"] == "stratified"
    assert out["is_temporal"] is False

    # Pass through Random Forest and Model Evaluator
    rf = RandomForestTrainerRecipe()
    rf_out = rf.execute(inputs=out, config={"task_type": "classification", "n_estimators": 5, "random_state": 42})

    eval_inputs = {**out, **rf_out}
    evaluator = ModelEvaluatorRecipe()
    eval_res = evaluator.execute(inputs=eval_inputs, config={})
    metrics = eval_res["metrics"]

    assert metrics["split_mode"] == "stratified"
    assert metrics["is_temporal"] is False
    assert metrics["temporal_column"] is None
    assert "report" in metrics
    assert "Temporal Order     : False (Shuffled / Non-chronological)" in metrics["report"]


def test_train_test_split_random_mode_sets_is_temporal_false(sample_dataset_with_timestamp):
    """
    Standard random train_test_split (stratify=False, time_series_mode=False) must have is_temporal=False.
    """
    recipe = TrainTestSplitRecipe()
    out = recipe.execute(
        inputs={"dataframe": sample_dataset_with_timestamp},
        config={"target_column": "target_churn", "test_size": 0.2, "stratify": False, "random_state": 42}
    )

    assert out["split_mode"] == "random"
    assert out["is_temporal"] is False

    rf = RandomForestTrainerRecipe()
    rf_out = rf.execute(inputs=out, config={"task_type": "classification", "n_estimators": 5, "random_state": 42})

    eval_inputs = {**out, **rf_out}
    evaluator = ModelEvaluatorRecipe()
    eval_res = evaluator.execute(inputs=eval_inputs, config={})
    metrics = eval_res["metrics"]

    assert metrics["split_mode"] == "random"
    assert metrics["is_temporal"] is False
    assert metrics["temporal_column"] is None


def test_train_test_split_stratify_flag_overrides_time_series_mode(sample_dataset_with_timestamp):
    """
    If stratify=True is specified, it must force chronological time_series_mode to False,
    setting is_temporal=False because class stratification breaks time order.
    """
    recipe = TrainTestSplitRecipe()
    out = recipe.execute(
        inputs={"dataframe": sample_dataset_with_timestamp},
        config={
            "target_column": "target_churn",
            "test_size": 0.2,
            "stratify": True,
            "time_series_mode": True,  # Conflicting flag: stratify MUST win
            "random_state": 42
        }
    )

    assert out["split_mode"] == "stratified"
    assert out["is_temporal"] is False

    evaluator = ModelEvaluatorRecipe()
    rf = RandomForestTrainerRecipe()
    rf_out = rf.execute(inputs=out, config={"task_type": "classification", "n_estimators": 5, "random_state": 42})
    eval_res = evaluator.execute(inputs={**out, **rf_out}, config={})
    metrics = eval_res["metrics"]

    assert metrics["split_mode"] == "stratified"
    assert metrics["is_temporal"] is False
    assert metrics["temporal_column"] is None


def test_chronological_time_series_split_sets_is_temporal_true(sample_dataset_with_timestamp):
    """
    Chronological time series split preserves temporal ordering, so is_temporal must be True.
    """
    recipe = TimeSeriesSplitRecipe()
    out = recipe.execute(
        inputs={"dataframe": sample_dataset_with_timestamp},
        config={"target_column": "target_churn", "time_column": "timestamp", "test_size": 0.2}
    )

    assert out["is_temporal"] is True
    assert out["split_mode"] == "chronological"

    rf = RandomForestTrainerRecipe()
    rf_out = rf.execute(inputs=out, config={"task_type": "classification", "n_estimators": 5, "random_state": 42})

    evaluator = ModelEvaluatorRecipe()
    eval_res = evaluator.execute(inputs={**out, **rf_out}, config={})
    metrics = eval_res["metrics"]

    assert metrics["split_mode"] == "chronological"
    assert metrics["is_temporal"] is True
    assert metrics["temporal_column"] == "timestamp"


def test_walk_forward_split_sets_is_temporal_true(sample_dataset_with_timestamp):
    """
    Walk-forward rolling window splitter preserves temporal ordering, so is_temporal must be True.
    """
    recipe = WalkForwardSplitRecipe()
    out = recipe.execute(
        inputs={"dataframe": sample_dataset_with_timestamp},
        config={"target_column": "target_churn", "time_column": "timestamp", "window_size": 50, "test_step_size": 10}
    )

    assert out["is_temporal"] is True
    assert out["split_mode"] == "walk_forward"

    rf = RandomForestTrainerRecipe()
    rf_out = rf.execute(inputs=out, config={"task_type": "classification", "n_estimators": 5, "random_state": 42})

    evaluator = ModelEvaluatorRecipe()
    eval_res = evaluator.execute(inputs={**out, **rf_out}, config={})
    metrics = eval_res["metrics"]

    assert metrics["split_mode"] == "walk_forward"
    assert metrics["is_temporal"] is True


def test_dag_workflow_stratified_split_suppresses_temporal_in_schema_and_metrics(sample_dataset_with_timestamp):
    """
    Verify end-to-end DAG execution:
    A pipeline with Stratified Train/Test Split on a dataset containing timestamp features
    must output is_temporal=False in final_metrics and has_temporal_feature=False in inference_schema.
    """
    nodes = [
        WorkflowNode(
            id="node_split",
            recipe_id="train_test_split",
            config={"target_column": "target_churn", "stratify": True, "test_size": 0.2, "random_state": 42}
        ),
        WorkflowNode(
            id="node_model",
            recipe_id="random_forest_trainer",
            config={"task_type": "classification", "n_estimators": 5, "random_state": 42}
        ),
        WorkflowNode(
            id="node_eval",
            recipe_id="model_evaluator",
            config={}
        )
    ]
    edges = [
        WorkflowEdge(source="node_split", target="node_model"),
        WorkflowEdge(source="node_split", target="node_eval"),
        WorkflowEdge(source="node_model", target="node_eval")
    ]
    graph = WorkflowGraph(nodes=nodes, edges=edges)

    exec_res = DAGExecutor.execute_workflow(
        execution_id="test_stratified_temporal_suppression_01",
        workflow=graph,
        initial_df=sample_dataset_with_timestamp
    )

    assert exec_res.status == "SUCCESS"
    assert exec_res.inference_schema is not None
    assert exec_res.inference_schema["has_temporal_feature"] is False
    assert exec_res.inference_schema["temporal_column"] is None

    assert exec_res.final_metrics is not None
    assert exec_res.final_metrics["split_mode"] == "stratified"
    assert exec_res.final_metrics["is_temporal"] is False
    assert exec_res.final_metrics["temporal_column"] is None
