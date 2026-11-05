import pytest
import numpy as np
import pandas as pd
from backend.app.recipes.training.xgboost_trainer import XGBoostTrainerRecipe
from backend.app.recipes.training.random_forest_trainer import RandomForestTrainerRecipe
from backend.app.recipes.evaluation.model_evaluator import ModelEvaluatorRecipe
from backend.app.engine.dag.graph import WorkflowGraph, WorkflowNode, WorkflowEdge


def test_xgboost_trainer_validate_config_discrete_target_regression_warning():
    """Verify recipe configuration validation emits an inline warning when Regression mode is paired with a binary target."""
    trainer = XGBoostTrainerRecipe()

    # 1. Regression on discrete churn target -> should emit inline warning
    warns = trainer.validate_config({"task_type": "regression", "target_column": "target_churn"})
    assert len(warns) == 1
    assert "⚠️ Inline Validation Warning" in warns[0]
    assert "target_churn" in warns[0]
    assert "sales_amount" in warns[0]

    # 2. Regression on sales_amount -> valid, no warning
    assert trainer.validate_config({"task_type": "regression", "target_column": "sales_amount"}) == []

    # 3. Classification on target_churn -> valid, no warning
    assert trainer.validate_config({"task_type": "classification", "target_column": "target_churn"}) == []


def test_random_forest_trainer_validate_config_discrete_target():
    """Verify RandomForest recipe also validates target column against task_type."""
    trainer = RandomForestTrainerRecipe()

    warns = trainer.validate_config({"task_type": "regression", "target_column": "is_fraud"})
    assert len(warns) == 1
    assert "⚠️ Inline Validation Warning" in warns[0]
    assert "is_fraud" in warns[0]

    assert trainer.validate_config({"task_type": "regression", "target_column": "house_price"}) == []


def test_xgboost_trainer_execute_regression_on_binary_target_emits_warning():
    """Verify trainer emits diagnostic warning when fit on binary 0/1 target in regression mode."""
    trainer = XGBoostTrainerRecipe()

    X_train = pd.DataFrame({"x1": [1.0, 2.0, 3.0, 4.0], "x2": [10.0, 20.0, 30.0, 40.0]})
    # Binary 0/1 churn targets
    y_train = pd.Series([0, 1, 0, 1], name="target_churn")

    res = trainer.execute(
        inputs={"X_train": X_train, "y_train": y_train},
        config={"task_type": "regression", "n_estimators": 5}
    )

    assert "warnings" in res
    warn_text = " ".join(res["warnings"])
    assert "[WARNING: DISCRETE TARGET IN REGRESSION]" in warn_text
    assert "sales_amount" in warn_text


def test_model_evaluator_regression_on_binary_target_emits_diagnostic_warning():
    """Verify Model Performance Evaluator catches misconfigured regression on binary 0/1 column."""
    evaluator = ModelEvaluatorRecipe()

    class MockRegressor:
        def predict(self, X):
            return np.array([0.2, 0.8, 0.3, 0.9])

    X_test = pd.DataFrame({"x": [1, 2, 3, 4]})
    y_test = pd.Series([0, 1, 0, 1])

    result = evaluator.execute(
        inputs={
            "model": MockRegressor(),
            "X_test": X_test,
            "y_test": y_test,
            "task_type": "regression"
        },
        config={"min_sample_threshold": 2}
    )

    metrics = result["metrics"]
    report = result["report"]

    assert metrics["is_discrete_binary_target"] is True
    warn_text = " ".join(metrics["warnings"])
    assert "[WARNING: DISCRETE 0/1 TARGET IN REGRESSION]" in warn_text
    assert "Target Type        : Discrete Binary (0/1) [WARNING: MISCONFIGURED REGRESSION]" in report
    assert "Diagnostics & Guardrail Warnings" in report


def test_workflow_dag_validation_flags_regression_on_churn_target():
    """Verify canvas DAG pre-flight validation alerts user when XGBoost is set to regression with target_churn."""
    nodes = [
        WorkflowNode(
            id="node_split",
            recipe_id="train_test_split",
            config={"target_column": "target_churn", "test_size": 0.2}
        ),
        WorkflowNode(
            id="node_xgb",
            recipe_id="xgboost_trainer",
            config={"task_type": "regression", "n_estimators": 100}
        ),
        WorkflowNode(
            id="node_eval",
            recipe_id="model_evaluator",
            config={}
        )
    ]
    edges = [
        WorkflowEdge(source="node_split", target="node_xgb"),
        WorkflowEdge(source="node_xgb", target="node_eval")
    ]

    graph = WorkflowGraph(nodes=nodes, edges=edges)
    diag = graph.get_diagnostics()

    # Should detect target_churn from upstream split node and generate an inline warning
    assert diag["is_valid"] is True  # Non-blocking warning
    warn_messages = [w["message"] for w in diag["warnings"]]
    churn_warn = [m for m in warn_messages if "target_churn" in m and "Regression mode is selected" in m]
    assert len(churn_warn) == 1
    assert "sales_amount" in churn_warn[0]
