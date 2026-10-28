import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier

from backend.app.recipes.base.registry import recipe_registry
from backend.app.recipes.training.hyperparameter_tuner import HyperparameterTunerRecipe


def test_hyperparameter_tuner_registry_and_aliases():
    """Verify HyperparameterTunerRecipe is registered with correct metadata and aliases."""
    assert recipe_registry.has("hyperparameter_tuner") is True
    meta = recipe_registry.get_metadata("hyperparameter_tuner")
    assert meta.enabled is True
    assert meta.group == "Modeling"
    assert meta.subgroup == "Deep Learning / AutoML"

    # Aliases
    assert recipe_registry.get("optuna").recipe_id == "hyperparameter_tuner"
    assert recipe_registry.get("hyperparameter_tuning").recipe_id == "hyperparameter_tuner"
    assert recipe_registry.get("bayesian_optimization").recipe_id == "hyperparameter_tuner"
    assert recipe_registry.get("grid_search").recipe_id == "hyperparameter_tuner"
    assert recipe_registry.get("random_search").recipe_id == "hyperparameter_tuner"
    assert recipe_registry.get("tuner").recipe_id == "hyperparameter_tuner"


def test_optuna_bayesian_tuning_classification():
    """Verify Optuna Bayesian optimization finds optimal hyperparameters and returns refitted model."""
    np.random.seed(42)
    X, y = make_classification(n_samples=150, n_features=6, n_informative=3, n_redundant=1, random_state=42)
    feature_cols = [f"feat_{i}" for i in range(6)]

    recipe = HyperparameterTunerRecipe()
    result = recipe.execute(
        inputs={
            "X_train": pd.DataFrame(X[:100], columns=feature_cols),
            "y_train": pd.Series(y[:100]),
            "X_test": pd.DataFrame(X[100:], columns=feature_cols),
            "y_test": pd.Series(y[100:])
        },
        config={
            "strategy": "optuna_bayesian",
            "model_type": "random_forest",
            "task_type": "classification",
            "scoring_metric": "f1_macro",
            "n_trials": 6,
            "cv_folds": 3
        }
    )

    model = result["model"]
    metrics = result["metrics"]
    report = result["report"]
    best_params = result["best_params"]

    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")
    assert "n_estimators" in best_params
    assert "max_depth" in best_params
    assert metrics["strategy"] == "optuna_bayesian"
    assert metrics["n_trials_completed"] == 6
    assert metrics["best_cv_score"] > 0.6
    assert len(metrics["top_trials"]) > 0
    assert "test_evaluation" in metrics
    assert "test_accuracy" in metrics["test_evaluation"]

    # Verify report formatting
    assert "# === Hyperparameter Optimization Tournament Report" in report
    assert "Optimal Hyperparameters Discovered:" in report
    assert "Top 5 Discovered Configurations:" in report


def test_grid_search_and_random_search_strategies():
    """Verify Grid Search and Random Search strategies execute cleanly."""
    np.random.seed(42)
    X, y = make_regression(n_samples=120, n_features=4, noise=0.1, random_state=42)

    recipe = HyperparameterTunerRecipe()

    # 1. Random Search on Regression
    res_rand = recipe.execute(
        inputs={
            "X_train": X[:80],
            "y_train": y[:80],
            "X_test": X[80:],
            "y_test": y[80:]
        },
        config={
            "strategy": "random_search",
            "model_type": "ridge",
            "task_type": "regression",
            "n_trials": 4,
            "cv_folds": 2
        }
    )
    assert res_rand["metrics"]["strategy"] == "random_search"
    assert "alpha" in res_rand["best_params"]
    assert res_rand["metrics"]["best_cv_score"] > 0.5

    # 2. Grid Search on Classification
    X_c, y_c = make_classification(n_samples=80, n_features=4, random_state=42)
    res_grid = recipe.execute(
        inputs={"X_train": X_c, "y_train": y_c},
        config={
            "strategy": "grid_search",
            "model_type": "logistic_regression",
            "n_trials": 4,
            "cv_folds": 2
        }
    )
    assert res_grid["metrics"]["strategy"] == "grid_search"
    assert "C" in res_grid["best_params"]
    assert res_grid["model"] is not None


def test_multiclass_and_string_target_tuning():
    """Verify multiclass (3-class) tuning with string targets and label encoder restoration."""
    np.random.seed(42)
    X, y_num = make_classification(n_samples=120, n_features=5, n_informative=3, n_classes=3, random_state=42)
    class_map = {0: "tier_low", 1: "tier_med", 2: "tier_high"}
    y_str = pd.Series([class_map[val] for val in y_num])

    recipe = HyperparameterTunerRecipe()
    result = recipe.execute(
        inputs={"X_train": X, "y_train": y_str},
        config={
            "model_type": "random_forest",
            "n_trials": 4,
            "cv_folds": 3
        }
    )

    assert result["task_type"] == "classification"
    assert "target_classes" in result
    assert set(result["target_classes"]) == {"tier_low", "tier_med", "tier_high"}
    assert result["target_encoder"] is not None
    assert result["model"] is not None


def test_tuning_upstream_model_instance():
    """Verify tuner recognizes and tunes an upstream model instance passed via inputs."""
    np.random.seed(42)
    X, y = make_classification(n_samples=90, n_features=4, random_state=42)
    upstream = RandomForestClassifier(n_estimators=10, random_state=42)

    recipe = HyperparameterTunerRecipe()
    result = recipe.execute(
        inputs={
            "X_train": X,
            "y_train": y,
            "model": upstream
        },
        config={"n_trials": 3, "cv_folds": 2}
    )

    assert result["metrics"]["model_type"] == "random_forest"
    assert result["model"] is not None


def test_hyperparameter_tuner_edge_cases():
    """Verify edge case guardrails: missing inputs, single class, adaptive CV on tiny datasets."""
    recipe = HyperparameterTunerRecipe()

    # Guard 1: Missing training inputs
    with pytest.raises(ValueError, match="HyperparameterTuner expects 'X_train' and 'y_train'"):
        recipe.execute(inputs={}, config={})

    # Guard 2: Single class rejected
    X_single = pd.DataFrame({"f1": [1, 2, 3, 4], "f2": [5, 6, 7, 8]})
    y_single = pd.Series([0, 0, 0, 0])
    with pytest.raises(ValueError, match="contains only 1 class"):
        recipe.execute(
            inputs={"X_train": X_single, "y_train": y_single},
            config={"task_type": "classification"}
        )

    # Guard 3: Small sample size (N=6, 3 per class) triggers adaptive CV fallback from 5 down to 3
    X_small = pd.DataFrame({
        "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "category": ["X", "Y", "X", "Y", "X", "Y"]
    })
    y_small = pd.Series([0, 1, 0, 1, 0, 1])

    res = recipe.execute(
        inputs={"X_train": X_small, "y_train": y_small},
        config={"n_trials": 3, "cv_folds": 5}
    )

    assert res["metrics"]["effective_cv_folds"] == 3
    assert any("ADAPTIVE CV" in w for w in res["metrics"]["warnings"])
    assert res["model"] is not None


if __name__ == "__main__":
    print("Running test_hyperparameter_tuner_registry_and_aliases...")
    test_hyperparameter_tuner_registry_and_aliases()
    print("PASS: test_hyperparameter_tuner_registry_and_aliases")

    print("Running test_optuna_bayesian_tuning_classification...")
    test_optuna_bayesian_tuning_classification()
    print("PASS: test_optuna_bayesian_tuning_classification")

    print("Running test_grid_search_and_random_search_strategies...")
    test_grid_search_and_random_search_strategies()
    print("PASS: test_grid_search_and_random_search_strategies")

    print("Running test_multiclass_and_string_target_tuning...")
    test_multiclass_and_string_target_tuning()
    print("PASS: test_multiclass_and_string_target_tuning")

    print("Running test_tuning_upstream_model_instance...")
    test_tuning_upstream_model_instance()
    print("PASS: test_tuning_upstream_model_instance")

    print("Running test_hyperparameter_tuner_edge_cases...")
    test_hyperparameter_tuner_edge_cases()
    print("PASS: test_hyperparameter_tuner_edge_cases")

    print("\nALL 6 HYPERPARAMETER TUNER TESTS PASSED SUCCESSFULLY!")
