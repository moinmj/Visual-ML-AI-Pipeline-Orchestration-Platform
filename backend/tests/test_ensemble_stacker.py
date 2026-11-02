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
from sklearn.linear_model import LogisticRegression

from backend.app.recipes.base.registry import recipe_registry
from backend.app.recipes.training.ensemble_stacker import EnsembleStackerRecipe


def test_ensemble_stacker_registry_and_aliases():
    """Verify EnsembleStackerRecipe is registered with correct metadata and aliases."""
    assert recipe_registry.has("ensemble_stacker") is True
    meta = recipe_registry.get_metadata("ensemble_stacker")
    assert meta.enabled is True
    assert meta.group == "Modeling"
    assert meta.subgroup == "Deep Learning / AutoML"

    # Aliases
    assert recipe_registry.get("stacking").recipe_id == "ensemble_stacker"
    assert recipe_registry.get("stacker").recipe_id == "ensemble_stacker"
    assert recipe_registry.get("stacking_classifier").recipe_id == "ensemble_stacker"
    assert recipe_registry.get("stacking_regressor").recipe_id == "ensemble_stacker"
    assert recipe_registry.get("ensemble_stacking").recipe_id == "ensemble_stacker"


def test_stacking_classifier_binary():
    """Verify StackingClassifier trains base models, meta-learner, and produces lift report."""
    np.random.seed(42)
    X, y = make_classification(n_samples=200, n_features=6, n_informative=3, n_redundant=1, random_state=42)
    X_train, y_train = X[:150], y[:150]
    X_test, y_test = X[150:], y[150:]

    feature_cols = [f"feat_{i}" for i in range(6)]
    recipe = EnsembleStackerRecipe()
    result = recipe.execute(
        inputs={
            "X_train": pd.DataFrame(X_train, columns=feature_cols),
            "y_train": pd.Series(y_train),
            "X_test": pd.DataFrame(X_test, columns=feature_cols),
            "y_test": pd.Series(y_test)
        },
        config={
            "task_type": "classification",
            "base_models_preset": "diverse_blend",
            "final_estimator": "logistic_regression",
            "cv_folds": 3
        }
    )

    model = result["model"]
    metrics = result["metrics"]
    report = result["report"]

    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")
    assert metrics["task_type"] == "classification"
    assert metrics["evaluated_split"] == "test"
    assert metrics["stacked_accuracy"] > 0.6
    assert metrics["stacked_f1_macro"] > 0.6
    assert len(metrics["base_models_metrics"]) == 3
    assert "best_base_model" in metrics
    assert "f1_lift_percentage" in metrics

    # Verify report formatting
    assert "# === Stacking Ensemble Meta-Learner Report" in report
    assert "Model Comparison Tournament:" in report
    assert "Stacked Ensemble" in report


def test_stacking_classifier_multiclass_and_string_targets():
    """Verify multiclass (3-class) stacking with string target labels and target encoder restoration."""
    np.random.seed(42)
    X, y_num = make_classification(n_samples=180, n_features=6, n_informative=4, n_classes=3, random_state=42)
    class_map = {0: "tier_bronze", 1: "tier_silver", 2: "tier_gold"}
    y_str = pd.Series([class_map[val] for val in y_num])

    recipe = EnsembleStackerRecipe()
    result = recipe.execute(
        inputs={
            "X_train": pd.DataFrame(X[:120], columns=[f"c_{i}" for i in range(6)]),
            "y_train": y_str[:120],
            "X_test": pd.DataFrame(X[120:], columns=[f"c_{i}" for i in range(6)]),
            "y_test": y_str[120:]
        },
        config={
            "base_models_preset": "tree_ensemble",
            "cv_folds": 3
        }
    )

    assert result["task_type"] == "classification"
    assert "target_classes" in result
    assert set(result["target_classes"]) == {"tier_bronze", "tier_silver", "tier_gold"}
    assert result["target_encoder"] is not None
    assert result["metrics"]["stacked_accuracy"] > 0.5


def test_stacking_regressor():
    """Verify StackingRegressor combines base regression models with R² and RMSE metrics."""
    np.random.seed(42)
    X, y = make_regression(n_samples=160, n_features=5, noise=0.1, random_state=42)
    feature_cols = [f"reg_{i}" for i in range(5)]

    recipe = EnsembleStackerRecipe()
    result = recipe.execute(
        inputs={
            "X_train": pd.DataFrame(X[:110], columns=feature_cols),
            "y_train": pd.Series(y[:110]),
            "X_test": pd.DataFrame(X[110:], columns=feature_cols),
            "y_test": pd.Series(y[110:])
        },
        config={
            "task_type": "regression",
            "final_estimator": "ridge",
            "cv_folds": 3
        }
    )

    model = result["model"]
    metrics = result["metrics"]
    report = result["report"]

    assert hasattr(model, "predict")
    assert metrics["task_type"] == "regression"
    assert metrics["stacked_r2"] > 0.8
    assert metrics["stacked_rmse"] >= 0.0
    assert metrics["stacked_mae"] >= 0.0
    assert len(metrics["base_models_metrics"]) >= 2
    assert "r2_lift_percentage" in metrics
    assert "R² Score" in report


def test_stacking_custom_base_models_and_passthrough():
    """Verify user can pass explicit base models into inputs and enable passthrough."""
    np.random.seed(42)
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    custom_models = [
        ("custom_rf", RandomForestClassifier(n_estimators=15, max_depth=3, random_state=42)),
        ("custom_lr", LogisticRegression(random_state=42))
    ]

    recipe = EnsembleStackerRecipe()
    result = recipe.execute(
        inputs={
            "X_train": X[:70],
            "y_train": y[:70],
            "X_test": X[70:],
            "y_test": y[70:],
            "base_models": custom_models
        },
        config={
            "passthrough": True,
            "cv_folds": 2
        }
    )

    assert result["passthrough"] is True
    assert result["base_models"] == ["custom_rf", "custom_lr"]
    assert result["metrics"]["passthrough"] is True
    assert len(result["metrics"]["base_models_metrics"]) == 2


def test_ensemble_stacker_edge_cases():
    """Verify edge case guardrails: single-class rejection, missing data rejection, adaptive CV on tiny data."""
    recipe = EnsembleStackerRecipe()

    # Guard 1: Missing training inputs
    with pytest.raises(ValueError, match="EnsembleStacker expects 'X_train' and 'y_train'"):
        recipe.execute(inputs={}, config={})

    # Guard 2: Single-class classification data rejected
    X_single = pd.DataFrame({"f1": [1, 2, 3, 4], "f2": [5, 6, 7, 8]})
    y_single = pd.Series([0, 0, 0, 0])
    with pytest.raises(ValueError, match="contains only 1 class"):
        recipe.execute(
            inputs={"X_train": X_single, "y_train": y_single},
            config={"task_type": "classification"}
        )

    # Guard 3: Small sample size (N=6, 3 per class) triggers adaptive CV fallback
    X_small = pd.DataFrame({
        "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "category": ["A", "B", "A", "B", "A", "B"]
    })
    y_small = pd.Series([0, 1, 0, 1, 0, 1])

    res = recipe.execute(
        inputs={"X_train": X_small, "y_train": y_small},
        config={"cv_folds": 5}
    )

    # Automatically reduced CV from 5 to 3 because min_class_count is 3
    assert res["metrics"]["effective_cv_folds"] == 3
    assert any("ADAPTIVE CV" in w for w in res["metrics"]["warnings"])
    assert res["model"] is not None


if __name__ == "__main__":
    print("Running test_ensemble_stacker_registry_and_aliases...")
    test_ensemble_stacker_registry_and_aliases()
    print("PASS: test_ensemble_stacker_registry_and_aliases")

    print("Running test_stacking_classifier_binary...")
    test_stacking_classifier_binary()
    print("PASS: test_stacking_classifier_binary")

    print("Running test_stacking_classifier_multiclass_and_string_targets...")
    test_stacking_classifier_multiclass_and_string_targets()
    print("PASS: test_stacking_classifier_multiclass_and_string_targets")

    print("Running test_stacking_regressor...")
    test_stacking_regressor()
    print("PASS: test_stacking_regressor")

    print("Running test_stacking_custom_base_models_and_passthrough...")
    test_stacking_custom_base_models_and_passthrough()
    print("PASS: test_stacking_custom_base_models_and_passthrough")

    print("Running test_ensemble_stacker_edge_cases...")
    test_ensemble_stacker_edge_cases()
    print("PASS: test_ensemble_stacker_edge_cases")

    print("\nALL 6 ENSEMBLE STACKER TESTS PASSED SUCCESSFULLY!")
