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

from backend.app.recipes.base.registry import recipe_registry
from backend.app.recipes.training.automl_search import AutoMLSearchRecipe


def test_automl_search_registry_and_aliases():
    """Verify AutoMLSearchRecipe is registered with correct metadata and aliases."""
    assert recipe_registry.has("automl_search") is True
    meta = recipe_registry.get_metadata("automl_search")
    assert meta.enabled is True
    assert meta.group == "Modeling"
    assert meta.subgroup == "Deep Learning / AutoML"

    # Aliases
    assert recipe_registry.get("automl").recipe_id == "automl_search"
    assert recipe_registry.get("model_tournament").recipe_id == "automl_search"
    assert recipe_registry.get("algorithm_selection").recipe_id == "automl_search"
    assert recipe_registry.get("auto_ml").recipe_id == "automl_search"
    assert recipe_registry.get("automl_classifier").recipe_id == "automl_search"
    assert recipe_registry.get("automl_regressor").recipe_id == "automl_search"


def test_automl_classification_tournament():
    """Verify AutoML tournament evaluates candidate models, crowns a champion, and returns refitted model."""
    np.random.seed(42)
    X, y = make_classification(n_samples=160, n_features=6, n_informative=3, n_redundant=1, random_state=42)
    feature_cols = [f"feat_{i}" for i in range(6)]

    recipe = AutoMLSearchRecipe()
    result = recipe.execute(
        inputs={
            "X_train": pd.DataFrame(X[:110], columns=feature_cols),
            "y_train": pd.Series(y[:110]),
            "X_test": pd.DataFrame(X[110:], columns=feature_cols),
            "y_test": pd.Series(y[110:])
        },
        config={
            "task_type": "classification",
            "candidate_pool": "all_candidates",
            "primary_metric": "f1_macro",
            "cv_folds": 3
        }
    )

    model = result["model"]
    metrics = result["metrics"]
    report = result["report"]

    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")
    assert metrics["task_type"] == "classification"
    assert metrics["primary_metric"] == "f1_macro"
    assert "champion_model" in metrics
    assert metrics["champion_score"] > 0.6
    assert len(metrics["leaderboard"]) >= 4
    assert metrics["leaderboard"][0]["rank"] == 1
    assert "test_evaluation" in metrics
    assert "test_f1_macro" in metrics["test_evaluation"]

    # Verify report formatting
    assert "# === AutoML Multi-Model Tournament Report" in report
    assert "Official Tournament Leaderboard:" in report
    assert "CHAMPION" in report


def test_automl_regression_tournament():
    """Verify AutoML regression tournament ranks models via R² and selects optimal regressor."""
    np.random.seed(42)
    X, y = make_regression(n_samples=140, n_features=5, noise=0.1, random_state=42)
    feature_cols = [f"reg_{i}" for i in range(5)]

    recipe = AutoMLSearchRecipe()
    result = recipe.execute(
        inputs={
            "X_train": pd.DataFrame(X[:90], columns=feature_cols),
            "y_train": pd.Series(y[:90]),
            "X_test": pd.DataFrame(X[90:], columns=feature_cols),
            "y_test": pd.Series(y[90:])
        },
        config={
            "task_type": "regression",
            "candidate_pool": "all_candidates",
            "primary_metric": "r2",
            "cv_folds": 3
        }
    )

    model = result["model"]
    metrics = result["metrics"]
    report = result["report"]

    assert hasattr(model, "predict")
    assert metrics["task_type"] == "regression"
    assert metrics["champion_score"] > 0.8
    assert len(metrics["leaderboard"]) >= 4
    assert "test_r2" in metrics["test_evaluation"]
    assert "Tournament Leaderboard:" in report


def test_automl_multiclass_and_string_targets():
    """Verify multiclass (3-class) tournament with string targets and label encoder restoration."""
    np.random.seed(42)
    X, y_num = make_classification(n_samples=120, n_features=6, n_informative=4, n_classes=3, random_state=42)
    class_map = {0: "tier_bronze", 1: "tier_silver", 2: "tier_gold"}
    y_str = pd.Series([class_map[val] for val in y_num])

    recipe = AutoMLSearchRecipe()
    result = recipe.execute(
        inputs={"X_train": X, "y_train": y_str},
        config={"cv_folds": 3}
    )

    assert result["task_type"] == "classification"
    assert "target_classes" in result
    assert set(result["target_classes"]) == {"tier_bronze", "tier_silver", "tier_gold"}
    assert result["target_encoder"] is not None
    assert result["champion_model"] is not None


def test_automl_candidate_pool_selection():
    """Verify specific candidate pools (e.g. linear_models, fast_candidates) limit the tournament."""
    np.random.seed(42)
    X, y = make_classification(n_samples=90, n_features=4, random_state=42)

    recipe = AutoMLSearchRecipe()
    res_linear = recipe.execute(
        inputs={"X_train": X, "y_train": y},
        config={"candidate_pool": "linear_models", "cv_folds": 2}
    )
    assert len(res_linear["metrics"]["leaderboard"]) == 2
    assert all("Logistic" in m["display_name"] or "Ridge" in m["display_name"] for m in res_linear["metrics"]["leaderboard"])


def test_automl_search_edge_cases():
    """Verify edge case guardrails: missing inputs, single class, adaptive CV on tiny datasets."""
    recipe = AutoMLSearchRecipe()

    # Guard 1: Missing training inputs
    with pytest.raises(ValueError, match="AutoMLSearch expects 'X_train' and 'y_train'"):
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
        "category": ["A", "B", "A", "B", "A", "B"]
    })
    y_small = pd.Series([0, 1, 0, 1, 0, 1])

    res = recipe.execute(
        inputs={"X_train": X_small, "y_train": y_small},
        config={"cv_folds": 5}
    )

    assert res["metrics"]["effective_cv_folds"] == 3
    assert any("ADAPTIVE CV" in w for w in res["metrics"]["warnings"])
    assert res["model"] is not None


if __name__ == "__main__":
    print("Running test_automl_search_registry_and_aliases...")
    test_automl_search_registry_and_aliases()
    print("PASS: test_automl_search_registry_and_aliases")

    print("Running test_automl_classification_tournament...")
    test_automl_classification_tournament()
    print("PASS: test_automl_classification_tournament")

    print("Running test_automl_regression_tournament...")
    test_automl_regression_tournament()
    print("PASS: test_automl_regression_tournament")

    print("Running test_automl_multiclass_and_string_targets...")
    test_automl_multiclass_and_string_targets()
    print("PASS: test_automl_multiclass_and_string_targets")

    print("Running test_automl_candidate_pool_selection...")
    test_automl_candidate_pool_selection()
    print("PASS: test_automl_candidate_pool_selection")

    print("Running test_automl_search_edge_cases...")
    test_automl_search_edge_cases()
    print("PASS: test_automl_search_edge_cases")

    print("\nALL 6 AUTOML SEARCH TESTS PASSED SUCCESSFULLY!")
