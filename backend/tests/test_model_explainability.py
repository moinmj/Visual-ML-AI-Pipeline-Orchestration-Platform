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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from backend.app.recipes.base.registry import recipe_registry
from backend.app.recipes.evaluation.model_explainability import ModelExplainabilityRecipe


def test_recipe_registry_and_aliases():
    """Verify ModelExplainabilityRecipe is properly registered with correct metadata and aliases."""
    assert recipe_registry.has("model_explainability") is True
    meta = recipe_registry.get_metadata("model_explainability")
    assert meta.enabled is True
    assert meta.group == "Modeling"
    assert meta.subgroup == "Specialized Models"

    # Verify registered aliases resolve
    assert recipe_registry.get("shap").recipe_id == "model_explainability"
    assert recipe_registry.get("lime").recipe_id == "model_explainability"
    assert recipe_registry.get("explainability").recipe_id == "model_explainability"


def test_shap_tree_classifier_explanation():
    """Test SHAP global and local attribution on a trained tree ensemble classifier."""
    np.random.seed(42)
    n = 100
    X = pd.DataFrame({
        "income": np.random.uniform(20000, 100000, n),
        "age": np.random.uniform(18, 70, n),
        "credit_score": np.random.uniform(300, 850, n),
        "debt_ratio": np.random.uniform(0.1, 0.9, n)
    })
    # Target strongly correlated with credit_score and income
    y = ((X["credit_score"] > 600) & (X["income"] > 40000)).astype(int)

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    recipe = ModelExplainabilityRecipe()
    result = recipe.execute(
        inputs={"model": model, "X_test": X, "task_type": "classification"},
        config={"method": "SHAP", "num_features_to_display": 4, "explain_row_index": 0}
    )

    metrics = result["metrics"]
    report = result["report"]

    assert metrics["task_type"] == "classification"
    assert metrics["method"] == "SHAP"
    assert len(metrics["global_feature_importance"]) == 4
    # Global rankings should feature importance > 0
    assert metrics["global_feature_importance"][0]["importance"] > 0
    assert "feature" in metrics["global_feature_importance"][0]

    # Local sample attribution
    assert len(metrics["local_attribution"]) == 4
    assert metrics["local_attribution"][0]["direction"] in ["positive", "negative"]

    assert "=== Model Explainability Report" in report
    assert "Global Feature Importance" in report
    assert "Local Sample #0 Attribution Breakdown" in report


def test_lime_mode_tabular_explanation():
    """Test LIME local surrogate explanations on tabular test data."""
    np.random.seed(42)
    n = 80
    X = pd.DataFrame({
        "sensor_a": np.random.randn(n),
        "sensor_b": np.random.randn(n),
        "sensor_c": np.random.randn(n)
    })
    y = (X["sensor_a"] + X["sensor_b"] > 0).astype(int)

    model = LogisticRegression()
    model.fit(X, y)

    recipe = ModelExplainabilityRecipe()
    result = recipe.execute(
        inputs={"model": model, "X_test": X},
        config={"method": "LIME", "num_features_to_display": 3, "explain_row_index": 2}
    )

    metrics = result["metrics"]
    report = result["report"]

    assert metrics["method"] == "LIME"
    assert "lime_local_explanation" in metrics
    assert len(metrics["lime_local_explanation"]) > 0
    assert "rule" in metrics["lime_local_explanation"][0]
    assert "weight" in metrics["lime_local_explanation"][0]
    assert "LIME Local Surrogate Rules" in report


def test_combined_shap_and_lime_mode():
    """Test combined mode running both SHAP and LIME together."""
    np.random.seed(42)
    n = 60
    X = pd.DataFrame({
        "feature_1": np.random.randn(n),
        "feature_2": np.random.randn(n)
    })
    y = (X["feature_1"] > 0).astype(int)

    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X, y)

    recipe = ModelExplainabilityRecipe()
    result = recipe.execute(
        inputs={"model": model, "X_test": X},
        config={"method": "Combined (SHAP + LIME)", "num_features_to_display": 2}
    )

    metrics = result["metrics"]
    assert "global_feature_importance" in metrics
    assert "lime_local_explanation" in metrics
    assert len(metrics["global_feature_importance"]) == 2
    assert len(metrics["lime_local_explanation"]) > 0


def test_regression_model_explanation():
    """Test explainability on continuous regression targets."""
    np.random.seed(42)
    n = 70
    X = pd.DataFrame({
        "sqft": np.random.uniform(500, 3000, n),
        "bedrooms": np.random.randint(1, 5, n),
        "lot_size": np.random.uniform(1000, 10000, n)
    })
    y = X["sqft"] * 200 + X["bedrooms"] * 10000 + np.random.randn(n) * 1000

    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)

    recipe = ModelExplainabilityRecipe()
    result = recipe.execute(
        inputs={"model": model, "X_test": X, "task_type": "regression"},
        config={"method": "SHAP", "num_features_to_display": 3}
    )

    metrics = result["metrics"]
    assert metrics["task_type"] == "regression"
    assert len(metrics["global_feature_importance"]) == 3
    # sqft should be the dominant feature
    top_feature = metrics["global_feature_importance"][0]["feature"]
    assert top_feature == "sqft"


def test_edge_cases():
    """Test critical edge cases: out-of-bounds row index, non-numeric column conversion, missing model/data."""
    recipe = ModelExplainabilityRecipe()

    # 1. Missing model error
    with pytest.raises(ValueError, match="expects a trained 'model'"):
        recipe.execute(inputs={"X_test": pd.DataFrame({"a": [1, 2]})}, config={})

    # 2. Missing data error
    class DummyModel:
        pass
    with pytest.raises(ValueError, match="expects feature data"):
        recipe.execute(inputs={"model": DummyModel()}, config={})

    # 3. Out-of-bounds row index clamped gracefully
    X = pd.DataFrame({"x1": [1.0, 2.0, 3.0], "x2": [10.0, 20.0, 30.0]})
    y = np.array([0, 1, 0])
    model = RandomForestClassifier(n_estimators=3, random_state=42).fit(X, y)

    # Requested index 999 on 3 rows -> clamped to 2
    res = recipe.execute(
        inputs={"model": model, "X_test": X},
        config={"explain_row_index": 999}
    )
    assert res["metrics"]["explained_row_index"] == 2

    # 4. Non-numeric columns in X_test converted safely
    X_str = pd.DataFrame({
        "num": [1, 2, 3, 4],
        "category": ["A", "B", "A", "C"]
    })
    y_str = np.array([0, 1, 0, 1])
    model2 = RandomForestClassifier(n_estimators=3, random_state=42).fit(pd.get_dummies(X_str), y_str)
    # Passing raw X_str should be safely coerced without throwing
    res2 = recipe.execute(
        inputs={"model": model2, "X_test": X_str},
        config={}
    )
    assert "global_feature_importance" in res2["metrics"]


if __name__ == "__main__":
    print("Running test_recipe_registry_and_aliases...")
    test_recipe_registry_and_aliases()
    print("PASS: test_recipe_registry_and_aliases")

    print("Running test_shap_tree_classifier_explanation...")
    test_shap_tree_classifier_explanation()
    print("PASS: test_shap_tree_classifier_explanation")

    print("Running test_lime_mode_tabular_explanation...")
    test_lime_mode_tabular_explanation()
    print("PASS: test_lime_mode_tabular_explanation")

    print("Running test_combined_shap_and_lime_mode...")
    test_combined_shap_and_lime_mode()
    print("PASS: test_combined_shap_and_lime_mode")

    print("Running test_regression_model_explanation...")
    test_regression_model_explanation()
    print("PASS: test_regression_model_explanation")

    print("Running test_edge_cases...")
    test_edge_cases()
    print("PASS: test_edge_cases")

    print("\nALL 6 MODEL EXPLAINABILITY TESTS PASSED SUCCESSFULLY!")
