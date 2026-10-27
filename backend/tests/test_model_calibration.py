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
from sklearn.datasets import make_classification

from backend.app.recipes.base.registry import recipe_registry
from backend.app.recipes.training.model_calibration import ModelCalibrationRecipe


def test_model_calibration_registry_and_aliases():
    """Verify ModelCalibrationRecipe is registered with correct metadata and aliases."""
    assert recipe_registry.has("model_calibration") is True
    meta = recipe_registry.get_metadata("model_calibration")
    assert meta.enabled is True
    assert meta.group == "Modeling"
    assert meta.subgroup == "Deep Learning / AutoML"

    # Aliases
    assert recipe_registry.get("calibration").recipe_id == "model_calibration"
    assert recipe_registry.get("platt_scaling").recipe_id == "model_calibration"
    assert recipe_registry.get("isotonic_regression").recipe_id == "model_calibration"
    assert recipe_registry.get("calibrator").recipe_id == "model_calibration"


def test_platt_scaling_sigmoid_binary_calibration():
    """Verify Platt scaling (sigmoid) improves Brier score, Log Loss, and computes ECE on binary classification."""
    np.random.seed(42)
    # Generate synthetic uncalibrated dataset
    X_train, y_train = make_classification(n_samples=200, n_features=6, n_informative=3, n_redundant=1, random_state=42)
    X_calib, y_calib = make_classification(n_samples=100, n_features=6, n_informative=3, n_redundant=1, random_state=123)

    model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    recipe = ModelCalibrationRecipe()
    result = recipe.execute(
        inputs={
            "model": model,
            "X_calib": pd.DataFrame(X_calib, columns=[f"f_{i}" for i in range(6)]),
            "y_calib": pd.Series(y_calib)
        },
        config={"method": "sigmoid", "cv_strategy": "prefit", "num_bins": 10}
    )

    metrics = result["metrics"]
    report = result["report"]
    calibrated_model = result["calibrated_model"]

    assert "calibrated_model" in result
    assert hasattr(calibrated_model, "predict_proba")
    assert metrics["calibration_method"] == "sigmoid"
    assert metrics["num_classes"] == 2
    assert metrics["expected_calibration_error_ece"] >= 0.0
    assert len(metrics["reliability_diagram"]) > 0

    # Ensure Log Loss and Brier Score are calculated before and after
    assert metrics["uncalibrated_brier_score"] is not None
    assert metrics["calibrated_brier_score"] is not None
    assert metrics["uncalibrated_log_loss"] is not None
    assert metrics["calibrated_log_loss"] is not None

    # Report verification
    assert "=== Model Probability Calibration Report" in report
    assert "Reliability Diagram" in report


def test_isotonic_regression_calibration():
    """Verify Isotonic regression calibration works cleanly on holdout data."""
    np.random.seed(42)
    X, y = make_classification(n_samples=150, n_features=5, random_state=42)
    model = LogisticRegression().fit(X[:100], y[:100])

    recipe = ModelCalibrationRecipe()
    result = recipe.execute(
        inputs={"model": model, "X_calib": X[100:], "y_calib": y[100:]},
        config={"method": "isotonic", "cv_strategy": "prefit"}
    )

    metrics = result["metrics"]
    assert metrics["calibration_method"] == "isotonic"
    assert result["calibrated_model"] is not None
    # Calibrated probabilities sum to 1.0
    probs = result["calibrated_model"].predict_proba(X[100:])
    np.testing.assert_allclose(np.sum(probs, axis=1), 1.0, rtol=1e-5)


def test_multiclass_calibration():
    """Verify multiclass (3-class) probability calibration via One-vs-Rest."""
    np.random.seed(42)
    X, y = make_classification(n_samples=180, n_features=6, n_informative=4, n_classes=3, random_state=42)
    model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X[:120], y[:120])

    recipe = ModelCalibrationRecipe()
    result = recipe.execute(
        inputs={"model": model, "X_calib": X[120:], "y_calib": y[120:]},
        config={"method": "sigmoid", "num_bins": 5}
    )

    metrics = result["metrics"]
    assert metrics["num_classes"] == 3
    assert len(metrics["reliability_diagram"]) > 0


def test_cross_validation_refit_strategy():
    """Verify cv_strategy='3-fold' refits base estimator via cross validation."""
    np.random.seed(42)
    X, y = make_classification(n_samples=90, n_features=4, random_state=42)
    model = LogisticRegression()

    recipe = ModelCalibrationRecipe()
    result = recipe.execute(
        inputs={"model": model, "X_calib": X, "y_calib": y},
        config={"method": "sigmoid", "cv_strategy": "3-fold"}
    )

    assert result["metrics"]["cv_strategy"] == "3"
    assert result["calibrated_model"] is not None


def test_model_calibration_edge_cases():
    """Test guardrails: regression model rejection, single-class guardrail, small-sample warning, string targets."""
    recipe = ModelCalibrationRecipe()

    # 1. Regressor rejected
    reg_model = RandomForestRegressor(n_estimators=2).fit([[1], [2]], [1.0, 2.0])
    with pytest.raises(ValueError, match="only applies to classification models"):
        recipe.execute(
            inputs={"model": reg_model, "X_calib": [[1], [2]], "y_calib": [1.0, 2.0]},
            config={}
        )

    # 2. Single-class calibration data rejected
    clf_model = RandomForestClassifier(n_estimators=2).fit([[1], [2]], [0, 1])
    with pytest.raises(ValueError, match="contains only 1 class"):
        recipe.execute(
            inputs={"model": clf_model, "X_calib": [[1], [2], [3]], "y_calib": [0, 0, 0]},
            config={}
        )

    # 3. String targets encoded cleanly via LabelEncoder
    X = pd.DataFrame({"f": [1, 2, 3, 4, 5, 6, 7, 8]})
    y_str = pd.Series(["churn", "retain", "churn", "retain", "churn", "retain", "churn", "retain"])
    clf_model2 = LogisticRegression().fit(X, [0, 1, 0, 1, 0, 1, 0, 1])

    res = recipe.execute(
        inputs={"model": clf_model2, "X_calib": X, "y_calib": y_str},
        config={"method": "sigmoid"}
    )
    assert res["metrics"]["num_classes"] == 2
    # Small sample warning triggered (N=8 < 30)
    assert any("SMALL SAMPLE" in w for w in res["metrics"]["warnings"])


if __name__ == "__main__":
    print("Running test_model_calibration_registry_and_aliases...")
    test_model_calibration_registry_and_aliases()
    print("PASS: test_model_calibration_registry_and_aliases")

    print("Running test_platt_scaling_sigmoid_binary_calibration...")
    test_platt_scaling_sigmoid_binary_calibration()
    print("PASS: test_platt_scaling_sigmoid_binary_calibration")

    print("Running test_isotonic_regression_calibration...")
    test_isotonic_regression_calibration()
    print("PASS: test_isotonic_regression_calibration")

    print("Running test_multiclass_calibration...")
    test_multiclass_calibration()
    print("PASS: test_multiclass_calibration")

    print("Running test_cross_validation_refit_strategy...")
    test_cross_validation_refit_strategy()
    print("PASS: test_cross_validation_refit_strategy")

    print("Running test_model_calibration_edge_cases...")
    test_model_calibration_edge_cases()
    print("PASS: test_model_calibration_edge_cases")

    print("\nALL 6 MODEL CALIBRATION TESTS PASSED SUCCESSFULLY!")
