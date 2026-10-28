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
from backend.app.recipes.training.neural_network_trainer import NeuralNetworkTrainerRecipe


def test_neural_network_trainer_registry_and_aliases():
    """Verify NeuralNetworkTrainerRecipe is registered with correct metadata and aliases."""
    assert recipe_registry.has("neural_network_trainer") is True
    meta = recipe_registry.get_metadata("neural_network_trainer")
    assert meta.enabled is True
    assert meta.group == "Modeling"
    assert meta.subgroup == "Deep Learning / AutoML"

    # Aliases
    assert recipe_registry.get("mlp").recipe_id == "neural_network_trainer"
    assert recipe_registry.get("mlp_classifier").recipe_id == "neural_network_trainer"
    assert recipe_registry.get("mlp_regressor").recipe_id == "neural_network_trainer"
    assert recipe_registry.get("neural_network").recipe_id == "neural_network_trainer"
    assert recipe_registry.get("feedforward_nn").recipe_id == "neural_network_trainer"
    assert recipe_registry.get("deep_learning_tabular").recipe_id == "neural_network_trainer"


def test_neural_network_classifier_binary():
    """Verify NeuralNetworkTrainer trains MLPClassifier with StandardScaler pipeline and loss curve."""
    np.random.seed(42)
    X, y = make_classification(n_samples=200, n_features=6, n_informative=3, n_redundant=1, random_state=42)
    feature_cols = [f"f_{i}" for i in range(6)]

    recipe = NeuralNetworkTrainerRecipe()
    result = recipe.execute(
        inputs={
            "X_train": pd.DataFrame(X[:140], columns=feature_cols),
            "y_train": pd.Series(y[:140]),
            "X_test": pd.DataFrame(X[140:], columns=feature_cols),
            "y_test": pd.Series(y[140:])
        },
        config={
            "task_type": "classification",
            "architecture_preset": "two_layer_medium",
            "activation": "relu",
            "optimizer": "adam",
            "max_iter": 100
        }
    )

    model = result["model"]
    metrics = result["metrics"]
    report = result["report"]

    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")
    assert metrics["task_type"] == "classification"
    assert metrics["architecture"] == [64, 32]
    assert metrics["epochs_trained"] > 0
    assert metrics["best_loss"] > 0.0
    assert "train_accuracy" in metrics["train_metrics"]
    assert "test_accuracy" in metrics["test_metrics"]
    assert len(metrics["loss_curve_sample"]) > 0

    # Verify report formatting
    assert "# === Deep Neural Network (MLP) Training Report" in report
    assert "Input(6) -> 64 -> 32 -> Output" in report
    assert "Training Loss Progression:" in report


def test_neural_network_regressor():
    """Verify NeuralNetworkTrainer trains MLPRegressor with R² and RMSE metrics."""
    np.random.seed(42)
    X, y = make_regression(n_samples=160, n_features=5, noise=0.1, random_state=42)
    feature_cols = [f"num_{i}" for i in range(5)]

    recipe = NeuralNetworkTrainerRecipe()
    result = recipe.execute(
        inputs={
            "X_train": pd.DataFrame(X[:110], columns=feature_cols),
            "y_train": pd.Series(y[:110]),
            "X_test": pd.DataFrame(X[110:], columns=feature_cols),
            "y_test": pd.Series(y[110:])
        },
        config={
            "task_type": "regression",
            "architecture_preset": "custom",
            "hidden_layers": "128, 64",
            "max_iter": 250
        }
    )

    model = result["model"]
    metrics = result["metrics"]
    report = result["report"]

    assert hasattr(model, "predict")
    assert metrics["task_type"] == "regression"
    assert metrics["architecture"] == [128, 64]
    assert metrics["train_metrics"]["train_r2"] > 0.5
    assert "test_r2" in metrics["test_metrics"]
    assert "Train R² Score" in report


def test_neural_network_multiclass_and_string_targets():
    """Verify multiclass (3-class) MLP training with string targets and label encoder restoration."""
    np.random.seed(42)
    X, y_num = make_classification(n_samples=150, n_features=6, n_informative=4, n_classes=3, random_state=42)
    class_map = {0: "tier_low", 1: "tier_mid", 2: "tier_high"}
    y_str = pd.Series([class_map[val] for val in y_num])

    recipe = NeuralNetworkTrainerRecipe()
    result = recipe.execute(
        inputs={"X_train": X, "y_train": y_str},
        config={
            "architecture_preset": "single_layer",
            "max_iter": 80
        }
    )

    assert result["task_type"] == "classification"
    assert "target_classes" in result
    assert set(result["target_classes"]) == {"tier_low", "tier_mid", "tier_high"}
    assert result["target_encoder"] is not None
    assert result["model"] is not None


def test_neural_network_activation_and_solver_presets():
    """Verify tanh activation and different network architectures."""
    np.random.seed(42)
    X, y = make_classification(n_samples=90, n_features=4, random_state=42)

    recipe = NeuralNetworkTrainerRecipe()
    res = recipe.execute(
        inputs={"X_train": X, "y_train": y},
        config={
            "architecture_preset": "wide_shallow",
            "activation": "tanh",
            "optimizer": "adam",
            "max_iter": 60
        }
    )

    assert res["metrics"]["activation"] == "tanh"
    assert res["metrics"]["architecture"] == [128]
    assert res["model"] is not None


def test_neural_network_trainer_edge_cases():
    """Verify edge case guardrails: missing data, single class, adaptive early stopping on tiny data."""
    recipe = NeuralNetworkTrainerRecipe()

    # Guard 1: Missing training inputs
    with pytest.raises(ValueError, match="NeuralNetworkTrainer expects 'X_train' and 'y_train'"):
        recipe.execute(inputs={}, config={})

    # Guard 2: Single class rejected
    X_single = pd.DataFrame({"f1": [1, 2, 3, 4], "f2": [5, 6, 7, 8]})
    y_single = pd.Series([0, 0, 0, 0])
    with pytest.raises(ValueError, match="contains only 1 class"):
        recipe.execute(
            inputs={"X_train": X_single, "y_train": y_single},
            config={"task_type": "classification"}
        )

    # Guard 3: Small sample size (N=8, 4 per class) triggers adaptive early stopping guard
    X_small = pd.DataFrame({
        "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "category": ["A", "B", "A", "B", "A", "B", "A", "B"]
    })
    y_small = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])

    res = recipe.execute(
        inputs={"X_train": X_small, "y_train": y_small},
        config={"early_stopping": True, "max_iter": 30}
    )

    assert any("ADAPTIVE REGULARIZATION" in w for w in res["metrics"]["warnings"])
    assert res["model"] is not None


if __name__ == "__main__":
    print("Running test_neural_network_trainer_registry_and_aliases...")
    test_neural_network_trainer_registry_and_aliases()
    print("PASS: test_neural_network_trainer_registry_and_aliases")

    print("Running test_neural_network_classifier_binary...")
    test_neural_network_classifier_binary()
    print("PASS: test_neural_network_classifier_binary")

    print("Running test_neural_network_regressor...")
    test_neural_network_regressor()
    print("PASS: test_neural_network_regressor")

    print("Running test_neural_network_multiclass_and_string_targets...")
    test_neural_network_multiclass_and_string_targets()
    print("PASS: test_neural_network_multiclass_and_string_targets")

    print("Running test_neural_network_activation_and_solver_presets...")
    test_neural_network_activation_and_solver_presets()
    print("PASS: test_neural_network_activation_and_solver_presets")

    print("Running test_neural_network_trainer_edge_cases...")
    test_neural_network_trainer_edge_cases()
    print("PASS: test_neural_network_trainer_edge_cases")

    print("\nALL 6 NEURAL NETWORK TRAINER TESTS PASSED SUCCESSFULLY!")
