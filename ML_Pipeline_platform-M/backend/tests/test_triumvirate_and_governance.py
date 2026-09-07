import pytest
import pandas as pd
import numpy as np
from backend.app.recipes.base.registry import recipe_registry
import backend.app.recipes


def test_lightgbm_trainer_with_special_characters():
    np.random.seed(42)
    n = 100
    df_train = pd.DataFrame({
        "Feature [A]": np.random.normal(0, 1, n),
        "Category: B": np.random.choice(["X", "Y", "Z"], n),
        "Target": np.random.choice([0, 1], n)
    })
    X_train = df_train[["Feature [A]", "Category: B"]]
    y_train = df_train["Target"]

    lgb_recipe = recipe_registry.get("lightgbm_trainer")
    assert lgb_recipe is not None

    res = lgb_recipe.execute({"X_train": X_train, "y_train": y_train}, {"task_type": "classification", "n_estimators": 20})
    assert res["model"] is not None
    assert "feature_importances" in res
    assert len(res["feature_importances"]) > 0


def test_catboost_trainer_with_native_strings():
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "Age": np.random.randint(20, 60, n).astype(float),
        "City": np.random.choice(["New York", "London", "Tokyo", "Berlin"], n),
        "Income": np.random.normal(50000, 10000, n),
        "Default": np.random.choice([0, 1], n)
    })
    X_train = df[["Age", "City", "Income"]]
    y_train = df["Default"]

    cb_recipe = recipe_registry.get("catboost_trainer")
    assert cb_recipe is not None

    res = cb_recipe.execute({"X_train": X_train, "y_train": y_train}, {"task_type": "classification", "iterations": 25})
    assert res["model"] is not None
    assert "feature_importances" in res
    assert "City" in res["feature_importances"]


def test_mlflow_governance_tracker(tmp_path):
    mlflow_recipe = recipe_registry.get("mlflow_tracker")
    assert mlflow_recipe is not None

    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=10).fit([[1, 2], [3, 4]], [0, 1])
    metrics = {"accuracy": 0.95, "f1_score": 0.94, "loss": 0.05}

    db_path = f"sqlite:///{str(tmp_path / 'mlflow.db').replace('\\', '/')}"
    res = mlflow_recipe.execute(
        {"model": model, "metrics": metrics},
        {
            "experiment_name": "Test_Experiment",
            "registered_model_name": "Test_Model",
            "stage": "Staging",
            "tracking_uri": db_path
        }
    )

    assert "governance_record" in res
    assert res["governance_record"]["stage"] == "Staging"
    assert res["governance_record"]["metrics_logged"] == 3
