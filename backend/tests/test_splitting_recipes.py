import pytest
import pandas as pd
import numpy as np
from backend.app.recipes.base.registry import recipe_registry
from backend.app.recipes.splitting.stratified_split import StratifiedSplitRecipe
from backend.app.recipes.splitting.time_series_split import TimeSeriesSplitRecipe
from backend.app.recipes.splitting.walk_forward_split import WalkForwardSplitRecipe
from backend.app.recipes.training.xgboost_trainer import XGBoostTrainerRecipe


def test_splitting_recipes_registered_in_catalog():
    for r_id in ["stratified_split", "time_series_split", "walk_forward_split"]:
        r = recipe_registry.get(r_id)
        assert r is not None, f"Recipe {r_id} not found in recipe_registry!"
        schema = r.get_schema()
        assert schema["type"] == "object"
        assert "properties" in schema
        code = r.to_code({})
        assert isinstance(code, str) and len(code) > 0


def test_stratified_split_preserves_class_distribution():
    # Construct an imbalanced dataset: 80% class 0, 20% class 1
    n_samples = 200
    n_ones = 40
    n_zeros = 160
    y = np.array([1] * n_ones + [0] * n_zeros)
    np.random.seed(42)
    np.random.shuffle(y)

    df = pd.DataFrame({
        "feature_1": np.random.randn(n_samples),
        "feature_2": np.random.randn(n_samples),
        "target": y
    })

    recipe = StratifiedSplitRecipe()
    res = recipe.execute({"dataframe": df}, {"target_column": "target", "test_size": 0.25, "random_state": 42})

    train_data = res["train_data"]
    test_data = res["test_data"]

    assert "X_train" in train_data and "y_train" in train_data
    assert "X_test" in test_data and "y_test" in test_data

    y_train = train_data["y_train"]
    y_test = test_data["y_test"]

    # Target ratio in train and test should both closely match 0.20
    train_ratio = (y_train == 1).mean()
    test_ratio = (y_test == 1).mean()

    assert abs(train_ratio - 0.20) < 0.03
    assert abs(test_ratio - 0.20) < 0.03
    assert len(train_data["X_train"]) == 150
    assert len(test_data["X_test"]) == 50


def test_stratified_split_single_member_safeguard():
    # Rare class with only 1 sample
    df = pd.DataFrame({
        "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "feature_2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        "target": ["A", "A", "A", "A", "A", "B"]  # B has only 1 sample
    })
    recipe = StratifiedSplitRecipe()
    # Should not raise ValueError even with single-member class B
    res = recipe.execute({"dataframe": df}, {"target_column": "target", "test_size": 0.3})
    assert res["train_data"]["row_count"] > 0
    assert res["test_data"]["row_count"] > 0


def test_time_series_split_chronological_order_and_gap():
    # 100 days of data in shuffled order to test auto-sorting
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    df = pd.DataFrame({
        "timestamp": dates,
        "value": np.arange(100, dtype=float),
        "target": np.arange(100, dtype=float) * 2
    })
    # Shuffle rows to ensure the recipe correctly sorts by timestamp
    df_shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    recipe = TimeSeriesSplitRecipe()

    # 1. Without gap (test_size = 0.20 -> 80 train, 20 test)
    res = recipe.execute({"dataframe": df_shuffled}, {
        "target_column": "target",
        "date_column": "timestamp",
        "test_size": 0.20,
        "gap": 0
    })

    train_data = res["train_data"]
    test_data = res["test_data"]

    assert len(train_data["X_train"]) == 80
    assert len(test_data["X_test"]) == 20

    # Ensure max train timestamp is strictly before min test timestamp (no future leakage!)
    assert train_data["X_train"]["timestamp"].max() < test_data["X_test"]["timestamp"].min()

    # 2. With purge/embargo gap of 5 observations
    res_gap = recipe.execute({"dataframe": df_shuffled}, {
        "target_column": "target",
        "date_column": "timestamp",
        "test_size": 0.20,
        "gap": 5
    })

    assert len(res_gap["train_data"]["X_train"]) == 75
    assert len(res_gap["test_data"]["X_test"]) == 20
    # Difference between min test value and max train value should be 6 steps (gap of 5)
    train_max_val = res_gap["train_data"]["X_train"]["value"].max()
    test_min_val = res_gap["test_data"]["X_test"]["value"].min()
    assert test_min_val - train_max_val == 6.0


def test_walk_forward_split_sliding_window():
    dates = pd.date_range("2024-01-01", periods=150, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "feature_1": np.arange(150),
        "target": np.arange(150) * 1.5
    })

    recipe = WalkForwardSplitRecipe()
    res = recipe.execute({"dataframe": df}, {
        "target_column": "target",
        "date_column": "date",
        "window_size": 60,
        "test_step_size": 15
    })

    train_data = res["train_data"]
    test_data = res["test_data"]

    assert train_data["row_count"] == 60
    assert test_data["row_count"] == 15

    # Train window immediately precedes test window
    assert train_data["X_train"]["date"].max() < test_data["X_test"]["date"].min()
    assert res["metadata"]["train_window_size"] == 60
    assert res["metadata"]["test_step_size"] == 15
    assert res["metadata"]["total_possible_folds"] == 6  # (150 - 60) // 15 = 6 folds


def test_split_to_model_trainer_handoff():
    # End-to-end integration: StratifiedSplitRecipe -> XGBoostTrainerRecipe
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "num_1": np.random.randn(n),
        "num_2": np.random.randn(n),
        "target": np.random.choice([0, 1], size=n)
    })

    splitter = StratifiedSplitRecipe()
    split_res = splitter.execute({"dataframe": df}, {"target_column": "target", "test_size": 0.2})

    trainer = XGBoostTrainerRecipe()
    # Trainer receives split outputs
    trainer_res = trainer.execute(
        inputs={"train_data": split_res["train_data"]},
        config={"task_type": "classification", "n_estimators": 10, "max_depth": 3}
    )

    assert "model" in trainer_res
    assert trainer_res["model"] is not None
    assert "feature_importances" in trainer_res
    assert trainer_res["task_type"] == "classification"

    # Evaluator receives model from trainer and test_data from splitter
    from backend.app.recipes.evaluation.model_evaluator import ModelEvaluatorRecipe
    evaluator = ModelEvaluatorRecipe()
    eval_res = evaluator.execute(
        inputs={
            "model": trainer_res["model"],
            "test_data": split_res["test_data"],
            "task_type": "classification"
        },
        config={}
    )
    assert "metrics" in eval_res
    assert "accuracy" in eval_res["metrics"]
