import pytest
import pandas as pd
import numpy as np
from backend.app.recipes.base.registry import recipe_registry
import backend.app.recipes


def test_recipe_registry_loading():
    categories = recipe_registry.get_categories()
    assert "preprocessing" in categories
    assert "training" in categories
    assert "evaluation" in categories

    recipes = recipe_registry.list_all()
    assert len(recipes) >= 7


def test_preprocessing_and_training_pipeline():
    # 1. Create dataset
    df = pd.DataFrame({
        "feature1": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "feature2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        "category": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })

    # 2. Impute
    imputer = recipe_registry.get("missing_value_imputer")
    out_impute = imputer.execute({"dataframe": df}, {"strategy": "mean"})
    assert out_impute["dataframe"]["feature1"].isna().sum() == 0

    # 3. Encode
    encoder = recipe_registry.get("categorical_encoder")
    out_encode = encoder.execute({"dataframe": out_impute["dataframe"]}, {"method": "one_hot"})
    assert "category_B" in out_encode["dataframe"].columns

    # 4. Split
    splitter = recipe_registry.get("train_test_split")
    out_split = splitter.execute({"dataframe": out_encode["dataframe"]}, {"target_column": "target", "test_size": 0.3})
    assert len(out_split["X_train"]) == 7
    assert len(out_split["X_test"]) == 3

    # 5. Train Random Forest
    rf = recipe_registry.get("random_forest_trainer")
    out_rf = rf.execute(out_split, {"task_type": "classification", "n_estimators": 10})
    assert "model" in out_rf

    # 6. Evaluate
    evaluator = recipe_registry.get("model_evaluator")
    eval_inputs = {
        "model": out_rf["model"],
        "X_test": out_split["X_test"],
        "y_test": out_split["y_test"]
    }
    out_eval = evaluator.execute(eval_inputs, {"metric_type": "classification"})
    assert "accuracy" in out_eval["metrics"]
    assert "confusion_matrix" in out_eval["metrics"]


def test_preprocessing_column_type_resilience():
    df = pd.DataFrame({
        "Age": [25, np.nan, 35, 40],
        "Salary": [50000, 60000, np.nan, 80000],
        "City": ["NYC", "London", np.nan, "Paris"],
        "Churn": [0, 1, 0, 1]
    })
    imputer = recipe_registry.get("missing_value_imputer")
    
    # 1. String single column name
    out1 = imputer.execute({"dataframe": df}, {"strategy": "median", "columns": "Age"})
    assert out1["dataframe"]["Age"].isna().sum() == 0
    assert out1["dataframe"]["Salary"].isna().sum() == 1

    # 2. String comma-separated
    out2 = imputer.execute({"dataframe": df}, {"strategy": "median", "columns": "Age, Salary"})
    assert out2["dataframe"]["Age"].isna().sum() == 0
    assert out2["dataframe"]["Salary"].isna().sum() == 0

    # 3. List of columns
    out3 = imputer.execute({"dataframe": df}, {"strategy": "median", "columns": ["Age", "City"]})
    assert out3["dataframe"]["Age"].isna().sum() == 0
    assert out3["dataframe"]["City"].isna().sum() == 0
