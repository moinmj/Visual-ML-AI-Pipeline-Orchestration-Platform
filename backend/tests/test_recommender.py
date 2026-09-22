import pytest
import pandas as pd
import numpy as np
from backend.app.recommendation.recommender import AIRecommender


def test_ai_recommender_classification():
    df = pd.DataFrame({
        "Age": [25, 30, np.nan, 45, 50],
        "Dept": ["Sales", "HR", "Sales", "IT", "HR"],
        "Target": [0, 1, 0, 1, 0]
    })
    rec = AIRecommender.recommend_pipeline(df)
    assert rec["task_type"] == "classification"
    assert len(rec["preprocessing_recommendations"]) > 0
    assert any(step["recipe_id"] == "missing_value_imputer" for step in rec["preprocessing_recommendations"])
    assert any(step["recipe_id"] == "categorical_encoder" for step in rec["preprocessing_recommendations"])
    assert rec["model_rankings"][0]["recipe_id"] in ["xgboost_trainer", "random_forest_trainer", "lightgbm_trainer", "catboost_trainer"]
    assert "recommended_dag" in rec
    dag_nodes = rec["recommended_dag"]["nodes"]
    assert any(n["id"] in ["node_eval", "eval_node"] and n["recipe_id"] in ["model_evaluator", "classification_evaluator", "regression_evaluator"] for n in dag_nodes)

    # Verify all recommended recipe IDs exist in recipe_registry
    from backend.app.recipes.base.registry import recipe_registry
    for n in dag_nodes:
        assert recipe_registry.has(n["recipe_id"]), f"Recipe {n['recipe_id']} not found in registry"


def test_recipe_registry_aliases():
    from backend.app.recipes.base.registry import recipe_registry
    # Ensure legacy or alias recipe IDs resolve without error
    cls_eval = recipe_registry.get("classification_evaluator")
    assert cls_eval.recipe_id == "model_evaluator"

    reg_eval = recipe_registry.get("regression_evaluator")
    assert reg_eval.recipe_id == "model_evaluator"


def test_ai_recommender_time_series():
    dates = pd.date_range("2026-01-01", periods=30, freq="D")
    sales = np.random.normal(500, 50, 30)
    df = pd.DataFrame({"Date": dates, "Sales": sales})

    rec = AIRecommender.recommend_pipeline(df)
    assert rec["task_type"] == "time_series_forecasting"
    assert rec["target_column"] == "Sales"
    assert rec["model_rankings"][0]["recipe_id"] in ["arima_forecaster", "prophet_forecaster"]
    assert "recommended_dag" in rec
    from backend.app.recipes.base.registry import recipe_registry
    for n in rec["recommended_dag"]["nodes"]:
        assert recipe_registry.has(n["recipe_id"]), f"Recipe {n['recipe_id']} not found in registry"


@pytest.mark.asyncio
async def test_llm_recommender_synthesis():
    from backend.app.recommendation.llm_recommender import LLMRecommender
    from backend.app.recipes.base.registry import recipe_registry

    df = pd.DataFrame({
        "sepal_length": [5.1, 4.9, 4.7, 4.6, 5.0],
        "petal_length": [1.4, 1.4, 1.3, 1.5, 1.4],
        "species": ["setosa", "setosa", "setosa", "setosa", "setosa"]
    })

    res = await LLMRecommender.recommend_pipeline_async(
        df=df,
        query="Train a high-accuracy classifier to identify Iris species and evaluate results",
        target_column="species"
    )

    assert "recommended_dag" in res
    assert len(res["recommended_dag"]["nodes"]) >= 3
    assert "explanation" in res
    assert res.get("llm_generated") is True
    for n in res["recommended_dag"]["nodes"]:
        assert recipe_registry.has(n["recipe_id"]), f"Recipe {n['recipe_id']} not found in registry"


@pytest.mark.asyncio
async def test_llm_recommender_provider_keys(monkeypatch):
    from backend.app.recommendation.llm_recommender import LLMRecommender
    from backend.app.core.config import settings

    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "test_gemini_key")
    monkeypatch.setattr(settings, "GROQ_API_KEY", None)

    # Calling with invalid test key falls back gracefully to AIRecommender
    res = await LLMRecommender.recommend_pipeline_async(df=df, query="predict y")
    assert "recommended_dag" in res
    assert res["target_column"] == "y"


def test_walmart_sales_target_and_scaler_exclusion():
    walmart_df = pd.DataFrame({
        "Store": [1, 1, 1, 1, 1],
        "Date": ["05-02-2010", "12-02-2010", "19-02-2010", "26-02-2010", "05-03-2010"],
        "Weekly_Sales": [24924.50, 46039.49, 41595.55, 19403.54, 21827.90],
        "Holiday_Flag": [0, 1, 0, 0, 0],
        "Temperature": [42.31, 38.51, 39.93, 46.63, 46.50],
        "Fuel_Price": [2.572, 2.548, 2.514, 2.561, 2.625],
        "CPI": [211.096, 211.242, 211.289, 211.319, 211.350],
        "Unemployment": [8.106, 8.106, 8.106, 8.106, 8.106]
    })
    rec = AIRecommender.recommend_pipeline(walmart_df)
    assert rec["target_column"] == "Weekly_Sales"

    # Verify DAG node configs pass target_column and exclude_target to Feature Scaler
    scaler_nodes = [n for n in rec["recommended_dag"]["nodes"] if n["recipe_id"] == "feature_scaler"]
    for sn in scaler_nodes:
        assert sn["config"].get("target_column") == "Weekly_Sales"
        assert sn["config"].get("exclude_target") is True




