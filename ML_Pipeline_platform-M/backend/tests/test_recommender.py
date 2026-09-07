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
    assert rec["model_rankings"][0]["recipe_id"] == "xgboost_trainer"


def test_ai_recommender_time_series():
    dates = pd.date_range("2026-01-01", periods=30, freq="D")
    sales = np.random.normal(500, 50, 30)
    df = pd.DataFrame({"Date": dates, "Sales": sales})

    rec = AIRecommender.recommend_pipeline(df)
    assert rec["task_type"] == "time_series_forecasting"
    assert rec["target_column"] == "Sales"
    assert rec["model_rankings"][0]["recipe_id"] == "prophet_forecaster"
