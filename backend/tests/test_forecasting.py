import pytest
import pandas as pd
import numpy as np
from backend.app.recipes.base.registry import recipe_registry
import backend.app.recipes


def test_lag_feature_engineering():
    dates = pd.date_range("2026-01-01", periods=60, freq="D")
    sales = np.sin(np.linspace(0, 10, 60)) * 50 + 100 + np.random.normal(0, 2, 60)
    df = pd.DataFrame({"Date": dates, "Sales": sales})

    lag_recipe = recipe_registry.get("lag_feature_engineering")
    assert lag_recipe is not None

    res = lag_recipe.execute({"dataframe": df}, {"lag_periods": "1, 2, 7", "rolling_windows": "7"})
    out_df = res["dataframe"]

    assert "Sales_lag_1" in out_df.columns
    assert "Sales_lag_7" in out_df.columns
    assert "Sales_roll_mean_7" in out_df.columns
    assert "cal_dayofweek" in out_df.columns
    assert len(out_df) == 60


def test_prophet_forecaster():
    dates = pd.date_range("2026-01-01", periods=45, freq="D")
    sales = np.linspace(50, 150, 45) + np.random.normal(0, 5, 45)
    df = pd.DataFrame({"Date": dates, "Sales": sales})

    prophet_recipe = recipe_registry.get("prophet_forecaster")
    assert prophet_recipe is not None

    res = prophet_recipe.execute({"dataframe": df}, {"horizon_periods": 7})
    forecast_df = res["forecast_df"]
    metrics = res["metrics"]

    assert len(forecast_df) == 45 + 7
    assert "yhat" in forecast_df.columns
    assert "yhat_lower" in forecast_df.columns
    assert metrics["task_type"] == "time_series_forecasting"
    assert "mape" in metrics


def test_arima_forecaster():
    dates = pd.date_range("2026-01-01", periods=50, freq="D")
    demand = np.linspace(100, 200, 50) + np.random.normal(0, 4, 50)
    df = pd.DataFrame({"Date": dates, "Demand": demand})

    arima_recipe = recipe_registry.get("arima_forecaster")
    assert arima_recipe is not None

    res = arima_recipe.execute({"dataframe": df}, {"p": 1, "d": 1, "q": 1, "horizon_periods": 10})
    forecast_df = res["forecast_df"]
    metrics = res["metrics"]

    assert len(forecast_df) == 50 + 10
    assert "yhat" in forecast_df.columns
    assert metrics["task_type"] == "time_series_forecasting"
    assert "aic" in metrics
