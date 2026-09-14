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

    # 1. Default handle_na="drop_rows" cleanly discards initial 7 unobserved days
    res_drop = lag_recipe.execute({"dataframe": df}, {"lag_periods": "1, 2, 7", "rolling_windows": "7", "handle_na": "drop_rows"})
    out_drop = res_drop["dataframe"]
    assert "Sales_lag_1" in out_drop.columns
    assert "Sales_lag_7" in out_drop.columns
    assert "Sales_roll_mean_7" in out_drop.columns
    assert "cal_dayofweek" in out_drop.columns
    assert len(out_drop) == 60 - 7  # 53 rows
    assert out_drop["Sales_lag_7"].isna().sum() == 0

    # 2. handle_na="bfill" keeps all rows
    res_bfill = lag_recipe.execute({"dataframe": df}, {"lag_periods": "1, 2, 7", "rolling_windows": "7", "handle_na": "bfill"})
    assert len(res_bfill["dataframe"]) == 60

    # 3. Multi-entity group_by_column (Store panel time-series)
    df_panel = pd.DataFrame({
        "Store": [1]*50 + [2]*50,
        "Date": list(dates[:50]) + list(dates[:50]),
        "Sales": list(sales[:50]) + list(sales[:50])
    })
    res_panel = lag_recipe.execute(
        {"dataframe": df_panel},
        {"lag_periods": "1, 10", "rolling_windows": "10", "group_by_column": "Store", "handle_na": "drop_rows"}
    )
    # Each of the 2 stores drops 10 rows: (50 - 10) * 2 = 80 rows
    assert len(res_panel["dataframe"]) == 80
    assert res_panel["rows_dropped"] == 20


def test_lag_feature_engineering_walmart_panel_52_weeks():
    # Exactly simulates the 45-store, 143-week Walmart panel dataset (6,435 total rows)
    stores = []
    np.random.seed(42)
    for s in range(1, 46):
        dates = pd.date_range("2010-02-05", periods=143, freq="7D")
        for d in dates:
            stores.append({
                "Store": s,
                "Date": d,
                "Weekly_Sales": float(np.random.uniform(15000, 45000))
            })
    df_walmart = pd.DataFrame(stores)
    assert len(df_walmart) == 6435

    lag_recipe = recipe_registry.get("lag_feature_engineering")
    res = lag_recipe.execute(
        {"dataframe": df_walmart},
        {
            "date_column": "Date",
            "target_column": "Weekly_Sales",
            "group_by_column": "Store",
            "lag_periods": "52",
            "rolling_windows": "52",
            "handle_na": "drop_rows"
        }
    )
    df_clean = res["dataframe"]

    # 1. Exactly 2,340 structural unobserved lag rows dropped (52 weeks * 45 stores)
    assert res["rows_dropped"] == 45 * 52  # 2340
    # 2. Exactly 4,095 clean valid historical rows remain
    assert len(df_clean) == 45 * (143 - 52)  # 4095
    assert len(df_clean) == 4095

    # 3. Verify Store 1 and Store 2 have zero fabricated 0s or NaNs in lag/rolling features
    assert df_clean["Weekly_Sales_lag_52"].isna().sum() == 0
    assert (df_clean["Weekly_Sales_lag_52"] == 0).sum() == 0
    assert df_clean["Weekly_Sales_roll_mean_52"].isna().sum() == 0

    # 4. Confirm each store has exactly 91 rows remaining
    assert set(df_clean.groupby("Store").size().values) == {91}


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
