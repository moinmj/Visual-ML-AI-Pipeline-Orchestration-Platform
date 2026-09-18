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


def test_prophet_weekly_frequency_and_export():
    """
    Verify that when Prophet is trained on weekly data or configured with Weekly frequency,
    future forecast records and CSV exports advance by 7-day weekly intervals, NOT daily steps.
    """
    from backend.app.engine.inference.pipeline_inferencer import PipelineInferencer
    from backend.app.engine.inference.schemas import PredictionRequest
    from backend.app.workflows.router import _convert_prediction_to_csv

    dates = pd.date_range("2020-01-03", periods=52, freq="W-FRI")
    sales = np.linspace(100000, 200000, 52) + np.random.normal(0, 5000, 52)
    df = pd.DataFrame({"Date": dates, "Weekly_Sales": sales})

    prophet_recipe = recipe_registry.get("prophet_forecaster")
    assert prophet_recipe is not None

    res = prophet_recipe.execute(
        inputs={"dataframe": df},
        config={"target_column": "Weekly_Sales", "date_column": "Date", "frequency": "W (Weekly)", "horizon_periods": 5}
    )

    metrics = res["metrics"]
    forecast_df = res["forecast_df"]
    model = res["model"]

    assert metrics["frequency"] == "W"
    # Check that future forecast rows are spaced by 7 days
    future_rows = forecast_df[forecast_df["is_future"] == 1].reset_index(drop=True)
    assert len(future_rows) == 5

    dt0 = pd.to_datetime(future_rows["ds"].iloc[0])
    dt1 = pd.to_datetime(future_rows["ds"].iloc[1])
    diff_days = (dt1 - dt0).days
    assert diff_days == 7, f"Future steps must be 7 days apart (weekly), found {diff_days} days"

    # Verify Inference bundle & live prediction endpoint
    bundle = {
        "execution_id": "test_weekly_prophet_exec",
        "task_type": "time_series_forecasting",
        "model": model,
        "target_column": "Weekly_Sales",
        "forecasting_summary": metrics,
        "frequency": "W",
        "freq": "W"
    }

    pred_res = PipelineInferencer.predict(bundle=bundle, request=PredictionRequest(forecast_horizon=5))
    assert pred_res.status == "SUCCESS"
    assert len(pred_res.forecast_records) == 5

    rec_d0 = pd.to_datetime(pred_res.forecast_records[0]["ds"])
    rec_d1 = pd.to_datetime(pred_res.forecast_records[1]["ds"])
    assert (rec_d1 - rec_d0).days == 7, "Prediction records must be weekly"

    # Verify CSV export
    csv_str = _convert_prediction_to_csv(pred_res)
    assert "ds,yhat" in csv_str
    csv_dates = [line.split(",")[0] for line in csv_str.strip().split("\n")[1:]]
    assert len(csv_dates) == 5
    d_first = pd.to_datetime(csv_dates[0])
    d_second = pd.to_datetime(csv_dates[1])
    assert (d_second - d_first).days == 7, "Exported CSV must contain weekly steps, not consecutive calendar days"


def test_prophet_panel_data_auto_aggregation_and_entity_filter():
    """
    Verify that multi-entity panel datasets (multiple stores sharing timestamps)
    are automatically aggregated across entities or filtered to a single entity,
    preventing duplicate timestamp distortion, catastrophic MAPE, and frequency misalignment.
    """
    # Create 5 stores across 25 weekly timestamps (125 total rows)
    stores = []
    dates = pd.date_range("2021-01-01", periods=25, freq="W-FRI")
    for s in range(1, 6):
        for i, d in enumerate(dates):
            stores.append({
                "Store": s,
                "Date": d,
                "Weekly_Sales": 10000.0 * s + (i * 100.0)
            })
    df_panel = pd.DataFrame(stores)
    assert len(df_panel) == 125
    assert df_panel["Date"].duplicated().sum() == 100  # 4 duplicate stores per date

    prophet_recipe = recipe_registry.get("prophet_forecaster")
    assert prophet_recipe is not None

    # 1. Default 'auto' strategy: Auto-aggregates by sum across stores
    res_agg = prophet_recipe.execute(
        inputs={"dataframe": df_panel},
        config={"target_column": "Weekly_Sales", "date_column": "Date", "horizon_periods": 4}
    )
    m_agg = res_agg["metrics"]
    assert m_agg["panel_data_detected"] is True
    assert m_agg["panel_strategy_applied"] == "aggregate_sum"
    assert m_agg["historical_points"] == 25  # exactly 25 unique weeks
    assert m_agg["frequency"] == "W"
    assert "Multi-entity panel data detected" in m_agg["panel_summary_info"]
    # Verify low MAPE on aggregated series
    assert m_agg["mape"] < 10.0

    # 2. 'filter_entity' strategy: Isolates Store 1
    res_filter = prophet_recipe.execute(
        inputs={"dataframe": df_panel},
        config={
            "target_column": "Weekly_Sales",
            "date_column": "Date",
            "group_by_column": "Store",
            "panel_strategy": "filter_entity",
            "entity_value": "1",
            "horizon_periods": 4
        }
    )
    m_fil = res_filter["metrics"]
    assert m_fil["panel_data_detected"] is True
    assert m_fil["panel_strategy_applied"] == "filter_entity"
    assert m_fil["historical_points"] == 25
    assert m_fil["entity_value"] == "1"
    assert m_fil["mape"] < 10.0

    # 3. ARIMA Forecaster also handles panel data with auto-aggregation
    arima_recipe = recipe_registry.get("arima_forecaster")
    assert arima_recipe is not None
    res_arima = arima_recipe.execute(
        inputs={"dataframe": df_panel},
        config={"target_column": "Weekly_Sales", "date_column": "Date", "p": 1, "d": 0, "q": 0, "horizon_periods": 4}
    )
    m_arima = res_arima["metrics"]
    assert m_arima["panel_data_detected"] is True
    assert m_arima["panel_strategy_applied"] == "aggregate_sum"
    assert m_arima["historical_points"] == 25


