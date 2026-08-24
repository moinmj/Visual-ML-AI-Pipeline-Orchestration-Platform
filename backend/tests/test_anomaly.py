import pytest
import pandas as pd
import numpy as np
from backend.app.recipes.base.registry import recipe_registry
import backend.app.recipes


def test_isolation_forest_anomaly_detector():
    np.random.seed(42)
    # Generate 100 normal points and 5 extreme outliers
    n = 100
    df = pd.DataFrame({
        "feature1": np.concatenate([np.random.normal(50, 5, n), [500, 600, -200, 700, 800]]),
        "feature2": np.concatenate([np.random.normal(100, 10, n), [1000, 1200, -500, 1500, 1800]]),
        "category": np.concatenate([np.random.choice(["A", "B"], size=n), ["A", "B", "A", "B", "A"]])
    })

    iso_recipe = recipe_registry.get("isolation_forest")
    assert iso_recipe is not None

    result = iso_recipe.execute({"dataframe": df}, {"contamination": 0.05, "n_estimators": 50})
    
    out_df = result["dataframe"]
    assert "is_anomaly" in out_df.columns
    assert "anomaly_score" in out_df.columns
    assert out_df["is_anomaly"].sum() > 0

    metrics = result["metrics"]
    assert metrics["task_type"] == "anomaly_detection"
    assert metrics["algorithm"] == "Isolation Forest"
    assert metrics["anomaly_count"] > 0
    assert "anomaly_percentage" in metrics


def test_isolation_forest_edge_cases_with_nans():
    np.random.seed(42)
    df = pd.DataFrame({
        "num1": [1.0, 2.0, np.nan, 4.0, 5.0, 1000.0],
        "num2": [10.0, np.nan, 30.0, 40.0, 50.0, 9999.0],
        "dept": ["IT", "Sales", np.nan, "HR", "IT", "Sales"]
    })

    iso_recipe = recipe_registry.get("isolation_forest")
    # Should execute without crashing despite NaNs and text strings
    result = iso_recipe.execute({"dataframe": df}, {"contamination": 0.2})
    assert len(result["dataframe"]) == 6
    assert result["dataframe"]["is_anomaly"].iloc[-1] == 1  # Extreme point


def test_statistical_guardrail_zscore_and_iqr():
    np.random.seed(42)
    # Normal data with one extreme outlier
    data = np.random.normal(50, 2, 50).tolist()
    data.append(500.0)  # Extreme outlier
    df = pd.DataFrame({"amount": data})

    guardrail = recipe_registry.get("statistical_guardrail")
    assert guardrail is not None

    # 1. Z-Score Flagging
    res_flag = guardrail.execute({"dataframe": df}, {"method": "z_score", "threshold": 3.0, "action": "flag"})
    assert res_flag["dataframe"]["is_outlier"].iloc[-1] == 1
    assert res_flag["metrics"]["outliers_detected"] >= 1

    # 2. IQR Filtering
    res_filter = guardrail.execute({"dataframe": df}, {"method": "iqr", "threshold": 1.5, "action": "filter"})
    assert len(res_filter["dataframe"]) < len(df)
    assert 500.0 not in res_filter["dataframe"]["amount"].values
