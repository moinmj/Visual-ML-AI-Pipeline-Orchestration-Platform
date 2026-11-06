import pandas as pd
import pytest

from backend.app.recipes.reporting.threshold_alert_monitor import ThresholdAlertMonitorRecipe


def test_threshold_alert_monitor_metadata_and_schema():
    recipe = ThresholdAlertMonitorRecipe()
    assert recipe.recipe_id == "threshold_alert_monitor"
    assert recipe.category == "reporting"
    assert "metrics" in recipe.output_types

    schema = recipe.get_schema()
    assert schema["type"] == "object"
    props = schema["properties"]
    assert "metric_column" in props
    assert "condition_operator" in props
    assert "threshold_value" in props
    assert "severity" in props

    for prop_name, prop_data in props.items():
        assert "description" in prop_data, f"Property {prop_name} missing description"
        assert "input_type" in prop_data, f"Property {prop_name} missing input_type"


def test_threshold_alert_monitor_fallback_row_breach():
    recipe = ThresholdAlertMonitorRecipe()
    result = recipe.execute({}, {"metric_column": "value", "condition_operator": ">", "threshold_value": 100.0})

    assert "output" in result
    assert "metrics" in result
    assert "breach_output" in result

    metrics = result["metrics"]
    assert metrics["status"] == "ALERT_TRIGGERED"
    assert metrics["breached"] is True
    assert metrics["breach_count"] > 0
    assert len(metrics["alerts"]) > 0

    breach_table = result["breach_output"]
    assert len(breach_table) == metrics["breach_count"]
    assert "_breach_rule" in breach_table.columns
    assert "_breached_value" in breach_table.columns


def test_threshold_alert_monitor_nominal_pass():
    recipe = ThresholdAlertMonitorRecipe()
    df = pd.DataFrame({"cpu_pct": [22.4, 35.1, 44.0, 19.8]})

    result = recipe.execute(
        {"input": df},
        {
            "metric_column": "cpu_pct",
            "condition_operator": ">",
            "threshold_value": 80.0,
        },
    )

    metrics = result["output"]
    assert metrics["status"] == "NOMINAL"
    assert metrics["breached"] is False
    assert metrics["breach_count"] == 0
    assert result["breach_output"].empty


def test_threshold_alert_monitor_aggregation_mean_and_outside_range():
    recipe = ThresholdAlertMonitorRecipe()
    df = pd.DataFrame({"latency_ms": [120.0, 140.0, 160.0]})  # mean is 140.0

    result = recipe.execute(
        {"dataframe": df},
        {
            "metric_column": "latency_ms",
            "aggregation_mode": "mean",
            "condition_operator": "outside_range",
            "threshold_value": 130.0,  # upper bound
            "threshold_secondary": 50.0,  # lower bound
        },
    )

    metrics = result["output"]
    assert metrics["breached"] is True
    assert metrics["aggregation_mode"] == "mean"
    assert metrics["evaluated_value"] == 140.0


def test_threshold_alert_monitor_input_metrics_dict():
    recipe = ThresholdAlertMonitorRecipe()
    result = recipe.execute(
        {"metrics": {"model_f1_score": 0.65}},
        {
            "metric_column": "model_f1_score",
            "condition_operator": "<",
            "threshold_value": 0.75,
            "severity": "fatal",
        },
    )

    metrics = result["metrics"]
    assert metrics["breached"] is True
    assert metrics["severity"] == "fatal"
    assert metrics["evaluated_value"] == 0.65


def test_threshold_alert_monitor_fail_pipeline_on_breach():
    recipe = ThresholdAlertMonitorRecipe()
    df = pd.DataFrame({"error_rate": [0.15]})

    with pytest.raises(ValueError) as exc_info:
        recipe.execute(
            {"input": df},
            {
                "metric_column": "error_rate",
                "condition_operator": ">",
                "threshold_value": 0.05,
                "fail_pipeline_on_breach": True,
            },
        )
    assert "Pipeline execution halted by ThresholdAlertMonitor" in str(exc_info.value)
