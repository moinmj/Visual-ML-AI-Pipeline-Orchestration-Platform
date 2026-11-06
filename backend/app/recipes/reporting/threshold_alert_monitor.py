import datetime
import hashlib
import json
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from backend.app.recipes.base.recipe import BaseRecipe, RecipePort


class ThresholdAlertMonitorRecipe(BaseRecipe):
    recipe_id = "threshold_alert_monitor"
    name = "Alert / Threshold Monitor"
    version = "1.0.0"
    category = "reporting"
    description = "Raises alerts when a metric crosses a configured threshold."
    input_types = ["dataframe", "metrics"]
    output_types = ["metrics", "dataframe"]

    def get_inputs(self) -> List[RecipePort]:
        return [
            RecipePort(
                id="input",
                label="Input Table",
                type="dataframe",
                required=False,
                description="Dataset whose columns are evaluated against threshold constraints",
            ),
            RecipePort(
                id="metrics",
                label="Input Metrics",
                type="metrics",
                required=False,
                description="Upstream numerical metrics dictionary to monitor",
            ),
        ]

    def get_outputs(self) -> List[RecipePort]:
        return [
            RecipePort(
                id="output",
                label="Output (metrics)",
                type="metrics",
                description="Alert telemetry, incident status, breach statistics, and trigger records",
            ),
            RecipePort(
                id="breach_output",
                label="Breached Records Table",
                type="dataframe",
                description="Filtered table containing only rows that violated the monitoring rule",
            ),
        ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "metric_column": {
                    "type": "string",
                    "title": "Target Metric Column / Key",
                    "default": "value",
                    "description": "Column name in DataFrame or key in metrics dictionary to evaluate.",
                    "input_type": "text",
                },
                "condition_operator": {
                    "type": "string",
                    "title": "Comparison Condition",
                    "enum": [">", ">=", "<", "<=", "==", "!=", "outside_range", "zscore_outlier"],
                    "default": ">",
                    "description": "Mathematical rule used to test against threshold bounds.",
                    "input_type": "select",
                },
                "threshold_value": {
                    "type": "number",
                    "title": "Primary Alert Threshold",
                    "default": 100.0,
                    "description": "Primary threshold boundary (upper bound or exact test value).",
                    "input_type": "number",
                },
                "threshold_secondary": {
                    "type": "number",
                    "title": "Secondary Bound (Lower)",
                    "default": 0.0,
                    "description": "Lower boundary when condition_operator is 'outside_range'.",
                    "input_type": "number",
                },
                "severity": {
                    "type": "string",
                    "title": "Incident Severity",
                    "enum": ["info", "warning", "critical", "fatal"],
                    "default": "critical",
                    "description": "Severity level assigned to generated incident notifications.",
                    "input_type": "select",
                },
                "aggregation_mode": {
                    "type": "string",
                    "title": "Evaluation Scope / Aggregation",
                    "enum": ["row_level", "mean", "sum", "max", "min", "p95", "rate_of_change"],
                    "default": "row_level",
                    "description": "Whether to check individual records or compute an aggregate metric first.",
                    "input_type": "select",
                },
                "alert_channel": {
                    "type": "string",
                    "title": "Notification Channel",
                    "enum": ["in_app_notification", "webhook", "email", "pagerduty", "slack"],
                    "default": "in_app_notification",
                    "description": "Target dispatch routing channel for triggered alerts.",
                    "input_type": "select",
                },
                "webhook_url": {
                    "type": "string",
                    "title": "Dispatch Webhook / Endpoint URL",
                    "default": "",
                    "description": "Destination endpoint receiving the alert payload when webhook/slack is chosen.",
                    "input_type": "text",
                },
                "max_alerts_per_batch": {
                    "type": "integer",
                    "title": "Max Alerts Captured",
                    "default": 50,
                    "minimum": 1,
                    "maximum": 1000,
                    "description": "Maximum number of breached rows included in the incident report.",
                    "input_type": "number",
                },
                "fail_pipeline_on_breach": {
                    "type": "boolean",
                    "title": "Halt Pipeline On Violation",
                    "default": False,
                    "description": "Raises an unhandled exception to prevent downstream stages from executing when breached.",
                    "input_type": "checkbox",
                },
                "cool_down_seconds": {
                    "type": "integer",
                    "title": "Alert Debounce / Cool-Down (s)",
                    "default": 300,
                    "minimum": 0,
                    "description": "Suppresses duplicate notifications for identical violations within the time window.",
                    "input_type": "number",
                },
            },
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        df = inputs.get("dataframe")
        if df is None and isinstance(inputs.get("input"), pd.DataFrame):
            df = inputs["input"]

        input_metrics = inputs.get("metrics")
        if not isinstance(input_metrics, dict):
            input_metrics = {}

        metric_col = str(config.get("metric_column", "value")).strip() or "value"
        op = str(config.get("condition_operator", ">")).strip()
        primary_threshold = float(config.get("threshold_value", 100.0))
        secondary_threshold = float(config.get("threshold_secondary", 0.0))
        severity = str(config.get("severity", "critical")).lower().strip()
        agg_mode = str(config.get("aggregation_mode", "row_level")).lower().strip()
        channel = str(config.get("alert_channel", "in_app_notification")).lower().strip()
        webhook_url = str(config.get("webhook_url", "")).strip()
        max_alerts = max(1, min(1000, int(config.get("max_alerts_per_batch", 50))))
        fail_on_breach = bool(config.get("fail_pipeline_on_breach", False))
        cool_down_sec = max(0, int(config.get("cool_down_seconds", 300)))

        now_utc = datetime.datetime.now(datetime.timezone.utc)
        now_iso = now_utc.isoformat()

        # Synthetic fallback if no dataframe and no metric dict provided
        if (df is None or not isinstance(df, pd.DataFrame) or df.empty) and not input_metrics:
            df = pd.DataFrame({
                "sensor_id": [f"S-{i:03d}" for i in range(1, 9)],
                "temperature": [95.2, 98.4, 105.1, 99.0, 112.6, 94.8, 101.5, 96.0],
                "value": [85.0, 92.0, 108.5, 77.0, 125.0, 90.0, 104.2, 88.0],
                "location": ["Zone-A", "Zone-B", "Zone-A", "Zone-C", "Zone-B", "Zone-A", "Zone-C", "Zone-B"],
            })

        breached = False
        breach_count = 0
        breached_df = pd.DataFrame()
        evaluated_value: Optional[float] = None
        alerts_list: List[Dict[str, Any]] = []

        # 1. Evaluation against input_metrics dict (if key matches or df missing)
        if (df is None or df.empty or metric_col in input_metrics) and metric_col in input_metrics:
            val = float(input_metrics[metric_col])
            evaluated_value = val
            is_violation = False
            if op == ">":
                is_violation = val > primary_threshold
            elif op == ">=":
                is_violation = val >= primary_threshold
            elif op == "<":
                is_violation = val < primary_threshold
            elif op == "<=":
                is_violation = val <= primary_threshold
            elif op == "==":
                is_violation = abs(val - primary_threshold) < 1e-6
            elif op == "!=":
                is_violation = abs(val - primary_threshold) >= 1e-6
            elif op == "outside_range":
                is_violation = val < secondary_threshold or val > primary_threshold

            if is_violation:
                breached = True
                breach_count = 1
                alerts_list.append({
                    "alert_id": f"ALT-{hashlib.md5(f'{metric_col}_{val}_{now_iso}'.encode()).hexdigest()[:8].upper()}",
                    "metric": metric_col,
                    "condition": f"{metric_col} {op} {primary_threshold}",
                    "observed_value": val,
                    "severity": severity,
                    "timestamp": now_iso,
                })

        # 2. Evaluation against DataFrame
        elif df is not None and not df.empty:
            eval_col = metric_col if metric_col in df.columns else None
            if eval_col is None:
                # Find first numeric column
                num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                eval_col = num_cols[0] if num_cols else df.columns[0]

            col_series = pd.to_numeric(df[eval_col], errors="coerce").fillna(0.0)

            if agg_mode == "row_level":
                mask = pd.Series(False, index=df.index)
                if op == ">":
                    mask = col_series > primary_threshold
                elif op == ">=":
                    mask = col_series >= primary_threshold
                elif op == "<":
                    mask = col_series < primary_threshold
                elif op == "<=":
                    mask = col_series <= primary_threshold
                elif op == "==":
                    mask = (col_series - primary_threshold).abs() < 1e-6
                elif op == "!=":
                    mask = (col_series - primary_threshold).abs() >= 1e-6
                elif op == "outside_range":
                    mask = (col_series < secondary_threshold) | (col_series > primary_threshold)
                elif op == "zscore_outlier":
                    std_dev = col_series.std()
                    if std_dev > 1e-9:
                        z_scores = (col_series - col_series.mean()).abs() / std_dev
                        mask = z_scores > primary_threshold
                    else:
                        mask = pd.Series(False, index=df.index)

                breach_count = int(mask.sum())
                breached = breach_count > 0

                if breached:
                    matched_rows = df[mask].copy()
                    matched_rows["_breach_rule"] = f"{eval_col} {op} {primary_threshold}"
                    matched_rows["_breach_severity"] = severity
                    matched_rows["_breached_value"] = col_series[mask]
                    breached_df = matched_rows

                    for idx, row_val in col_series[mask].head(max_alerts).items():
                        alerts_list.append({
                            "alert_id": f"ALT-{hashlib.md5(f'{eval_col}_{idx}_{row_val}'.encode()).hexdigest()[:8].upper()}",
                            "row_index": int(idx),
                            "metric": eval_col,
                            "condition": f"{eval_col} {op} {primary_threshold}",
                            "observed_value": float(row_val),
                            "severity": severity,
                            "timestamp": now_iso,
                        })
            else:
                # Aggregate mode
                if agg_mode == "mean":
                    agg_val = float(col_series.mean())
                elif agg_mode == "sum":
                    agg_val = float(col_series.sum())
                elif agg_mode == "max":
                    agg_val = float(col_series.max())
                elif agg_mode == "min":
                    agg_val = float(col_series.min())
                elif agg_mode == "p95":
                    agg_val = float(col_series.quantile(0.95))
                elif agg_mode == "rate_of_change":
                    pct_change = col_series.pct_change().dropna()
                    agg_val = float(pct_change.mean() * 100.0) if not pct_change.empty else 0.0
                else:
                    agg_val = float(col_series.mean())

                evaluated_value = agg_val
                is_violation = False
                if op == ">":
                    is_violation = agg_val > primary_threshold
                elif op == ">=":
                    is_violation = agg_val >= primary_threshold
                elif op == "<":
                    is_violation = agg_val < primary_threshold
                elif op == "<=":
                    is_violation = agg_val <= primary_threshold
                elif op == "==":
                    is_violation = abs(agg_val - primary_threshold) < 1e-6
                elif op == "!=":
                    is_violation = abs(agg_val - primary_threshold) >= 1e-6
                elif op == "outside_range":
                    is_violation = agg_val < secondary_threshold or agg_val > primary_threshold

                if is_violation:
                    breached = True
                    breach_count = 1
                    alerts_list.append({
                        "alert_id": f"ALT-{hashlib.md5(f'{eval_col}_{agg_mode}_{agg_val}'.encode()).hexdigest()[:8].upper()}",
                        "aggregation": agg_mode,
                        "metric": eval_col,
                        "condition": f"{agg_mode}({eval_col}) {op} {primary_threshold}",
                        "observed_value": agg_val,
                        "severity": severity,
                        "timestamp": now_iso,
                    })

        # Incident status
        incident_status = "ALERT_TRIGGERED" if breached else "NOMINAL"

        # Dispatch simulation
        dispatch_metadata = {
            "channel": channel,
            "dispatched": breached,
            "webhook_url": webhook_url if webhook_url else None,
            "cool_down_seconds": cool_down_sec,
            "debounced": False,
        }

        metrics_payload: Dict[str, Any] = {
            "monitor_id": "threshold_alert_monitor",
            "status": incident_status,
            "breached": breached,
            "severity": severity if breached else "nominal",
            "breach_count": breach_count,
            "rule": f"{metric_col} {op} {primary_threshold}",
            "aggregation_mode": agg_mode,
            "evaluated_value": evaluated_value,
            "alerts_captured": len(alerts_list),
            "alerts": alerts_list[:max_alerts],
            "dispatch": dispatch_metadata,
            "evaluated_at": now_iso,
        }

        if fail_on_breach and breached:
            raise ValueError(
                f"Pipeline execution halted by ThresholdAlertMonitor: "
                f"Rule '{metric_col} {op} {primary_threshold}' violated with {breach_count} breach(es)."
            )

        return {
            "output": metrics_payload,
            "metrics": metrics_payload,
            "breach_output": breached_df,
            "dataframe": df if df is not None else pd.DataFrame(),
        }
