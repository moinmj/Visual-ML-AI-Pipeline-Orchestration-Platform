import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from backend.app.recipes.base.recipe import BaseRecipe

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except (ImportError, OSError, Exception):
    Prophet = None
    PROPHET_AVAILABLE = False


class ProphetForecasterRecipe(BaseRecipe):
    recipe_id = "prophet_forecaster"
    name = "Prophet Time-Series Forecaster"
    version = "1.1.0"
    category = "forecasting"
    description = "Meta Prophet additive model with chronological out-of-sample backtesting, non-linear trends, and daily/weekly/yearly seasonality."
    input_types = ["dataframe"]
    output_types = ["forecast", "metrics", "model"]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "date_column": {
                    "type": "string",
                    "title": "Date Column",
                    "description": "Timestamp column (ds)."
                },
                "target_column": {
                    "type": "string",
                    "title": "Target Metric (Y)",
                    "description": "The time-series value to forecast."
                },
                "horizon_periods": {
                    "type": "integer",
                    "title": "Forecast Horizon (Steps ahead)",
                    "default": 14,
                    "minimum": 1,
                    "maximum": 365
                },
                "frequency": {
                    "type": "string",
                    "title": "Data Frequency",
                    "enum": ["D (Daily)", "W (Weekly)", "M (Monthly)", "H (Hourly)"],
                    "default": "D (Daily)"
                },
                "seasonality_mode": {
                    "type": "string",
                    "title": "Seasonality Mode",
                    "enum": ["additive", "multiplicative"],
                    "default": "additive"
                },
                "test_size_pct": {
                    "type": "number",
                    "title": "Holdout Test Size (%)",
                    "default": 0.2,
                    "minimum": 0.05,
                    "maximum": 0.4,
                    "description": "Fraction of chronologically latest observations to hold out for out-of-sample evaluation."
                }
            },
            "required": ["target_column"]
        }

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        errors = []
        target_col = config.get("target_column")
        if not target_col or not str(target_col).strip() or str(target_col).strip() in ["-- Select Column --", "(None)"]:
            errors.append("Target variable 'target_column' is required for Prophet Forecaster and cannot be empty.")
        return errors

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        if not PROPHET_AVAILABLE:
            raise ValueError("Prophet is not installed in the environment. Please run 'pip install prophet'.")

        df: pd.DataFrame = inputs.get("dataframe")
        if df is None:
            if context and isinstance(context, dict) and "dataframe" in context:
                df = context["dataframe"]
            else:
                raise ValueError("ProphetForecaster expects 'dataframe' in inputs.")

        df = df.copy()

        # 1. Validate and resolve target column
        target_col = config.get("target_column") or inputs.get("target_column")
        if not target_col or not str(target_col).strip() or str(target_col).strip() in ["-- Select Column --", "(None)"]:
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if num_cols:
                target_col = num_cols[-1]
            else:
                raise ValueError(
                    "Target variable 'target_column' is required for Prophet Forecaster, but was left empty. "
                    "Please configure which numeric column to forecast."
                )
        target_col = str(target_col).strip()
        if target_col not in df.columns:
            matching = [c for c in df.columns if c.lower() == target_col.lower()]
            if matching:
                target_col = matching[0]
            else:
                raise ValueError(
                    f"Specified target column '{target_col}' was not found in dataset columns: {list(df.columns)}. "
                    "Please select an existing numeric column."
                )

        # 2. Identify date column or resolve valid datetime series
        date_col = config.get("date_column") or inputs.get("date_column")
        valid_ds = None

        def _safe_parse_datetime(series: pd.Series) -> Optional[pd.Series]:
            if pd.api.types.is_numeric_dtype(series):
                num_s = pd.to_numeric(series, errors="coerce").dropna()
                if len(num_s) >= 5 and num_s.between(1800, 2200).all():
                    res = pd.to_datetime(series.astype(str) + "-01-01", errors="coerce")
                    if res.notna().sum() >= 5:
                        return res
            res = pd.to_datetime(series, errors="coerce")
            if res.notna().sum() >= 5:
                return res
            return None

        if date_col and str(date_col).strip():
            date_col = str(date_col).strip()
            if date_col not in df.columns:
                matching_date = [c for c in df.columns if c.lower() == date_col.lower()]
                if matching_date:
                    date_col = matching_date[0]
                else:
                    raise ValueError(
                        f"Specified date column '{date_col}' was not found in dataset columns: {list(df.columns)}. "
                        "Please select an existing timestamp column."
                    )
            converted = _safe_parse_datetime(df[date_col])
            if converted is not None and converted.notna().sum() >= 5:
                valid_ds = converted
            else:
                raise ValueError(f"Date column '{date_col}' does not contain at least 5 valid datetime values.")

        if valid_ds is None:
            # Auto-search for any valid datetime column across dataframe
            for col in df.columns:
                if col == target_col:
                    continue
                cand = _safe_parse_datetime(df[col])
                if cand is not None and cand.notna().sum() >= 5:
                    valid_ds = cand
                    date_col = col
                    break

        if valid_ds is None:
            raise ValueError(
                "No valid date/datetime column found in dataset for Prophet forecasting. "
                "Please ensure your dataset contains a valid timestamp column and configure 'date_column'."
            )

        horizon = int(config.get("horizon_periods", 14))
        freq_code = config.get("frequency", "D (Daily)").split()[0]
        seas_mode = config.get("seasonality_mode", "additive")
        test_size_pct = float(config.get("test_size_pct", 0.2))

        # Build clean chronologically sorted dataframe for Prophet
        prophet_df = pd.DataFrame({
            "ds": valid_ds,
            "y": pd.to_numeric(df[target_col], errors="coerce")
        }).dropna().sort_values(by="ds").reset_index(drop=True)

        if len(prophet_df) < 5:
            raise ValueError(f"Prophet requires at least 5 valid time-series observations, found {len(prophet_df)}.")

        # ─────────────────────────────────────────────────────────────
        # CHRONOLOGICAL OUT-OF-SAMPLE BACKTESTING
        # Split last `test_size_pct` of observations as unseen holdout.
        # Fit only on the training slice, predict test dates, compute OOS metrics.
        # Then refit on FULL data for the final production forecast.
        # ─────────────────────────────────────────────────────────────
        n = len(prophet_df)
        # Minimum 5 test points; if dataset is too small fall back to all-data in-sample
        min_test_pts = 5
        split_idx = max(min_test_pts, int(n * (1.0 - test_size_pct)))

        eval_type = "out_of_sample_holdout"
        train_df = prophet_df.iloc[:split_idx].reset_index(drop=True)
        test_df  = prophet_df.iloc[split_idx:].reset_index(drop=True)

        if len(test_df) < 2:
            # Dataset too small for a meaningful holdout — fall back gracefully
            train_df = prophet_df.copy()
            test_df  = prophet_df.copy()
            eval_type = "in_sample_fallback"

        # --- Phase 1: Fit on training slice, evaluate on held-out test ---
        eval_model = Prophet(
            seasonality_mode=seas_mode,
            yearly_seasonality="auto",
            weekly_seasonality="auto",
            daily_seasonality="auto"
        )
        eval_model.fit(train_df)
        test_forecast = eval_model.predict(test_df[["ds"]])

        oos_actuals = test_df["y"].values
        oos_preds   = test_forecast["yhat"].values

        mae  = float(round(np.mean(np.abs(oos_actuals - oos_preds)), 4))
        rmse = float(round(np.sqrt(np.mean((oos_actuals - oos_preds) ** 2)), 4))

        non_zero_mask = oos_actuals != 0
        if np.any(non_zero_mask):
            mape = float(round(
                np.mean(np.abs((oos_actuals[non_zero_mask] - oos_preds[non_zero_mask]) / oos_actuals[non_zero_mask])) * 100,
                2
            ))
        else:
            mape = 0.0

        # Coverage: fraction of actuals within the 95% CI
        lower = test_forecast["yhat_lower"].values
        upper = test_forecast["yhat_upper"].values
        coverage_95 = float(round(np.mean((oos_actuals >= lower) & (oos_actuals <= upper)) * 100, 2))

        # --- Phase 2: Refit on FULL data for the production forecast ---
        model = Prophet(
            seasonality_mode=seas_mode,
            yearly_seasonality="auto",
            weekly_seasonality="auto",
            daily_seasonality="auto"
        )
        model.fit(prophet_df)

        # Generate Future Dataframe
        future   = model.make_future_dataframe(periods=horizon, freq=freq_code)
        forecast = model.predict(future)

        last_date       = prophet_df["ds"].max()
        is_future_flags = (forecast["ds"] > last_date).astype(int).tolist()

        metrics = {
            "task_type":          "time_series_forecasting",
            "algorithm":          "Meta Prophet",
            "eval_type":          eval_type,
            "date_column":        date_col,
            "target_column":      target_col,
            "historical_points":  len(prophet_df),
            "train_size":         len(train_df),
            "test_size":          len(test_df),
            "holdout_test_start": str(test_df["ds"].iloc[0].date()) if len(test_df) > 0 else None,
            "holdout_test_end":   str(test_df["ds"].iloc[-1].date()) if len(test_df) > 0 else None,
            "forecast_horizon":   horizon,
            "trend_direction":    "Upward" if float(forecast["yhat"].iloc[-1]) >= float(forecast["yhat"].iloc[0]) else "Downward",
            "horizon_periods":    horizon,
            # ── True out-of-sample accuracy metrics ──
            "mae":                mae,
            "rmse":               rmse,
            "mape":               mape,
            "coverage_95pct":     coverage_95,
            "evaluation_note":    (
                f"Metrics computed on chronologically held-out last {len(test_df)} observations "
                f"({test_size_pct*100:.0f}% of data). "
                f"Model was retrained on all {len(prophet_df)} points for future forecasting."
                if eval_type == "out_of_sample_holdout"
                else "Dataset too small for holdout — metrics are in-sample."
            )
        }

        # Build clean visualization table (historical + future)
        result_df = pd.DataFrame({
            "ds": forecast["ds"].dt.strftime("%Y-%m-%d") if hasattr(forecast["ds"].dt, "strftime") else forecast["ds"].astype(str),
            "yhat":       np.round(forecast["yhat"],       2),
            "yhat_lower": np.round(forecast["yhat_lower"], 2),
            "yhat_upper": np.round(forecast["yhat_upper"], 2),
            "is_future":  is_future_flags
        })

        # Embed forecast points directly into metrics/summary for lightweight charting
        metrics["forecast_data"] = result_df.to_dict(orient="records")

        return {
            "forecast_df":        result_df,
            "dataframe":          result_df,
            "metrics":            metrics,
            "forecasting_summary": metrics,
            "model":              model,
            "task_type":          "time_series_forecasting"
        }

    def to_code(self, config: Dict[str, Any]) -> str:
        horizon = config.get("horizon_periods", 14)
        seas    = config.get("seasonality_mode", "additive")
        return (
            f"from prophet import Prophet\n\n"
            f"# Chronological train/test split (80/20)\n"
            f"n = len(df); split = int(n * 0.8)\n"
            f"train_df, test_df = df.iloc[:split], df.iloc[split:]\n\n"
            f"# Evaluate on held-out test set\n"
            f"eval_model = Prophet(seasonality_mode='{seas}')\n"
            f"eval_model.fit(train_df[['ds', 'y']])\n"
            f"oos_preds = eval_model.predict(test_df[['ds']])\n\n"
            f"# Refit on full data for final forecast\n"
            f"model = Prophet(seasonality_mode='{seas}')\n"
            f"model.fit(df[['ds', 'y']])\n"
            f"future = model.make_future_dataframe(periods={horizon})\n"
            f"forecast = model.predict(future)"
        )
