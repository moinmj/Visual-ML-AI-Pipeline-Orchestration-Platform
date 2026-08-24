import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from backend.app.recipes.base.recipe import BaseRecipe

try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False


class ARIMAForecasterRecipe(BaseRecipe):
    recipe_id = "arima_forecaster"
    name = "ARIMA Statistical Forecaster"
    version = "1.0.0"
    category = "forecasting"
    description = "Classical AutoRegressive Integrated Moving Average (ARIMA) for statistical univariate time-series forecasting (Tier-1 Baseline)."
    input_types = ["dataframe"]
    output_types = ["forecast", "metrics", "model"]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "date_column": {
                    "type": "string",
                    "title": "Date Column",
                    "description": "Timestamp column."
                },
                "target_column": {
                    "type": "string",
                    "title": "Target Metric (Y)",
                    "description": "The time-series value to model."
                },
                "p": {
                    "type": "integer",
                    "title": "AR Order (p)",
                    "default": 1,
                    "minimum": 0,
                    "maximum": 10,
                    "description": "Auto-regressive lag order."
                },
                "d": {
                    "type": "integer",
                    "title": "Differencing Order (d)",
                    "default": 1,
                    "minimum": 0,
                    "maximum": 2,
                    "description": "Degree of differencing for stationarity."
                },
                "q": {
                    "type": "integer",
                    "title": "MA Order (q)",
                    "default": 1,
                    "minimum": 0,
                    "maximum": 10,
                    "description": "Moving average window order."
                },
                "horizon_periods": {
                    "type": "integer",
                    "title": "Forecast Horizon (Steps ahead)",
                    "default": 14,
                    "minimum": 1,
                    "maximum": 365
                }
            }
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        if not ARIMA_AVAILABLE:
            raise ValueError("statsmodels is not installed. Please run 'pip install statsmodels'.")

        df: pd.DataFrame = inputs.get("dataframe")
        if df is None:
            if context and isinstance(context, dict) and "dataframe" in context:
                df = context["dataframe"]
            else:
                raise ValueError("ARIMAForecaster expects 'dataframe' in inputs.")

        df = df.copy()

        # Identify date column
        date_col = config.get("date_column") or inputs.get("date_column")
        if not date_col or date_col not in df.columns:
            date_cands = [c for c in df.columns if "date" in c.lower() or "time" in c.lower() or "timestamp" in c.lower() or pd.api.types.is_datetime64_any_dtype(df[c])]
            date_col = date_cands[0] if date_cands else df.columns[0]

        # Identify target column
        target_col = config.get("target_column") or inputs.get("target_column")
        if not target_col or target_col not in df.columns:
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != date_col]
            target_col = num_cols[-1] if num_cols else df.columns[-1]

        p = int(config.get("p", 1))
        d = int(config.get("d", 1))
        q = int(config.get("q", 1))
        horizon = int(config.get("horizon_periods", 14))

        ts_df = pd.DataFrame({
            "ds": pd.to_datetime(df[date_col], errors="coerce"),
            "y": pd.to_numeric(df[target_col], errors="coerce")
        }).dropna().sort_values(by="ds").reset_index(drop=True)

        if len(ts_df) < (p + d + q + 3):
            raise ValueError(f"ARIMA({p},{d},{q}) requires at least {p + d + q + 3} observations, found {len(ts_df)}.")

        # Fit ARIMA Model
        model = ARIMA(ts_df["y"].values, order=(p, d, q))
        fitted_model = model.fit()

        # In-sample fitted values
        fitted_values = fitted_model.fittedvalues
        actuals = ts_df["y"].values

        # Out-of-sample Forecast
        forecast_res = fitted_model.get_forecast(steps=horizon)
        future_means = forecast_res.predicted_mean
        conf_int = forecast_res.conf_int(alpha=0.05)

        # Generate future dates
        last_date = ts_df["ds"].iloc[-1]
        freq = pd.infer_freq(ts_df["ds"]) or "D"
        future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=freq)[1:]

        # Metrics on in-sample
        valid_idx = slice(d, None)
        mae = float(round(np.mean(np.abs(actuals[valid_idx] - fitted_values[valid_idx])), 4))
        rmse = float(round(np.sqrt(np.mean((actuals[valid_idx] - fitted_values[valid_idx]) ** 2)), 4))

        non_zero = actuals[valid_idx] != 0
        if np.any(non_zero):
            mape = float(round(np.mean(np.abs((actuals[valid_idx][non_zero] - fitted_values[valid_idx][non_zero]) / actuals[valid_idx][non_zero])) * 100, 2))
        else:
            mape = 0.0

        metrics = {
            "task_type": "time_series_forecasting",
            "algorithm": f"ARIMA({p},{d},{q})",
            "aic": float(round(fitted_model.aic, 2)),
            "bic": float(round(fitted_model.bic, 2)),
            "horizon_periods": horizon,
            "mae": mae,
            "rmse": rmse,
            "mape": mape
        }

        # Build clean visualization table
        hist_df = pd.DataFrame({
            "ds": ts_df["ds"],
            "yhat": np.round(fitted_values, 2),
            "yhat_lower": np.round(fitted_values, 2),
            "yhat_upper": np.round(fitted_values, 2),
            "is_future": 0
        })

        fut_df = pd.DataFrame({
            "ds": future_dates,
            "yhat": np.round(future_means, 2),
            "yhat_lower": np.round(conf_int[:, 0], 2),
            "yhat_upper": np.round(conf_int[:, 1], 2),
            "is_future": 1
        })

        full_forecast_df = pd.concat([hist_df, fut_df], ignore_index=True)

        return {
            "forecast_df": full_forecast_df,
            "dataframe": full_forecast_df,
            "metrics": metrics,
            "forecasting_summary": metrics,
            "model": fitted_model
        }
