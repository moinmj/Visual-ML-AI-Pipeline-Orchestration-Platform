import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from backend.app.recipes.base.recipe import BaseRecipe

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False


class ProphetForecasterRecipe(BaseRecipe):
    recipe_id = "prophet_forecaster"
    name = "Prophet Time-Series Forecaster"
    version = "1.0.0"
    category = "forecasting"
    description = "Meta Prophet additive model for non-linear trends with daily/weekly/yearly seasonality and future prediction bands."
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
                }
            }
        }

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

        # Identify date column or resolve valid datetime series
        date_col = config.get("date_column") or inputs.get("date_column")
        valid_ds = None

        if date_col and date_col in df.columns:
            converted = pd.to_datetime(df[date_col], errors="coerce")
            if converted.notna().sum() >= 5:
                valid_ds = converted

        if valid_ds is None:
            # Search for any valid datetime column across dataframe
            for col in df.columns:
                converted = pd.to_datetime(df[col], errors="coerce")
                if converted.notna().sum() >= 5:
                    valid_ds = converted
                    date_col = col
                    break

        if valid_ds is None:
            # Fall back to synthetic sequential daily timeline
            date_col = "ds_synthetic"
            valid_ds = pd.date_range(end=pd.Timestamp.today().normalize(), periods=len(df), freq="D")

        # Identify target column
        target_col = config.get("target_column") or inputs.get("target_column")
        if not target_col or target_col not in df.columns or target_col == date_col:
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != date_col]
            target_col = num_cols[-1] if num_cols else df.columns[-1]

        horizon = int(config.get("horizon_periods", 14))
        freq_code = config.get("frequency", "D (Daily)").split()[0]
        seas_mode = config.get("seasonality_mode", "additive")

        # Format DataFrame for Prophet (ds and y)
        prophet_df = pd.DataFrame({
            "ds": valid_ds,
            "y": pd.to_numeric(df[target_col], errors="coerce")
        }).dropna().sort_values(by="ds").reset_index(drop=True)

        if len(prophet_df) < 5:
            raise ValueError(f"Prophet requires at least 5 valid time-series observations, found {len(prophet_df)}.")

        # Train Prophet
        model = Prophet(
            seasonality_mode=seas_mode,
            yearly_seasonality="auto",
            weekly_seasonality="auto",
            daily_seasonality="auto"
        )
        model.fit(prophet_df)

        # In-sample predictions on actual historical observations
        in_sample = model.predict(prophet_df[["ds"]])
        actuals = prophet_df["y"].values
        fitted_preds = in_sample["yhat"].values
        
        mae = float(round(np.mean(np.abs(actuals - fitted_preds)), 4))
        rmse = float(round(np.sqrt(np.mean((actuals - fitted_preds) ** 2)), 4))
        
        non_zero_mask = actuals != 0
        if np.any(non_zero_mask):
            mape = float(round(np.mean(np.abs((actuals[non_zero_mask] - fitted_preds[non_zero_mask]) / actuals[non_zero_mask])) * 100, 2))
        else:
            mape = 0.0

        # Generate Future Dataframe
        future = model.make_future_dataframe(periods=horizon, freq=freq_code)
        forecast = model.predict(future)

        last_date = prophet_df["ds"].max()
        is_future_flags = (forecast["ds"] > last_date).astype(int).tolist()

        metrics = {
            "task_type": "time_series_forecasting",
            "algorithm": "Meta Prophet",
            "date_column": date_col,
            "target_column": target_col,
            "historical_points": len(prophet_df),
            "forecast_horizon": horizon,
            "trend_direction": "Upward" if float(forecast["yhat"].iloc[-1]) >= float(forecast["yhat"].iloc[0]) else "Downward",
            "horizon_periods": horizon,
            "mae": mae,
            "rmse": rmse,
            "mape": mape
        }

        # Build clean visualization table
        result_df = pd.DataFrame({
            "ds": forecast["ds"],
            "yhat": np.round(forecast["yhat"], 2),
            "yhat_lower": np.round(forecast["yhat_lower"], 2),
            "yhat_upper": np.round(forecast["yhat_upper"], 2),
            "is_future": is_future_flags
        })

        return {
            "forecast_df": result_df,
            "dataframe": result_df,
            "metrics": metrics,
            "forecasting_summary": metrics,
            "model": model
        }

    def to_code(self, config: Dict[str, Any]) -> str:
        horizon = config.get("horizon_periods", 14)
        seas = config.get("seasonality_mode", "additive")
        return f"from prophet import Prophet\n\nmodel = Prophet(seasonality_mode='{seas}')\nmodel.fit(df[['ds', 'y']])\nfuture = model.make_future_dataframe(periods={horizon})\nforecast = model.predict(future)"
