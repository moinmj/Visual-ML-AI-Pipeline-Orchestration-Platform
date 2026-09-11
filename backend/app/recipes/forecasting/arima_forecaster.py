import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from backend.app.recipes.base.recipe import BaseRecipe

try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False


class ARIMAForecasterRecipe(BaseRecipe):
    recipe_id = "arima_forecaster"
    name = "ARIMA Statistical Forecaster"
    version = "1.1.0"
    category = "forecasting"
    description = "Classical ARIMA for statistical time-series forecasting with chronological out-of-sample backtesting (Tier-1 Baseline)."
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
            errors.append("Target variable 'target_column' is required for ARIMA Forecaster and cannot be empty.")
        return errors

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

        # 1. Validate and resolve target column
        target_col = config.get("target_column") or inputs.get("target_column")
        if not target_col or not str(target_col).strip() or str(target_col).strip() in ["-- Select Column --", "(None)"]:
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if num_cols:
                target_col = num_cols[-1]
            else:
                raise ValueError(
                    "Target variable 'target_column' is required for ARIMA Forecaster, but was left empty. "
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
            # Search for any valid datetime column across dataframe
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
                "No valid date/datetime column found in dataset for ARIMA forecasting. "
                "Please ensure your dataset contains a valid timestamp column and configure 'date_column'."
            )

        p = int(config.get("p", 1))
        d = int(config.get("d", 1))
        q = int(config.get("q", 1))
        horizon = int(config.get("horizon_periods", 14))
        test_size_pct = float(config.get("test_size_pct", 0.2))

        ts_df = pd.DataFrame({
            "ds": valid_ds,
            "y": pd.to_numeric(df[target_col], errors="coerce")
        }).dropna().sort_values(by="ds").reset_index(drop=True)

        min_obs = p + d + q + 3
        if len(ts_df) < min_obs:
            raise ValueError(f"ARIMA({p},{d},{q}) requires at least {min_obs} observations, found {len(ts_df)}.")

        # ─────────────────────────────────────────────────────────────
        # CHRONOLOGICAL OUT-OF-SAMPLE BACKTESTING
        # Split last `test_size_pct` as unseen holdout, fit ARIMA on
        # training slice, forecast the exact number of test steps,
        # compute OOS metrics. Then refit on full data for production.
        # ─────────────────────────────────────────────────────────────
        n = len(ts_df)
        min_test_pts = max(2, p + d + q + 1)
        split_idx = max(min_test_pts, int(n * (1.0 - test_size_pct)))

        eval_type = "out_of_sample_holdout"
        train_series = ts_df["y"].values[:split_idx]
        test_series  = ts_df["y"].values[split_idx:]

        if len(test_series) < 2:
            train_series = ts_df["y"].values
            test_series  = ts_df["y"].values
            eval_type    = "in_sample_fallback"

        # --- Phase 1: Fit on training slice, forecast test horizon ---
        try:
            eval_arima        = ARIMA(train_series, order=(p, d, q))
            eval_fitted        = eval_arima.fit()
            oos_forecast_res   = eval_fitted.get_forecast(steps=len(test_series))
            oos_preds          = oos_forecast_res.predicted_mean

            oos_actuals = test_series
            mae  = float(round(np.mean(np.abs(oos_actuals - oos_preds)), 4))
            rmse = float(round(np.sqrt(np.mean((oos_actuals - oos_preds) ** 2)), 4))

            non_zero = oos_actuals != 0
            if np.any(non_zero):
                mape = float(round(
                    np.mean(np.abs((oos_actuals[non_zero] - oos_preds[non_zero]) / oos_actuals[non_zero])) * 100,
                    2
                ))
            else:
                mape = 0.0
        except Exception:
            # Graceful fallback — in-sample residuals if OOS fitting fails
            eval_type   = "in_sample_fallback"
            eval_arima  = ARIMA(ts_df["y"].values, order=(p, d, q))
            eval_fitted  = eval_arima.fit()
            fv           = eval_fitted.fittedvalues
            oos_actuals  = ts_df["y"].values[d:]
            oos_preds    = fv[d:]
            mae          = float(round(np.mean(np.abs(oos_actuals - oos_preds)), 4))
            rmse         = float(round(np.sqrt(np.mean((oos_actuals - oos_preds) ** 2)), 4))
            non_zero     = oos_actuals != 0
            mape         = float(round(np.mean(np.abs(
                (oos_actuals[non_zero] - oos_preds[non_zero]) / oos_actuals[non_zero]
            )) * 100, 2)) if np.any(non_zero) else 0.0

        # --- Phase 2: Refit on FULL dataset for production forecast ---
        full_arima   = ARIMA(ts_df["y"].values, order=(p, d, q))
        fitted_model = full_arima.fit()

        forecast_res  = fitted_model.get_forecast(steps=horizon)
        future_means  = forecast_res.predicted_mean
        conf_int      = forecast_res.conf_int(alpha=0.05)
        fitted_values = fitted_model.fittedvalues

        # Generate future dates
        last_date    = ts_df["ds"].iloc[-1]
        freq         = pd.infer_freq(ts_df["ds"]) or "D"
        future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=freq)[1:]

        metrics = {
            "task_type":          "time_series_forecasting",
            "algorithm":          f"ARIMA({p},{d},{q})",
            "eval_type":          eval_type,
            "aic":                float(round(fitted_model.aic, 2)),
            "bic":                float(round(fitted_model.bic, 2)),
            "historical_points":  len(ts_df),
            "train_size":         split_idx,
            "test_size":          n - split_idx,
            "holdout_test_start": str(ts_df["ds"].iloc[split_idx].date()) if split_idx < n else None,
            "holdout_test_end":   str(ts_df["ds"].iloc[-1].date()),
            "forecast_horizon":   horizon,
            "trend_direction":    "Upward" if float(future_means[-1]) >= float(ts_df["y"].iloc[-1]) else "Downward",
            "horizon_periods":    horizon,
            # ── True out-of-sample accuracy metrics ──
            "mae":                mae,
            "rmse":               rmse,
            "mape":               mape,
            "evaluation_note":    (
                f"Metrics computed on chronologically held-out last {n - split_idx} observations "
                f"({test_size_pct*100:.0f}% of data). "
                f"Model was retrained on all {len(ts_df)} points for future forecasting."
                if eval_type == "out_of_sample_holdout"
                else "Dataset too small for holdout — metrics are in-sample."
            )
        }

        # Build clean visualization table (historical fitted + future forecast)
        hist_df = pd.DataFrame({
            "ds":         ts_df["ds"],
            "yhat":       np.round(fitted_values, 2),
            "yhat_lower": np.round(fitted_values, 2),
            "yhat_upper": np.round(fitted_values, 2),
            "is_future":  0
        })

        fut_df = pd.DataFrame({
            "ds":         future_dates,
            "yhat":       np.round(future_means, 2),
            # conf_int may be a DataFrame or ndarray depending on statsmodels version
            "yhat_lower": np.round(
                conf_int.iloc[:, 0].values if hasattr(conf_int, "iloc") else conf_int[:, 0],
                2
            ),
            "yhat_upper": np.round(
                conf_int.iloc[:, 1].values if hasattr(conf_int, "iloc") else conf_int[:, 1],
                2
            ),
            "is_future":  1
        })

        full_forecast_df = pd.concat([hist_df, fut_df], ignore_index=True)
        if hasattr(full_forecast_df["ds"].dt, "strftime"):
            full_forecast_df["ds"] = full_forecast_df["ds"].dt.strftime("%Y-%m-%d")
        else:
            full_forecast_df["ds"] = full_forecast_df["ds"].astype(str)

        # Embed forecast points into metrics for lightweight charting
        metrics["forecast_data"] = full_forecast_df.to_dict(orient="records")

        return {
            "forecast_df":         full_forecast_df,
            "dataframe":           full_forecast_df,
            "metrics":             metrics,
            "forecasting_summary": metrics,
            "model":               fitted_model,
            "task_type":           "time_series_forecasting"
        }
