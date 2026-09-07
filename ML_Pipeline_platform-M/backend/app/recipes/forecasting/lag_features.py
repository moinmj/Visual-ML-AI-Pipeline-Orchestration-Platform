import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from backend.app.recipes.base.recipe import BaseRecipe


class LagFeatureEngineeringRecipe(BaseRecipe):
    recipe_id = "lag_feature_engineering"
    name = "Lag & Time Feature Engineer"
    version = "1.0.0"
    category = "forecasting"
    description = "Extracts time-series lag features (t-1..t-n), rolling window aggregations, and calendar attributes to enable Tree-Based Forecasting."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "date_column": {
                    "type": "string",
                    "title": "Date / Timestamp Column",
                    "description": "The datetime column representing the chronological axis."
                },
                "target_column": {
                    "type": "string",
                    "title": "Target Series Column",
                    "description": "The numerical metric to forecast (e.g. Sales, Demand)."
                },
                "lag_periods": {
                    "type": "string",
                    "title": "Lag Periods (Comma-separated)",
                    "default": "1, 2, 3, 7, 14",
                    "description": "Past time steps to create as input features (e.g. '1, 2, 7')."
                },
                "rolling_windows": {
                    "type": "string",
                    "title": "Rolling Windows (Comma-separated)",
                    "default": "7, 14",
                    "description": "Window sizes for rolling mean/std features (e.g. '7, 14')."
                },
                "include_calendar_features": {
                    "type": "boolean",
                    "title": "Extract Calendar Features (Day, Month, Weekend)",
                    "default": True
                }
            }
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        df: pd.DataFrame = inputs.get("dataframe")
        if df is None:
            if context and isinstance(context, dict) and "dataframe" in context:
                df = context["dataframe"]
            else:
                raise ValueError("LagFeatureEngineering expects 'dataframe' in inputs.")

        df_out = df.copy()

        # 1. Identify Date Column
        date_col = config.get("date_column")
        if not date_col or date_col not in df_out.columns:
            # Auto-detect date column
            date_candidates = [c for c in df_out.columns if "date" in c.lower() or "time" in c.lower() or "timestamp" in c.lower() or pd.api.types.is_datetime64_any_dtype(df_out[c])]
            date_col = date_candidates[0] if date_candidates else df_out.columns[0]

        # Convert to datetime and sort
        df_out[date_col] = pd.to_datetime(df_out[date_col], errors="coerce")
        df_out = df_out.sort_values(by=date_col).reset_index(drop=True)

        # 2. Identify Target Column
        target_col = config.get("target_column")
        if not target_col or target_col not in df_out.columns:
            num_cols = [c for c in df_out.columns if pd.api.types.is_numeric_dtype(df_out[c]) and c != date_col]
            target_col = num_cols[-1] if num_cols else df_out.columns[-1]

        # 3. Create Lag Features
        lag_str = str(config.get("lag_periods", "1, 2, 3, 7, 14"))
        lags = [int(p.strip()) for p in lag_str.split(",") if p.strip().isdigit()]
        for lag in lags:
            df_out[f"{target_col}_lag_{lag}"] = df_out[target_col].shift(lag)

        # 4. Create Rolling Window Features
        roll_str = str(config.get("rolling_windows", "7, 14"))
        windows = [int(w.strip()) for w in roll_str.split(",") if w.strip().isdigit()]
        for w in windows:
            df_out[f"{target_col}_roll_mean_{w}"] = df_out[target_col].shift(1).rolling(window=w, min_periods=1).mean()
            df_out[f"{target_col}_roll_std_{w}"] = df_out[target_col].shift(1).rolling(window=w, min_periods=1).std().fillna(0)

        # 5. Extract Calendar Attributes
        if config.get("include_calendar_features", True):
            dt_series = df_out[date_col]
            df_out["cal_dayofweek"] = dt_series.dt.dayofweek
            df_out["cal_month"] = dt_series.dt.month
            df_out["cal_day"] = dt_series.dt.day
            df_out["cal_is_weekend"] = dt_series.dt.dayofweek.isin([5, 6]).astype(int)

        # Forward fill or drop leading NaNs caused by lags
        df_out = df_out.bfill().fillna(0)

        return {
            "dataframe": df_out,
            "date_column": date_col,
            "target_column": target_col,
            "feature_names": [c for c in df_out.columns if c not in [date_col, target_col]]
        }
