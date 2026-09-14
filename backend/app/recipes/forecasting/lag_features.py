import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from backend.app.recipes.base.recipe import BaseRecipe


class LagFeatureEngineeringRecipe(BaseRecipe):
    recipe_id = "lag_feature_engineering"
    name = "Lag & Time Feature Engineer"
    version = "1.1.0"
    category = "forecasting"
    description = "Extracts time-series lag features (t-1..t-n), rolling window aggregations, and calendar attributes with per-entity grouping and safe NaN handling."
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
                    "description": "The numerical metric to forecast (e.g. Weekly_Sales, Demand)."
                },
                "group_by_column": {
                    "type": "string",
                    "title": "Group / Entity Column (e.g. Store, SKU)",
                    "description": "Optional column to compute lags and rolling stats independently per entity (prevents cross-store leakage).",
                    "default": ""
                },
                "lag_periods": {
                    "type": "string",
                    "title": "Lag Periods (Comma-separated)",
                    "default": "1, 2, 3, 7, 14, 52",
                    "description": "Past time steps to create as input features (e.g. '1, 2, 7, 52')."
                },
                "rolling_windows": {
                    "type": "string",
                    "title": "Rolling Windows (Comma-separated)",
                    "default": "7, 14, 52",
                    "description": "Window sizes for rolling mean/std features (e.g. '7, 14, 52')."
                },
                "handle_na": {
                    "type": "string",
                    "title": "NaN Lag Handling Strategy",
                    "enum": ["drop_rows", "keep_nan", "fillna_zero", "bfill"],
                    "default": "drop_rows",
                    "description": "'drop_rows' discards initial unobserved lag periods (e.g. first 52 weeks); 'keep_nan' keeps NaNs for downstream imputer or native LightGBM handling; 'fillna_zero' fills with 0; 'bfill' backfills."
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
            date_candidates = [c for c in df_out.columns if "date" in c.lower() or "time" in c.lower() or "timestamp" in c.lower() or pd.api.types.is_datetime64_any_dtype(df_out[c])]
            date_col = date_candidates[0] if date_candidates else df_out.columns[0]

        df_out[date_col] = pd.to_datetime(df_out[date_col], errors="coerce")

        # Optional Group / Entity Column (e.g. Store)
        group_col = config.get("group_by_column")
        if group_col and str(group_col).strip() in df_out.columns:
            group_col = str(group_col).strip()
            df_out = df_out.sort_values(by=[group_col, date_col]).reset_index(drop=True)
        else:
            group_col = None
            df_out = df_out.sort_values(by=date_col).reset_index(drop=True)

        # 2. Identify Target Column
        target_col = config.get("target_column")
        if not target_col or target_col not in df_out.columns:
            num_cols = [c for c in df_out.columns if pd.api.types.is_numeric_dtype(df_out[c]) and c not in [date_col, group_col]]
            target_col = num_cols[-1] if num_cols else df_out.columns[-1]

        created_features = []

        # 3. Create Lag Features
        lag_str = str(config.get("lag_periods", "1, 2, 3, 7, 14, 52"))
        lags = [int(p.strip()) for p in lag_str.split(",") if p.strip().isdigit()]
        for lag in lags:
            col_name = f"{target_col}_lag_{lag}"
            if group_col:
                df_out[col_name] = df_out.groupby(group_col)[target_col].shift(lag)
            else:
                df_out[col_name] = df_out[target_col].shift(lag)
            created_features.append(col_name)

        # 4. Create Rolling Window Features
        roll_str = str(config.get("rolling_windows", "7, 14, 52"))
        windows = [int(w.strip()) for w in roll_str.split(",") if w.strip().isdigit()]
        for w in windows:
            mean_col = f"{target_col}_roll_mean_{w}"
            std_col = f"{target_col}_roll_std_{w}"
            if group_col:
                df_out[mean_col] = df_out.groupby(group_col)[target_col].transform(
                    lambda s: s.shift(1).rolling(window=w, min_periods=w).mean()
                )
                df_out[std_col] = df_out.groupby(group_col)[target_col].transform(
                    lambda s: s.shift(1).rolling(window=w, min_periods=w).std()
                )
            else:
                df_out[mean_col] = df_out[target_col].shift(1).rolling(window=w, min_periods=w).mean()
                df_out[std_col] = df_out[target_col].shift(1).rolling(window=w, min_periods=w).std()
            created_features.extend([mean_col, std_col])

        # 5. Extract Calendar Attributes
        if config.get("include_calendar_features", True):
            dt_series = df_out[date_col]
            df_out["cal_dayofweek"] = dt_series.dt.dayofweek
            df_out["cal_month"] = dt_series.dt.month
            df_out["cal_day"] = dt_series.dt.day
            df_out["cal_is_weekend"] = dt_series.dt.dayofweek.isin([5, 6]).astype(int)

        # 6. Safe NaN Lag Handling
        handle_na = config.get("handle_na", "drop_rows")
        if handle_na == "drop_rows":
            # Cleanly drop structural unobserved lag periods (e.g. first 52 weeks of each store)
            df_out = df_out.dropna(subset=created_features).reset_index(drop=True)
        elif handle_na == "keep_nan":
            # Retain NaNs for downstream imputer or native LightGBM NaN handling
            pass
        elif handle_na == "fillna_zero":
            df_out[created_features] = df_out[created_features].fillna(0)
        elif handle_na == "bfill":
            if group_col:
                df_out[created_features] = df_out.groupby(group_col)[created_features].bfill().fillna(0)
            else:
                df_out[created_features] = df_out[created_features].bfill().fillna(0)

        return {
            "dataframe": df_out,
            "date_column": date_col,
            "target_column": target_col,
            "group_by_column": group_col,
            "feature_names": [c for c in df_out.columns if c not in [date_col, target_col]],
            "lag_features": created_features,
            "rows_before": len(df),
            "rows_after": len(df_out),
            "rows_dropped": len(df) - len(df_out)
        }

    def to_code(self, config: Dict[str, Any]) -> str:
        date_col = config.get("date_column", "Date")
        target_col = config.get("target_column", "target")
        group_col = config.get("group_by_column", "")
        lag_str = config.get("lag_periods", "1, 2, 7, 52")
        roll_str = config.get("rolling_windows", "7, 14, 52")
        handle_na = config.get("handle_na", "drop_rows")

        lines = [
            "# Lag & Temporal Feature Engineering",
            f"df['{date_col}'] = pd.to_datetime(df['{date_col}'], errors='coerce')",
        ]
        if group_col:
            lines.append(f"df = df.sort_values(by=['{group_col}', '{date_col}']).reset_index(drop=True)")
            lines.append(f"# Per-entity group lags: {lag_str}")
            lines.append(f"for lag in [{lag_str}]:")
            lines.append(f"    df[f'{target_col}_lag_{{lag}}'] = df.groupby('{group_col}')['{target_col}'].shift(lag)")
        else:
            lines.append(f"df = df.sort_values(by='{date_col}').reset_index(drop=True)")
            lines.append(f"# Time series lags: {lag_str}")
            lines.append(f"for lag in [{lag_str}]:")
            lines.append(f"    df[f'{target_col}_lag_{{lag}}'] = df['{target_col}'].shift(lag)")

        if handle_na == "drop_rows":
            lines.append("# Drop rows with unobserved initial lag periods")
            lines.append(f"lag_cols = [c for c in df.columns if c.startswith('{target_col}_lag_') or c.startswith('{target_col}_roll_')]")
            lines.append("df = df.dropna(subset=lag_cols).reset_index(drop=True)")
        return "\n".join(lines)
