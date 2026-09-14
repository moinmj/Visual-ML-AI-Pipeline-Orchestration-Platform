import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from backend.app.recipes.base.recipe import BaseRecipe


class TimeSeriesSplitRecipe(BaseRecipe):
    recipe_id = "time_series_split"
    name = "Time-Series Chronological Splitter"
    version = "1.0.0"
    category = "splitting"
    description = "Partitions data chronologically by date/timestamp to prevent future data leakage (train on the past, test on the future)."
    input_types = ["dataframe"]
    output_types = ["train_data", "test_data"]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "target_column": {
                    "type": "string",
                    "title": "Target Variable (Y)",
                    "description": "The metric or column to forecast/predict."
                },
                "date_column": {
                    "type": "string",
                    "title": "Date / Timestamp Column",
                    "description": "Chronological column used to order observations. If left empty, automatically inferred from column names and types."
                },
                "test_size": {
                    "type": "number",
                    "title": "Test Split Ratio",
                    "default": 0.2,
                    "minimum": 0.05,
                    "maximum": 0.5,
                    "description": "Fraction of the latest chronological observations reserved for testing."
                },
                "gap": {
                    "type": "integer",
                    "title": "Purge / Embargo Gap",
                    "default": 0,
                    "minimum": 0,
                    "description": "Number of observations to drop between the end of train and start of test to eliminate lag autocorrelation leakage."
                }
            },
            "required": ["target_column"]
        }

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        errors = []
        target_col = config.get("target_column")
        if not target_col or not str(target_col).strip() or str(target_col).strip() in ["-- Select Column --", "(None)"]:
            errors.append("Target variable 'target_column' is required for Time-Series Split and cannot be empty.")
        return errors

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        df: pd.DataFrame = inputs.get("dataframe")
        if df is None:
            if context and isinstance(context, dict) and "dataframe" in context:
                df = context["dataframe"]
            else:
                raise ValueError("TimeSeriesSplitRecipe expects 'dataframe' in inputs.")

        target_col = config.get("target_column")
        if not target_col or not str(target_col).strip() or str(target_col).strip() in ["-- Select Column --", "(None)"]:
            candidates = [c for c in df.columns if any(k in c.lower() for k in ["target", "sales", "revenue", "demand", "price", "value", "y"])]
            target_col = candidates[0] if candidates else df.columns[-1]

        target_col = str(target_col).strip()
        if target_col not in df.columns:
            matching = [c for c in df.columns if c.lower() == target_col.lower()]
            if matching:
                target_col = matching[0]
            else:
                raise ValueError(f"Specified target column '{target_col}' not found in dataset columns: {list(df.columns)}")

        test_size = float(config.get("test_size", 0.2))
        gap = max(0, int(config.get("gap", 0)))
        time_col = (config.get("date_column") or "").strip()

        # 1. Resolve date / chronological column
        if not time_col or time_col not in df.columns:
            temporal_keywords = ["year", "date", "time", "period", "timestamp", "month", "ds", "week", "day"]
            candidates = [c for c in df.columns if c != target_col and any(kw in c.lower() for kw in temporal_keywords)]
            if candidates:
                time_col = candidates[0]
            else:
                # Check for datetime dtypes
                dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
                time_col = dt_cols[0] if dt_cols else None

        # 2. Sort chronologically
        if time_col and time_col in df.columns:
            try:
                num_s = pd.to_numeric(df[time_col], errors="coerce")
                if num_s.isna().mean() > 0.5:
                    dt_s = pd.to_datetime(df[time_col], errors="coerce")
                    order = dt_s.argsort()
                else:
                    order = num_s.argsort()
                df_sorted = df.iloc[order.values].reset_index(drop=True)
            except Exception:
                df_sorted = df.copy().reset_index(drop=True)
        else:
            df_sorted = df.copy().reset_index(drop=True)

        n_total = len(df_sorted)
        n_test = max(1, int(n_total * test_size))
        n_train = max(1, n_total - n_test - gap)

        df_train = df_sorted.iloc[:n_train].reset_index(drop=True)
        df_test = df_sorted.iloc[n_total - n_test:].reset_index(drop=True)

        X_tr = df_train.drop(columns=[target_col])
        y_tr = df_train[target_col]
        X_te = df_test.drop(columns=[target_col])
        y_te = df_test[target_col]

        return {
            "X_train": X_tr,
            "y_train": y_tr,
            "X_test": X_te,
            "y_test": y_te,
            "feature_names": list(X_tr.columns),
            "target_column": target_col,
            "time_column": time_col or "auto",
            "split_mode": "chronological",
            "train_data": {
                "X_train": X_tr,
                "y_train": y_tr,
                "columns": list(X_tr.columns),
                "row_count": len(X_tr),
                "target_column": target_col,
                "time_column": time_col
            },
            "test_data": {
                "X_test": X_te,
                "y_test": y_te,
                "columns": list(X_te.columns),
                "row_count": len(X_te),
                "target_column": target_col,
                "time_column": time_col
            },
            "dataframe_train": df_train,
            "dataframe_test": df_test,
            "metadata": {
                "split_mode": "chronological",
                "gap": gap,
                "test_size": test_size,
                "train_rows": len(X_tr),
                "test_rows": len(X_te),
                "time_column": time_col
            }
        }

    def to_code(self, config: Dict[str, Any]) -> str:
        tgt = config.get("target_column", "target")
        time_col = config.get("date_column", "date")
        ts = config.get("test_size", 0.2)
        gap = config.get("gap", 0)
        return (
            f"# Time-Series Chronological Split (No random shuffle / No future data leakage)\n"
            f"df = df.sort_values(by='{time_col}').reset_index(drop=True)\n"
            f"n_test = int(len(df) * {ts})\n"
            f"n_train = len(df) - n_test - {gap}\n"
            f"df_train = df.iloc[:n_train]\n"
            f"df_test = df.iloc[len(df) - n_test:]\n"
            f"X_train, y_train = df_train.drop(columns=['{tgt}']), df_train['{tgt}']\n"
            f"X_test, y_test = df_test.drop(columns=['{tgt}']), df_test['{tgt}']"
        )
