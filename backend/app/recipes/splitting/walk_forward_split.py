import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from backend.app.recipes.base.recipe import BaseRecipe
from backend.app.recipes.training.encoder_utils import find_date_column, extract_test_dates_from_column


class WalkForwardSplitRecipe(BaseRecipe):
    recipe_id = "walk_forward_split"
    name = "Walk-Forward Rolling Window Splitter"
    version = "1.0.0"
    category = "splitting"
    description = "Partitions sequential/temporal data into sliding rolling windows to simulate real-world production backtesting and continuous retraining."
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
                    "description": "Chronological column used to order observations. Auto-inferred if left blank."
                },
                "window_size": {
                    "type": "integer",
                    "title": "Training Window Size (Rows)",
                    "default": 100,
                    "minimum": 5,
                    "description": "Fixed number of historical observations included in each rolling training window."
                },
                "test_step_size": {
                    "type": "integer",
                    "title": "Test Horizon Size (Rows)",
                    "default": 20,
                    "minimum": 1,
                    "description": "Number of consecutive future observations evaluated in each forward step."
                }
            },
            "required": ["target_column"]
        }

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        errors = []
        target_col = config.get("target_column")
        if not target_col or not str(target_col).strip() or str(target_col).strip() in ["-- Select Column --", "(None)"]:
            errors.append("Target variable 'target_column' is required for Walk-Forward Split and cannot be empty.")
        return errors

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        df: pd.DataFrame = inputs.get("dataframe")
        if df is None:
            if context and isinstance(context, dict) and "dataframe" in context:
                df = context["dataframe"]
            else:
                raise ValueError("WalkForwardSplitRecipe expects 'dataframe' in inputs.")

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

        time_col = (config.get("date_column") or "").strip()
        req_window = max(5, int(config.get("window_size", 100)))
        req_step = max(1, int(config.get("test_step_size", 20)))

        # 1. Resolve date column
        if not time_col or time_col not in df.columns:
            temporal_keywords = ["year", "date", "time", "period", "timestamp", "month", "ds", "week", "day"]
            candidates = [c for c in df.columns if c != target_col and any(kw in c.lower() for kw in temporal_keywords)]
            if candidates:
                time_col = candidates[0]
            else:
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

        # Scale window/step if dataset size is smaller than defaults
        effective_step = min(req_step, max(1, int(n_total * 0.2)))
        effective_window = min(req_window, max(2, n_total - effective_step))

        # Most recent sliding window partition
        test_end = n_total
        test_start = max(effective_window, n_total - effective_step)
        train_start = max(0, test_start - effective_window)
        train_end = test_start

        df_train = df_sorted.iloc[train_start:train_end].reset_index(drop=True)
        df_test = df_sorted.iloc[test_start:test_end].reset_index(drop=True)

        X_tr = df_train.drop(columns=[target_col])
        y_tr = df_train[target_col]
        X_te = df_test.drop(columns=[target_col])
        y_te = df_test[target_col]

        # Calculate number of sliding folds possible across the historical dataset
        stride = effective_step
        num_folds = max(1, (n_total - effective_window) // stride) if stride > 0 else 1

        _wf_result = {
            "X_train": X_tr,
            "y_train": y_tr,
            "X_test": X_te,
            "y_test": y_te,
            "feature_names": list(X_tr.columns),
            "target_column": target_col,
            "time_column": time_col or "auto",
            "split_mode": "walk_forward",
            "train_data": {
                "X_train": X_tr,
                "y_train": y_tr,
                "columns": list(X_tr.columns),
                "row_count": len(X_tr),
                "target_column": target_col,
                "time_column": time_col,
                "rolling_folds_count": num_folds,
                "window_size": effective_window
            },
            "test_data": {
                "X_test": X_te,
                "y_test": y_te,
                "columns": list(X_te.columns),
                "row_count": len(X_te),
                "target_column": target_col,
                "time_column": time_col,
                "step_size": effective_step
            },
            "dataframe_train": df_train,
            "dataframe_test": df_test,
            "metadata": {
                "split_mode": "walk_forward",
                "train_window_size": effective_window,
                "test_step_size": effective_step,
                "total_possible_folds": num_folds,
                "time_column": time_col
            }
        }

        # ── Thread date sidecar to evaluator ────────────────────────────────
        # extract_test_dates_from_column rejects integer year columns.
        _wf_test_dates = None
        _wf_date_col = None

        if time_col and time_col in df_test.columns:
            _wf_test_dates = extract_test_dates_from_column(df_test[time_col].reset_index(drop=True))
            if _wf_test_dates is not None:
                _wf_date_col = time_col

        if _wf_test_dates is None:
            _real_date_col = find_date_column(df_test, exclude_cols=[target_col])
            if _real_date_col:
                _wf_test_dates = extract_test_dates_from_column(df_test[_real_date_col].reset_index(drop=True))
                if _wf_test_dates is not None:
                    _wf_date_col = _real_date_col

        if _wf_test_dates is not None:
            _wf_result["test_dates"] = _wf_test_dates
            _wf_result["date_column_name"] = _wf_date_col

        return _wf_result


    def to_code(self, config: Dict[str, Any]) -> str:
        tgt = config.get("target_column", "target")
        time_col = config.get("date_column", "date")
        win = config.get("window_size", 100)
        step = config.get("test_step_size", 20)
        return (
            f"# Walk-Forward Rolling Window Split (Continuous production simulation)\n"
            f"df = df.sort_values(by='{time_col}').reset_index(drop=True)\n"
            f"test_start = len(df) - {step}\n"
            f"train_start = max(0, test_start - {win})\n"
            f"df_train = df.iloc[train_start:test_start]\n"
            f"df_test = df.iloc[test_start:]\n"
            f"X_train, y_train = df_train.drop(columns=['{tgt}']), df_train['{tgt}']\n"
            f"X_test, y_test = df_test.drop(columns=['{tgt}']), df_test['{tgt}']"
        )
