import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from backend.app.recipes.base.recipe import BaseRecipe


class TrainTestSplitRecipe(BaseRecipe):
    recipe_id = "train_test_split"
    name = "Train / Test Splitter"
    version = "1.1.0"
    category = "splitting"
    description = "Splits a dataset into training and testing subsets. Supports random split for tabular data and chronological split for time-series data to prevent future data leakage."
    input_types = ["dataframe"]
    output_types = ["train_data", "test_data"]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "target_column": {
                    "type": "string",
                    "title": "Target Variable (Y)",
                    "description": "The column to predict."
                },
                "test_size": {
                    "type": "number",
                    "title": "Test Split Ratio",
                    "default": 0.2,
                    "minimum": 0.05,
                    "maximum": 0.5
                },
                "random_state": {
                    "type": "integer",
                    "title": "Random Seed",
                    "default": 42
                },
                "stratify": {
                    "type": "boolean",
                    "title": "Stratified Split",
                    "default": False,
                    "description": "Preserve target class distribution in splits (classification only)."
                },
                "time_series_mode": {
                    "type": "boolean",
                    "title": "Time-Series Mode (Chronological Split)",
                    "default": False,
                    "description": (
                        "When enabled, sorts data by the time column and uses the chronologically latest "
                        "test_size fraction as the test set. Prevents data leakage from future observations "
                        "into the training set — required for any dataset with temporal features (year, date, period)."
                    )
                },
                "time_column": {
                    "type": "string",
                    "title": "Time / Date Column (for Time-Series Mode)",
                    "description": (
                        "Column to sort by for chronological splitting. Auto-detected if not specified "
                        "(searches for 'year', 'date', 'time', 'period', 'timestamp' in column names)."
                    )
                }
            },
            "required": ["target_column"]
        }

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        errors = []
        target_col = config.get("target_column")
        if not target_col or not str(target_col).strip() or str(target_col).strip() in ["-- Select Column --", "(None)"]:
            errors.append("Target variable 'target_column' (Y) is required for Train / Test Split and cannot be empty.")
        return errors

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        df: pd.DataFrame = inputs.get("dataframe")
        if df is None:
            raise ValueError("TrainTestSplit expects 'dataframe' in inputs.")

        target_col = config.get("target_column")
        if not target_col or not str(target_col).strip() or str(target_col).strip() in ["-- Select Column --", "(None)"]:
            raise ValueError(
                "Target variable 'target_column' (Y) is required for Train / Test Split, but was left empty. "
                "Please configure a valid target column to predict."
            )
        target_col = str(target_col).strip()
        if target_col not in df.columns:
            matching = [c for c in df.columns if c.lower() == target_col.lower()]
            if matching:
                target_col = matching[0]
            else:
                raise ValueError(
                    f"Specified target column '{target_col}' was not found in dataset columns: {list(df.columns)}. "
                    "Please select an existing column."
                )

        test_size        = float(config.get("test_size", 0.2))
        random_state     = int(config.get("random_state", 42))
        stratify_flag    = config.get("stratify", False)
        ts_mode          = bool(config.get("time_series_mode", False))
        time_col_cfg     = config.get("time_column", "") or ""

        X = df.drop(columns=[target_col])
        y = df[target_col]

        # ─────────────────────────────────────────────────────────
        # AUTO-DETECT TIME-SERIES MODE
        # If the user hasn't explicitly set time_series_mode but the
        # dataset clearly has temporal features, enable it automatically
        # to prevent data leakage.
        # ─────────────────────────────────────────────────────────
        temporal_keywords = ["year", "date", "time", "period", "timestamp", "month", "ds", "week"]
        if not ts_mode:
            detected_temporal_cols = [
                c for c in X.columns
                if any(kw in c.lower() for kw in temporal_keywords)
            ]
            if detected_temporal_cols:
                ts_mode = True

        # ─────────────────────────────────────────────────────────
        # TIME-SERIES CHRONOLOGICAL SPLIT (no random shuffle)
        # Sort the full dataframe by the time column, then take the
        # chronologically last `test_size` fraction as test.
        # ─────────────────────────────────────────────────────────
        if ts_mode:
            # Resolve time column
            time_col = None
            if time_col_cfg.strip() and time_col_cfg.strip() in df.columns:
                time_col = time_col_cfg.strip()
            else:
                # Auto-detect: prefer 'year', then 'date', then 'timestamp'
                for kw in temporal_keywords:
                    candidates = [c for c in X.columns if kw in c.lower()]
                    if candidates:
                        time_col = candidates[0]
                        break

            if time_col and time_col in df.columns:
                # Try numeric sort (year integers) then datetime
                try:
                    sort_series = pd.to_numeric(df[time_col], errors="coerce")
                    if sort_series.isna().mean() > 0.5:
                        sort_series = pd.to_datetime(df[time_col], errors="coerce")
                    sort_order = sort_series.argsort()
                    df_sorted = df.iloc[sort_order.values].reset_index(drop=True)
                except Exception:
                    df_sorted = df.copy()
            else:
                df_sorted = df.copy()

            n_total  = len(df_sorted)
            n_test   = max(1, int(n_total * test_size))
            n_train  = n_total - n_test

            train_df = df_sorted.iloc[:n_train]
            test_df  = df_sorted.iloc[n_train:]

            X_train = train_df.drop(columns=[target_col])
            y_train = train_df[target_col]
            X_test  = test_df.drop(columns=[target_col])
            y_test  = test_df[target_col]

            # Encode target labels if needed
            target_classes  = None
            target_encoder  = None
            if not pd.api.types.is_numeric_dtype(y_train):
                le = LabelEncoder()
                le.fit(pd.concat([y_train, y_test]).astype(str))
                y_train = pd.Series(le.transform(y_train.astype(str)), index=y_train.index, name=target_col)
                y_test  = pd.Series(le.transform(y_test.astype(str)),  index=y_test.index,  name=target_col)
                target_classes = [str(c) for c in le.classes_]
                target_encoder = le
            elif pd.api.types.is_float_dtype(y_train) and y_train.nunique() <= 10:
                y_train = y_train.astype(int)
                y_test  = y_test.astype(int)

            res = {
                "X_train":       X_train,
                "X_test":        X_test,
                "y_train":       y_train,
                "y_test":        y_test,
                "feature_names": list(X_train.columns),
                "target_column": target_col,
                "split_mode":    "chronological",
                "time_column":   time_col or "auto"
            }
            if target_classes:
                res["target_classes"] = target_classes
                res["target_encoder"] = target_encoder
            return res

        # ─────────────────────────────────────────────────────────
        # STANDARD RANDOM SPLIT (non-temporal tabular data)
        # ─────────────────────────────────────────────────────────
        # If y is categorical / string or continuous float in classification, cast or encode
        target_classes = None
        target_encoder = None
        if not pd.api.types.is_numeric_dtype(y):
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y.astype(str)), index=y.index, name=target_col)
            target_classes = [str(c) for c in le.classes_]
            target_encoder = le
        elif pd.api.types.is_float_dtype(y) and y.nunique() <= 10:
            # Discrete float classes like 0.0, 1.0 -> cast to integer
            y = y.astype(int)

        strat = y if (stratify_flag and y.nunique() > 1 and y.value_counts().min() > 1) else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=strat
        )

        res = {
            "X_train":       X_train,
            "X_test":        X_test,
            "y_train":       y_train,
            "y_test":        y_test,
            "feature_names": list(X.columns),
            "target_column": target_col,
            "split_mode":    "random"
        }
        if target_classes:
            res["target_classes"] = target_classes
            res["target_encoder"] = target_encoder
        return res
