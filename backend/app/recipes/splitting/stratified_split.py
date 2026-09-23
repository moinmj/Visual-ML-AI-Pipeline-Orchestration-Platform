import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.model_selection import train_test_split
from backend.app.recipes.base.recipe import BaseRecipe


class StratifiedSplitRecipe(BaseRecipe):
    recipe_id = "stratified_split"
    name = "Stratified Train / Test Splitter"
    version = "1.0.0"
    category = "splitting"
    description = "Partitions data while strictly preserving target class distribution between training and testing subsets (ideal for imbalanced classification)."
    input_types = ["dataframe"]
    output_types = ["train_data", "test_data"]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "target_column": {
                    "type": "string",
                    "title": "Target Variable (Y)",
                    "description": "The categorical target column to stratify on."
                },
                "test_size": {
                    "type": "number",
                    "title": "Test Split Ratio",
                    "default": 0.2,
                    "minimum": 0.05,
                    "maximum": 0.5,
                    "description": "Fraction of dataset allocated for testing (e.g. 0.2 for 80/20 train/test split)."
                },
                "random_state": {
                    "type": "integer",
                    "title": "Random Seed",
                    "default": 42,
                    "description": "Random seed for reproducible shuffling."
                }
            },
            "required": ["target_column"]
        }

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        errors = []
        target_col = config.get("target_column")
        if not target_col or not str(target_col).strip() or str(target_col).strip() in ["-- Select Column --", "(None)"]:
            errors.append("Target variable 'target_column' is required for Stratified Split and cannot be empty.")
        return errors

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        df: pd.DataFrame = inputs.get("dataframe")
        if df is None:
            if context and isinstance(context, dict) and "dataframe" in context:
                df = context["dataframe"]
            else:
                raise ValueError("StratifiedSplitRecipe expects 'dataframe' in inputs.")

        target_col = config.get("target_column")
        if not target_col or not str(target_col).strip() or str(target_col).strip() in ["-- Select Column --", "(None)"]:
            # Auto-detect target column from common keywords if missing
            candidates = [c for c in df.columns if any(k in c.lower() for k in ["target", "churn", "label", "survived", "class", "y"])]
            target_col = candidates[0] if candidates else df.columns[-1]

        target_col = str(target_col).strip() if target_col else ""
        if not target_col or target_col not in df.columns:
            matching = [c for c in df.columns if target_col and c.lower() == target_col.lower()]
            if matching:
                target_col = matching[0]
            else:
                candidates = [c for c in df.columns if any(k in c.lower() for k in ["sales", "weekly_sales", "price", "amount", "revenue", "demand", "target", "churn", "label", "class", "y"])]
                if candidates:
                    target_col = candidates[0]
                else:
                    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if not any(d in c.lower() for d in ["year", "date", "month", "day", "week"])]
                    target_col = num_cols[-1] if num_cols else df.columns[-1]

        test_size = float(config.get("test_size", 0.2))
        random_state = int(config.get("random_state", 42))

        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Check for rare classes with only 1 sample which would cause scikit-learn stratification to crash
        class_counts = y.value_counts()
        single_members = class_counts[class_counts < 2].index.tolist()

        if len(single_members) > 0 and len(class_counts) > len(single_members):
            # Duplicate single member rows once so stratify can partition them into train and test
            single_mask = y.isin(single_members)
            df_augmented = pd.concat([df, df[single_mask]], ignore_index=True)
            X = df_augmented.drop(columns=[target_col])
            y = df_augmented[target_col]

        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
        except Exception:
            # Fallback to standard train_test_split if target has too many continuous unique values
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

        # Standardize index
        X_tr = X_tr.reset_index(drop=True)
        X_te = X_te.reset_index(drop=True)
        y_tr = pd.Series(y_tr.values, name=target_col)
        y_te = pd.Series(y_te.values, name=target_col)

        df_train = pd.concat([X_tr, y_tr], axis=1)
        df_test = pd.concat([X_te, y_te], axis=1)

        return {
            "X_train": X_tr,
            "y_train": y_tr,
            "X_test": X_te,
            "y_test": y_te,
            "feature_names": list(X_tr.columns),
            "target_column": target_col,
            "split_mode": "stratified",
            "train_data": {
                "X_train": X_tr,
                "y_train": y_tr,
                "columns": list(X_tr.columns),
                "row_count": len(X_tr),
                "target_column": target_col
            },
            "test_data": {
                "X_test": X_te,
                "y_test": y_te,
                "columns": list(X_te.columns),
                "row_count": len(X_te),
                "target_column": target_col
            },
            "dataframe_train": df_train,
            "dataframe_test": df_test,
            "metadata": {
                "split_mode": "stratified",
                "test_size": test_size,
                "train_rows": len(X_tr),
                "test_rows": len(X_te)
            }
        }

    def to_code(self, config: Dict[str, Any]) -> str:
        tgt = config.get("target_column", "target")
        ts = config.get("test_size", 0.2)
        seed = config.get("random_state", 42)
        return (
            f"# Stratified Train / Test Split (Preserves class balance)\n"
            f"from sklearn.model_selection import train_test_split\n\n"
            f"X = df.drop(columns=['{tgt}'])\n"
            f"y = df['{tgt}']\n"
            f"X_train, X_test, y_train, y_test = train_test_split(\n"
            f"    X, y, test_size={ts}, random_state={seed}, stratify=y\n"
            f")"
        )
