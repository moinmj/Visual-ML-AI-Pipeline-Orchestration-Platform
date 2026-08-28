import re
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from backend.app.recipes.base.recipe import BaseRecipe

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except (ImportError, OSError, Exception):
    lgb = None
    LIGHTGBM_AVAILABLE = False


def sanitize_lgb_colnames(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitizes column names to be strictly JSON-safe for LightGBM."""
    clean_cols = [re.sub(r'[\[\]\{\}:",\s]', '_', str(c)) for c in df.columns]
    df_clean = df.copy()
    df_clean.columns = clean_cols
    return df_clean


class LightGBMTrainerRecipe(BaseRecipe):
    recipe_id = "lightgbm_trainer"
    name = "LightGBM Classifier / Regressor"
    version = "1.0.0"
    category = "training"
    description = "Fast, distributed, high-performance gradient boosting framework utilizing leaf-wise tree growth and GOSS (Tier-1 Enterprise Standard)."
    input_types = ["train_data"]
    output_types = ["model"]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task_type": {
                    "type": "string",
                    "title": "ML Task Type",
                    "enum": ["classification", "regression"],
                    "default": "classification"
                },
                "n_estimators": {
                    "type": "integer",
                    "title": "Number of Trees (n_estimators)",
                    "default": 100,
                    "minimum": 10,
                    "maximum": 2000
                },
                "num_leaves": {
                    "type": "integer",
                    "title": "Max Tree Leaves (num_leaves)",
                    "default": 31,
                    "minimum": 2,
                    "maximum": 256
                },
                "learning_rate": {
                    "type": "number",
                    "title": "Learning Rate",
                    "default": 0.1,
                    "minimum": 0.001,
                    "maximum": 1.0
                },
                "random_state": {
                    "type": "integer",
                    "title": "Random Seed",
                    "default": 42
                }
            },
            "required": ["task_type"]
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        if not LIGHTGBM_AVAILABLE:
            raise ValueError("LightGBM is not installed. Please run 'pip install lightgbm'.")

        X_train = inputs.get("X_train")
        y_train = inputs.get("y_train")
        X_test = inputs.get("X_test")
        y_test = inputs.get("y_test")

        if X_train is None or y_train is None:
            raise ValueError("LightGBMTrainer expects 'X_train' and 'y_train' in inputs. Please connect a Train/Test Split node before this trainer.")

        from backend.app.recipes.training.encoder_utils import safe_prepare_training_data
        X_train, X_test = safe_prepare_training_data(X_train, X_test)

        # Edge Case 2: Sanitize column names for LightGBM (replaces [, ], {, }, :, ", spaces)
        X_train = sanitize_lgb_colnames(X_train)
        if X_test is not None:
            X_test = sanitize_lgb_colnames(X_test)

        task_type = config.get("task_type", "classification")
        n_estimators = int(config.get("n_estimators", 100))
        num_leaves = int(config.get("num_leaves", 31))
        lr = float(config.get("learning_rate", 0.1))
        random_state = int(config.get("random_state", 42))

        if task_type == "classification":
            model = lgb.LGBMClassifier(
                n_estimators=n_estimators,
                num_leaves=num_leaves,
                learning_rate=lr,
                random_state=random_state,
                verbose=-1
            )
        else:
            model = lgb.LGBMRegressor(
                n_estimators=n_estimators,
                num_leaves=num_leaves,
                learning_rate=lr,
                random_state=random_state,
                verbose=-1
            )

        model.fit(X_train, y_train)

        # Feature importances
        feature_names = list(X_train.columns)
        importances = {}
        if hasattr(model, "feature_importances_") and feature_names:
            for feat, imp in zip(feature_names, model.feature_importances_):
                importances[feat] = float(round(imp, 4))

        output = {
            "model": model,
            "task_type": task_type,
            "feature_importances": importances,
            "feature_names": feature_names
        }

        if X_test is not None:
            output["X_test"] = X_test
        if y_test is not None:
            output["y_test"] = y_test

        return output
