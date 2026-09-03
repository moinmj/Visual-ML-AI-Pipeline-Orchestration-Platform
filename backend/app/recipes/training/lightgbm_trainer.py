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
    version = "1.1.0"
    category = "training"
    description = "Fast, distributed gradient boosting utilizing leaf-wise tree growth with full regularization, subsampling, and class imbalance tuning."
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
                "subsample": {
                    "type": "number",
                    "title": "Row Subsample Ratio",
                    "default": 1.0,
                    "minimum": 0.3,
                    "maximum": 1.0,
                    "description": "Fraction of data to be used per iteration (bagging_fraction)."
                },
                "colsample_bytree": {
                    "type": "number",
                    "title": "Feature Subsample Ratio",
                    "default": 1.0,
                    "minimum": 0.3,
                    "maximum": 1.0,
                    "description": "Fraction of features sampled per tree (feature_fraction)."
                },
                "reg_alpha": {
                    "type": "number",
                    "title": "L1 Regularization",
                    "default": 0.0,
                    "minimum": 0.0,
                    "maximum": 100.0
                },
                "reg_lambda": {
                    "type": "number",
                    "title": "L2 Regularization",
                    "default": 0.0,
                    "minimum": 0.0,
                    "maximum": 100.0
                },
                "scale_pos_weight": {
                    "type": "number",
                    "title": "Class Imbalance Weight (scale_pos_weight)",
                    "default": 1.0,
                    "minimum": 0.1,
                    "maximum": 1000.0,
                    "description": "Weight of positive class for imbalanced classification (fraud, churn)."
                },
                "min_child_samples": {
                    "type": "integer",
                    "title": "Min Data in Leaf",
                    "default": 20,
                    "minimum": 1,
                    "maximum": 200
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
            raise ValueError("LightGBM is not installed. Please install it or use Random Forest / Logistic Regression.")

        X_train = inputs.get("X_train")
        y_train = inputs.get("y_train")
        X_test = inputs.get("X_test")
        y_test = inputs.get("y_test")

        if X_train is None or y_train is None:
            raise ValueError("LightGBMTrainer expects 'X_train' and 'y_train' in inputs.")

        from backend.app.recipes.training.encoder_utils import safe_prepare_training_data
        X_train, X_test = safe_prepare_training_data(X_train, X_test)

        # Sanitize column names for LightGBM
        X_train = sanitize_lgb_colnames(X_train)
        if X_test is not None:
            X_test = sanitize_lgb_colnames(X_test)

        task_type = config.get("task_type", "classification")
        n_estimators = int(config.get("n_estimators", 100))
        num_leaves = int(config.get("num_leaves", 31))
        lr = float(config.get("learning_rate", 0.1))
        subsample = float(config.get("subsample", 1.0))
        colsample_bytree = float(config.get("colsample_bytree", 1.0))
        reg_alpha = float(config.get("reg_alpha", 0.0))
        reg_lambda = float(config.get("reg_lambda", 0.0))
        scale_pos_weight = float(config.get("scale_pos_weight", 1.0))
        min_child_samples = int(config.get("min_child_samples", 20))
        random_state = int(config.get("random_state", 42))

        is_continuous = False
        if pd.api.types.is_float_dtype(y_train) and y_train.nunique() > 20:
            is_continuous = True

        if task_type == "classification" and not is_continuous:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_train = pd.Series(le.fit_transform(y_train), index=y_train.index if hasattr(y_train, 'index') else None)
            if y_test is not None:
                try:
                    y_test = pd.Series(le.transform(y_test), index=y_test.index if hasattr(y_test, 'index') else None)
                except Exception:
                    pass

            model = lgb.LGBMClassifier(
                n_estimators=n_estimators,
                num_leaves=num_leaves,
                learning_rate=lr,
                subsample=subsample,
                subsample_freq=1 if subsample < 1.0 else 0,
                colsample_bytree=colsample_bytree,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                scale_pos_weight=scale_pos_weight,
                min_child_samples=min_child_samples,
                random_state=random_state,
                verbose=-1
            )
        else:
            task_type = "regression"
            model = lgb.LGBMRegressor(
                n_estimators=n_estimators,
                num_leaves=num_leaves,
                learning_rate=lr,
                subsample=subsample,
                subsample_freq=1 if subsample < 1.0 else 0,
                colsample_bytree=colsample_bytree,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                min_child_samples=min_child_samples,
                random_state=random_state,
                verbose=-1
            )

        model.fit(X_train, y_train)

        feature_names = list(X_train.columns)
        importances = {}
        if hasattr(model, "feature_importances_") and feature_names:
            total_imp = sum(model.feature_importances_) or 1.0
            for feat, imp in zip(feature_names, model.feature_importances_):
                importances[feat] = float(round(imp / total_imp, 4))

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

    def to_code(self, config: Dict[str, Any]) -> str:
        n_est = config.get("n_estimators", 100)
        leaves = config.get("num_leaves", 31)
        lr = config.get("learning_rate", 0.1)
        task = config.get("task_type", "classification")
        cls_name = "LGBMClassifier" if task == "classification" else "LGBMRegressor"
        return f"import lightgbm as lgb\n\nmodel = lgb.{cls_name}(n_estimators={n_est}, num_leaves={leaves}, learning_rate={lr}, random_state=42, verbose=-1)\nmodel.fit(X_train, y_train)"
