import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from backend.app.recipes.base.recipe import BaseRecipe

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


class CatBoostTrainerRecipe(BaseRecipe):
    recipe_id = "catboost_trainer"
    name = "CatBoost Classifier / Regressor"
    version = "1.0.0"
    category = "training"
    description = "Yandex CatBoost gradient boosting algorithm utilizing oblivious decision trees with state-of-the-art native categorical feature handling (Tier-1 Enterprise Standard)."
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
                "iterations": {
                    "type": "integer",
                    "title": "Iterations / Trees",
                    "default": 100,
                    "minimum": 10,
                    "maximum": 2000
                },
                "depth": {
                    "type": "integer",
                    "title": "Tree Depth",
                    "default": 6,
                    "minimum": 1,
                    "maximum": 12
                },
                "learning_rate": {
                    "type": "number",
                    "title": "Learning Rate",
                    "default": 0.1,
                    "minimum": 0.001,
                    "maximum": 1.0
                },
                "random_seed": {
                    "type": "integer",
                    "title": "Random Seed",
                    "default": 42
                }
            },
            "required": ["task_type"]
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        if not CATBOOST_AVAILABLE:
            raise ValueError("CatBoost is not installed. Please run 'pip install catboost'.")

        X_train = inputs.get("X_train")
        y_train = inputs.get("y_train")
        X_test = inputs.get("X_test")
        y_test = inputs.get("y_test")

        if X_train is None or y_train is None:
            raise ValueError("CatBoostTrainer expects 'X_train' and 'y_train' in inputs. Please connect a Train/Test Split node before this trainer.")

        X_train = X_train.copy()
        if X_test is not None:
            X_test = X_test.copy()

        # Identify categorical features for native CatBoost processing
        cat_features = [c for c in X_train.columns if not pd.api.types.is_numeric_dtype(X_train[c])]
        
        # Clean NaNs in categorical features
        for c in cat_features:
            X_train[c] = X_train[c].fillna("missing").astype(str)
            if X_test is not None and c in X_test.columns:
                X_test[c] = X_test[c].fillna("missing").astype(str)

        # Clean NaNs in numeric features
        num_features = [c for c in X_train.columns if c not in cat_features]
        for c in num_features:
            med = X_train[c].median()
            X_train[c] = X_train[c].fillna(med)
            if X_test is not None and c in X_test.columns:
                X_test[c] = X_test[c].fillna(med)

        task_type = config.get("task_type", "classification")
        iterations = int(config.get("iterations", 100))
        depth = int(config.get("depth", 6))
        lr = float(config.get("learning_rate", 0.1))
        random_seed = int(config.get("random_seed", 42))

        if task_type == "classification":
            model = CatBoostClassifier(
                iterations=iterations,
                depth=depth,
                learning_rate=lr,
                random_seed=random_seed,
                cat_features=cat_features if cat_features else None,
                verbose=0
            )
        else:
            model = CatBoostRegressor(
                iterations=iterations,
                depth=depth,
                learning_rate=lr,
                random_seed=random_seed,
                cat_features=cat_features if cat_features else None,
                verbose=0
            )

        model.fit(X_train, y_train)

        # Feature importances
        feature_names = list(X_train.columns)
        importances = {}
        if hasattr(model, "get_feature_importance") and feature_names:
            raw_imp = model.get_feature_importance()
            for feat, imp in zip(feature_names, raw_imp):
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
