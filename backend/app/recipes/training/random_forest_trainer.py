import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from typing import Dict, Any, Optional
from backend.app.recipes.base.recipe import BaseRecipe


class RandomForestTrainerRecipe(BaseRecipe):
    recipe_id = "random_forest_trainer"
    name = "Random Forest Classifier / Regressor"
    version = "1.0.0"
    category = "training"
    description = "Trains an ensemble of decision trees using Scikit-Learn's Random Forest."
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
                    "title": "Number of Trees",
                    "default": 100,
                    "minimum": 10,
                    "maximum": 1000
                },
                "max_depth": {
                    "type": "integer",
                    "title": "Max Depth",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 50
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
        X_train = inputs.get("X_train")
        y_train = inputs.get("y_train")
        X_test = inputs.get("X_test")
        y_test = inputs.get("y_test")

        if X_train is None or y_train is None:
            raise ValueError("RandomForestTrainer expects 'X_train' and 'y_train' in inputs. Please connect a Train/Test Split node before this trainer.")

        from backend.app.recipes.training.encoder_utils import safe_prepare_training_data
        X_train, X_test = safe_prepare_training_data(X_train, X_test)

        task_type = config.get("task_type", "classification")
        n_estimators = int(config.get("n_estimators", 100))
        max_depth = int(config.get("max_depth", 10))
        random_state = int(config.get("random_state", 42))

        if task_type == "classification":
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state
            )
        else:
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state
            )

        model.fit(X_train, y_train)

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
