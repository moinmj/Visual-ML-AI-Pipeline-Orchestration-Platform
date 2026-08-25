import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from typing import Dict, Any, Optional
from backend.app.recipes.base.recipe import BaseRecipe


class LogisticRegressionTrainerRecipe(BaseRecipe):
    recipe_id = "linear_trainer"
    name = "Logistic / Ridge Linear Model"
    version = "1.0.0"
    category = "training"
    description = "Trains a linear model (Logistic Regression for classification or Ridge for regression)."
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
                "max_iter": {
                    "type": "integer",
                    "title": "Max Iterations",
                    "default": 200
                },
                "C": {
                    "type": "number",
                    "title": "Inverse Regularization Strength (C)",
                    "default": 1.0
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
            raise ValueError("LinearTrainer expects 'X_train' and 'y_train' in inputs. Please connect a Train/Test Split node before this trainer.")

        from backend.app.recipes.training.encoder_utils import safe_prepare_training_data
        X_train, X_test = safe_prepare_training_data(X_train, X_test)

        task_type = config.get("task_type", "classification")
        max_iter = int(config.get("max_iter", 200))
        c_val = float(config.get("C", 1.0))

        if task_type == "classification":
            model = LogisticRegression(max_iter=max_iter, C=c_val)
        else:
            model = Ridge(alpha=1.0 / c_val)

        model.fit(X_train, y_train)

        output = {
            "model": model,
            "task_type": task_type,
            "feature_names": list(X_train.columns)
        }

        if X_test is not None:
            output["X_test"] = X_test
        if y_test is not None:
            output["y_test"] = y_test

        return output
