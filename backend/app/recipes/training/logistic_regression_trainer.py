import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from typing import Dict, Any, Optional
from backend.app.recipes.base.recipe import BaseRecipe


class LogisticRegressionTrainerRecipe(BaseRecipe):
    recipe_id = "linear_trainer"
    name = "Logistic / Ridge Linear Model"
    version = "1.1.0"
    category = "training"
    description = "Trains an interpretable linear model (Logistic Regression for classification or Ridge for regression) with penalty regularization and class weighting."
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
                    "default": 200,
                    "minimum": 50,
                    "maximum": 5000
                },
                "C": {
                    "type": "number",
                    "title": "Inverse Regularization Strength (C)",
                    "default": 1.0,
                    "minimum": 0.001,
                    "maximum": 1000.0,
                    "description": "Smaller values specify stronger regularization."
                },
                "penalty": {
                    "type": "string",
                    "title": "Regularization Penalty",
                    "enum": ["l2", "l1", "none"],
                    "default": "l2",
                    "description": "L1 performs feature selection (shrinks non-important weights to 0); L2 prevents extreme weights."
                },
                "class_weight": {
                    "type": "string",
                    "title": "Class Weighting (Imbalance)",
                    "enum": ["none", "balanced"],
                    "default": "none",
                    "description": "Adjusts weights inversely proportional to class frequencies for imbalanced data."
                },
                "solver": {
                    "type": "string",
                    "title": "Optimization Solver",
                    "enum": ["auto", "lbfgs", "liblinear", "saga"],
                    "default": "auto"
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

        task_type = str(config.get("task_type", "classification")).lower()
        max_iter = int(config.get("max_iter", 200))
        c_val = float(config.get("C", 1.0))
        penalty = config.get("penalty", "l2")
        class_weight_str = config.get("class_weight", "none")
        class_weight = None if class_weight_str == "none" else class_weight_str
        solver_cfg = config.get("solver", "auto")

        if task_type == "classification":
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_train = pd.Series(le.fit_transform(y_train), index=y_train.index if hasattr(y_train, 'index') else None)
            if y_test is not None:
                try:
                    known_classes = set(le.classes_)
                    y_test = pd.Series(
                        [le.transform([val])[0] if val in known_classes else 0 for val in y_test],
                        index=y_test.index if hasattr(y_test, 'index') else None
                    )
                except Exception:
                    pass

            # Auto-resolve solver compatibility
            if solver_cfg == "auto":
                solver = "liblinear" if penalty == "l1" else "lbfgs"
            else:
                solver = solver_cfg

            pen = None if penalty == "none" else penalty

            model = LogisticRegression(
                max_iter=max_iter,
                C=c_val,
                penalty=pen,
                solver=solver,
                class_weight=class_weight,
                random_state=42
            )
        else:
            task_type = "regression"
            model = Ridge(alpha=1.0 / max(c_val, 1e-5), random_state=42)

        model.fit(X_train, y_train)

        feature_names = list(X_train.columns)
        importances = {}
        if hasattr(model, "coef_") and feature_names:
            coefs = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
            for feat, coef in zip(feature_names, coefs):
                importances[feat] = float(round(abs(coef), 4))

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
        max_iter = config.get("max_iter", 200)
        c_val = config.get("C", 1.0)
        task = config.get("task_type", "classification")
        if task == "classification":
            return f"from sklearn.linear_model import LogisticRegression\n\nmodel = LogisticRegression(max_iter={max_iter}, C={c_val}, random_state=42)\nmodel.fit(X_train, y_train)"
        return f"from sklearn.linear_model import Ridge\n\nmodel = Ridge(alpha={1.0 / c_val}, random_state=42)\nmodel.fit(X_train, y_train)"
