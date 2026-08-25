import pandas as pd
from typing import Dict, Any, Optional
from backend.app.recipes.base.recipe import BaseRecipe

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class XGBoostTrainerRecipe(BaseRecipe):
    recipe_id = "xgboost_trainer"
    name = "XGBoost Classifier / Regressor"
    version = "1.0.0"
    category = "training"
    description = "Trains a gradient-boosted decision tree ensemble using XGBoost."
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
                "max_depth": {
                    "type": "integer",
                    "title": "Max Depth",
                    "default": 6,
                    "minimum": 1,
                    "maximum": 20
                },
                "learning_rate": {
                    "type": "number",
                    "title": "Learning Rate (eta)",
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
        if not XGBOOST_AVAILABLE:
            raise ValueError("XGBoost is not installed in the environment. Please run 'pip install xgboost' or choose Random Forest / Logistic Regression.")

        X_train = inputs.get("X_train")
        y_train = inputs.get("y_train")
        X_test = inputs.get("X_test")
        y_test = inputs.get("y_test")

        if X_train is None or y_train is None:
            raise ValueError("XGBoostTrainer expects 'X_train' and 'y_train' in inputs. Please connect a Train/Test Split node before this trainer.")

        X_train = X_train.copy()
        if X_test is not None:
            X_test = X_test.copy()

        # Auto-encode any remaining string / object columns if user omitted Categorical Encoder
        non_numeric = [c for c in X_train.columns if not pd.api.types.is_numeric_dtype(X_train[c])]
        if non_numeric:
            X_train = pd.get_dummies(X_train, columns=non_numeric, drop_first=False, dtype=int)
            if X_test is not None:
                X_test = pd.get_dummies(X_test, columns=non_numeric, drop_first=False, dtype=int)
                # Align columns between train and test
                X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

        task_type = config.get("task_type", "classification")
        n_estimators = int(config.get("n_estimators", 100))
        max_depth = int(config.get("max_depth", 6))
        lr = float(config.get("learning_rate", 0.1))
        random_state = int(config.get("random_state", 42))

        # Check if y_train is continuous float for classification
        is_continuous = False
        if pd.api.types.is_float_dtype(y_train) and y_train.nunique() > 20:
            is_continuous = True

        if task_type == "classification" and not is_continuous:
            # Encode target labels to 0..N-1
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_train = pd.Series(le.fit_transform(y_train), index=y_train.index if hasattr(y_train, 'index') else None)
            if y_test is not None:
                # Handle unseen labels gracefully
                try:
                    y_test = pd.Series(le.transform(y_test), index=y_test.index if hasattr(y_test, 'index') else None)
                except Exception:
                    pass

            model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=lr,
                random_state=random_state,
                eval_metric="logloss"
            )
        else:
            task_type = "regression"
            model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=lr,
                random_state=random_state
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

    def to_code(self, config: Dict[str, Any]) -> str:
        n_est = config.get("n_estimators", 100)
        depth = config.get("max_depth", 6)
        lr = config.get("learning_rate", 0.1)
        task = config.get("task_type", "classification")
        cls_name = "XGBClassifier" if task == "classification" else "XGBRegressor"
        return f"import xgboost as xgb\n\nmodel = xgb.{cls_name}(\n    n_estimators={n_est},\n    max_depth={depth},\n    learning_rate={lr},\n    random_state=42\n)\nmodel.fit(X_train, y_train)"
