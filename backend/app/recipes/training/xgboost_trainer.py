import pandas as pd
from typing import Dict, Any, Optional
from backend.app.recipes.base.recipe import BaseRecipe

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except (ImportError, OSError, Exception):
    xgb = None
    XGBOOST_AVAILABLE = False


class XGBoostTrainerRecipe(BaseRecipe):
    recipe_id = "xgboost_trainer"
    name = "XGBoost Classifier / Regressor"
    version = "1.1.0"
    category = "training"
    description = "Trains a gradient-boosted decision tree ensemble using XGBoost with full regularization, subsampling, and class imbalance business tuning."
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
                "subsample": {
                    "type": "number",
                    "title": "Row Subsample Ratio",
                    "default": 1.0,
                    "minimum": 0.3,
                    "maximum": 1.0,
                    "description": "Fraction of training rows randomly sampled per tree (prevents overfitting)."
                },
                "colsample_bytree": {
                    "type": "number",
                    "title": "Feature Subsample Ratio",
                    "default": 1.0,
                    "minimum": 0.3,
                    "maximum": 1.0,
                    "description": "Fraction of columns randomly sampled per tree."
                },
                "reg_alpha": {
                    "type": "number",
                    "title": "L1 Regularization (reg_alpha)",
                    "default": 0.0,
                    "minimum": 0.0,
                    "maximum": 100.0,
                    "description": "L1 penalty on leaf weights (encourages sparsity in noisy datasets)."
                },
                "reg_lambda": {
                    "type": "number",
                    "title": "L2 Regularization (reg_lambda)",
                    "default": 1.0,
                    "minimum": 0.0,
                    "maximum": 100.0,
                    "description": "L2 penalty on leaf weights (smooths predictions)."
                },
                "scale_pos_weight": {
                    "type": "number",
                    "title": "Class Imbalance Weight (scale_pos_weight)",
                    "default": 1.0,
                    "minimum": 0.1,
                    "maximum": 1000.0,
                    "description": "Crucial business parameter for imbalanced data (e.g. Fraud or Churn). Set to (Negative Cases / Positive Cases)."
                },
                "min_child_weight": {
                    "type": "number",
                    "title": "Min Child Weight",
                    "default": 1.0,
                    "minimum": 0.1,
                    "maximum": 50.0
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

        from backend.app.recipes.training.encoder_utils import safe_prepare_training_data
        X_train, X_test = safe_prepare_training_data(X_train, X_test)

        task_type = str(config.get("task_type", "classification")).lower()
        n_estimators = int(config.get("n_estimators", 100))
        max_depth = int(config.get("max_depth", 6))
        lr = float(config.get("learning_rate", 0.1))
        subsample = float(config.get("subsample", 1.0))
        colsample_bytree = float(config.get("colsample_bytree", 1.0))
        reg_alpha = float(config.get("reg_alpha", 0.0))
        reg_lambda = float(config.get("reg_lambda", 1.0))
        scale_pos_weight = float(config.get("scale_pos_weight", 1.0))
        min_child_weight = float(config.get("min_child_weight", 1.0))
        random_state = int(config.get("random_state", 42))

        if task_type == "classification":
            # Encode target labels to 0..N-1
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

            n_classes = int(y_train.nunique())
            eval_metric = "logloss" if n_classes <= 2 else "mlogloss"

            xgb_kwargs = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": lr,
                "subsample": subsample,
                "colsample_bytree": colsample_bytree,
                "reg_alpha": reg_alpha,
                "reg_lambda": reg_lambda,
                "min_child_weight": min_child_weight,
                "random_state": random_state,
                "eval_metric": eval_metric
            }
            if n_classes <= 2 and scale_pos_weight != 1.0:
                xgb_kwargs["scale_pos_weight"] = scale_pos_weight

            model = xgb.XGBClassifier(**xgb_kwargs)
        else:
            task_type = "regression"
            model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=lr,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                min_child_weight=min_child_weight,
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
        subsample = config.get("subsample", 1.0)
        task = config.get("task_type", "classification")
        cls_name = "XGBClassifier" if task == "classification" else "XGBRegressor"
        return f"import xgboost as xgb\n\nmodel = xgb.{cls_name}(\n    n_estimators={n_est},\n    max_depth={depth},\n    learning_rate={lr},\n    subsample={subsample},\n    random_state=42\n)\nmodel.fit(X_train, y_train)"
