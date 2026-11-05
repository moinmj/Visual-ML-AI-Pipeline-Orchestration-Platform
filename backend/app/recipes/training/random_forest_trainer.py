import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from typing import Dict, Any, Optional, List
from backend.app.recipes.base.recipe import BaseRecipe


class RandomForestTrainerRecipe(BaseRecipe):
    recipe_id = "random_forest_trainer"
    name = "Random Forest Classifier / Regressor"
    version = "1.1.0"
    category = "training"
    description = "Trains an ensemble of decision trees using Scikit-Learn's Random Forest with full hyperparameter controls and class imbalance weighting."
    input_types = ["train_data"]
    output_types = ["model"]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task_type": {
                    "type": "string",
                    "title": "ML Task Type",
                    "input_type": "dropdown",
                    "enum": ["classification", "regression"],
                    "default": "classification",
                    "description": "Supervised learning objective ('classification' or 'regression')."
                },
                "target_column": {
                    "type": "string",
                    "title": "Target Column (Optional)",
                    "description": "Target column to predict. If left blank, automatically inherited from upstream Train/Test Split node."
                },
                "n_estimators": {
                    "type": "integer",
                    "title": "Number of Trees",
                    "input_type": "slider",
                    "default": 100,
                    "minimum": 10,
                    "maximum": 1000,
                    "description": 'Number of decision trees in the ensemble forest.'
                },
                "max_depth": {
                    "type": "integer",
                    "title": "Max Depth",
                    "input_type": "slider",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 50,
                    "description": 'Maximum depth of individual decision trees.'
                },
                "min_samples_split": {
                    "type": "integer",
                    "title": "Min Samples to Split",
                    "input_type": "slider",
                    "default": 2,
                    "minimum": 2,
                    "maximum": 50,
                    "description": "The minimum number of samples required to split an internal node."
                },
                "min_samples_leaf": {
                    "type": "integer",
                    "title": "Min Samples in Leaf",
                    "input_type": "slider",
                    "default": 1,
                    "minimum": 1,
                    "maximum": 50,
                    "description": "The minimum number of samples required to be at a leaf node (higher values smooth predictions and prevent overfitting)."
                },
                "max_features": {
                    "type": "string",
                    "title": "Max Features per Split",
                    "input_type": "dropdown",
                    "enum": ["sqrt", "log2", "all"],
                    "default": "sqrt",
                    "description": "Number of features to examine when searching for the best split ('sqrt', 'log2', or float)."
                },
                "class_weight": {
                    "type": "string",
                    "title": "Class Weighting (Imbalance)",
                    "input_type": "dropdown",
                    "enum": ["none", "balanced", "balanced_subsample"],
                    "default": "none",
                    "description": "Crucial business setting for fraud/churn: automatically adjusts weights inversely proportional to class frequencies."
                },
                "random_state": {
                    "type": "integer",
                    "title": "Random Seed",
                    "input_type": "number",
                    "default": 42,
                    "description": 'Random seed for reproducible bootstrap bagging and splits.'
                }
            },
            "required": ["task_type"]
        }


    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        errors = []
        task_type = str(config.get("task_type", "classification")).lower()
        target_col = str(config.get("target_column", "")).strip()

        if task_type == "regression" and target_col:
            binary_keywords = ["churn", "target_churn", "fraud", "default", "label", "binary", "class", "target_class", "survived", "promoted", "converted"]
            t_lower = target_col.lower()
            if any(k in t_lower for k in binary_keywords) or t_lower.startswith("is_"):
                errors.append(
                    f"⚠️ Inline Validation Warning: Target column '{target_col}' appears to be a discrete 0/1 category column, "
                    f"but Regression mode is selected. Regression is intended for continuous numeric quantities (such as sales_amount, price, temperature). "
                    f"For binary 0/1 outcomes like churn or fraud, switch task_type to 'classification' or select a continuous numeric target like 'sales_amount'."
                )
        return errors

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        from backend.app.recipes.training.encoder_utils import safe_prepare_training_data, extract_train_test_data, pass_through_metadata
        X_train, y_train, X_test, y_test = extract_train_test_data(inputs)

        if X_train is None or y_train is None:
            raise ValueError("RandomForestTrainer expects 'X_train' and 'y_train' in inputs. Please connect a Train/Test Split node before this trainer.")

        X_train, X_test = safe_prepare_training_data(X_train, X_test)

        task_type = str(config.get("task_type", "classification")).lower()
        n_estimators = int(config.get("n_estimators", 100))
        max_depth = int(config.get("max_depth", 10))
        min_samples_split = int(config.get("min_samples_split", 2))
        min_samples_leaf = int(config.get("min_samples_leaf", 1))
        max_feat_str = config.get("max_features", "sqrt")
        max_feat = None if max_feat_str == "all" else max_feat_str
        class_weight_str = config.get("class_weight", "none")
        class_weight = None if class_weight_str == "none" else class_weight_str
        random_state = int(config.get("random_state", 42))

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

            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_feat,
                class_weight=class_weight,
                random_state=random_state
            )
        else:
            task_type = "regression"
            warnings_list = []
            y_vals = pd.Series(y_train).dropna()
            u_vals = sorted(list(set(y_vals.unique())))
            if len(u_vals) <= 2 and (set(u_vals).issubset({0, 1, 0.0, 1.0, True, False, "0", "1", "yes", "no"}) or len(u_vals) <= 2):
                discrete_warn = (
                    f"[WARNING: DISCRETE TARGET IN REGRESSION] Target column appears to be a discrete 0/1 category column "
                    f"(unique values: {u_vals}), but Regression mode was selected. "
                    f"Regression is intended for continuous numeric quantities (such as sales_amount, price, or temperature). "
                    f"Please switch task_type to 'classification' for binary targets like churn, or choose a continuous numeric target like 'sales_amount'."
                )
                warnings_list.append(discrete_warn)

            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_feat,
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
            "feature_names": feature_names,
            "warnings": locals().get("warnings_list", [])
        }

        upstream_classes = inputs.get("target_classes") or (context.get("target_classes") if isinstance(context, dict) else None)
        if upstream_classes:
            output["target_classes"] = upstream_classes
            if "target_encoder" in inputs:
                output["target_encoder"] = inputs["target_encoder"]
        elif task_type == "classification" and "le" in locals():
            output["target_classes"] = [str(c) for c in le.classes_]
            output["target_encoder"] = le

        if X_test is not None:
            output["X_test"] = X_test
        if y_test is not None:
            output["y_test"] = y_test

        return pass_through_metadata(inputs, output, context)

    def to_code(self, config: Dict[str, Any]) -> str:
        n_est = config.get("n_estimators", 100)
        depth = config.get("max_depth", 10)
        task = config.get("task_type", "classification")
        cls_name = "RandomForestClassifier" if task == "classification" else "RandomForestRegressor"
        return f"from sklearn.ensemble import {cls_name}\n\nmodel = {cls_name}(n_estimators={n_est}, max_depth={depth}, random_state=42)\nmodel.fit(X_train, y_train)"
