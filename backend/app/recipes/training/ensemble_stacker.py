import math
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union

from sklearn.ensemble import (
    StackingClassifier,
    StackingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor
)
from sklearn.linear_model import (
    LogisticRegression,
    RidgeClassifier,
    Ridge,
    ElasticNet
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    r2_score,
    mean_squared_error,
    mean_absolute_error
)
from sklearn.preprocessing import LabelEncoder
from sklearn.base import is_classifier, is_regressor, clone

from backend.app.recipes.base.recipe import BaseRecipe, RecipePort
from backend.app.recipes.training.encoder_utils import (
    safe_prepare_training_data,
    extract_train_test_data,
    pass_through_metadata
)


class EnsembleStackerRecipe(BaseRecipe):
    """
    Ensemble Stacker (Stacked Generalization) Recipe.
    Combines diverse base machine learning models (e.g. Random Forest, Gradient Boosting,
    Logistic Regression / Ridge) and trains a meta-learner on cross-validated out-of-fold predictions.
    Produces state-of-the-art predictive performance surpassing individual single-model baselines.
    """

    recipe_id = "ensemble_stacker"
    name = "Ensemble Stacker (Stacked Generalization)"
    version = "1.0.0"
    category = "training"
    group = "Modeling"
    subgroup = "Deep Learning / AutoML"
    description = (
        "Trains a multi-model Stacking Ensemble (meta-learner) that combines predictions from diverse base models "
        "using cross-validation out-of-fold features for superior predictive accuracy."
    )
    input_types = ["train_data", "model"]
    output_types = ["model", "metrics", "report"]

    inputs: List[RecipePort] = [
        RecipePort(
            id="train_data",
            label="Train/Test Split Data",
            type="train_data",
            required=True,
            max_connections=1,
            description="Training dataset containing X_train, y_train and optional X_test, y_test."
        ),
        RecipePort(
            id="base_models",
            label="Optional Pre-Configured Models",
            type="model",
            required=False,
            max_connections=5,
            description="Up to 5 base estimators to stack. If omitted, diverse default models are automatically generated."
        )
    ]

    outputs: List[RecipePort] = [
        RecipePort(
            id="model",
            label="Stacked Ensemble Model",
            type="model",
            description="Trained StackingClassifier or StackingRegressor pipeline with out-of-fold meta-learner."
        ),
        RecipePort(
            id="metrics",
            label="Stacking Metrics",
            type="metrics",
            description="Performance metrics for individual base models vs. the stacked ensemble, including lift percentage."
        ),
        RecipePort(
            id="report",
            label="Executive Summary Report",
            type="report",
            description="Markdown report with performance comparisons, base model weights, and out-of-fold diagnostic metrics."
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task_type": {
                    "type": "string",
                    "title": "ML Task Type",
                    "input_type": "dropdown",
                    "enum": ["auto", "classification", "regression"],
                    "default": "auto",
                    "description": "Auto-detects classification vs. regression based on target variable."
                },
                "base_models_preset": {
                    "type": "string",
                    "title": "Base Estimator Selection",
                    "input_type": "dropdown",
                    "enum": [
                        "diverse_blend",
                        "tree_ensemble",
                        "linear_tree_blend",
                        "high_capacity"
                    ],
                    "default": "diverse_blend",
                    "description": "Diversity is key to stacking success. 'diverse_blend' blends trees, boosting, and regularized linear models."
                },
                "final_estimator": {
                    "type": "string",
                    "title": "Meta-Learner (Final Estimator)",
                    "input_type": "dropdown",
                    "enum": ["auto", "logistic_regression", "ridge", "random_forest", "gradient_boosting", "elastic_net"],
                    "default": "auto",
                    "description": "The algorithm trained on out-of-fold predictions. Logistic Regression / Ridge is recommended to prevent meta-overfitting."
                },
                "cv_folds": {
                    "type": "integer",
                    "title": "Internal CV Folds",
                    "input_type": "slider",
                    "default": 5,
                    "minimum": 2,
                    "maximum": 10,
                    "description": "Number of cross-validation folds used to generate out-of-fold meta-features."
                },
                "passthrough": {
                    "type": "boolean",
                    "title": "Feature Passthrough",
                    "input_type": "toggle",
                    "default": False,
                    "description": "When enabled, passes original features alongside predictions into the meta-learner."
                },
                "stack_method": {
                    "type": "string",
                    "title": "Classification Stacking Method",
                    "input_type": "dropdown",
                    "enum": ["auto", "predict_proba", "decision_function", "predict"],
                    "default": "auto",
                    "description": "Methods used to generate meta-features for classifiers (predict_proba gives rich probability distributions)."
                },
                "random_state": {
                    "type": "integer",
                    "title": "Random Seed",
                    "input_type": "number",
                    "default": 42
                }
            },
            "required": []
        }

    def _infer_task_type(self, y: Any) -> str:
        y_arr = np.array(y)
        if y_arr.dtype.kind in ("U", "S", "O", "b"):
            return "classification"
        unique_vals = np.unique(y_arr[~pd.isna(y_arr)])
        if len(unique_vals) <= 10 and np.all(unique_vals == unique_vals.astype(int)):
            return "classification"
        return "regression"

    def _build_default_base_estimators(
        self,
        task_type: str,
        preset: str,
        random_state: int
    ) -> List[Tuple[str, Any]]:
        estimators: List[Tuple[str, Any]] = []

        if task_type == "classification":
            if preset == "tree_ensemble":
                estimators = [
                    ("rf", RandomForestClassifier(n_estimators=40, max_depth=6, random_state=random_state)),
                    ("gb", GradientBoostingClassifier(n_estimators=40, max_depth=3, random_state=random_state)),
                    ("et", ExtraTreesClassifier(n_estimators=40, max_depth=6, random_state=random_state))
                ]
            elif preset == "linear_tree_blend":
                estimators = [
                    ("rf", RandomForestClassifier(n_estimators=40, max_depth=6, random_state=random_state)),
                    ("lr", LogisticRegression(max_iter=1000, random_state=random_state)),
                    ("ridge", RidgeClassifier(random_state=random_state))
                ]
            else:  # diverse_blend or high_capacity
                estimators = [
                    ("rf", RandomForestClassifier(n_estimators=40, max_depth=6, random_state=random_state)),
                    ("gb", GradientBoostingClassifier(n_estimators=40, max_depth=3, random_state=random_state)),
                    ("lr", LogisticRegression(max_iter=1000, random_state=random_state))
                ]
        else:  # regression
            if preset == "tree_ensemble":
                estimators = [
                    ("rf", RandomForestRegressor(n_estimators=40, max_depth=6, random_state=random_state)),
                    ("gb", GradientBoostingRegressor(n_estimators=40, max_depth=3, random_state=random_state)),
                    ("et", ExtraTreesRegressor(n_estimators=40, max_depth=6, random_state=random_state))
                ]
            elif preset == "linear_tree_blend":
                estimators = [
                    ("rf", RandomForestRegressor(n_estimators=40, max_depth=6, random_state=random_state)),
                    ("ridge", Ridge(random_state=random_state)),
                    ("elastic", ElasticNet(random_state=random_state))
                ]
            else:  # diverse_blend or high_capacity
                estimators = [
                    ("rf", RandomForestRegressor(n_estimators=40, max_depth=6, random_state=random_state)),
                    ("gb", GradientBoostingRegressor(n_estimators=40, max_depth=3, random_state=random_state)),
                    ("ridge", Ridge(random_state=random_state))
                ]

        return estimators

    def _resolve_final_estimator(
        self,
        task_type: str,
        final_estimator_cfg: str,
        random_state: int
    ) -> Any:
        cfg = str(final_estimator_cfg).lower().strip()
        if task_type == "classification":
            if cfg in ("auto", "logistic_regression", "lr"):
                return LogisticRegression(max_iter=1000, random_state=random_state)
            elif cfg == "ridge":
                return RidgeClassifier(random_state=random_state)
            elif cfg == "random_forest":
                return RandomForestClassifier(n_estimators=30, max_depth=4, random_state=random_state)
            elif cfg == "gradient_boosting":
                return GradientBoostingClassifier(n_estimators=30, max_depth=2, random_state=random_state)
            return LogisticRegression(max_iter=1000, random_state=random_state)
        else:
            if cfg in ("auto", "ridge"):
                return Ridge(random_state=random_state)
            elif cfg in ("elastic_net", "elastic"):
                return ElasticNet(random_state=random_state)
            elif cfg == "random_forest":
                return RandomForestRegressor(n_estimators=30, max_depth=4, random_state=random_state)
            elif cfg == "gradient_boosting":
                return GradientBoostingRegressor(n_estimators=30, max_depth=2, random_state=random_state)
            return Ridge(random_state=random_state)

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        # 1. Extract Training Data
        X_train, y_train, X_test, y_test = extract_train_test_data(inputs)
        if (X_train is None or y_train is None) and isinstance(context, dict):
            X_train, y_train, X_test, y_test = extract_train_test_data(context)

        # Fallback to direct dataframe input
        if (X_train is None or y_train is None) and "dataframe" in inputs:
            df = inputs["dataframe"]
            target_col = config.get("target_column") or (context.get("target_column") if isinstance(context, dict) else "")
            if target_col and target_col in df.columns:
                X_train = df.drop(columns=[target_col])
                y_train = df[target_col]

        if X_train is None or y_train is None:
            raise ValueError(
                "EnsembleStacker expects 'X_train' and 'y_train' in inputs. "
                "Please connect a Train/Test Split node before this trainer."
            )

        if isinstance(X_train, np.ndarray):
            X_train = pd.DataFrame(X_train, columns=[f"feature_{i}" for i in range(X_train.shape[1])])
        elif not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train)

        if isinstance(X_test, np.ndarray):
            X_test = pd.DataFrame(X_test, columns=X_train.columns)
        elif X_test is not None and not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test)

        # Clean non-numeric features safely
        X_train, X_test = safe_prepare_training_data(X_train, X_test)
        feature_names = list(X_train.columns)

        # 2. Determine Task Type
        configured_task = str(config.get("task_type", "auto")).lower()
        if configured_task in ("classification", "regression"):
            task_type = configured_task
        else:
            task_type = self._infer_task_type(y_train)

        # 3. Target Variable Guardrails & Encoding
        y_train_arr = np.array(y_train)
        target_encoder = None
        target_classes: Optional[List[str]] = None

        if task_type == "classification":
            unique_classes, counts = np.unique(y_train_arr, return_counts=True)
            if len(unique_classes) < 2:
                raise ValueError(
                    f"Classification target variable contains only 1 class ({unique_classes[0]}). "
                    "At least 2 classes are required to train a classifier."
                )

            # String or categorical label encoding
            if isinstance(unique_classes[0], (str, bytes)):
                target_encoder = LabelEncoder()
                y_train_arr = target_encoder.fit_transform(y_train_arr)
                target_classes = [str(c) for c in target_encoder.classes_]
            else:
                target_classes = [str(c) for c in unique_classes]

            if y_test is not None and target_encoder is not None:
                y_test_arr = np.array(y_test)
                y_test_clean = target_encoder.transform(y_test_arr)
            elif y_test is not None:
                y_test_clean = np.array(y_test)
            else:
                y_test_clean = None
        else:
            # Regression numeric conversion
            try:
                y_train_arr = pd.to_numeric(pd.Series(y_train_arr)).to_numpy()
            except Exception:
                raise ValueError("Regression target variable must be numeric.")
            if y_test is not None:
                y_test_clean = pd.to_numeric(pd.Series(np.array(y_test))).to_numpy()
            else:
                y_test_clean = None

        # 4. Resolve Base Estimators
        random_state = int(config.get("random_state", 42))
        preset = config.get("base_models_preset", "diverse_blend")
        user_models = inputs.get("base_models") or inputs.get("models")

        base_estimators: List[Tuple[str, Any]] = []
        if isinstance(user_models, list) and len(user_models) >= 2:
            for idx, item in enumerate(user_models):
                if isinstance(item, tuple) and len(item) == 2:
                    base_estimators.append((str(item[0]), item[1]))
                else:
                    base_estimators.append((f"model_{idx+1}_{item.__class__.__name__}", item))
        else:
            base_estimators = self._build_default_base_estimators(task_type, preset, random_state)

        # 5. Adaptive CV Strategy Guardrail
        requested_cv = int(config.get("cv_folds", 5))
        warnings: List[str] = []

        if task_type == "classification":
            _, class_counts = np.unique(y_train_arr, return_counts=True)
            min_class_count = int(np.min(class_counts))
            if min_class_count < 2:
                raise ValueError(
                    f"At least 2 samples per class are required for cross-validation out-of-fold stacking, "
                    f"but smallest class has only {min_class_count} sample(s)."
                )
            if min_class_count < requested_cv:
                effective_cv = min_class_count
                warnings.append(
                    f"[INFO: ADAPTIVE CV] Minimum class sample size ({min_class_count}) is smaller than requested "
                    f"CV folds ({requested_cv}). Adjusted internal CV folds to {effective_cv} to ensure fold stratification."
                )
            else:
                effective_cv = requested_cv
        else:
            n_samples = len(y_train_arr)
            if n_samples < 4:
                raise ValueError(f"EnsembleStacker requires at least 4 observations for regression, got {n_samples}.")
            effective_cv = max(2, min(requested_cv, n_samples // 2))
            if effective_cv < requested_cv:
                warnings.append(
                    f"[INFO: ADAPTIVE CV] Training set size ({n_samples}) adjusted internal CV folds to {effective_cv}."
                )

        # 6. Instantiate Meta-Learner and Stacking Ensemble
        final_est_cfg = config.get("final_estimator", "auto")
        final_est = self._resolve_final_estimator(task_type, final_est_cfg, random_state)
        passthrough = bool(config.get("passthrough", False))
        stack_method = str(config.get("stack_method", "auto"))

        if task_type == "classification":
            stacked_model = StackingClassifier(
                estimators=base_estimators,
                final_estimator=final_est,
                cv=effective_cv,
                stack_method=stack_method,
                passthrough=passthrough,
                n_jobs=1
            )
        else:
            stacked_model = StackingRegressor(
                estimators=base_estimators,
                final_estimator=final_est,
                cv=effective_cv,
                passthrough=passthrough,
                n_jobs=1
            )

        # 7. Fit Stacked Ensemble
        stacked_model.fit(X_train, y_train_arr)

        # 8. Evaluate Base Models vs Stacked Ensemble
        eval_X = X_test if X_test is not None else X_train
        eval_y = y_test_clean if y_test_clean is not None else y_train_arr
        eval_split_label = "test" if X_test is not None else "train"

        base_evaluations: List[Dict[str, Any]] = []
        best_base_metric = -float("inf")
        best_base_name = ""

        # Evaluate fitted base estimators inside the stacked model
        for name, fitted_base in zip(stacked_model.named_estimators_.keys(), stacked_model.named_estimators_.values()):
            preds = fitted_base.predict(eval_X)
            if task_type == "classification":
                acc = float(round(accuracy_score(eval_y, preds), 4))
                f1 = float(round(f1_score(eval_y, preds, average="macro", zero_division=0), 4))
                score_for_rank = f1
                base_evaluations.append({
                    "model_name": name,
                    "accuracy": acc,
                    "f1_macro": f1
                })
            else:
                r2 = float(round(r2_score(eval_y, preds), 4))
                rmse = float(round(math.sqrt(mean_squared_error(eval_y, preds)), 4))
                mae = float(round(mean_absolute_error(eval_y, preds), 4))
                score_for_rank = r2
                base_evaluations.append({
                    "model_name": name,
                    "r2": r2,
                    "rmse": rmse,
                    "mae": mae
                })

            if score_for_rank > best_base_metric:
                best_base_metric = score_for_rank
                best_base_name = name

        # Evaluate the Stacked Model
        stack_preds = stacked_model.predict(eval_X)
        if task_type == "classification":
            stack_acc = float(round(accuracy_score(eval_y, stack_preds), 4))
            stack_f1 = float(round(f1_score(eval_y, stack_preds, average="macro", zero_division=0), 4))
            stack_metric = stack_f1
            lift_pct = round(((stack_metric - best_base_metric) / max(0.0001, abs(best_base_metric))) * 100.0, 2)
            metrics = {
                "task_type": "classification",
                "evaluated_split": eval_split_label,
                "stacked_accuracy": stack_acc,
                "stacked_f1_macro": stack_f1,
                "best_base_model": best_base_name,
                "best_base_f1_macro": best_base_metric,
                "f1_lift_percentage": lift_pct,
                "base_models_metrics": base_evaluations,
                "effective_cv_folds": effective_cv,
                "passthrough": passthrough,
                "final_estimator_name": final_est.__class__.__name__,
                "warnings": warnings
            }
        else:
            stack_r2 = float(round(r2_score(eval_y, stack_preds), 4))
            stack_rmse = float(round(math.sqrt(mean_squared_error(eval_y, stack_preds)), 4))
            stack_mae = float(round(mean_absolute_error(eval_y, stack_preds), 4))
            stack_metric = stack_r2
            denom = abs(best_base_metric) if abs(best_base_metric) > 0.0001 else 1.0
            lift_pct = round(((stack_metric - best_base_metric) / denom) * 100.0, 2)
            metrics = {
                "task_type": "regression",
                "evaluated_split": eval_split_label,
                "stacked_r2": stack_r2,
                "stacked_rmse": stack_rmse,
                "stacked_mae": stack_mae,
                "best_base_model": best_base_name,
                "best_base_r2": best_base_metric,
                "r2_lift_percentage": lift_pct,
                "base_models_metrics": base_evaluations,
                "effective_cv_folds": effective_cv,
                "passthrough": passthrough,
                "final_estimator_name": final_est.__class__.__name__,
                "warnings": warnings
            }

        # 9. Format Executive Report
        report_lines = [
            f"# === Stacking Ensemble Meta-Learner Report ({task_type.capitalize()}) ===",
            f"- Meta-Learner (Final Estimator): {final_est.__class__.__name__}",
            f"- Out-Of-Fold CV Folds: {effective_cv}",
            f"- Feature Passthrough: {'Enabled' if passthrough else 'Disabled'}",
            f"- Evaluated Split: {eval_split_label.upper()} (N={len(eval_y)})",
            "",
            "## Model Comparison Tournament:"
        ]

        if task_type == "classification":
            report_lines.append("| Model | Accuracy | Macro F1 | Status |")
            report_lines.append("| :--- | :--- | :--- | :--- |")
            for b in base_evaluations:
                tag = " (Best Base)" if b["model_name"] == best_base_name else ""
                report_lines.append(f"| Base: {b['model_name']}{tag} | {b['accuracy']:.4f} | {b['f1_macro']:.4f} | Base Model |")
            report_lines.append(f"| **Stacked Ensemble** | **{stack_acc:.4f}** | **{stack_f1:.4f}** | **Meta-Learner** |")
            report_lines.append("")
            report_lines.append(f"Ensemble Performance Lift over '{best_base_name}': **{lift_pct:+.2f}%**")
        else:
            report_lines.append("| Model | R² Score | RMSE | MAE | Status |")
            report_lines.append("| :--- | :--- | :--- | :--- | :--- |")
            for b in base_evaluations:
                tag = " (Best Base)" if b["model_name"] == best_base_name else ""
                report_lines.append(f"| Base: {b['model_name']}{tag} | {b['r2']:.4f} | {b['rmse']:.4f} | {b['mae']:.4f} | Base Model |")
            report_lines.append(f"| **Stacked Ensemble** | **{stack_r2:.4f}** | **{stack_rmse:.4f}** | **{stack_mae:.4f}** | **Meta-Learner** |")
            report_lines.append("")
            report_lines.append(f"Ensemble R² Lift over '{best_base_name}': **{lift_pct:+.2f}%**")

        if warnings:
            report_lines.append("")
            report_lines.append("## Guardrail Alerts:")
            for w in warnings:
                report_lines.append(f"- {w}")

        report_markdown = "\n".join(report_lines)

        output: Dict[str, Any] = {
            "model": stacked_model,
            "stacked_model": stacked_model,
            "task_type": task_type,
            "metrics": metrics,
            "report": report_markdown,
            "base_models": [name for name, _ in base_estimators],
            "meta_learner": final_est.__class__.__name__,
            "effective_cv": effective_cv,
            "passthrough": passthrough,
            "feature_names": feature_names
        }

        if target_classes:
            output["target_classes"] = target_classes
        if target_encoder:
            output["target_encoder"] = target_encoder

        if X_test is not None:
            output["X_test"] = X_test
        if y_test is not None:
            output["y_test"] = y_test

        return pass_through_metadata(inputs, output, context)

    def to_code(self, config: Dict[str, Any]) -> str:
        task_type = config.get("task_type", "classification")
        cv = config.get("cv_folds", 5)
        passthrough = config.get("passthrough", False)
        if task_type == "regression":
            return (
                "from sklearn.ensemble import StackingRegressor, RandomForestRegressor, GradientBoostingRegressor\n"
                "from sklearn.linear_model import Ridge\n\n"
                "estimators = [\n"
                "    ('rf', RandomForestRegressor(n_estimators=50, random_state=42)),\n"
                "    ('gb', GradientBoostingRegressor(n_estimators=50, random_state=42))\n"
                "]\n"
                f"model = StackingRegressor(estimators=estimators, final_estimator=Ridge(), cv={cv}, passthrough={passthrough})\n"
                "model.fit(X_train, y_train)"
            )
        return (
            "from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier\n"
            "from sklearn.linear_model import LogisticRegression\n\n"
            "estimators = [\n"
            "    ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),\n"
            "    ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42))\n"
            "]\n"
            f"model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), cv={cv}, passthrough={passthrough})\n"
            "model.fit(X_train, y_train)"
        )
