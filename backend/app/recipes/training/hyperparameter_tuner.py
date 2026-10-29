import math
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union

import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor
)
from sklearn.linear_model import (
    LogisticRegression,
    RidgeClassifier,
    Ridge,
    ElasticNet
)
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

# Suppress verbose Optuna logging during automated pipeline execution
optuna.logging.set_verbosity(optuna.logging.WARNING)


class HyperparameterTunerRecipe(BaseRecipe):
    """
    Hyperparameter Tuner Recipe (Bayesian Optimization via Optuna / Random Search / Grid Search).
    Explores and tunes high-dimensional hyperparameter search spaces using Tree-structured Parzen
    Estimators (TPE) with cross-validation to maximize predictive performance while guarding against overfitting.
    """

    recipe_id = "hyperparameter_tuner"
    name = "Hyperparameter Tuner (Optuna / Bayesian / Grid)"
    version = "1.0.0"
    category = "training"
    group = "Modeling"
    subgroup = "Deep Learning / AutoML"
    description = (
        "Optimizes model hyperparameters using Bayesian Optimization (Optuna TPE), Random Search, "
        "or Grid Search with cross-validation."
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
            description="Dataset containing X_train, y_train and optional X_test, y_test."
        ),
        RecipePort(
            id="model",
            label="Optional Baseline Model",
            type="model",
            required=False,
            max_connections=1,
            description="Estimator instance to tune. If not connected, model_type config is used."
        )
    ]

    outputs: List[RecipePort] = [
        RecipePort(
            id="model",
            label="Best Tuned Model",
            type="model",
            description="The optimal estimator refitted on the full training dataset with best discovered hyperparameters."
        ),
        RecipePort(
            id="metrics",
            label="Tuning Metrics",
            type="metrics",
            description="Best parameters, CV score, parameter importances, and trial trajectory."
        ),
        RecipePort(
            id="report",
            label="Tuning Summary Report",
            type="report",
            description="Markdown report containing optimization tournament and parameter sensitivities."
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "strategy": {
                    "type": "string",
                    "title": "Optimization Strategy",
                    "input_type": "dropdown",
                    "enum": ["optuna_bayesian", "random_search", "grid_search"],
                    "default": "optuna_bayesian",
                    "description": "Optuna TPE (Bayesian) efficiently finds optimal parameters by modeling the objective probability distribution."
                },
                "model_type": {
                    "type": "string",
                    "title": "Model Algorithm",
                    "input_type": "dropdown",
                    "enum": ["auto", "random_forest", "gradient_boosting", "logistic_regression", "ridge"],
                    "default": "auto",
                    "description": "Model family to tune if no upstream model is connected."
                },
                "task_type": {
                    "type": "string",
                    "title": "ML Task Type",
                    "input_type": "dropdown",
                    "enum": ["auto", "classification", "regression"],
                    "default": "auto"
                },
                "scoring_metric": {
                    "type": "string",
                    "title": "Optimization Metric",
                    "input_type": "dropdown",
                    "enum": ["auto", "accuracy", "f1_macro", "f1_weighted", "roc_auc", "r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"],
                    "default": "auto"
                },
                "n_trials": {
                    "type": "integer",
                    "title": "Number of Trials",
                    "input_type": "slider",
                    "default": 15,
                    "minimum": 3,
                    "maximum": 100,
                    "description": "Total iterations / parameter combinations explored."
                },
                "cv_folds": {
                    "type": "integer",
                    "title": "Internal CV Folds",
                    "input_type": "slider",
                    "default": 3,
                    "minimum": 2,
                    "maximum": 10
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

    def _resolve_model_type(self, model: Optional[Any], config_model: str, task_type: str) -> str:
        if model is not None:
            name = model.__class__.__name__.lower()
            if "forest" in name:
                return "random_forest"
            elif "gradient" in name or "gbm" in name:
                return "gradient_boosting"
            elif "logistic" in name:
                return "logistic_regression"
            elif "ridge" in name:
                return "ridge"

        cfg = str(config_model).lower().strip()
        if cfg != "auto":
            return cfg

        return "random_forest"

    def _resolve_scoring_metric(self, scoring_cfg: str, task_type: str) -> str:
        cfg = str(scoring_cfg).lower().strip()
        if cfg != "auto":
            return cfg
        return "f1_macro" if task_type == "classification" else "r2"

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
                "HyperparameterTuner expects 'X_train' and 'y_train' in inputs. "
                "Please connect a Train/Test Split node before this tuner."
            )

        if isinstance(X_train, np.ndarray):
            X_train = pd.DataFrame(X_train, columns=[f"feat_{i}" for i in range(X_train.shape[1])])
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
                    "At least 2 classes are required for training and tuning."
                )

            # String or categorical label encoding
            if isinstance(unique_classes[0], (str, bytes)):
                target_encoder = LabelEncoder()
                y_train_arr = target_encoder.fit_transform(y_train_arr)
                target_classes = [str(c) for c in target_encoder.classes_]
            else:
                target_classes = [str(c) for c in unique_classes]

            if y_test is not None and target_encoder is not None:
                y_test_clean = target_encoder.transform(np.array(y_test))
            elif y_test is not None:
                y_test_clean = np.array(y_test)
            else:
                y_test_clean = None
        else:
            try:
                y_train_arr = pd.to_numeric(pd.Series(y_train_arr)).to_numpy()
            except Exception:
                raise ValueError("Regression target variable must be numeric.")
            if y_test is not None:
                y_test_clean = pd.to_numeric(pd.Series(np.array(y_test))).to_numpy()
            else:
                y_test_clean = None

        # 4. Resolve Model Architecture & Search Space
        upstream_model = inputs.get("model")
        if upstream_model is None and isinstance(context, dict):
            upstream_model = context.get("model")
        model_type = self._resolve_model_type(upstream_model, config.get("model_type", "auto"), task_type)
        random_state = int(config.get("random_state", 42))

        # 5. Adaptive Cross-Validation Strategy Guardrail
        requested_cv = int(config.get("cv_folds", 3))
        warnings: List[str] = []

        if task_type == "classification":
            _, class_counts = np.unique(y_train_arr, return_counts=True)
            min_class_count = int(np.min(class_counts))
            if min_class_count < 2:
                raise ValueError(
                    f"At least 2 samples per class are required for cross-validation tuning, "
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
            cv_splitter = StratifiedKFold(n_splits=effective_cv, shuffle=True, random_state=random_state)
        else:
            n_samples = len(y_train_arr)
            if n_samples < 4:
                raise ValueError(f"HyperparameterTuner requires at least 4 observations for regression, got {n_samples}.")
            effective_cv = max(2, min(requested_cv, n_samples // 2))
            if effective_cv < requested_cv:
                warnings.append(
                    f"[INFO: ADAPTIVE CV] Training set size ({n_samples}) adjusted internal CV folds to {effective_cv}."
                )
            cv_splitter = KFold(n_splits=effective_cv, shuffle=True, random_state=random_state)

        scoring = self._resolve_scoring_metric(config.get("scoring_metric", "auto"), task_type)

        # Baseline evaluation with default parameters
        baseline_model = self._instantiate_model(model_type, task_type, {}, random_state)
        try:
            baseline_scores = cross_val_score(baseline_model, X_train, y_train_arr, cv=cv_splitter, scoring=scoring, n_jobs=1)
            baseline_score = float(round(float(np.mean(baseline_scores)), 4))
        except Exception:
            baseline_score = 0.0

        # 6. Optuna Objective Definition & Optimization
        n_trials = int(config.get("n_trials", 15))
        strategy = config.get("strategy", "optuna_bayesian")

        def objective(trial: optuna.Trial) -> float:
            params = self._sample_hyperparameters(trial, model_type, task_type, strategy)
            estimator = self._instantiate_model(model_type, task_type, params, random_state)
            scores = cross_val_score(estimator, X_train, y_train_arr, cv=cv_splitter, scoring=scoring, n_jobs=1)
            return float(np.mean(scores))

        # Sampler setup
        if strategy == "random_search":
            sampler = optuna.samplers.RandomSampler(seed=random_state)
        elif strategy == "grid_search":
            # Optuna GridSampler requires discrete search space
            search_space = self._get_grid_search_space(model_type, task_type)
            sampler = optuna.samplers.GridSampler(search_space, seed=random_state)
            total_grid_combinations = math.prod(len(v) for v in search_space.values())
            n_trials = min(n_trials, total_grid_combinations)
        else:
            sampler = optuna.samplers.TPESampler(seed=random_state)

        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        best_cv_score = float(round(study.best_value, 4))

        # 7. Refit Best Model on Full Training Set
        best_model = self._instantiate_model(model_type, task_type, best_params, random_state)
        best_model.fit(X_train, y_train_arr)

        # Calculate Parameter Importances if available
        param_importances: Dict[str, float] = {}
        try:
            if len(study.trials) >= 3 and len(best_params) > 1:
                raw_importances = optuna.importance.get_param_importances(study)
                param_importances = {k: float(round(v, 4)) for k, v in raw_importances.items()}
        except Exception:
            pass

        # 8. Test Set Evaluation (if X_test provided)
        test_metrics: Dict[str, Any] = {}
        if X_test is not None and y_test_clean is not None:
            preds = best_model.predict(X_test)
            if task_type == "classification":
                test_metrics["test_accuracy"] = float(round(accuracy_score(y_test_clean, preds), 4))
                test_metrics["test_f1_macro"] = float(round(f1_score(y_test_clean, preds, average="macro", zero_division=0), 4))
            else:
                test_metrics["test_r2"] = float(round(r2_score(y_test_clean, preds), 4))
                test_metrics["test_rmse"] = float(round(math.sqrt(mean_squared_error(y_test_clean, preds)), 4))
                test_metrics["test_mae"] = float(round(mean_absolute_error(y_test_clean, preds), 4))

        # Trial History
        trial_history = []
        for t in study.trials:
            if t.state == optuna.trial.TrialState.COMPLETE:
                trial_history.append({
                    "trial_number": t.number,
                    "score": float(round(t.value, 4)),
                    "params": t.params
                })
        trial_history.sort(key=lambda x: x["score"], reverse=True)

        lift_pct = round(((best_cv_score - baseline_score) / max(0.0001, abs(baseline_score))) * 100.0, 2)

        metrics = {
            "strategy": strategy,
            "model_type": model_type,
            "task_type": task_type,
            "scoring_metric": scoring,
            "n_trials_completed": len(study.trials),
            "effective_cv_folds": effective_cv,
            "baseline_cv_score": baseline_score,
            "best_cv_score": best_cv_score,
            "score_lift_percentage": lift_pct,
            "best_params": best_params,
            "param_importances": param_importances,
            "top_trials": trial_history[:5],
            "test_evaluation": test_metrics,
            "warnings": warnings
        }

        # 9. Format Executive Report
        report_lines = [
            f"# === Hyperparameter Optimization Tournament Report ({strategy.replace('_', ' ').title()}) ===",
            f"- Model Architecture: {best_model.__class__.__name__}",
            f"- Task Type: {task_type.capitalize()}",
            f"- Optimization Metric: {scoring}",
            f"- Total Trials Completed: {len(study.trials)}",
            f"- CV Folds: {effective_cv}",
            "",
            "## Score Comparison:",
            f"- **Default Baseline Score**: {baseline_score:.4f}",
            f"- **Best Tuned Score**: **{best_cv_score:.4f}** ({lift_pct:+.2f}% lift)",
            "",
            "## Optimal Hyperparameters Discovered:",
            "| Parameter | Optimal Value |",
            "| :--- | :--- |"
        ]
        for k, v in best_params.items():
            report_lines.append(f"| `{k}` | `{v}` |")

        if param_importances:
            report_lines.append("")
            report_lines.append("## Hyperparameter Sensitivity & Importance:")
            report_lines.append("| Parameter | Importance Weight |")
            report_lines.append("| :--- | :--- |")
            for k, imp in param_importances.items():
                bar = "■" * max(1, int(imp * 20))
                report_lines.append(f"| `{k}` | {imp:.4f} ({bar}) |")

        report_lines.append("")
        report_lines.append("## Top 5 Discovered Configurations:")
        report_lines.append("| Rank | Trial # | CV Score | Parameters |")
        report_lines.append("| :--- | :--- | :--- | :--- |")
        for rank, t in enumerate(trial_history[:5], start=1):
            param_str = ", ".join(f"{pk}={pv}" for pk, pv in t["params"].items())
            report_lines.append(f"| #{rank} | Trial {t['trial_number']} | **{t['score']:.4f}** | `{param_str}` |")

        if warnings:
            report_lines.append("")
            report_lines.append("## Guardrail Alerts:")
            for w in warnings:
                report_lines.append(f"- {w}")

        report_markdown = "\n".join(report_lines)

        output: Dict[str, Any] = {
            "model": best_model,
            "best_model": best_model,
            "task_type": task_type,
            "best_params": best_params,
            "metrics": metrics,
            "report": report_markdown,
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

    def _sample_hyperparameters(
        self,
        trial: optuna.Trial,
        model_type: str,
        task_type: str,
        strategy: str
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {}

        if model_type == "random_forest":
            params["n_estimators"] = trial.suggest_int("n_estimators", 20, 100, step=10)
            params["max_depth"] = trial.suggest_int("max_depth", 3, 12)
            params["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 8)
        elif model_type == "gradient_boosting":
            params["n_estimators"] = trial.suggest_int("n_estimators", 20, 80, step=10)
            params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
            params["max_depth"] = trial.suggest_int("max_depth", 2, 6)
        elif model_type == "logistic_regression":
            params["C"] = trial.suggest_float("C", 0.01, 10.0, log=True)
            params["max_iter"] = trial.suggest_categorical("max_iter", [500, 1000])
        elif model_type == "ridge":
            params["alpha"] = trial.suggest_float("alpha", 0.01, 10.0, log=True)
        else:
            params["n_estimators"] = trial.suggest_int("n_estimators", 20, 80, step=10)
            params["max_depth"] = trial.suggest_int("max_depth", 3, 10)

        return params

    def _get_grid_search_space(self, model_type: str, task_type: str) -> Dict[str, List[Any]]:
        if model_type == "random_forest":
            return {
                "n_estimators": [30, 60, 90],
                "max_depth": [4, 8],
                "min_samples_split": [2, 5]
            }
        elif model_type == "gradient_boosting":
            return {
                "n_estimators": [30, 60],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 5]
            }
        elif model_type == "logistic_regression":
            return {
                "C": [0.1, 1.0, 10.0],
                "max_iter": [500, 1000]
            }
        elif model_type == "ridge":
            return {
                "alpha": [0.1, 1.0, 10.0]
            }
        return {
            "n_estimators": [30, 60],
            "max_depth": [4, 8]
        }

    def _instantiate_model(
        self,
        model_type: str,
        task_type: str,
        params: Dict[str, Any],
        random_state: int
    ) -> Any:
        p = dict(params)
        p["random_state"] = random_state

        if task_type == "classification":
            if model_type == "random_forest":
                return RandomForestClassifier(**p)
            elif model_type == "gradient_boosting":
                return GradientBoostingClassifier(**p)
            elif model_type == "logistic_regression":
                return LogisticRegression(**p)
            elif model_type == "ridge":
                return RidgeClassifier(**p)
            return RandomForestClassifier(**p)
        else:
            if model_type == "random_forest":
                return RandomForestRegressor(**p)
            elif model_type == "gradient_boosting":
                return GradientBoostingRegressor(**p)
            elif model_type == "ridge":
                return Ridge(**p)
            elif model_type == "elastic_net":
                return ElasticNet(**p)
            return RandomForestRegressor(**p)

    def to_code(self, config: Dict[str, Any]) -> str:
        strategy = config.get("strategy", "optuna_bayesian")
        n_trials = config.get("n_trials", 15)
        model_type = config.get("model_type", "random_forest")
        return (
            "import optuna\n"
            "from sklearn.ensemble import RandomForestClassifier\n"
            "from sklearn.model_selection import cross_val_score\n\n"
            "def objective(trial):\n"
            "    n_est = trial.suggest_int('n_estimators', 20, 100, step=10)\n"
            "    depth = trial.suggest_int('max_depth', 3, 12)\n"
            "    clf = RandomForestClassifier(n_estimators=n_est, max_depth=depth, random_state=42)\n"
            "    return cross_val_score(clf, X_train, y_train, cv=3, scoring='f1_macro').mean()\n\n"
            f"study = optuna.create_study(direction='maximize')\n"
            f"study.optimize(objective, n_trials={n_trials})\n"
            "best_model = RandomForestClassifier(**study.best_params, random_state=42)\n"
            "best_model.fit(X_train, y_train)"
        )
