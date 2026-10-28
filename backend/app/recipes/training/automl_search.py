import math
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union

from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor
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


class AutoMLSearchRecipe(BaseRecipe):
    """
    AutoML Search Recipe (Multi-Algorithm Tournament & Automatic Model Selection).
    Executes a tournament across diverse candidate model architectures (Random Forest,
    Gradient Boosting, Extra Trees, HistGradientBoosting, and Regularized Linear/Ridge),
    ranks models via cross-validation, and refits the optimal champion model.
    """

    recipe_id = "automl_search"
    name = "AutoML Search (Multi-Model Tournament)"
    version = "1.0.0"
    category = "training"
    group = "Modeling"
    subgroup = "Deep Learning / AutoML"
    description = (
        "Conducts an automated multi-algorithm tournament across diverse model families (Random Forest, "
        "Gradient Boosting, Extra Trees, Linear/Ridge) with cross-validation and selects the champion model."
    )
    input_types = ["train_data"]
    output_types = ["model", "metrics", "report"]

    inputs: List[RecipePort] = [
        RecipePort(
            id="train_data",
            label="Train/Test Split Data",
            type="train_data",
            required=True,
            max_connections=1,
            description="Training dataset containing X_train, y_train and optional X_test, y_test."
        )
    ]

    outputs: List[RecipePort] = [
        RecipePort(
            id="model",
            label="Champion Model",
            type="model",
            description="The winning estimator from the tournament refitted on the full training set."
        ),
        RecipePort(
            id="metrics",
            label="Leaderboard Metrics",
            type="metrics",
            description="Complete leaderboard table, model scores, training latencies, and victory margins."
        ),
        RecipePort(
            id="report",
            label="Tournament Report",
            type="report",
            description="Markdown tournament summary with podium standings, score rankings, and latency benchmarks."
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
                "candidate_pool": {
                    "type": "string",
                    "title": "Algorithm Candidate Pool",
                    "input_type": "dropdown",
                    "enum": ["all_candidates", "tree_ensembles", "fast_candidates", "linear_models"],
                    "default": "all_candidates",
                    "description": "Candidate model families included in the tournament."
                },
                "primary_metric": {
                    "type": "string",
                    "title": "Primary Ranking Metric",
                    "input_type": "dropdown",
                    "enum": ["auto", "f1_macro", "accuracy", "roc_auc", "r2", "rmse", "mae"],
                    "default": "auto",
                    "description": "Metric used to crown the tournament winner."
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

    def _get_candidate_models(
        self,
        task_type: str,
        pool: str,
        random_state: int
    ) -> List[Tuple[str, str, Any]]:
        """
        Returns list of tuples: (model_id, display_name, estimator_instance)
        """
        candidates: List[Tuple[str, str, Any]] = []

        if task_type == "classification":
            if pool == "tree_ensembles":
                candidates = [
                    ("random_forest", "Random Forest", RandomForestClassifier(n_estimators=40, max_depth=6, random_state=random_state)),
                    ("gradient_boosting", "Gradient Boosting", GradientBoostingClassifier(n_estimators=40, max_depth=3, random_state=random_state)),
                    ("extra_trees", "Extra Trees", ExtraTreesClassifier(n_estimators=40, max_depth=6, random_state=random_state)),
                    ("hist_gbm", "Hist Gradient Boosting", HistGradientBoostingClassifier(max_iter=40, random_state=random_state))
                ]
            elif pool == "linear_models":
                candidates = [
                    ("logistic_regression", "Logistic Regression", LogisticRegression(max_iter=1000, random_state=random_state)),
                    ("ridge_classifier", "Ridge Classifier", RidgeClassifier(random_state=random_state))
                ]
            elif pool == "fast_candidates":
                candidates = [
                    ("logistic_regression", "Logistic Regression", LogisticRegression(max_iter=1000, random_state=random_state)),
                    ("random_forest", "Random Forest", RandomForestClassifier(n_estimators=30, max_depth=5, random_state=random_state)),
                    ("hist_gbm", "Hist Gradient Boosting", HistGradientBoostingClassifier(max_iter=30, random_state=random_state))
                ]
            else:  # all_candidates
                candidates = [
                    ("random_forest", "Random Forest", RandomForestClassifier(n_estimators=40, max_depth=6, random_state=random_state)),
                    ("gradient_boosting", "Gradient Boosting", GradientBoostingClassifier(n_estimators=40, max_depth=3, random_state=random_state)),
                    ("extra_trees", "Extra Trees", ExtraTreesClassifier(n_estimators=40, max_depth=6, random_state=random_state)),
                    ("hist_gbm", "Hist Gradient Boosting", HistGradientBoostingClassifier(max_iter=40, random_state=random_state)),
                    ("logistic_regression", "Logistic Regression", LogisticRegression(max_iter=1000, random_state=random_state))
                ]
        else:  # regression
            if pool == "tree_ensembles":
                candidates = [
                    ("random_forest", "Random Forest Regressor", RandomForestRegressor(n_estimators=40, max_depth=6, random_state=random_state)),
                    ("gradient_boosting", "Gradient Boosting Regressor", GradientBoostingRegressor(n_estimators=40, max_depth=3, random_state=random_state)),
                    ("extra_trees", "Extra Trees Regressor", ExtraTreesRegressor(n_estimators=40, max_depth=6, random_state=random_state)),
                    ("hist_gbm", "Hist Gradient Boosting Regressor", HistGradientBoostingRegressor(max_iter=40, random_state=random_state))
                ]
            elif pool == "linear_models":
                candidates = [
                    ("ridge", "Ridge Regression", Ridge(random_state=random_state)),
                    ("elastic_net", "ElasticNet", ElasticNet(random_state=random_state))
                ]
            elif pool == "fast_candidates":
                candidates = [
                    ("ridge", "Ridge Regression", Ridge(random_state=random_state)),
                    ("random_forest", "Random Forest Regressor", RandomForestRegressor(n_estimators=30, max_depth=5, random_state=random_state)),
                    ("hist_gbm", "Hist Gradient Boosting Regressor", HistGradientBoostingRegressor(max_iter=30, random_state=random_state))
                ]
            else:  # all_candidates
                candidates = [
                    ("random_forest", "Random Forest Regressor", RandomForestRegressor(n_estimators=40, max_depth=6, random_state=random_state)),
                    ("gradient_boosting", "Gradient Boosting Regressor", GradientBoostingRegressor(n_estimators=40, max_depth=3, random_state=random_state)),
                    ("extra_trees", "Extra Trees Regressor", ExtraTreesRegressor(n_estimators=40, max_depth=6, random_state=random_state)),
                    ("hist_gbm", "Hist Gradient Boosting Regressor", HistGradientBoostingRegressor(max_iter=40, random_state=random_state)),
                    ("ridge", "Ridge Regression", Ridge(random_state=random_state))
                ]

        return candidates

    def _resolve_scoring_metric(self, scoring_cfg: str, task_type: str) -> str:
        cfg = str(scoring_cfg).lower().strip()
        if cfg != "auto":
            if cfg == "rmse":
                return "neg_root_mean_squared_error"
            elif cfg == "mae":
                return "neg_mean_absolute_error"
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
                "AutoMLSearch expects 'X_train' and 'y_train' in inputs. "
                "Please connect a Train/Test Split node before this recipe."
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
                    "At least 2 classes are required to run an AutoML tournament."
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

        # 4. Resolve Candidate Model Pool & CV Strategy
        pool = config.get("candidate_pool", "all_candidates")
        random_state = int(config.get("random_state", 42))
        candidates = self._get_candidate_models(task_type, pool, random_state)

        requested_cv = int(config.get("cv_folds", 3))
        warnings: List[str] = []

        if task_type == "classification":
            _, class_counts = np.unique(y_train_arr, return_counts=True)
            min_class_count = int(np.min(class_counts))
            if min_class_count < 2:
                raise ValueError(
                    f"At least 2 samples per class are required for cross-validation tournament, "
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
                raise ValueError(f"AutoMLSearch requires at least 4 observations for regression, got {n_samples}.")
            effective_cv = max(2, min(requested_cv, n_samples // 2))
            if effective_cv < requested_cv:
                warnings.append(
                    f"[INFO: ADAPTIVE CV] Training set size ({n_samples}) adjusted internal CV folds to {effective_cv}."
                )
            cv_splitter = KFold(n_splits=effective_cv, shuffle=True, random_state=random_state)

        primary_metric = self._resolve_scoring_metric(config.get("primary_metric", "auto"), task_type)

        # 5. Execute AutoML Tournament
        leaderboard: List[Dict[str, Any]] = []
        champion_estimator = None
        best_score = -float("inf")
        champion_info: Dict[str, Any] = {}

        for model_id, display_name, estimator in candidates:
            start_t = time.perf_counter()
            try:
                cv_scores = cross_val_score(
                    estimator,
                    X_train,
                    y_train_arr,
                    cv=cv_splitter,
                    scoring=primary_metric,
                    n_jobs=1
                )
                mean_score = float(round(float(np.mean(cv_scores)), 4))
                std_score = float(round(float(np.std(cv_scores)), 4))
            except Exception as e:
                warnings.append(f"[WARNING: MODEL FAILED] Candidate '{display_name}' failed CV evaluation: {str(e)[:100]}")
                continue

            elapsed_ms = round((time.perf_counter() - start_t) * 1000.0, 1)

            entry = {
                "model_id": model_id,
                "display_name": display_name,
                "score": mean_score,
                "score_std": std_score,
                "training_latency_ms": elapsed_ms,
                "estimator_class": estimator.__class__.__name__
            }
            leaderboard.append(entry)

            if mean_score > best_score:
                best_score = mean_score
                champion_estimator = estimator
                champion_info = entry

        if not leaderboard or champion_estimator is None:
            raise RuntimeError("AutoML tournament failed to train any candidate algorithms.")

        # Sort leaderboard descending by score
        leaderboard.sort(key=lambda x: x["score"], reverse=True)
        for rank, item in enumerate(leaderboard, start=1):
            item["rank"] = rank

        # Calculate victory margin over runner-up
        runner_up_score = leaderboard[1]["score"] if len(leaderboard) > 1 else leaderboard[0]["score"]
        margin_pct = round(((best_score - runner_up_score) / max(0.0001, abs(runner_up_score))) * 100.0, 2)

        # 6. Refit Champion on Full Training Set
        refit_start = time.perf_counter()
        champion_model = clone(champion_estimator)
        champion_model.fit(X_train, y_train_arr)
        refit_elapsed_ms = round((time.perf_counter() - refit_start) * 1000.0, 1)

        # 7. Evaluate on Test Split (if provided)
        test_metrics: Dict[str, Any] = {}
        if X_test is not None and y_test_clean is not None:
            test_preds = champion_model.predict(X_test)
            if task_type == "classification":
                test_metrics["test_accuracy"] = float(round(accuracy_score(y_test_clean, test_preds), 4))
                test_metrics["test_f1_macro"] = float(round(f1_score(y_test_clean, test_preds, average="macro", zero_division=0), 4))
            else:
                test_metrics["test_r2"] = float(round(r2_score(y_test_clean, test_preds), 4))
                test_metrics["test_rmse"] = float(round(math.sqrt(mean_squared_error(y_test_clean, test_preds)), 4))
                test_metrics["test_mae"] = float(round(mean_absolute_error(y_test_clean, test_preds), 4))

        metrics = {
            "task_type": task_type,
            "primary_metric": primary_metric,
            "effective_cv_folds": effective_cv,
            "champion_model": champion_info.get("display_name"),
            "champion_model_id": champion_info.get("model_id"),
            "champion_score": best_score,
            "margin_over_runner_up_pct": margin_pct,
            "refit_latency_ms": refit_elapsed_ms,
            "leaderboard": leaderboard,
            "test_evaluation": test_metrics,
            "warnings": warnings
        }

        # 8. Format Tournament Report
        medals = {1: "1st (Gold)", 2: "2nd (Silver)", 3: "3rd (Bronze)"}
        report_lines = [
            f"# === AutoML Multi-Model Tournament Report ({task_type.capitalize()}) ===",
            f"- **Champion Algorithm**: **{champion_info.get('display_name')}** ({champion_info.get('estimator_class')})",
            f"- **Winning Cross-Validation Score**: **{best_score:.4f}** ({primary_metric})",
            f"- **Victory Margin**: +{margin_pct:.2f}% over runner-up",
            f"- Internal CV Folds: {effective_cv}",
            "",
            "## Official Tournament Leaderboard:",
            "| Standing | Model Architecture | Mean Score | Std Dev | Latency | Status |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |"
        ]

        for item in leaderboard:
            standing_tag = medals.get(item["rank"], f"#{item['rank']}")
            status_tag = "**CHAMPION**" if item["rank"] == 1 else "Contender"
            report_lines.append(
                f"| {standing_tag} | {item['display_name']} | **{item['score']:.4f}** | ±{item['score_std']:.4f} | {item['training_latency_ms']} ms | {status_tag} |"
            )

        if test_metrics:
            report_lines.append("")
            report_lines.append("## Holdout Test Set Performance:")
            for k, v in test_metrics.items():
                report_lines.append(f"- {k.replace('_', ' ').title()}: **{v}**")

        if warnings:
            report_lines.append("")
            report_lines.append("## Guardrail Alerts:")
            for w in warnings:
                report_lines.append(f"- {w}")

        report_markdown = "\n".join(report_lines)

        output: Dict[str, Any] = {
            "model": champion_model,
            "champion_model": champion_model,
            "task_type": task_type,
            "champion_name": champion_info.get("display_name"),
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

    def to_code(self, config: Dict[str, Any]) -> str:
        task_type = config.get("task_type", "classification")
        cv = config.get("cv_folds", 3)
        return (
            "from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier\n"
            "from sklearn.linear_model import LogisticRegression\n"
            "from sklearn.model_selection import cross_val_score\n\n"
            "candidates = [\n"
            "    ('rf', RandomForestClassifier(random_state=42)),\n"
            "    ('gb', GradientBoostingClassifier(random_state=42)),\n"
            "    ('lr', LogisticRegression(max_iter=1000, random_state=42))\n"
            "]\n"
            "scores = {name: cross_val_score(m, X_train, y_train, cv=3).mean() for name, m in candidates}\n"
            "best_name, _ = max(scores.items(), key=lambda x: x[1])\n"
            "champion = dict(candidates)[best_name].fit(X_train, y_train)"
        )
