import time
import math
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)

from backend.app.recipes.base.recipe import BaseRecipe, RecipePort


class ChampionChallengerRecipe(BaseRecipe):
    """
    Champion / Challenger Comparison Recipe.
    Compares a candidate model (Challenger) against the current production model (Champion)
    on identical evaluation data. Evaluates head-to-head metric differentials, inference latency,
    statistical significance, and provides automated promotion gating decisions.
    """

    recipe_id = "champion_challenger"
    name = "Champion / Challenger Comparison"
    version = "1.0.0"
    category = "evaluation"
    group = "Evaluation & Governance"
    subgroup = "Evaluation"
    description = (
        "Compares a candidate model (Challenger) against the current production model (Champion) "
        "on identical holdout data to evaluate metric lift, latency differential, and promotion gating."
    )
    input_types = ["model", "dataframe"]
    output_types = ["dataframe", "report"]

    inputs: List[RecipePort] = [
        RecipePort(
            id="champion_model",
            label="Champion Model (Prod)",
            type="model",
            required=False,
            max_connections=1,
            description="Active production model serving live traffic"
        ),
        RecipePort(
            id="challenger_model",
            label="Challenger Model (Candidate)",
            type="model",
            required=False,
            max_connections=1,
            description="New candidate model evaluated for deployment"
        ),
        RecipePort(
            id="dataframe",
            label="Evaluation / Test Table",
            type="dataframe",
            required=False,
            max_connections=1,
            description="Shared test dataset containing features and ground truth target"
        )
    ]

    outputs: List[RecipePort] = [
        RecipePort(
            id="dataframe",
            label="Side-by-Side Predictions",
            type="dataframe",
            description="Evaluation dataset enriched with Champion and Challenger predictions, residuals, and agreement flags"
        ),
        RecipePort(
            id="report",
            label="Promotion Gating Report",
            type="report",
            description="Comprehensive model governance report with metric differentials, latency benchmark, and promotion verdict"
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "comparison_metric": {
                    "type": "string",
                    "title": "Primary Decision Metric",
                    "enum": ["f1", "accuracy", "precision", "recall", "auc_roc", "rmse", "mae", "r2"],
                    "default": "f1",
                    "description": "Primary KPI used to determine whether the Challenger outperforms the Champion."
                },
                "task_type": {
                    "type": "string",
                    "title": "ML Task Type",
                    "enum": ["classification", "regression"],
                    "default": "classification",
                    "description": "Classification or regression problem type."
                },
                "target_column": {
                    "type": "string",
                    "title": "Ground Truth Target Column",
                    "default": "target",
                    "description": "Column in evaluation dataset containing true labels or values."
                },
                "champion_model_name": {
                    "type": "string",
                    "title": "Champion Model Identifier",
                    "default": "Champion-v1",
                    "description": "Display name or version tag of current production champion."
                },
                "challenger_model_name": {
                    "type": "string",
                    "title": "Challenger Model Identifier",
                    "default": "Challenger-v2",
                    "description": "Display name or version tag of candidate challenger model."
                },
                "champion_baseline_score": {
                    "type": "number",
                    "title": "Champion Baseline Score (Optional)",
                    "default": 0.82,
                    "description": "Optional known production score if comparing challenger against production SLA without live champion model artifact."
                },
                "min_lift_threshold_pct": {
                    "type": "number",
                    "title": "Minimum Required Lift (%)",
                    "default": 1.0,
                    "minimum": 0.0,
                    "maximum": 50.0,
                    "description": "Minimum relative improvement required to justify promoting challenger to production."
                },
                "max_latency_regression_pct": {
                    "type": "number",
                    "title": "Maximum Allowed Latency Increase (%)",
                    "default": 20.0,
                    "minimum": 0.0,
                    "maximum": 200.0,
                    "description": "Maximum permissible percentage increase in inference latency before challenger promotion is blocked."
                }
            }
        }

    def execute(
        self,
        inputs: Dict[str, Any],
        config: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        # 1. Resolve Parameters
        metric_choice = str(config.get("comparison_metric") or "f1").strip().lower()
        task_type = str(config.get("task_type") or "classification").strip().lower()
        target_col = str(config.get("target_column") or "target").strip()
        champ_name = str(config.get("champion_model_name") or "Champion-v1").strip()
        chall_name = str(config.get("challenger_model_name") or "Challenger-v2").strip()
        min_lift = float(config.get("min_lift_threshold_pct") or 1.0)
        max_lat_reg = float(config.get("max_latency_regression_pct") or 20.0)
        champ_base_cfg = config.get("champion_baseline_score")

        # 2. Resolve Test DataFrame
        df = None
        for candidate in [inputs.get("dataframe"), inputs.get("test_data"), inputs.get("test_df"), inputs.get("input"), inputs.get("left_dataframe")]:
            if candidate is not None and isinstance(candidate, pd.DataFrame):
                df = candidate
                break

        if df is None and context and isinstance(context, dict):
            for candidate in [context.get("test_data"), context.get("test_df"), context.get("X_test"), context.get("dataframe")]:
                if candidate is not None and isinstance(candidate, pd.DataFrame):
                    df = candidate
                    break

        # Standalone mock generation if executed in isolation
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            np.random.seed(42)
            n_eval = 500
            x1 = np.random.randn(n_eval)
            x2 = np.random.randn(n_eval)
            if task_type == "regression":
                y = 2.5 * x1 - 1.2 * x2 + np.random.randn(n_eval) * 0.5
            else:
                p = 1.0 / (1.0 + np.exp(-(1.8 * x1 - 1.5 * x2)))
                y = np.random.binomial(1, p, n_eval)
            df = pd.DataFrame({"feature_1": x1, "feature_2": x2, "target": y})
            target_col = "target"

        out_df = df.copy()

        # Resolve ground truth target
        if target_col not in out_df.columns:
            if context and isinstance(context, dict) and context.get("target_column") in out_df.columns:
                target_col = context.get("target_column")
            else:
                candidates = ["target", "label", "churn", "converted", "y", "actual", "class"]
                found_target = next((c for c in candidates if c in out_df.columns), None)
                if found_target:
                    target_col = found_target
                else:
                    # Pick last column
                    target_col = out_df.columns[-1]

        y_true = pd.to_numeric(out_df[target_col], errors="coerce").fillna(0).values
        feature_cols = [c for c in out_df.columns if c != target_col and pd.api.types.is_numeric_dtype(out_df[c])]
        if not feature_cols:
            feature_cols = [c for c in out_df.columns if c != target_col]
        X_eval = out_df[feature_cols].copy().fillna(0).to_numpy(dtype=float)

        # 3. Resolve Models
        champ_model = inputs.get("champion_model") or inputs.get("model") or inputs.get("model_1")
        chall_model = inputs.get("challenger_model") or inputs.get("candidate_model") or inputs.get("model_2")

        if champ_model is None and context and isinstance(context, dict):
            champ_model = context.get("model")

        # Mock fallback models if testing in isolation without pre-trained model artifacts
        if chall_model is None:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            if task_type == "regression":
                chall_model = RandomForestRegressor(n_estimators=30, max_depth=5, random_state=42)
            else:
                chall_model = RandomForestClassifier(n_estimators=30, max_depth=5, random_state=42)
            chall_model.fit(X_eval, y_true)

        if champ_model is None:
            from sklearn.linear_model import LogisticRegression, Ridge
            if task_type == "regression":
                champ_model = Ridge()
            else:
                champ_model = LogisticRegression()
            champ_model.fit(X_eval, y_true)

        # 4. Benchmarking Inference Latency
        def _benchmark_model(m, X) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
            t0 = time.perf_counter()
            preds = m.predict(X)
            t_ms = (time.perf_counter() - t0) * 1000.0
            probs = None
            if hasattr(m, "predict_proba"):
                try:
                    probs_all = m.predict_proba(X)
                    probs = probs_all[:, 1] if probs_all.shape[1] > 1 else probs_all[:, 0]
                except Exception:
                    pass
            return preds, probs, round(t_ms, 3)

        champ_preds, champ_probs, champ_lat_ms = _benchmark_model(champ_model, X_eval)
        chall_preds, chall_probs, chall_lat_ms = _benchmark_model(chall_model, X_eval)

        # 5. Compute Metrics
        champ_metrics: Dict[str, float] = {}
        chall_metrics: Dict[str, float] = {}

        if task_type == "regression":
            champ_metrics["mae"] = round(float(mean_absolute_error(y_true, champ_preds)), 4)
            champ_metrics["rmse"] = round(float(np.sqrt(mean_squared_error(y_true, champ_preds))), 4)
            champ_metrics["r2"] = round(float(r2_score(y_true, champ_preds)), 4)

            chall_metrics["mae"] = round(float(mean_absolute_error(y_true, chall_preds)), 4)
            chall_metrics["rmse"] = round(float(np.sqrt(mean_squared_error(y_true, chall_preds))), 4)
            chall_metrics["r2"] = round(float(r2_score(y_true, chall_preds)), 4)

            # Residual paired t-test
            res_champ = np.abs(y_true - champ_preds)
            res_chall = np.abs(y_true - chall_preds)
            stat_res = stats.ttest_rel(res_chall, res_champ)
            p_value = round(float(stat_res.pvalue), 5)
            stat_test_name = "Paired T-Test on Absolute Residuals"

            # Determine primary metric
            if metric_choice not in ["mae", "rmse", "r2"]:
                metric_choice = "rmse"

            champ_score = champ_metrics[metric_choice]
            chall_score = chall_metrics[metric_choice]
            # For MAE/RMSE, lower is better
            if metric_choice in ["mae", "rmse"]:
                lift_pct = round(((champ_score - chall_score) / champ_score) * 100.0, 2) if champ_score > 0 else 0.0
            else:
                lift_pct = round(((chall_score - champ_score) / champ_score) * 100.0, 2) if abs(champ_score) > 1e-9 else 0.0

        else:  # Classification
            # Binary conversion of predictions
            champ_pred_bin = (champ_preds >= 0.5).astype(int) if set(np.unique(champ_preds)).issubset({0, 1}) else champ_preds
            chall_pred_bin = (chall_preds >= 0.5).astype(int) if set(np.unique(chall_preds)).issubset({0, 1}) else chall_preds
            y_bin = (y_true >= 0.5).astype(int)

            champ_metrics["accuracy"] = round(float(accuracy_score(y_bin, champ_pred_bin)), 4)
            champ_metrics["f1"] = round(float(f1_score(y_bin, champ_pred_bin, average="weighted", zero_division=0)), 4)
            champ_metrics["precision"] = round(float(precision_score(y_bin, champ_pred_bin, average="weighted", zero_division=0)), 4)
            champ_metrics["recall"] = round(float(recall_score(y_bin, champ_pred_bin, average="weighted", zero_division=0)), 4)

            chall_metrics["accuracy"] = round(float(accuracy_score(y_bin, chall_pred_bin)), 4)
            chall_metrics["f1"] = round(float(f1_score(y_bin, chall_pred_bin, average="weighted", zero_division=0)), 4)
            chall_metrics["precision"] = round(float(precision_score(y_bin, chall_pred_bin, average="weighted", zero_division=0)), 4)
            chall_metrics["recall"] = round(float(recall_score(y_bin, chall_pred_bin, average="weighted", zero_division=0)), 4)

            if champ_probs is not None and len(np.unique(y_bin)) == 2:
                try:
                    champ_metrics["auc_roc"] = round(float(roc_auc_score(y_bin, champ_probs)), 4)
                except Exception:
                    pass
            if chall_probs is not None and len(np.unique(y_bin)) == 2:
                try:
                    chall_metrics["auc_roc"] = round(float(roc_auc_score(y_bin, chall_probs)), 4)
                except Exception:
                    pass

            # McNemar's Test for statistical significance of disagreement
            champ_correct = (champ_pred_bin == y_bin)
            chall_correct = (chall_pred_bin == y_bin)
            b = int(np.sum(~champ_correct & chall_correct))  # Champion wrong, Challenger right
            c = int(np.sum(champ_correct & ~chall_correct))  # Champion right, Challenger wrong

            if b + c > 0:
                mcnemar_stat = ((abs(b - c) - 1.0) ** 2) / (b + c)
                p_value = round(float(stats.chi2.sf(mcnemar_stat, 1)), 5)
            else:
                p_value = 1.0
            stat_test_name = "McNemar's Test for Paired Classifiers"

            # Apply baseline override if explicitly configured
            if champ_base_cfg is not None and float(champ_base_cfg) > 0:
                champ_metrics[metric_choice] = round(float(champ_base_cfg), 4)

            if metric_choice not in champ_metrics:
                metric_choice = "f1"

            champ_score = champ_metrics[metric_choice]
            chall_score = chall_metrics[metric_choice]
            lift_pct = round(((chall_score - champ_score) / champ_score) * 100.0, 2) if champ_score > 0 else 0.0

        # Latency differential
        latency_diff_pct = round(((chall_lat_ms - champ_lat_ms) / champ_lat_ms) * 100.0, 2) if champ_lat_ms > 0 else 0.0

        # Agreement Rate
        agreement_count = int(np.sum(champ_preds == chall_preds))
        agreement_pct = round((agreement_count / len(out_df)) * 100.0, 2)

        # Only treat latency as an SLA breach if both percentage exceeds threshold AND absolute difference is noticeable (> 5ms)
        is_latency_breach = (latency_diff_pct > max_lat_reg) and ((chall_lat_ms - champ_lat_ms) > 5.0)

        # 6. Promotion Decision Gating Logic
        if lift_pct >= min_lift and not is_latency_breach:
            decision = "PROMOTE_CHALLENGER"
            recommendation = (
                f"Promote Challenger ({chall_name}) to Production — Relative lift of +{lift_pct:.2f}% "
                f"in {metric_choice.upper()} exceeds the required threshold (+{min_lift:.1f}%) "
                f"with acceptable latency impact ({latency_diff_pct:+.1f}%)."
            )
        elif is_latency_breach:
            decision = "REJECT_LATENCY_BREACH"
            recommendation = (
                f"Reject Challenger ({chall_name}) — Inference latency increased by +{latency_diff_pct:.1f}% (+{chall_lat_ms - champ_lat_ms:.2f}ms), "
                f"breaching the maximum permitted SLA degradation (+{max_lat_reg:.1f}%)."
            )
        elif lift_pct < 0:
            decision = "REJECT_METRIC_DROP"
            recommendation = (
                f"Reject Challenger ({chall_name}) — Candidate model underperformed current Champion "
                f"by {lift_pct:.2f}% in {metric_choice.upper()}."
            )
        else:
            decision = "RETAIN_CHAMPION"
            recommendation = (
                f"Retain Current Champion ({champ_name}) — Challenger lift of +{lift_pct:.2f}% "
                f"is below the minimum required threshold (+{min_lift:.1f}%)."
            )

        # 7. Comparison Bar Chart Payload for UI
        common_metrics = sorted(list(set(champ_metrics.keys()).intersection(set(chall_metrics.keys()))))
        comparison_bar_chart = []
        for m_key in common_metrics:
            c_val = champ_metrics[m_key]
            ch_val = chall_metrics[m_key]
            if m_key in ["mae", "rmse"]:
                m_lift = round(((c_val - ch_val) / c_val) * 100.0, 2) if c_val > 0 else 0.0
            else:
                m_lift = round(((ch_val - c_val) / c_val) * 100.0, 2) if abs(c_val) > 1e-9 else 0.0
            comparison_bar_chart.append({
                "metric": m_key.upper(),
                "champion": c_val,
                "challenger": ch_val,
                "lift_pct": m_lift,
                "champion_name": champ_name,
                "challenger_name": chall_name
            })

        # 8. Enrich Output DataFrame
        out_df["champion_prediction"] = champ_preds
        out_df["challenger_prediction"] = chall_preds
        out_df["models_agree"] = (champ_preds == chall_preds).astype(int)
        if task_type == "regression":
            out_df["champion_residual"] = np.round(np.abs(y_true - champ_preds), 4)
            out_df["challenger_residual"] = np.round(np.abs(y_true - chall_preds), 4)

        metrics_payload: Dict[str, Any] = {
            "task_type": task_type,
            "decision_metric": metric_choice,
            "champion_model_name": champ_name,
            "challenger_model_name": chall_name,
            "champion_score": champ_score,
            "challenger_score": chall_score,
            "relative_lift_pct": lift_pct,
            "min_lift_threshold_pct": min_lift,
            "champion_latency_ms": champ_lat_ms,
            "challenger_latency_ms": chall_lat_ms,
            "latency_delta_pct": latency_diff_pct,
            "max_latency_regression_pct": max_lat_reg,
            "statistical_test": stat_test_name,
            "p_value": p_value,
            "is_statistically_significant": bool(p_value < 0.05),
            "model_agreement_pct": agreement_pct,
            "evaluation_samples": len(out_df),
            "decision": decision,
            "recommendation": recommendation,
            "champion_metrics": champ_metrics,
            "challenger_metrics": chall_metrics,
            "comparison_bar_chart": comparison_bar_chart,
            "bar_chart": comparison_bar_chart
        }

        report_lines = [
            "# Champion vs. Challenger Model Comparison",
            f"**Decision Metric:** `{metric_choice.upper()}` | **Task Type:** `{task_type.capitalize()}`",
            "",
            "## Head-to-Head Performance",
            f"- **Champion ({champ_name}):** `{metric_choice.upper()}` = `{champ_score:.4f}` (Latency: `{champ_lat_ms:.2f}ms`)",
            f"- **Challenger ({chall_name}):** `{metric_choice.upper()}` = `{chall_score:.4f}` (Latency: `{chall_lat_ms:.2f}ms`)",
            f"- **Relative Metric Lift:** `{lift_pct:+.2f}%` (Min Required: `+{min_lift:.1f}%`)",
            f"- **Latency Impact:** `{latency_diff_pct:+.1f}%` (Max Allowed: `+{max_lat_reg:.1f}%`)",
            f"- **Statistical Test ({stat_test_name}):** p-value = `{p_value:.5f}`",
            f"- **Prediction Agreement Rate:** `{agreement_pct:.1f}%`",
            "",
            "## Governance & Promotion Verdict",
            f"> **Decision:** `{decision}`  ",
            f"> {recommendation}"
        ]
        report_text = "\n".join(report_lines)

        summary_payload = {
            "title": "Champion / Challenger Comparison",
            "decision": decision,
            "champion": champ_name,
            "challenger": chall_name,
            "lift_pct": f"{lift_pct:+.2f}%",
            "decision_metric": metric_choice.upper(),
            "recommendation": recommendation,
            "bar_chart": comparison_bar_chart,
            "status": "SUCCESS"
        }

        return {
            "dataframe": out_df,
            "report": report_text,
            "metrics": metrics_payload,
            "output_summary": summary_payload,
            "comparison_summary": metrics_payload
        }

    def to_code(self, config: Dict[str, Any]) -> str:
        metric = repr(config.get("comparison_metric", "f1"))
        min_lift = config.get("min_lift_threshold_pct", 1.0)
        return (
            "# Champion / Challenger Model Comparison\n"
            f"# Decision metric: {metric}, Min Lift Required: {min_lift}%\n"
            "champ_pred = champion_model.predict(X_test)\n"
            "chall_pred = challenger_model.predict(X_test)\n"
            "df['champion_pred'] = champ_pred\n"
            "df['challenger_pred'] = chall_pred\n"
        )
