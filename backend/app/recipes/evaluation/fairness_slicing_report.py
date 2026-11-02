import math
from collections import defaultdict
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union

from backend.app.recipes.base.recipe import BaseRecipe, RecipePort
from backend.app.recipes.training.encoder_utils import pass_through_metadata


class FairnessSlicingReportRecipe(BaseRecipe):
    """
    Fairness Slicing Report Recipe (by protected demographic group).
    Disaggregates model performance (Accuracy, Precision, Recall, F1, FPR, FNR, MAE, RMSE)
    across protected demographic attributes or intersectional slices.
    Detects severe slice-level performance degradation, minority under-performance,
    and small-sample vulnerability.
    """

    recipe_id = "fairness_slicing_report"
    name = "Fairness Slicing Report (by protected group)"
    version = "1.0.0"
    category = "evaluation"
    group = "Evaluation & Governance"
    subgroup = "Fairness & Compliance"
    description = (
        "Disaggregates model performance across demographic slices and intersectional groups, "
        "identifying severe performance drops and disparities against global baseline."
    )
    input_types = ["dataframe"]
    output_types = ["dataframe", "metrics", "report"]

    inputs: List[RecipePort] = [
        RecipePort(
            id="test_data",
            label="Evaluation / Test Dataset",
            type="dataframe",
            required=True,
            max_connections=1,
            description="Dataset containing targets, predictions, and sensitive attribute / slice columns."
        ),
        RecipePort(
            id="model",
            label="Optional Trained Model",
            type="model",
            required=False,
            max_connections=1,
            description="Optional trained model to generate predictions if not already in dataframe."
        )
    ]

    outputs: List[RecipePort] = [
        RecipePort(
            id="slices_table",
            label="Demographic Slices Table",
            type="dataframe",
            description="Disaggregated metrics table with sample size, error rates, and disparity vs baseline for each slice."
        ),
        RecipePort(
            id="metrics",
            label="Slicing Metrics Summary",
            type="metrics",
            description="Global baseline metrics, worst-performing slice, max disparity spread, and degradation alerts."
        ),
        RecipePort(
            id="report",
            label="Executive Fairness Slicing Report",
            type="report",
            description="Markdown report featuring slice leaderboard, degradation alerts, and intersectional performance."
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "slice_columns": {
                    "type": "string",
                    "title": "Demographic Slice Columns (comma-separated)",
                    "input_type": "text",
                    "default": "gender",
                    "description": "Column(s) used to slice data. Single column or multiple columns for intersectional slices (e.g. 'gender,age_group')."
                },
                "target_col": {
                    "type": "string",
                    "title": "Ground Truth Target Column",
                    "input_type": "text",
                    "default": "target",
                    "description": "Ground truth column name."
                },
                "prediction_col": {
                    "type": "string",
                    "title": "Model Prediction Column",
                    "input_type": "text",
                    "default": "prediction",
                    "description": "Predicted output column name (optional if model is connected)."
                },
                "problem_type": {
                    "type": "string",
                    "title": "Task Nature",
                    "input_type": "dropdown",
                    "enum": ["auto", "classification", "regression"],
                    "default": "auto",
                    "description": "Type of predictive task (classification or regression)."
                },
                "degradation_tolerance_pct": {
                    "type": "number",
                    "title": "Max Acceptable Degradation Drop (%)",
                    "input_type": "number",
                    "default": 15.0,
                    "description": "Threshold (%) below global baseline that flags a slice as severely underperforming."
                },
                "min_slice_sample_size": {
                    "type": "integer",
                    "title": "Minimum Slice Sample Size Guardrail",
                    "input_type": "number",
                    "default": 30,
                    "description": "Slices with fewer rows than this threshold trigger a small-sample statistical warning."
                }
            },
            "required": []
        }

    def _auto_detect_columns(self, df: pd.DataFrame, slice_cfg: str, tgt_cfg: str, pred_cfg: str) -> Tuple[List[str], str, str]:
        cols_lower = {c.lower(): c for c in df.columns}

        # Resolve slice columns (supports comma-separated list)
        raw_slice_items = [s.strip() for s in slice_cfg.split(",") if s.strip()]
        resolved_slices = []
        for item in raw_slice_items:
            if item in df.columns:
                resolved_slices.append(item)
            elif item.lower() in cols_lower:
                resolved_slices.append(cols_lower[item.lower()])

        if not resolved_slices:
            candidates = ["protected_attribute", "gender", "sex", "race", "ethnicity", "age_group", "age_band", "segment", "group"]
            for cand in candidates:
                if cand in cols_lower:
                    resolved_slices.append(cols_lower[cand])
                    break

        # Resolve target
        tgt_col = ""
        if tgt_cfg in df.columns:
            tgt_col = tgt_cfg
        elif tgt_cfg.lower() in cols_lower:
            tgt_col = cols_lower[tgt_cfg.lower()]
        else:
            candidates = ["target", "label", "actual", "ground_truth", "y_true", "y", "outcome"]
            for cand in candidates:
                if cand in cols_lower:
                    tgt_col = cols_lower[cand]
                    break

        # Resolve prediction
        pred_col = ""
        if pred_cfg in df.columns:
            pred_col = pred_cfg
        elif pred_cfg.lower() in cols_lower:
            pred_col = cols_lower[pred_cfg.lower()]
        else:
            candidates = ["prediction", "pred", "y_pred", "predicted", "model_prediction", "output"]
            for cand in candidates:
                if cand in cols_lower:
                    pred_col = cols_lower[cand]
                    break

        return resolved_slices, tgt_col, pred_col

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        # 1. Resolve Dataframe
        df = inputs.get("dataframe")
        if df is None:
            df = inputs.get("test_data")
        if df is None:
            df = inputs.get("data")
        if df is None and isinstance(context, dict):
            df = context.get("dataframe")
            if df is None:
                df = context.get("test_data")

        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("FairnessSlicingReport expects a non-empty pandas DataFrame in inputs.")

        df = df.copy()

        # 2. Detect & Validate Columns
        slice_cfg = config.get("slice_columns", "gender")
        tgt_cfg = config.get("target_col", "target")
        pred_cfg = config.get("prediction_col", "prediction")

        slice_cols, tgt_col, pred_col = self._auto_detect_columns(df, slice_cfg, tgt_cfg, pred_cfg)

        if not slice_cols:
            raise ValueError(
                f"FairnessSlicingReport could not locate slice column(s) '{slice_cfg}'. Available columns: {list(df.columns)}"
            )
        if not tgt_col or tgt_col not in df.columns:
            raise ValueError(
                f"FairnessSlicingReport could not locate target column '{tgt_cfg}'. Available columns: {list(df.columns)}"
            )

        # Generate model predictions if missing
        model = inputs.get("model")
        if (not pred_col or pred_col not in df.columns) and model is not None:
            feature_cols = [c for c in df.columns if c not in slice_cols and c != tgt_col]
            try:
                df["_model_prediction"] = model.predict(df[feature_cols])
                pred_col = "_model_prediction"
            except Exception as e:
                raise ValueError(f"Failed to generate predictions with provided model: {e}")

        if not pred_col or pred_col not in df.columns:
            raise ValueError(
                f"FairnessSlicingReport could not locate prediction column '{pred_cfg}'. Provide predictions or connect a trained model."
            )

        df = df.dropna(subset=slice_cols + [tgt_col, pred_col])
        n_total = len(df)
        if n_total < 2:
            raise ValueError("FairnessSlicingReport dataset contains insufficient rows after dropping missing values.")

        # 3. Determine Task Type (Classification vs Regression)
        task_type = config.get("problem_type", "auto")
        if task_type == "auto":
            u_tgt = df[tgt_col].nunique()
            if u_tgt <= 10 and (df[tgt_col].dtype.kind in ("i", "b", "O", "U") or df[tgt_col].apply(lambda x: float(x).is_integer()).all()):
                task_type = "classification"
            else:
                task_type = "regression"

        # 4. Create Intersectional Slice Identifier
        if len(slice_cols) == 1:
            df["_slice_id"] = df[slice_cols[0]].astype(str)
        else:
            df["_slice_id"] = df[slice_cols].astype(str).agg(" x ".join, axis=1)

        unique_slices = sorted(df["_slice_id"].unique().tolist())
        if len(unique_slices) < 2:
            raise ValueError(
                f"FairnessSlicingReport requires at least 2 distinct slices in {slice_cols}, found only: {unique_slices}"
            )

        deg_tol = float(config.get("degradation_tolerance_pct", 15.0))
        min_n = int(config.get("min_slice_sample_size", 30))

        # 5. Global Baseline Calculation
        warnings: List[str] = []
        global_metrics: Dict[str, Any] = {}
        slice_rows: List[Dict[str, Any]] = []

        if task_type == "classification":
            y_t = df[tgt_col].to_numpy()
            y_p = df[pred_col].to_numpy()
            global_acc = float(round(float(np.mean(y_t == y_p)) * 100.0, 2))

            # Binary metrics if target in (0, 1)
            is_binary = set(np.unique(y_t)).issubset({0, 1, 0.0, 1.0, True, False, "0", "1"})
            global_f1 = None
            global_precision = None
            global_recall = None

            if is_binary:
                y_t_bin = pd.to_numeric(df[tgt_col]).astype(int).to_numpy()
                y_p_bin = pd.to_numeric(df[pred_col]).astype(int).to_numpy()

                tp = int(np.sum((y_t_bin == 1) & (y_p_bin == 1)))
                fp = int(np.sum((y_t_bin == 0) & (y_p_bin == 1)))
                fn = int(np.sum((y_t_bin == 1) & (y_p_bin == 0)))
                tn = int(np.sum((y_t_bin == 0) & (y_p_bin == 0)))

                prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
                rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
                f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

                global_precision = float(round(prec * 100.0, 2))
                global_recall = float(round(rec * 100.0, 2))
                global_f1 = float(round(f1 * 100.0, 2))

            global_metrics = {
                "task_type": "classification",
                "total_samples": n_total,
                "global_accuracy_pct": global_acc,
                "global_f1_score_pct": global_f1,
                "global_precision_pct": global_precision,
                "global_recall_pct": global_recall
            }

            # Sliced evaluation
            for s_id in unique_slices:
                sub = df[df["_slice_id"] == s_id]
                n_s = len(sub)
                pct_share = float(round((n_s / n_total) * 100.0, 2))

                y_t_s = sub[tgt_col].to_numpy()
                y_p_s = sub[pred_col].to_numpy()
                s_acc = float(round(float(np.mean(y_t_s == y_p_s)) * 100.0, 2))
                acc_diff = float(round(s_acc - global_acc, 2))

                record: Dict[str, Any] = {
                    "slice_id": s_id,
                    "sample_size": n_s,
                    "share_of_dataset_pct": pct_share,
                    "accuracy_pct": s_acc,
                    "accuracy_diff_vs_baseline": acc_diff,
                    "is_severely_degraded": False
                }

                if is_binary:
                    y_tb_s = pd.to_numeric(sub[tgt_col]).astype(int).to_numpy()
                    y_pb_s = pd.to_numeric(sub[pred_col]).astype(int).to_numpy()

                    tp_s = int(np.sum((y_tb_s == 1) & (y_pb_s == 1)))
                    fp_s = int(np.sum((y_tb_s == 0) & (y_pb_s == 1)))
                    fn_s = int(np.sum((y_tb_s == 1) & (y_pb_s == 0)))
                    tn_s = int(np.sum((y_tb_s == 0) & (y_pb_s == 0)))

                    prec_s = (tp_s / (tp_s + fp_s)) if (tp_s + fp_s) > 0 else 0.0
                    rec_s = (tp_s / (tp_s + fn_s)) if (tp_s + fn_s) > 0 else 0.0
                    f1_s = (2 * prec_s * rec_s / (prec_s + rec_s)) if (prec_s + rec_s) > 0 else 0.0
                    tpr_s = rec_s
                    fpr_s = (fp_s / (fp_s + tn_s)) if (fp_s + tn_s) > 0 else 0.0

                    s_f1 = float(round(f1_s * 100.0, 2))
                    f1_diff = float(round(s_f1 - global_f1, 2)) if global_f1 is not None else 0.0

                    record["f1_score_pct"] = s_f1
                    record["f1_diff_vs_baseline"] = f1_diff
                    record["precision_pct"] = float(round(prec_s * 100.0, 2))
                    record["recall_pct"] = float(round(rec_s * 100.0, 2))
                    record["false_positive_rate_fpr"] = float(round(fpr_s, 4))
                    record["false_negative_rate_fnr"] = float(round(1.0 - tpr_s, 4))

                    # Degradation check: if F1 drops by > deg_tol %
                    if global_f1 and (f1_diff < -deg_tol):
                        record["is_severely_degraded"] = True
                        warnings.append(
                            f"[ALERT: SEVERE SLICE DEGRADATION] Demographic slice '{s_id}' F1-Score ({s_f1:.1f}%) "
                            f"is {abs(f1_diff):.1f}% lower than global baseline ({global_f1:.1f}%)."
                        )
                else:
                    if acc_diff < -deg_tol:
                        record["is_severely_degraded"] = True
                        warnings.append(
                            f"[ALERT: SEVERE SLICE DEGRADATION] Slice '{s_id}' Accuracy ({s_acc:.1f}%) "
                            f"is {abs(acc_diff):.1f}% lower than global baseline ({global_acc:.1f}%)."
                        )

                if n_s < min_n:
                    warnings.append(
                        f"[WARNING: SMALL SLICE SAMPLE SIZE] Slice '{s_id}' contains only N={n_s} samples (< {min_n} guardrail). "
                        "Metrics may be statistically unstable."
                    )

                slice_rows.append(record)

        else:
            # Regression Task Slicing (MAE, RMSE, R-squared)
            y_t = pd.to_numeric(df[tgt_col], errors="coerce").fillna(0.0).to_numpy()
            y_p = pd.to_numeric(df[pred_col], errors="coerce").fillna(0.0).to_numpy()

            errors = np.abs(y_t - y_p)
            global_mae = float(round(float(np.mean(errors)), 4))
            global_rmse = float(round(float(np.sqrt(np.mean((y_t - y_p)**2))), 4))

            global_metrics = {
                "task_type": "regression",
                "total_samples": n_total,
                "global_mae": global_mae,
                "global_rmse": global_rmse
            }

            for s_id in unique_slices:
                sub = df[df["_slice_id"] == s_id]
                n_s = len(sub)
                pct_share = float(round((n_s / n_total) * 100.0, 2))

                y_t_s = pd.to_numeric(sub[tgt_col], errors="coerce").fillna(0.0).to_numpy()
                y_p_s = pd.to_numeric(sub[pred_col], errors="coerce").fillna(0.0).to_numpy()

                s_mae = float(round(float(np.mean(np.abs(y_t_s - y_p_s))), 4))
                s_rmse = float(round(float(np.sqrt(np.mean((y_t_s - y_p_s)**2))), 4))
                mae_diff = float(round(s_mae - global_mae, 4))
                mae_pct_increase = float(round(((s_mae - global_mae) / max(0.0001, global_mae)) * 100.0, 2))

                record = {
                    "slice_id": s_id,
                    "sample_size": n_s,
                    "share_of_dataset_pct": pct_share,
                    "mae": s_mae,
                    "rmse": s_rmse,
                    "mae_diff_vs_baseline": mae_diff,
                    "mae_increase_pct": mae_pct_increase,
                    "is_severely_degraded": False
                }

                if mae_pct_increase > deg_tol:
                    record["is_severely_degraded"] = True
                    warnings.append(
                        f"[ALERT: SEVERE ERROR DEGRADATION] Slice '{s_id}' MAE ({s_mae:.4f}) is {mae_pct_increase:+.1f}% "
                        f"higher than baseline ({global_mae:.4f})."
                    )

                if n_s < min_n:
                    warnings.append(
                        f"[WARNING: SMALL SLICE SAMPLE SIZE] Slice '{s_id}' contains only N={n_s} samples (< {min_n} guardrail)."
                    )

                slice_rows.append(record)

        slices_df = pd.DataFrame(slice_rows)

        # 6. Find Best and Worst Performing Slices
        if task_type == "classification":
            primary_col = "f1_score_pct" if "f1_score_pct" in slices_df.columns else "accuracy_pct"
            best_slice = slices_df.loc[slices_df[primary_col].idxmax()]["slice_id"]
            worst_slice = slices_df.loc[slices_df[primary_col].idxmin()]["slice_id"]
            max_disparity = float(round(slices_df[primary_col].max() - slices_df[primary_col].min(), 2))
        else:
            best_slice = slices_df.loc[slices_df["mae"].idxmin()]["slice_id"]
            worst_slice = slices_df.loc[slices_df["mae"].idxmax()]["slice_id"]
            max_disparity = float(round(slices_df["mae"].max() - slices_df["mae"].min(), 4))

        severely_degraded_count = int(slices_df["is_severely_degraded"].sum())

        metrics = {
            "slice_columns": slice_cols,
            "total_slices_audited": len(unique_slices),
            "global_baseline": global_metrics,
            "best_performing_slice": best_slice,
            "worst_performing_slice": worst_slice,
            "maximum_performance_disparity": max_disparity,
            "severely_degraded_slices_count": severely_degraded_count,
            "slices_summary": slice_rows,
            "warnings": warnings
        }

        # 7. Generate Executive Markdown Report
        report_lines = [
            f"# === Demographic Fairness Slicing Report ({task_type.capitalize()}) ===",
            f"- **Slicing Columns**: `{', '.join(slice_cols)}` ({len(unique_slices)} total demographic cohort slices)",
            f"- **Best Performing Slice**: `{best_slice}` | **Worst Performing Slice**: `{worst_slice}`",
            f"- **Max Performance Disparity Gap**: `{max_disparity}`",
            f"- **Severely Degraded Slices**: `{severely_degraded_count}` slice(s) exceeding {deg_tol:.1f}% drop tolerance",
            "",
            "## Disaggregated Demographic Slices Leaderboard:",
        ]

        if task_type == "classification":
            has_f1 = "f1_score_pct" in slices_df.columns
            if has_f1:
                report_lines.append("| Demographic Slice | Sample Size (N) | Share % | Accuracy | Diff vs Base | F1-Score | F1 Diff vs Base | Status |")
                report_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
                for r in slice_rows:
                    status_badge = "DEGRADED" if r["is_severely_degraded"] else "NOMINAL"
                    report_lines.append(
                        f"| `{r['slice_id']}` | {r['sample_size']:,} | {r['share_of_dataset_pct']:.1f}% | {r['accuracy_pct']:.1f}% | {r['accuracy_diff_vs_baseline']:+.1f}% | **{r.get('f1_score_pct', 0.0):.1f}%** | {r.get('f1_diff_vs_baseline', 0.0):+.1f}% | `{status_badge}` |"
                    )
            else:
                report_lines.append("| Demographic Slice | Sample Size (N) | Share % | Accuracy | Diff vs Baseline | Status |")
                report_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
                for r in slice_rows:
                    status_badge = "DEGRADED" if r["is_severely_degraded"] else "NOMINAL"
                    report_lines.append(
                        f"| `{r['slice_id']}` | {r['sample_size']:,} | {r['share_of_dataset_pct']:.1f}% | **{r['accuracy_pct']:.1f}%** | {r['accuracy_diff_vs_baseline']:+.1f}% | `{status_badge}` |"
                    )
        else:
            report_lines.append("| Demographic Slice | Sample Size (N) | Share % | MAE | MAE vs Base | RMSE | Degradation % | Status |")
            report_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
            for r in slice_rows:
                status_badge = "DEGRADED" if r["is_severely_degraded"] else "NOMINAL"
                report_lines.append(
                    f"| `{r['slice_id']}` | {r['sample_size']:,} | {r['share_of_dataset_pct']:.1f}% | **{r['mae']:.4f}** | {r['mae_diff_vs_baseline']:+.4f} | {r['rmse']:.4f} | {r['mae_increase_pct']:+.1f}% | `{status_badge}` |"
                )

        if warnings:
            report_lines.append("")
            report_lines.append("## Slice Integrity & Risk Warnings:")
            for w in warnings:
                report_lines.append(f"- {w}")

        report_markdown = "\n".join(report_lines)

        output: Dict[str, Any] = {
            "slices_table": slices_df,
            "dataframe": slices_df,
            "metrics": metrics,
            "report": report_markdown
        }

        return pass_through_metadata(inputs, output, context)

    def to_code(self, config: Dict[str, Any]) -> str:
        s_col = config.get("slice_columns", "gender")
        tgt = config.get("target_col", "target")
        pred = config.get("prediction_col", "prediction")
        return (
            "import pandas as pd\n"
            "import numpy as np\n\n"
            f"# Disaggregate performance by demographic slice '{s_col}'\n"
            f"for slice_val, group in df.groupby('{s_col}'):\n"
            f"    acc = (group['{tgt}'] == group['{pred}']).mean() * 100.0\n"
            f"    print(f'Slice [{slice_val}] N={{len(group)}}: Accuracy = {{acc:.1f}}%')"
        )
