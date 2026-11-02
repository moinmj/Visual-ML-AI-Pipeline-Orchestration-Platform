import math
from collections import defaultdict
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union

from backend.app.recipes.base.recipe import BaseRecipe, RecipePort
from backend.app.recipes.training.encoder_utils import pass_through_metadata


class BiasFairnessAuditRecipe(BaseRecipe):
    """
    Bias & Fairness Audit Recipe.
    Audits algorithmic model bias across protected demographic attributes (e.g., gender, race,
    age group, disability status) using industry-standard algorithmic fairness metrics:
      - Disparate Impact Ratio (EEOC Four-Fifths 80% Rule)
      - Statistical / Demographic Parity Difference
      - Equal Opportunity Difference (True Positive Rate Parity)
      - False Positive Rate Parity (Predictive Equality)
      - Average Odds Difference
    Generates actionable remediation alerts and executive compliance reports.
    """

    recipe_id = "bias_fairness_audit"
    name = "Bias & Fairness Audit (Disparate Impact & Parity)"
    version = "1.0.0"
    category = "evaluation"
    group = "Evaluation & Governance"
    subgroup = "Fairness & Compliance"
    description = (
        "Audits algorithmic fairness across protected demographic groups, evaluating Disparate Impact "
        "(EEOC 4/5ths rule), Demographic Parity, Equal Opportunity, and False Positive parity."
    )
    input_types = ["dataframe"]
    output_types = ["metrics", "report", "dataframe"]

    inputs: List[RecipePort] = [
        RecipePort(
            id="test_data",
            label="Evaluation / Test Dataset",
            type="dataframe",
            required=True,
            max_connections=1,
            description="Dataset containing actual targets, model predictions, and sensitive protected attribute columns."
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
            id="metrics",
            label="Fairness Metrics Summary",
            type="metrics",
            description="Disparate impact ratio, demographic parity, equal opportunity difference, and 4/5ths rule compliance."
        ),
        RecipePort(
            id="report",
            label="Executive Fairness Audit Report",
            type="report",
            description="Markdown audit report with compliance status badges, four-fifths rule alerts, and group tables."
        ),
        RecipePort(
            id="audit_table",
            label="Demographic Group Metrics Table",
            type="dataframe",
            description="Disaggregated metrics table showing selection rate, accuracy, TPR, and FPR per group."
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "protected_attribute": {
                    "type": "string",
                    "title": "Protected / Sensitive Attribute Column",
                    "input_type": "text",
                    "default": "gender",
                    "description": "Column defining demographic groups (e.g., gender, race, age_band, veteran_status)."
                },
                "target_col": {
                    "type": "string",
                    "title": "Ground Truth Target Column",
                    "input_type": "text",
                    "default": "target",
                    "description": "Ground truth binary outcome (0/1, approved/rejected, hired/declined)."
                },
                "prediction_col": {
                    "type": "string",
                    "title": "Model Prediction Column",
                    "input_type": "text",
                    "default": "prediction",
                    "description": "Binary predictions from the model (optional if trained model is connected)."
                },
                "probability_col": {
                    "type": "string",
                    "title": "Optional Probability Score Column",
                    "input_type": "text",
                    "default": "",
                    "description": "Predicted positive class probabilities (0.0 to 1.0)."
                },
                "privileged_group": {
                    "type": "string",
                    "title": "Privileged Group Value",
                    "input_type": "text",
                    "default": "auto",
                    "description": "Demographic group used as reference baseline (default 'auto' picks group with highest favorable rate)."
                },
                "favorable_label": {
                    "type": "string",
                    "title": "Favorable Outcome Label",
                    "input_type": "text",
                    "default": "1",
                    "description": "Positive / favorable label value (e.g., 1, true, approved, hired)."
                },
                "disparate_impact_threshold": {
                    "type": "number",
                    "title": "Disparate Impact Threshold (Four-Fifths Rule)",
                    "input_type": "number",
                    "default": 0.80,
                    "description": "Ratios below this threshold (e.g. 0.80) indicate potential disparate impact violation."
                },
                "parity_difference_threshold": {
                    "type": "number",
                    "title": "Statistical Parity Tolerance",
                    "input_type": "number",
                    "default": 0.10,
                    "description": "Max acceptable absolute difference in favorable selection rates between groups (e.g. 10%)."
                }
            },
            "required": []
        }

    def _auto_detect_columns(self, df: pd.DataFrame, prot_cfg: str, tgt_cfg: str, pred_cfg: str) -> Tuple[str, str, str]:
        cols_lower = {c.lower(): c for c in df.columns}

        # Resolve protected attribute
        prot_col = ""
        if prot_cfg in df.columns:
            prot_col = prot_cfg
        elif prot_cfg.lower() in cols_lower:
            prot_col = cols_lower[prot_cfg.lower()]
        else:
            candidates = ["protected_attribute", "protected_group", "sensitive_attribute", "gender", "sex", "race", "ethnicity", "age_group", "age_band", "group"]
            for cand in candidates:
                if cand in cols_lower:
                    prot_col = cols_lower[cand]
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
            candidates = ["prediction", "pred", "y_pred", "predicted", "model_prediction", "decision", "approved"]
            for cand in candidates:
                if cand in cols_lower:
                    pred_col = cols_lower[cand]
                    break

        return prot_col, tgt_col, pred_col

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
            raise ValueError("BiasFairnessAudit expects a non-empty pandas DataFrame in inputs.")

        df = df.copy()

        # 2. Detect & Validate Columns
        prot_cfg = config.get("protected_attribute", "gender")
        tgt_cfg = config.get("target_col", "target")
        pred_cfg = config.get("prediction_col", "prediction")
        prob_cfg = config.get("probability_col", "")

        prot_col, tgt_col, pred_col = self._auto_detect_columns(df, prot_cfg, tgt_cfg, pred_cfg)

        if not prot_col or prot_col not in df.columns:
            raise ValueError(
                f"BiasFairnessAudit could not locate protected attribute column '{prot_cfg}'. Available columns: {list(df.columns)}"
            )

        # 3. If model passed and predictions missing, generate predictions
        model = inputs.get("model")
        if (not pred_col or pred_col not in df.columns) and model is not None:
            feature_cols = [c for c in df.columns if c not in [prot_col, tgt_col]]
            try:
                preds = model.predict(df[feature_cols])
                df["_model_prediction"] = preds
                pred_col = "_model_prediction"
            except Exception as e:
                raise ValueError(f"Failed to generate predictions with provided model: {e}")

        if not pred_col or pred_col not in df.columns:
            raise ValueError(
                f"BiasFairnessAudit could not locate prediction column '{pred_cfg}'. Provide predictions in the dataset or connect a trained model."
            )

        # Drop rows with nulls in protected or prediction columns
        cols_to_check = [prot_col, pred_col]
        has_ground_truth = bool(tgt_col and tgt_col in df.columns)
        if has_ground_truth:
            cols_to_check.append(tgt_col)

        df = df.dropna(subset=cols_to_check)
        if len(df) < 2:
            raise ValueError("BiasFairnessAudit dataset contains insufficient valid rows after dropping missing values.")

        # 4. Standardize Favorable Outcome to 1, Unfavorable to 0
        fav_cfg_str = str(config.get("favorable_label", "1")).strip().lower()

        def _binarize(val: Any) -> int:
            if pd.isna(val):
                return 0
            v_str = str(val).strip().lower()
            if v_str in [fav_cfg_str, "1", "1.0", "true", "yes", "approved", "hired", "accepted", "positive"]:
                return 1
            return 0

        y_pred = df[pred_col].apply(_binarize).to_numpy()
        y_true = df[tgt_col].apply(_binarize).to_numpy() if has_ground_truth else None
        groups = df[prot_col].astype(str).to_numpy()

        unique_groups = sorted(list(set(groups)))
        if len(unique_groups) < 2:
            raise ValueError(
                f"BiasFairnessAudit requires at least 2 distinct groups in '{prot_col}', found only: {unique_groups}"
            )

        # 5. Determine Privileged and Unprivileged Groups
        # Compute selection rate per group: P(y_pred = 1 | group)
        group_counts: Dict[str, int] = {}
        group_positives: Dict[str, int] = {}
        group_selection_rates: Dict[str, float] = {}

        for g in unique_groups:
            idx = (groups == g)
            n_g = int(np.sum(idx))
            pos_g = int(np.sum(y_pred[idx]))
            group_counts[g] = n_g
            group_positives[g] = pos_g
            group_selection_rates[g] = (pos_g / n_g) if n_g > 0 else 0.0

        priv_cfg = str(config.get("privileged_group", "auto")).strip()
        matched_priv = next((g for g in unique_groups if g.lower() == priv_cfg.lower()), None)
        if not matched_priv or priv_cfg.lower() == "auto":
            # Auto-pick group with highest favorable selection rate as benchmark privileged group
            matched_priv = max(unique_groups, key=lambda g: group_selection_rates[g])

        unprivileged_groups = [g for g in unique_groups if g != matched_priv]
        priv_selection_rate = group_selection_rates[matched_priv]

        # 6. Compute Disaggregated Performance & Error Rates per Group
        group_audit_records: List[Dict[str, Any]] = []

        for g in unique_groups:
            idx = (groups == g)
            n_g = group_counts[g]
            sel_rate = group_selection_rates[g]

            record = {
                "group": g,
                "is_privileged": (g == matched_priv),
                "sample_size": n_g,
                "favorable_predictions": group_positives[g],
                "selection_rate_pct": float(round(sel_rate * 100.0, 2))
            }

            if has_ground_truth:
                y_t_g = y_true[idx]
                y_p_g = y_pred[idx]

                tp = int(np.sum((y_t_g == 1) & (y_p_g == 1)))
                fp = int(np.sum((y_t_g == 0) & (y_p_g == 1)))
                tn = int(np.sum((y_t_g == 0) & (y_p_g == 0)))
                fn = int(np.sum((y_t_g == 1) & (y_p_g == 0)))

                actual_pos = tp + fn
                actual_neg = fp + tn

                tpr = (tp / actual_pos) if actual_pos > 0 else 0.0
                fpr = (fp / actual_neg) if actual_neg > 0 else 0.0
                acc = ((tp + tn) / n_g) if n_g > 0 else 0.0

                record["actual_positives"] = actual_pos
                record["true_positive_rate_tpr"] = float(round(tpr, 4))
                record["false_positive_rate_fpr"] = float(round(fpr, 4))
                record["group_accuracy_pct"] = float(round(acc * 100.0, 2))

            group_audit_records.append(record)

        audit_df = pd.DataFrame(group_audit_records)

        # 7. Compute Algorithmic Fairness Metrics vs Privileged Group
        di_threshold = float(config.get("disparate_impact_threshold", 0.80))
        parity_diff_threshold = float(config.get("parity_difference_threshold", 0.10))

        pairwise_evaluations: List[Dict[str, Any]] = []
        overall_four_fifths_passed = True
        overall_demographic_parity_passed = True
        warnings: List[str] = []

        priv_rec = next(r for r in group_audit_records if r["group"] == matched_priv)
        priv_tpr = priv_rec.get("true_positive_rate_tpr", 0.0)
        priv_fpr = priv_rec.get("false_positive_rate_fpr", 0.0)

        for unpriv_g in unprivileged_groups:
            u_rec = next(r for r in group_audit_records if r["group"] == unpriv_g)
            u_sel_rate = group_selection_rates[unpriv_g]

            # Disparate Impact Ratio: SR_unpriv / SR_priv
            if priv_selection_rate > 0:
                di_ratio = float(round(u_sel_rate / priv_selection_rate, 4))
            else:
                di_ratio = 1.0 if u_sel_rate == 0 else float("inf")

            # Demographic / Statistical Parity Difference: SR_unpriv - SR_priv
            sp_diff = float(round(u_sel_rate - priv_selection_rate, 4))

            # Compliance check
            ff_passed = (di_ratio >= di_threshold) and (di_ratio <= 1.25)
            sp_passed = abs(sp_diff) <= parity_diff_threshold

            if not ff_passed:
                overall_four_fifths_passed = False
                warnings.append(
                    f"[ALERT: FOUR-FIFTHS RULE VIOLATION] Unprivileged group '{unpriv_g}' has Disparate Impact ratio of {di_ratio:.3f} "
                    f"(< {di_threshold:.2f} threshold relative to '{matched_priv}'). Potential adverse impact under EEOC guidelines."
                )

            if not sp_passed:
                overall_demographic_parity_passed = False

            eval_item: Dict[str, Any] = {
                "unprivileged_group": unpriv_g,
                "disparate_impact_ratio": di_ratio,
                "four_fifths_rule_passed": ff_passed,
                "statistical_parity_difference": sp_diff,
                "statistical_parity_passed": sp_passed
            }

            # If ground truth available, calculate Equal Opportunity & Equalized Odds
            if has_ground_truth:
                u_tpr = u_rec.get("true_positive_rate_tpr", 0.0)
                u_fpr = u_rec.get("false_positive_rate_fpr", 0.0)

                # Equal Opportunity Difference: TPR_unpriv - TPR_priv
                eq_opp_diff = float(round(u_tpr - priv_tpr, 4))
                # False Positive Rate Difference: FPR_unpriv - FPR_priv
                fpr_diff = float(round(u_fpr - priv_fpr, 4))
                # Average Odds Difference: 0.5 * (|TPR_diff| + |FPR_diff|)
                avg_odds_diff = float(round(0.5 * (abs(eq_opp_diff) + abs(fpr_diff)), 4))

                eval_item["equal_opportunity_difference"] = eq_opp_diff
                eval_item["false_positive_rate_difference"] = fpr_diff
                eval_item["average_odds_difference"] = avg_odds_diff

                if abs(eq_opp_diff) > 0.10:
                    warnings.append(
                        f"[ALERT: EQUAL OPPORTUNITY DISPARITY] Group '{unpriv_g}' True Positive Rate ({u_tpr*100:.1f}%) diverges "
                        f"from reference '{matched_priv}' ({priv_tpr*100:.1f}%) by {eq_opp_diff*100:+.1f}%."
                    )

            pairwise_evaluations.append(eval_item)

        # Min Disparate Impact across all unprivileged groups
        min_di_ratio = min([e["disparate_impact_ratio"] for e in pairwise_evaluations]) if pairwise_evaluations else 1.0

        overall_status = "COMPLIANT" if (overall_four_fifths_passed and overall_demographic_parity_passed) else "DISPARATE_IMPACT_DETECTED"

        # 8. Assemble Metrics
        metrics = {
            "protected_attribute": prot_col,
            "privileged_group": matched_priv,
            "overall_fairness_status": overall_status,
            "four_fifths_rule_compliant": overall_four_fifths_passed,
            "minimum_disparate_impact_ratio": min_di_ratio,
            "disparate_impact_threshold": di_threshold,
            "total_groups_audited": len(unique_groups),
            "pairwise_evaluations": pairwise_evaluations,
            "group_summaries": group_audit_records,
            "warnings": warnings
        }

        # 9. Format Executive Audit Report
        badge = "PASSED: COMPLIANT WITH 4/5THS RULE" if overall_four_fifths_passed else "FAIL: ADVERSE DISPARATE IMPACT DETECTED"

        report_lines = [
            "# === Algorithmic Bias & Fairness Compliance Audit ===",
            f"- **Protected Demographic Attribute**: `{prot_col}`",
            f"- **Reference (Privileged) Group**: `{matched_priv}` (Selection Rate: **{priv_selection_rate*100:.1f}%**)",
            f"- **Four-Fifths (80%) Rule Compliance**: **[{badge}]**",
            f"- **Minimum Disparate Impact Ratio**: `{min_di_ratio:.4f}` (Threshold: `{di_threshold:.2f}`)",
            "",
            "## Demographic Group Performance Breakdown:",
        ]

        if has_ground_truth:
            report_lines.append("| Demographic Group | Sample Size (N) | Selection Rate | Accuracy | True Positive Rate (TPR) | False Positive Rate (FPR) |")
            report_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
            for r in group_audit_records:
                tag = " (Reference)" if r["is_privileged"] else ""
                report_lines.append(
                    f"| `{r['group']}`{tag} | {r['sample_size']:,} | {r['selection_rate_pct']:.1f}% | {r.get('group_accuracy_pct', 0.0):.1f}% | {r.get('true_positive_rate_tpr', 0.0)*100:.1f}% | {r.get('false_positive_rate_fpr', 0.0)*100:.1f}% |"
                )
        else:
            report_lines.append("| Demographic Group | Sample Size (N) | Favorable Outcomes | Selection Rate |")
            report_lines.append("| :--- | :--- | :--- | :--- |")
            for r in group_audit_records:
                tag = " (Reference)" if r["is_privileged"] else ""
                report_lines.append(
                    f"| `{r['group']}`{tag} | {r['sample_size']:,} | {r['favorable_predictions']:,} | **{r['selection_rate_pct']:.1f}%** |"
                )

        report_lines.append("")
        report_lines.append("## Algorithmic Fairness Comparison (vs Privileged Reference):")
        if has_ground_truth:
            report_lines.append("| Unprivileged Group | Disparate Impact (4/5ths) | Statistical Parity Diff | Equal Opportunity Diff | Average Odds Diff | Compliance |")
            report_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
            for pw in pairwise_evaluations:
                status_str = "PASS" if pw["four_fifths_rule_passed"] else "VIOLATION"
                report_lines.append(
                    f"| `{pw['unprivileged_group']}` | **{pw['disparate_impact_ratio']:.3f}x** | {pw['statistical_parity_difference']*100:+.1f}% | {pw.get('equal_opportunity_difference', 0.0)*100:+.1f}% | {pw.get('average_odds_difference', 0.0)*100:.1f}% | **{status_str}** |"
                )
        else:
            report_lines.append("| Unprivileged Group | Disparate Impact (4/5ths) | Statistical Parity Diff | 4/5ths Rule Status |")
            report_lines.append("| :--- | :--- | :--- | :--- |")
            for pw in pairwise_evaluations:
                status_str = "PASS" if pw["four_fifths_rule_passed"] else "VIOLATION"
                report_lines.append(
                    f"| `{pw['unprivileged_group']}` | **{pw['disparate_impact_ratio']:.3f}x** | {pw['statistical_parity_difference']*100:+.1f}% | **{status_str}** |"
                )

        if warnings:
            report_lines.append("")
            report_lines.append("## Governance & Remediation Alerts:")
            for w in warnings:
                report_lines.append(f"- {w}")
            report_lines.append("")
            report_lines.append("> [!WARNING]")
            report_lines.append(
                "> Models exhibiting adverse disparate impact may expose deployments to regulatory scrutiny under "
                "EEOC Title VII, EU AI Act, and Fair Lending Act guidelines. Consider re-weighting sample classes, "
                "adjusting group classification thresholds, or adversarial debiasing."
            )

        report_markdown = "\n".join(report_lines)

        output: Dict[str, Any] = {
            "metrics": metrics,
            "report": report_markdown,
            "audit_table": audit_df,
            "dataframe": audit_df
        }

        return pass_through_metadata(inputs, output, context)

    def to_code(self, config: Dict[str, Any]) -> str:
        prot = config.get("protected_attribute", "gender")
        pred = config.get("prediction_col", "prediction")
        return (
            "import numpy as np\n"
            "import pandas as pd\n\n"
            f"# 1. Compute selection rate per demographic group in '{prot}'\n"
            f"rates = df.groupby('{prot}')['{pred}'].mean().to_dict()\n"
            "priv_group = max(rates, key=rates.get)\n"
            "priv_rate = rates[priv_group]\n\n"
            "# 2. Calculate Disparate Impact (Four-Fifths Rule)\n"
            "for g, r in rates.items():\n"
            "    di_ratio = r / priv_rate if priv_rate > 0 else 1.0\n"
            "    status = 'PASS' if di_ratio >= 0.8 else 'ALERT: Adverse Impact'\n"
            "    print(f'Group {g}: DI Ratio = {di_ratio:.3f} ({status})')"
        )
