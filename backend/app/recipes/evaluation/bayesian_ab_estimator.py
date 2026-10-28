import math
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from scipy import stats

from backend.app.recipes.base.recipe import BaseRecipe, RecipePort
from backend.app.recipes.training.encoder_utils import pass_through_metadata


class BayesianABEstimatorRecipe(BaseRecipe):
    """
    Bayesian A/B Test Estimator Recipe.
    Evaluates online and offline experiments using Bayesian Beta-Binomial conjugate priors
    and Monte Carlo posterior sampling. Computes exact P(Treatment > Control), Expected Loss
    (financial/conversion risk of picking the wrong variant), and 95% Credible Intervals.
    """

    recipe_id = "bayesian_ab_estimator"
    name = "Bayesian A/B Test Estimator (Expected Loss & Posterior Lift)"
    version = "1.0.0"
    category = "evaluation"
    group = "Evaluation & Governance"
    subgroup = "Evaluation"
    description = (
        "Evaluates experiment variants using Bayesian Beta-Binomial and Gaussian posteriors, computing "
        "P(B > A), Expected Loss (risk of choosing wrong), and 95% Credible Intervals."
    )
    input_types = ["dataframe"]
    output_types = ["metrics", "report", "dataframe"]

    inputs: List[RecipePort] = [
        RecipePort(
            id="data",
            label="Experiment Observations Table",
            type="dataframe",
            required=True,
            max_connections=1,
            description="Dataset containing variant assignments (Control/Treatment) and outcome metrics."
        )
    ]

    outputs: List[RecipePort] = [
        RecipePort(
            id="metrics",
            label="Bayesian Posterior Metrics",
            type="metrics",
            description="P(B > A), Expected Loss for Control and Treatment, 95% Credible Intervals, and decision."
        ),
        RecipePort(
            id="report",
            label="Executive Decision Report",
            type="report",
            description="Markdown report with posterior probability curves, risk thresholds, and deploy recommendations."
        ),
        RecipePort(
            id="summary_table",
            label="Variant Summary Table",
            type="dataframe",
            description="Summary table of variants with posterior parameters, sample sizes, and win probabilities."
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "variant_col": {
                    "type": "string",
                    "title": "Variant Assignment Column",
                    "input_type": "text",
                    "default": "variant",
                    "description": "Column defining experiment arm (e.g., control, treatment_a, treatment_b)."
                },
                "outcome_col": {
                    "type": "string",
                    "title": "Outcome Metric Column",
                    "input_type": "text",
                    "default": "converted",
                    "description": "Binary conversion indicator (0/1) or continuous revenue metric."
                },
                "control_variant": {
                    "type": "string",
                    "title": "Control Variant Name",
                    "input_type": "text",
                    "default": "control",
                    "description": "Baseline variant against which treatment arms are compared."
                },
                "metric_type": {
                    "type": "string",
                    "title": "Metric Nature",
                    "input_type": "dropdown",
                    "enum": ["auto", "binary_conversion", "continuous_revenue"],
                    "default": "auto",
                    "description": "Auto-detects binary conversion (Beta-Binomial) vs continuous revenue (Gaussian-LogNormal)."
                },
                "prior_alpha": {
                    "type": "number",
                    "title": "Prior Alpha (Beta Successes)",
                    "input_type": "number",
                    "default": 1.0,
                    "description": "Prior pseudo-conversions (default 1.0 represents an uninformative Uniform prior)."
                },
                "prior_beta": {
                    "type": "number",
                    "title": "Prior Beta (Beta Non-Conversions)",
                    "input_type": "number",
                    "default": 1.0,
                    "description": "Prior pseudo-non-conversions."
                },
                "loss_threshold": {
                    "type": "number",
                    "title": "Decision Loss Threshold (Threshold of Caring)",
                    "input_type": "number",
                    "default": 0.001,
                    "description": "Maximum acceptable Expected Loss to declare a definitive winner and deploy."
                },
                "n_simulations": {
                    "type": "integer",
                    "title": "Monte Carlo Posterior Samples",
                    "input_type": "slider",
                    "default": 20000,
                    "minimum": 5000,
                    "maximum": 50000
                }
            },
            "required": []
        }

    def _auto_detect_columns(self, df: pd.DataFrame, var_cfg: str, out_cfg: str) -> Tuple[str, str]:
        cols_lower = {c.lower(): c for c in df.columns}

        # Resolve variant
        var_col = ""
        if var_cfg in df.columns:
            var_col = var_cfg
        elif var_cfg.lower() in cols_lower:
            var_col = cols_lower[var_cfg.lower()]
        else:
            candidates = ["variant", "group", "arm", "treatment", "ab_group", "test_group", "segment"]
            for cand in candidates:
                if cand in cols_lower:
                    var_col = cols_lower[cand]
                    break

        # Resolve outcome
        out_col = ""
        if out_cfg in df.columns:
            out_col = out_cfg
        elif out_cfg.lower() in cols_lower:
            out_col = cols_lower[out_cfg.lower()]
        else:
            candidates = ["converted", "conversion", "revenue", "value", "purchased", "success", "metric", "y", "target"]
            for cand in candidates:
                if cand in cols_lower:
                    out_col = cols_lower[cand]
                    break

        return var_col, out_col

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        # 1. Resolve Dataframe
        df = inputs.get("dataframe")
        if df is None:
            df = inputs.get("data")
        if df is None and isinstance(context, dict):
            df = context.get("dataframe")
            if df is None:
                df = context.get("data")

        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("BayesianABEstimator expects a non-empty pandas DataFrame in inputs.")

        df = df.copy()

        # 2. Detect and Validate Columns
        var_cfg = config.get("variant_col", "variant")
        out_cfg = config.get("outcome_col", "converted")
        var_col, out_col = self._auto_detect_columns(df, var_cfg, out_cfg)

        if not var_col or var_col not in df.columns:
            raise ValueError(
                f"BayesianABEstimator could not locate variant column '{var_cfg}'. Available columns: {list(df.columns)}"
            )
        if not out_col or out_col not in df.columns:
            raise ValueError(
                f"BayesianABEstimator could not locate outcome column '{out_cfg}'. Available columns: {list(df.columns)}"
            )

        df = df.dropna(subset=[var_col, out_col])

        unique_variants = [str(v) for v in df[var_col].unique()]
        if len(unique_variants) < 2:
            raise ValueError(
                f"BayesianABEstimator requires at least 2 distinct variants for comparison, got: {unique_variants}"
            )

        # 3. Identify Control vs Treatment Variants
        control_name = config.get("control_variant", "control").strip()
        matched_control = next((v for v in unique_variants if v.lower() == control_name.lower()), None)
        if not matched_control:
            matched_control = next((v for v in unique_variants if "control" in v.lower() or "base" in v.lower() or "a" == v.lower()), unique_variants[0])

        treatment_variants = [v for v in unique_variants if v != matched_control]

        # 4. Determine Metric Nature (Binary vs Continuous)
        metric_nature_cfg = config.get("metric_type", "auto").lower()
        outcome_vals = df[out_col]
        is_binary = False

        if metric_nature_cfg == "binary_conversion":
            is_binary = True
        elif metric_nature_cfg == "continuous_revenue":
            is_binary = False
        else:
            clean_str = set(outcome_vals.dropna().astype(str).str.lower().str.strip().unique())
            valid_binary_tokens = {"0", "1", "0.0", "1.0", "true", "false", "yes", "no", "converted"}
            if len(clean_str) <= 2 and clean_str.issubset(valid_binary_tokens):
                is_binary = True

        # Clean outcome values
        if is_binary:
            if outcome_vals.dtype.kind in ("U", "S", "O", "b"):
                df[out_col] = outcome_vals.astype(str).str.lower().str.strip().isin(["1", "1.0", "true", "yes", "converted"]).astype(int)
            else:
                df[out_col] = pd.to_numeric(outcome_vals).fillna(0).astype(int).clip(0, 1)
        else:
            df[out_col] = pd.to_numeric(outcome_vals, errors="coerce").fillna(0.0)

        # 5. Extract Prior & Simulation Parameters
        prior_alpha = float(config.get("prior_alpha", 1.0))
        prior_beta = float(config.get("prior_beta", 1.0))
        loss_threshold = float(config.get("loss_threshold", 0.001))
        n_sims = int(config.get("n_simulations", 20000))
        np.random.seed(42)

        # 6. Sample Posterior Distributions per Variant
        posteriors: Dict[str, np.ndarray] = {}
        variant_stats: List[Dict[str, Any]] = []

        for v in unique_variants:
            v_data = df[df[var_col].astype(str) == v][out_col].to_numpy()
            n = len(v_data)

            if is_binary:
                k = int(np.sum(v_data))
                alpha_post = prior_alpha + k
                beta_post = prior_beta + (n - k)
                samples = np.random.beta(alpha_post, beta_post, size=n_sims)
                conv_rate = round((k / max(1, n)) * 100.0, 2)
                post_mean = float(round(float(np.mean(samples)), 4))
                ci_low, ci_high = np.percentile(samples, [2.5, 97.5])
                variant_stats.append({
                    "variant": v,
                    "is_control": (v == matched_control),
                    "sample_size": n,
                    "conversions": k,
                    "observed_rate": f"{conv_rate}%",
                    "posterior_mean": post_mean,
                    "ci_95_low": float(round(ci_low, 4)),
                    "ci_95_high": float(round(ci_high, 4))
                })
            else:
                mean = float(np.mean(v_data)) if n > 0 else 0.0
                std = float(np.std(v_data, ddof=1)) if n > 1 else 1.0
                se = std / math.sqrt(max(1, n))
                samples = np.random.normal(loc=mean, scale=se, size=n_sims)
                ci_low, ci_high = np.percentile(samples, [2.5, 97.5])
                variant_stats.append({
                    "variant": v,
                    "is_control": (v == matched_control),
                    "sample_size": n,
                    "observed_mean": float(round(mean, 4)),
                    "posterior_mean": float(round(float(np.mean(samples)), 4)),
                    "ci_95_low": float(round(ci_low, 4)),
                    "ci_95_high": float(round(ci_high, 4))
                })

            posteriors[v] = samples

        summary_df = pd.DataFrame(variant_stats)

        # 7. Compute Bayesian Pairwise Comparisons (Treatment vs Control)
        ctrl_samples = posteriors[matched_control]
        treatment_evaluations: List[Dict[str, Any]] = []

        overall_decision = "CONTINUE_EXPERIMENT"
        champion_treatment = None
        best_p_better = 0.0

        for t_name in treatment_variants:
            t_samples = posteriors[t_name]

            # P(Treatment > Control)
            diff = t_samples - ctrl_samples
            p_treatment_better = float(round(float(np.mean(t_samples > ctrl_samples)), 4))

            # Relative uplift: (T - C) / C
            relative_lift_samples = diff / np.where(ctrl_samples != 0, ctrl_samples, 0.0001)
            median_lift_pct = float(round(float(np.median(relative_lift_samples)) * 100.0, 2))
            lift_ci_low = float(round(float(np.percentile(relative_lift_samples, 2.5)) * 100.0, 2))
            lift_ci_high = float(round(float(np.percentile(relative_lift_samples, 97.5)) * 100.0, 2))

            # Expected Loss:
            # Loss of choosing Treatment over Control: max(0, ctrl - t)
            # Loss of choosing Control over Treatment: max(0, t - ctrl)
            expected_loss_treatment = float(round(float(np.mean(np.maximum(0, -diff))), 6))
            expected_loss_control = float(round(float(np.mean(np.maximum(0, diff))), 6))

            # Automated Decision Recommendation
            if expected_loss_treatment < loss_threshold and p_treatment_better >= 0.95:
                recommendation = "DEPLOY_TREATMENT"
                rationale = f"High probability of superiority ({p_treatment_better*100:.1f}%) and Expected Loss ({expected_loss_treatment:.6f}) below threshold {loss_threshold}."
            elif expected_loss_control < loss_threshold and (1.0 - p_treatment_better) >= 0.95:
                recommendation = "RETAIN_CONTROL"
                rationale = f"Control is definitively superior (P(Control > {t_name}) = {(1.0 - p_treatment_better)*100:.1f}%). Deploying {t_name} risks significant negative loss."
            else:
                recommendation = "CONTINUE_EXPERIMENT"
                rationale = f"Insufficient certainty (P={p_treatment_better*100:.1f}%). Expected loss ({expected_loss_treatment:.6f}) exceeds acceptable threshold."

            treatment_evaluations.append({
                "treatment_variant": t_name,
                "p_treatment_superior": p_treatment_better,
                "relative_lift_median_pct": median_lift_pct,
                "relative_lift_95_ci": [lift_ci_low, lift_ci_high],
                "expected_loss_choose_treatment": expected_loss_treatment,
                "expected_loss_choose_control": expected_loss_control,
                "recommendation": recommendation,
                "rationale": rationale
            })

        # Determine overall tournament decision
        deployable = [te for te in treatment_evaluations if te["recommendation"] == "DEPLOY_TREATMENT"]
        if deployable:
            deployable.sort(key=lambda x: x["p_treatment_superior"], reverse=True)
            champion_treatment = deployable[0]["treatment_variant"]
            overall_decision = "DEPLOY_TREATMENT"
        elif treatment_evaluations and all(te["recommendation"] == "RETAIN_CONTROL" for te in treatment_evaluations):
            overall_decision = "RETAIN_CONTROL"
            champion_treatment = matched_control
        else:
            overall_decision = "CONTINUE_EXPERIMENT"
            champion_treatment = matched_control

        # 8. Assemble Metrics
        metrics = {
            "control_variant": matched_control,
            "metric_nature": "binary_conversion" if is_binary else "continuous_revenue",
            "loss_threshold": loss_threshold,
            "overall_decision": overall_decision,
            "champion_variant": champion_treatment if overall_decision == "DEPLOY_TREATMENT" else matched_control,
            "treatments_evaluated": treatment_evaluations,
            "variant_summaries": variant_stats,
            "n_simulations": n_sims
        }

        # 9. Format Executive Report
        decision_badges = {
            "DEPLOY_TREATMENT": "SUCCESS: DEPLOY TREATMENT VARIANT",
            "RETAIN_CONTROL": "STOP: RETAIN CONTROL BASELINE",
            "CONTINUE_EXPERIMENT": "INFO: CONTINUE GATHERING DATA"
        }

        report_lines = [
            "# === Bayesian A/B Experimentation Decision Report ===",
            f"- **Control Baseline**: `{matched_control}`",
            f"- **Metric Type**: {'Binary Conversion Rate' if is_binary else 'Continuous Value'}",
            f"- **Loss Threshold (Caring Horizon)**: {loss_threshold}",
            f"- **Strategic Decision**: **[{decision_badges.get(overall_decision, overall_decision)}]**",
            "",
            "## Variant Performance & Posterior Estimates:"
        ]

        if is_binary:
            report_lines.append("| Variant | Sample Size (N) | Conversions | Observed Rate | Posterior Mean | 95% Credible Interval |")
            report_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
            for vs in variant_stats:
                tag = " (Control)" if vs["is_control"] else ""
                report_lines.append(
                    f"| `{vs['variant']}`{tag} | {vs['sample_size']:,} | {vs['conversions']:,} | {vs['observed_rate']} | **{vs['posterior_mean']*100:.2f}%** | [{vs['ci_95_low']*100:.2f}%, {vs['ci_95_high']*100:.2f}%] |"
                )
        else:
            report_lines.append("| Variant | Sample Size (N) | Observed Mean | Posterior Mean | 95% Credible Interval |")
            report_lines.append("| :--- | :--- | :--- | :--- | :--- |")
            for vs in variant_stats:
                tag = " (Control)" if vs["is_control"] else ""
                report_lines.append(
                    f"| `{vs['variant']}`{tag} | {vs['sample_size']:,} | {vs['observed_mean']:.4f} | **{vs['posterior_mean']:.4f}** | [{vs['ci_95_low']:.4f}, {vs['ci_95_high']:.4f}] |"
                )

        report_lines.append("")
        report_lines.append("## Bayesian Risk & Uplift Tournament:")
        report_lines.append("| Treatment Variant | P(Treatment > Control) | Median Uplift (95% CI) | Expected Loss (Risk) | Decision |")
        report_lines.append("| :--- | :--- | :--- | :--- | :--- |")

        for te in treatment_evaluations:
            ci_str = f"[{te['relative_lift_95_ci'][0]:+.1f}%, {te['relative_lift_95_ci'][1]:+.1f}%]"
            report_lines.append(
                f"| `{te['treatment_variant']}` | **{te['p_treatment_superior']*100:.1f}%** | **{te['relative_lift_median_pct']:+.2f}%** ({ci_str}) | `{te['expected_loss_choose_treatment']:.6f}` | **{te['recommendation']}** |"
            )

        report_lines.append("")
        report_lines.append("## Executive Guidance & Rationale:")
        for te in treatment_evaluations:
            report_lines.append(f"- **`{te['treatment_variant']}`**: {te['rationale']}")

        report_markdown = "\n".join(report_lines)

        output: Dict[str, Any] = {
            "metrics": metrics,
            "report": report_markdown,
            "summary_table": summary_df,
            "dataframe": summary_df
        }

        return pass_through_metadata(inputs, output, context)

    def to_code(self, config: Dict[str, Any]) -> str:
        var = config.get("variant_col", "variant")
        out = config.get("outcome_col", "converted")
        n_sim = config.get("n_simulations", 20000)
        return (
            "import numpy as np\n\n"
            "# 1. Calculate Beta posterior parameters\n"
            f"k_ctrl = df[df['{var}'] == 'control']['{out}'].sum()\n"
            f"n_ctrl = len(df[df['{var}'] == 'control'])\n"
            f"k_treat = df[df['{var}'] == 'treatment']['{out}'].sum()\n"
            f"n_treat = len(df[df['{var}'] == 'treatment'])\n\n"
            f"ctrl_samples = np.random.beta(1 + k_ctrl, 1 + n_ctrl - k_ctrl, size={n_sim})\n"
            f"treat_samples = np.random.beta(1 + k_treat, 1 + n_treat - k_treat, size={n_sim})\n\n"
            "# 2. Calculate P(Treatment > Control) and Expected Loss\n"
            "p_better = np.mean(treat_samples > ctrl_samples)\n"
            "expected_loss = np.mean(np.maximum(0, ctrl_samples - treat_samples))\n"
            "print(f'P(B > A): {p_better:.4f}, Expected Loss: {expected_loss:.6f}')"
        )
