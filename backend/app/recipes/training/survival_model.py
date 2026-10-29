import math
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.utils import concordance_index

from backend.app.recipes.base.recipe import BaseRecipe, RecipePort
from backend.app.recipes.training.encoder_utils import (
    safe_prepare_training_data,
    pass_through_metadata
)


class SurvivalModelRecipe(BaseRecipe):
    """
    Survival Analysis Recipe (Cox Proportional Hazards & Kaplan-Meier).
    Models time-to-event outcomes with right-censoring for customer churn, equipment reliability,
    and patient retention. Provides non-parametric Kaplan-Meier survival curves and semi-parametric
    Cox PH hazard ratio regressions with concordance index evaluation.
    """

    recipe_id = "survival_model"
    name = "Survival Analysis (Cox Proportional Hazards & Kaplan-Meier)"
    version = "1.0.0"
    category = "training"
    group = "Modeling"
    subgroup = "Specialized Models"
    description = (
        "Fits Kaplan-Meier survival curves and Cox Proportional Hazards regression for time-to-event analysis, "
        "customer churn duration, and censored reliability modeling."
    )
    input_types = ["dataframe"]
    output_types = ["model", "metrics", "report"]

    inputs: List[RecipePort] = [
        RecipePort(
            id="data",
            label="Survival Dataset",
            type="dataframe",
            required=True,
            max_connections=1,
            description="Dataset containing duration/time, event/censored status, and covariate features."
        )
    ]

    outputs: List[RecipePort] = [
        RecipePort(
            id="model",
            label="Fitted Survival Model",
            type="model",
            description="Trained CoxPHFitter and/or KaplanMeierFitter model object."
        ),
        RecipePort(
            id="metrics",
            label="Survival Metrics",
            type="metrics",
            description="Concordance index (C-index), median survival time, hazard ratios, and survival milestones."
        ),
        RecipePort(
            id="report",
            label="Survival Analysis Report",
            type="report",
            description="Markdown report containing hazard ratios, p-values, median survival times, and survival curves."
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "duration_col": {
                    "type": "string",
                    "title": "Duration Column",
                    "input_type": "text",
                    "default": "duration",
                    "description": "Column representing time elapsed until event or censoring (e.g. tenure, days, months)."
                },
                "event_col": {
                    "type": "string",
                    "title": "Event Column (1=Event, 0=Censored)",
                    "input_type": "text",
                    "default": "event",
                    "description": "Binary indicator column (1 = event occurred like churn/failure, 0 = right-censored/active)."
                },
                "model_type": {
                    "type": "string",
                    "title": "Survival Model Type",
                    "input_type": "dropdown",
                    "enum": ["both", "cox_ph", "kaplan_meier"],
                    "default": "both",
                    "description": "Kaplan-Meier estimates baseline survival probability; Cox PH estimates feature hazard ratios."
                },
                "strata_col": {
                    "type": "string",
                    "title": "Optional Stratification Column",
                    "input_type": "text",
                    "default": "",
                    "description": "Categorical column to compare Kaplan-Meier survival curves across customer cohorts."
                },
                "penalizer": {
                    "type": "number",
                    "title": "L2 Ridge Penalizer",
                    "input_type": "number",
                    "default": 0.1,
                    "description": "L2 regularization penalty for Cox PH to prevent singular matrix inversion."
                }
            },
            "required": []
        }

    def _auto_detect_columns(self, df: pd.DataFrame, duration_cfg: str, event_cfg: str) -> Tuple[str, str]:
        cols_lower = {c.lower(): c for c in df.columns}

        # Resolve duration
        duration_col = ""
        if duration_cfg in df.columns:
            duration_col = duration_cfg
        elif duration_cfg.lower() in cols_lower:
            duration_col = cols_lower[duration_cfg.lower()]
        else:
            candidates = ["duration", "tenure", "time", "months", "days", "tte", "survival_time", "period"]
            for cand in candidates:
                if cand in cols_lower:
                    duration_col = cols_lower[cand]
                    break

        # Resolve event
        event_col = ""
        if event_cfg in df.columns:
            event_col = event_cfg
        elif event_cfg.lower() in cols_lower:
            event_col = cols_lower[event_cfg.lower()]
        else:
            candidates = ["event", "churn", "status", "censored", "death", "observed", "failed", "is_churn"]
            for cand in candidates:
                if cand in cols_lower:
                    event_col = cols_lower[cand]
                    break

        return duration_col, event_col

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        # 1. Resolve Dataframe
        df = inputs.get("dataframe")
        if df is None:
            df = inputs.get("data")
        if df is None and isinstance(context, dict):
            df = context.get("dataframe")
            if df is None:
                df = context.get("data")
        if df is None and "train_data" in inputs and isinstance(inputs["train_data"], dict):
            X = inputs["train_data"].get("X_train")
            y = inputs["train_data"].get("y_train")
            if isinstance(X, pd.DataFrame):
                df = X.copy()
                if y is not None:
                    df["event"] = y

        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("SurvivalModel expects a non-empty pandas DataFrame in inputs.")

        df = df.copy()

        # 2. Detect and Validate Duration and Event Columns
        duration_cfg = config.get("duration_col", "duration")
        event_cfg = config.get("event_col", "event")
        dur_col, ev_col = self._auto_detect_columns(df, duration_cfg, event_cfg)

        if not dur_col or dur_col not in df.columns:
            raise ValueError(
                f"SurvivalModel could not locate duration column '{duration_cfg}'. "
                f"Available columns: {list(df.columns)}"
            )
        if not ev_col or ev_col not in df.columns:
            raise ValueError(
                f"SurvivalModel could not locate event column '{event_cfg}'. "
                f"Available columns: {list(df.columns)}"
            )

        # 3. Clean Duration & Event Data
        df = df.dropna(subset=[dur_col, ev_col])
        try:
            df[dur_col] = pd.to_numeric(df[dur_col])
        except Exception:
            raise ValueError(f"Duration column '{dur_col}' must contain numeric values.")

        if (df[dur_col] < 0).any():
            raise ValueError(f"Duration column '{dur_col}' contains negative values. Time-to-event must be non-negative.")

        # Ensure positive duration for Cox PH log transforms
        df[dur_col] = df[dur_col].clip(lower=0.0001)

        # Clean event column to binary 0/1
        ev_series = df[ev_col]
        if ev_series.dtype.kind in ("U", "S", "O"):
            ev_clean = ev_series.astype(str).str.lower().str.strip()
            # 1 for churn/event/dead/true, 0 for retained/censored/alive/false
            df[ev_col] = ev_clean.isin(["1", "true", "yes", "churn", "dead", "event", "failed"]).astype(int)
        elif pd.api.types.is_bool_dtype(ev_series):
            df[ev_col] = ev_series.astype(int)
        else:
            df[ev_col] = pd.to_numeric(ev_series).fillna(0).astype(int).clip(0, 1)

        N = len(df)
        events_count = int(df[ev_col].sum())
        censored_count = N - events_count
        censoring_rate_pct = round((censored_count / max(1, N)) * 100.0, 2)

        warnings: List[str] = []
        if events_count == 0:
            raise ValueError("Event column contains 0 observed events. At least 1 event is required for survival estimation.")
        if events_count < 5:
            warnings.append(
                f"[WARNING: FEW EVENTS] Only {events_count} events observed out of N={N}. "
                "Cox Proportional Hazards parameter estimates may exhibit large variance."
            )

        model_type = config.get("model_type", "both")
        strata_col = config.get("strata_col", "").strip()
        penalizer = float(config.get("penalizer", 0.1))

        kmf = None
        cph = None
        c_index = None
        median_survival = None
        stratified_medians: Dict[str, Any] = {}
        hazard_ratios_dict: Dict[str, float] = {}
        km_curve_sample: List[Dict[str, Any]] = []

        # 4. Kaplan-Meier Estimation
        if model_type in ("both", "kaplan_meier"):
            kmf = KaplanMeierFitter()
            kmf.fit(durations=df[dur_col], event_observed=df[ev_col], label="Overall Population")
            med = kmf.median_survival_time_
            median_survival = float(round(med, 2)) if not (np.isnan(med) or np.isinf(med)) else None

            # Sample timeline survival points (up to 10 points)
            timeline = kmf.timeline
            step = max(1, len(timeline) // 10)
            sampled_times = timeline[::step]
            for t in sampled_times:
                surv_prob = float(round(float(kmf.survival_function_at_times(t).iloc[0]), 4))
                km_curve_sample.append({
                    "time": float(round(t, 2)),
                    "survival_probability": surv_prob
                })

            # Stratified KM if strata column is provided
            if strata_col and strata_col in df.columns:
                unique_strata = df[strata_col].dropna().unique()
                for s in unique_strata[:6]:  # Limit to 6 groups
                    mask = df[strata_col] == s
                    if mask.sum() >= 3 and df.loc[mask, ev_col].sum() > 0:
                        kmf_sub = KaplanMeierFitter()
                        kmf_sub.fit(durations=df.loc[mask, dur_col], event_observed=df.loc[mask, ev_col], label=str(s))
                        s_med = kmf_sub.median_survival_time_
                        stratified_medians[str(s)] = float(round(s_med, 2)) if not (np.isnan(s_med) or np.isinf(s_med)) else "Not Reached"

        # 5. Cox Proportional Hazards Regression
        if model_type in ("both", "cox_ph"):
            cph = CoxPHFitter(penalizer=penalizer)
            # Prepare covariate features dataframe
            covariate_df = df.copy()
            # Drop non-feature columns
            drop_cols = [dur_col, ev_col]
            if strata_col and strata_col in covariate_df.columns and strata_col != dur_col and strata_col != ev_col:
                drop_cols.append(strata_col)

            covariate_features = covariate_df.drop(columns=drop_cols, errors="ignore")
            # Encode non-numeric features cleanly
            if not covariate_features.empty:
                covariate_features, _ = safe_prepare_training_data(covariate_features)
                # Keep numeric columns only and remove zero variance
                numeric_cols = [c for c in covariate_features.columns if pd.api.types.is_numeric_dtype(covariate_features[c])]
                covariate_features = covariate_features[numeric_cols]
                # Drop constant columns
                covariate_features = covariate_features.loc[:, covariate_features.nunique() > 1]

                # Combine back duration and event for lifelines fit
                fit_df = pd.concat([covariate_features, df[[dur_col, ev_col]]], axis=1)

                try:
                    cph.fit(fit_df, duration_col=dur_col, event_col=ev_col)
                    c_index = float(round(float(cph.concordance_index_), 4))
                    for feat, hr in cph.hazard_ratios_.items():
                        hazard_ratios_dict[str(feat)] = float(round(float(hr), 4))
                except Exception as e:
                    warnings.append(f"[WARNING: COX PH FIT FAILED] {str(e)[:120]}. Falling back to Kaplan-Meier baseline.")
            else:
                warnings.append("[INFO: NO COVARIATES] No feature covariates available for Cox PH regression; Kaplan-Meier curve produced.")

        # Primary fitted model to return
        primary_model = cph if cph is not None and hasattr(cph, "hazard_ratios_") else kmf

        # 6. Build Metrics Dictionary
        metrics = {
            "model_type": model_type,
            "duration_column": dur_col,
            "event_column": ev_col,
            "total_observations": N,
            "events_observed": events_count,
            "censored_observations": censored_count,
            "censoring_rate_pct": censoring_rate_pct,
            "median_survival_time": median_survival,
            "concordance_index": c_index,
            "hazard_ratios": hazard_ratios_dict,
            "stratified_median_survival": stratified_medians,
            "survival_curve_milestones": km_curve_sample,
            "warnings": warnings
        }

        # 7. Format Executive Report
        report_lines = [
            "# === Survival Analysis & Cohort Retention Report ===",
            f"- **Cohort Size**: N={N} observations ({events_count} events, {censored_count} right-censored)",
            f"- **Censoring Rate**: {censoring_rate_pct:.1f}%",
            f"- **Median Survival Time**: {median_survival if median_survival is not None else 'Not Reached (> observation period)'}",
        ]
        if c_index is not None:
            report_lines.append(f"- **Cox PH Concordance Index (C-Index)**: **{c_index:.4f}** (Discriminative Ability)")

        if hazard_ratios_dict:
            report_lines.append("")
            report_lines.append("## Cox Proportional Hazards: Feature Hazard Ratios (HR):")
            report_lines.append("| Feature | Hazard Ratio (HR) | Risk Impact |")
            report_lines.append("| :--- | :--- | :--- |")
            for feat, hr in hazard_ratios_dict.items():
                if hr > 1.05:
                    impact = f"+{(hr - 1.0)*100.0:.1f}% higher event risk"
                elif hr < 0.95:
                    impact = f"-{(1.0 - hr)*100.0:.1f}% protective (lower risk)"
                else:
                    impact = "Neutral (negligible effect)"
                report_lines.append(f"| `{feat}` | **{hr:.4f}** | {impact} |")

        if stratified_medians:
            report_lines.append("")
            report_lines.append(f"## Stratified Cohort Median Survival ({strata_col}):")
            report_lines.append("| Cohort Segment | Median Survival Duration |")
            report_lines.append("| :--- | :--- |")
            for grp, med_val in stratified_medians.items():
                report_lines.append(f"| Segment: {grp} | **{med_val}** |")

        if km_curve_sample:
            report_lines.append("")
            report_lines.append("## Kaplan-Meier Survival Curve Milestones:")
            report_lines.append("| Timeline Elapsed | Probability Remaining (S(t)) |")
            report_lines.append("| :--- | :--- |")
            for pt in km_curve_sample[:8]:
                bar = "█" * int(pt["survival_probability"] * 15)
                report_lines.append(f"| Time = {pt['time']} | **{pt['survival_probability'] * 100:.1f}%** ({bar}) |")

        if warnings:
            report_lines.append("")
            report_lines.append("## Guardrail Alerts:")
            for w in warnings:
                report_lines.append(f"- {w}")

        report_markdown = "\n".join(report_lines)

        output: Dict[str, Any] = {
            "model": primary_model,
            "survival_model": primary_model,
            "kmf": kmf,
            "cph": cph,
            "metrics": metrics,
            "report": report_markdown
        }

        return pass_through_metadata(inputs, output, context)

    def to_code(self, config: Dict[str, Any]) -> str:
        dur = config.get("duration_col", "duration")
        ev = config.get("event_col", "event")
        penalizer = config.get("penalizer", 0.1)
        return (
            "from lifelines import KaplanMeierFitter, CoxPHFitter\n\n"
            "# 1. Non-parametric Kaplan-Meier Curve\n"
            "kmf = KaplanMeierFitter()\n"
            f"kmf.fit(df['{dur}'], event_observed=df['{ev}'], label='Overall Cohort')\n"
            "print('Median Survival Time:', kmf.median_survival_time_)\n\n"
            "# 2. Semi-parametric Cox Proportional Hazards Regression\n"
            f"cph = CoxPHFitter(penalizer={penalizer})\n"
            f"cph.fit(df, duration_col='{dur}', event_col='{ev}')\n"
            "cph.print_summary()"
        )
