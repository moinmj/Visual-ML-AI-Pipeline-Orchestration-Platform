from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from backend.app.recipes.base.recipe import BaseRecipe, RecipePort


class DataDriftDetectorRecipe(BaseRecipe):
    recipe_id = "data_drift_detector"
    name = "Data Drift Detector (input distribution)"
    version = "1.0.0"
    category = "monitoring"
    description = "Detects when incoming data no longer matches the training data distribution."
    input_types = ["dataframe"]
    output_types = ["metrics", "dataframe"]

    def get_inputs(self) -> List[RecipePort]:
        return [
            RecipePort(id="input", label="Input Table", type="dataframe", required=True, description="Primary inference or combined dataset"),
            RecipePort(id="baseline_data", label="Baseline Table (Optional)", type="dataframe", required=False, description="Baseline reference dataset")
        ]

    def get_outputs(self) -> List[RecipePort]:
        return [
            RecipePort(id="output", label="Output (metrics)", type="metrics", description="Data drift summary metrics and alert indicators"),
            RecipePort(id="output_table", label="Drift Analysis Table", type="dataframe", description="Feature-by-feature drift metrics and p-values")
        ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "detection_method": {
                    "type": "string",
                    "title": "Drift Detection Method",
                    "enum": [
                        "Population Stability Index (PSI)",
                        "Kolmogorov-Smirnov (KS-Test)",
                        "Wasserstein Distance"
                    ],
                    "default": "Population Stability Index (PSI)",
                    "description": "Statistical test used to compare baseline vs inference numeric distributions."
                },
                "drift_threshold": {
                    "type": "number",
                    "title": "Drift Alert Threshold",
                    "default": 0.10,
                    "minimum": 0.001,
                    "maximum": 1.0,
                    "description": "Threshold above which feature distribution is flagged as drifted (e.g. PSI >= 0.10 or KS p-value <= 0.05)."
                },
                "feature_columns": {
                    "type": "string",
                    "title": "Feature Columns to Monitor",
                    "default": "",
                    "description": "Comma-separated feature list (monitors all features if left blank)."
                },
                "baseline_split_ratio": {
                    "type": "number",
                    "title": "Baseline Split Ratio",
                    "default": 0.5,
                    "minimum": 0.1,
                    "maximum": 0.9,
                    "description": "Proportion of input data treated as baseline reference when a single dataset is provided."
                },
                "dataset_drift_ratio_threshold": {
                    "type": "number",
                    "title": "Dataset Drift Share Threshold",
                    "default": 0.25,
                    "minimum": 0.05,
                    "maximum": 1.0,
                    "description": "Fraction of drifted features required to trigger a full dataset-level drift alert (e.g. 0.25 = 25%)."
                }
            }
        }

    def _compute_psi(self, baseline: np.ndarray, current: np.ndarray, num_bins: int = 10) -> float:
        """Computes Population Stability Index (PSI) with epsilon smoothing."""
        try:
            baseline = baseline[~np.isnan(baseline)]
            current = current[~np.isnan(current)]
            if len(baseline) == 0 or len(current) == 0:
                return 0.0

            # Quantile bin edges based on baseline
            quantiles = np.linspace(0, 100, num_bins + 1)
            bin_edges = np.percentile(baseline, quantiles)
            bin_edges[0] = -np.inf
            bin_edges[-1] = np.inf

            # Deduplicate bin edges if data has repeated values
            bin_edges = np.unique(bin_edges)
            if len(bin_edges) <= 2:
                # If unique edges collapsed, use linear spacing
                min_v = min(np.min(baseline), np.min(current))
                max_v = max(np.max(baseline), np.max(current))
                if min_v == max_v:
                    return 0.0
                bin_edges = np.linspace(min_v - 1e-5, max_v + 1e-5, num_bins + 1)

            base_counts, _ = np.histogram(baseline, bins=bin_edges)
            curr_counts, _ = np.histogram(current, bins=bin_edges)

            # Convert to relative proportions with smoothing
            base_pct = (base_counts + 1e-4) / (len(baseline) + 1e-4 * len(base_counts))
            curr_pct = (curr_counts + 1e-4) / (len(current) + 1e-4 * len(curr_counts))

            psi_val = np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct))
            return float(max(0.0, psi_val))
        except Exception:
            return 0.0

    def _compute_ks(self, baseline: np.ndarray, current: np.ndarray) -> Tuple[float, float]:
        """Computes 2-sample Kolmogorov-Smirnov test."""
        try:
            baseline = baseline[~np.isnan(baseline)]
            current = current[~np.isnan(current)]
            if len(baseline) == 0 or len(current) == 0:
                return 0.0, 1.0
            stat, p_val = stats.ks_2samp(baseline, current)
            return float(stat), float(p_val)
        except Exception:
            return 0.0, 1.0

    def _compute_wasserstein(self, baseline: np.ndarray, current: np.ndarray) -> float:
        """Computes normalized Earth Mover's (Wasserstein) Distance."""
        try:
            baseline = baseline[~np.isnan(baseline)]
            current = current[~np.isnan(current)]
            if len(baseline) == 0 or len(current) == 0:
                return 0.0
            raw_dist = stats.wasserstein_distance(baseline, current)
            # Normalize by standard deviation of baseline
            std_base = float(np.std(baseline))
            norm_dist = (raw_dist / std_base) if std_base > 1e-6 else raw_dist
            return float(norm_dist)
        except Exception:
            return 0.0

    def _compute_categorical_psi(self, base_series: pd.Series, curr_series: pd.Series) -> float:
        """Computes categorical stability distance across categories."""
        try:
            all_cats = list(set(base_series.dropna().unique()).union(set(curr_series.dropna().unique())))
            if not all_cats:
                return 0.0
            base_counts = base_series.value_counts(normalize=True).to_dict()
            curr_counts = curr_series.value_counts(normalize=True).to_dict()

            psi = 0.0
            for cat in all_cats:
                b_p = base_counts.get(cat, 1e-4)
                c_p = curr_counts.get(cat, 1e-4)
                psi += (c_p - b_p) * np.log(c_p / b_p)
            return float(max(0.0, psi))
        except Exception:
            return 0.0

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        # 1. Resolve Baseline vs Current (Inference) Datasets
        baseline_df = None
        current_df = None

        if isinstance(inputs.get("left_dataframe"), pd.DataFrame) and isinstance(inputs.get("right_dataframe"), pd.DataFrame):
            baseline_df = inputs["left_dataframe"]
            current_df = inputs["right_dataframe"]
        elif isinstance(inputs.get("baseline_data"), pd.DataFrame) and isinstance(inputs.get("input"), pd.DataFrame):
            baseline_df = inputs["baseline_data"]
            current_df = inputs["input"]
        elif isinstance(inputs.get("X_train"), pd.DataFrame) and isinstance(inputs.get("X_test"), pd.DataFrame):
            baseline_df = inputs["X_train"]
            current_df = inputs["X_test"]
        else:
            primary_df = inputs.get("dataframe")
            if primary_df is None:
                primary_df = inputs.get("input")
            if primary_df is None:
                primary_df = inputs.get("input_table")
            if primary_df is None or not isinstance(primary_df, pd.DataFrame) or primary_df.empty:
                raise ValueError("Valid input dataframe is required for Data Drift Detector.")

            # Chronological / sequential split
            split_ratio = float(config.get("baseline_split_ratio", 0.5))
            split_idx = int(len(primary_df) * split_ratio)
            baseline_df = primary_df.iloc[:split_idx].copy()
            current_df = primary_df.iloc[split_idx:].copy()

        method = config.get("detection_method", "Population Stability Index (PSI)")
        drift_threshold = float(config.get("drift_threshold", 0.10))
        dataset_drift_ratio = float(config.get("dataset_drift_ratio_threshold", 0.25))

        # 2. Select Features to Monitor
        raw_feat_cols = config.get("feature_columns") or config.get("columns") or ""
        target_features: List[str] = []
        if isinstance(raw_feat_cols, list):
            target_features = [str(c).strip() for c in raw_feat_cols if str(c).strip()]
        elif isinstance(raw_feat_cols, str) and raw_feat_cols.strip():
            target_features = [c.strip() for c in raw_feat_cols.split(",") if c.strip()]

        shared_cols = [c for c in baseline_df.columns if c in current_df.columns]
        if target_features:
            features = [c for c in target_features if c in shared_cols]
        else:
            features = [c for c in shared_cols if c.lower() not in ["id", "uuid", "created_at", "timestamp", "row_id"]]
        if not features:
            features = shared_cols

        # 3. Analyze Feature-by-Feature Distribution Shifts
        feature_rows: List[Dict[str, Any]] = []
        drifted_count = 0
        max_drift_score = 0.0
        top_drifted_feature = "None"

        for feat in features:
            base_col = baseline_df[feat].dropna()
            curr_col = current_df[feat].dropna()

            if len(base_col) == 0 or len(curr_col) == 0:
                continue

            is_numeric = np.issubdtype(base_col.dtype, np.number) and np.issubdtype(curr_col.dtype, np.number)

            if is_numeric:
                b_vals = base_col.values.astype(float)
                c_vals = curr_col.values.astype(float)

                if "KS-Test" in method:
                    ks_stat, p_val = self._compute_ks(b_vals, c_vals)
                    score = ks_stat
                    # For KS-test, p-value < 0.05 indicates significant drift
                    is_drifted = p_val < min(0.05, drift_threshold)
                    p_val_display = round(p_val, 5)
                    test_label = "KS-Test"
                elif "Wasserstein" in method:
                    w_dist = self._compute_wasserstein(b_vals, c_vals)
                    score = w_dist
                    is_drifted = score > drift_threshold
                    p_val_display = None
                    test_label = "Wasserstein Distance"
                else:
                    psi_val = self._compute_psi(b_vals, c_vals)
                    score = psi_val
                    is_drifted = score >= drift_threshold
                    p_val_display = None
                    test_label = "PSI"

                dtype_label = "Numeric"
            else:
                score = self._compute_categorical_psi(base_col, curr_col)
                is_drifted = score >= drift_threshold
                p_val_display = None
                test_label = "Categorical PSI"
                dtype_label = "Categorical"

            if is_drifted:
                drifted_count += 1
                status_str = "DRIFT_DETECTED"
                rec_str = "Retrain model or update feature preprocessing transform."
            else:
                status_str = "STABLE"
                rec_str = "Distribution matches baseline within tolerance."

            if score > max_drift_score:
                max_drift_score = score
                top_drifted_feature = feat

            feature_rows.append({
                "feature": feat,
                "data_type": dtype_label,
                "method": test_label,
                "drift_score": round(score, 4),
                "threshold": drift_threshold,
                "p_value": p_val_display if p_val_display is not None else "N/A",
                "status": status_str,
                "recommendation": rec_str
            })

        # 4. Synthesize Overall Dataset Drift Summary
        total_feats = max(1, len(feature_rows))
        drift_share = drifted_count / total_feats
        dataset_drift_flag = drift_share >= dataset_drift_ratio

        drift_analysis_df = pd.DataFrame(feature_rows)

        status_overall = "DATASET_DRIFT_ALERT" if dataset_drift_flag else ("MINOR_DRIFT" if drifted_count > 0 else "HEALTHY")

        metrics = {
            "status": status_overall,
            "dataset_drift_detected": dataset_drift_flag,
            "drifted_features_count": drifted_count,
            "total_features_monitored": total_feats,
            "drift_share_percentage": round(drift_share * 100.0, 2),
            "max_drift_score": round(max_drift_score, 4),
            "highest_drift_feature": top_drifted_feature
        }

        report = {
            "title": "Data Distribution Drift Report",
            "overall_status": status_overall,
            "method": method,
            "dataset_drift_detected": dataset_drift_flag,
            "drifted_features_count": f"{drifted_count} of {total_feats} ({round(drift_share * 100.0, 1)}%)",
            "highest_drift_feature": f"{top_drifted_feature} (Score: {round(max_drift_score, 4)})",
            "baseline_samples": len(baseline_df),
            "current_samples": len(current_df)
        }

        return {
            "metrics": metrics,
            "dataframe": drift_analysis_df,
            "output_table": drift_analysis_df,
            "report": report,
            "input_dataframe": current_df
        }
