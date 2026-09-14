import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.feature_selection import (
    SelectKBest,
    SelectPercentile,
    VarianceThreshold,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression
)
from sklearn.preprocessing import LabelEncoder
from backend.app.recipes.base.recipe import BaseRecipe


class FeatureSelectorRecipe(BaseRecipe):
    recipe_id = "feature_selector"
    name = "Feature Selector (SelectKBest / Mutual Info)"
    version = "1.0.0"
    category = "preprocessing"
    description = "Ranks and filters top features using supervised statistical scoring (SelectKBest, Mutual Information, Percentile, or Variance Threshold)."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "title": "Selection Method",
                    "enum": ["select_k_best", "mutual_info", "percentile", "variance_threshold"],
                    "default": "select_k_best",
                    "description": "'select_k_best' uses ANOVA F-value; 'mutual_info' measures non-linear mutual dependence; 'percentile' keeps top % of features; 'variance_threshold' removes low-variance features."
                },
                "k": {
                    "type": "integer",
                    "title": "Top K Features",
                    "default": 10,
                    "minimum": 1,
                    "description": "Number of top features to retain (used when method='select_k_best')."
                },
                "percentile": {
                    "type": "integer",
                    "title": "Top Percentile (%)",
                    "default": 50,
                    "minimum": 1,
                    "maximum": 100,
                    "description": "Percentage of top features to retain (used when method='percentile')."
                },
                "target_column": {
                    "type": "string",
                    "title": "Target Variable",
                    "description": "Target variable to calculate feature importance against. If empty, automatically detected from common target keywords."
                },
                "task_type": {
                    "type": "string",
                    "title": "Problem Task Type",
                    "enum": ["auto", "classification", "regression"],
                    "default": "auto",
                    "description": "'auto' infers whether the target is discrete (classification) or continuous (regression)."
                },
                "feature_columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "title": "Candidate Feature Columns",
                    "description": "Specific columns to evaluate. If empty, all columns except target are evaluated."
                }
            },
            "required": ["method"]
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        df: pd.DataFrame = inputs.get("dataframe")
        if df is None:
            if context and isinstance(context, dict) and "dataframe" in context:
                df = context["dataframe"]
            else:
                raise ValueError("FeatureSelectorRecipe expects 'dataframe' in inputs.")

        if len(df) == 0 or len(df.columns) <= 1:
            return {"dataframe": df.copy()}

        df_out = df.copy()
        method = config.get("method", "select_k_best")
        k_val = int(config.get("k", 10))
        pct_val = int(config.get("percentile", 50))
        target_col = (config.get("target_column") or "").strip()
        task_type = config.get("task_type", "auto")
        feat_cols_cfg = config.get("feature_columns", [])

        # 1. Resolve Target Column if not provided
        if not target_col or target_col not in df_out.columns:
            target_keywords = ["target", "churn", "survived", "label", "price", "sales", "revenue", "y"]
            named_matches = [c for c in df_out.columns if any(k in c.lower() for k in target_keywords)]
            target_col = named_matches[0] if named_matches else list(df_out.columns)[-1]

        target_exists = target_col in df_out.columns
        target_series = df_out[target_col] if target_exists else None

        # 2. Candidate Features
        if isinstance(feat_cols_cfg, str):
            specified_feats = [c.strip() for c in feat_cols_cfg.split(",") if c.strip() and c in df_out.columns]
        elif isinstance(feat_cols_cfg, (list, tuple)):
            specified_feats = [str(c).strip() for c in feat_cols_cfg if str(c).strip() in df_out.columns]
        else:
            specified_feats = []

        if specified_feats:
            feature_cols = [c for c in specified_feats if c != target_col]
        else:
            feature_cols = [c for c in df_out.columns if c != target_col]

        if not feature_cols:
            return {"dataframe": df_out}

        # 3. Variance Threshold (Unsupervised)
        if method == "variance_threshold":
            numeric_feats = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df_out[c])]
            non_numeric = [c for c in feature_cols if c not in numeric_feats]
            if numeric_feats:
                vt = VarianceThreshold(threshold=0.01)
                vt.fit(df_out[numeric_feats].fillna(0))
                selected_num = [numeric_feats[i] for i, mask in enumerate(vt.get_support()) if mask]
                selected_features = (selected_num if selected_num else numeric_feats) + non_numeric
            else:
                selected_features = feature_cols

            final_cols = [c for c in selected_features if c in df_out.columns]
            if target_exists and target_col not in final_cols:
                final_cols.append(target_col)
            return {"dataframe": df_out[final_cols]}

        # 4. Supervised Feature Selection: Infer task type if auto
        is_classification = True
        if target_exists:
            t_clean = target_series.dropna()
            nunique = t_clean.nunique()
            if task_type == "classification":
                is_classification = True
            elif task_type == "regression":
                is_classification = False
            else:
                if pd.api.types.is_numeric_dtype(t_clean) and nunique > 20:
                    is_classification = False
                else:
                    is_classification = True

        # Prepare X and y for scoring
        X_df = df_out[feature_cols].copy()
        for col in X_df.columns:
            if not pd.api.types.is_numeric_dtype(X_df[col]):
                le = LabelEncoder()
                X_df[col] = le.fit_transform(X_df[col].astype(str))
            else:
                X_df[col] = X_df[col].fillna(X_df[col].median() if not np.isnan(X_df[col].median()) else 0)

        y_raw = target_series.copy() if target_series is not None else pd.Series(np.zeros(len(df_out)))
        if is_classification:
            y_encoded = LabelEncoder().fit_transform(y_raw.astype(str))
        else:
            y_encoded = pd.to_numeric(y_raw, errors="coerce").fillna(0).values

        # Determine effective k
        k_effective = min(k_val, len(feature_cols))

        try:
            if method == "select_k_best":
                score_func = f_classif if is_classification else f_regression
                selector = SelectKBest(score_func=score_func, k=k_effective)
                selector.fit(X_df, y_encoded)
                selected_mask = selector.get_support()
                selected_features = [feature_cols[i] for i, mask in enumerate(selected_mask) if mask]

            elif method == "mutual_info":
                score_func = mutual_info_classif if is_classification else mutual_info_regression
                selector = SelectKBest(score_func=score_func, k=k_effective)
                selector.fit(X_df, y_encoded)
                selected_mask = selector.get_support()
                selected_features = [feature_cols[i] for i, mask in enumerate(selected_mask) if mask]

            elif method == "percentile":
                score_func = f_classif if is_classification else f_regression
                selector = SelectPercentile(score_func=score_func, percentile=pct_val)
                selector.fit(X_df, y_encoded)
                selected_mask = selector.get_support()
                selected_features = [feature_cols[i] for i, mask in enumerate(selected_mask) if mask]

            else:
                selected_features = feature_cols[:k_effective]

        except Exception:
            # Fallback if statistical scoring encounters constant variance or numerical singularities
            selected_features = feature_cols[:k_effective]

        if not selected_features:
            selected_features = feature_cols[:k_effective]

        # Ensure target column is strictly preserved
        final_cols = [c for c in selected_features if c in df_out.columns]
        if target_exists and target_col not in final_cols:
            final_cols.append(target_col)

        return {"dataframe": df_out[final_cols]}

    def to_code(self, config: Dict[str, Any]) -> str:
        method = config.get("method", "select_k_best")
        k = config.get("k", 10)
        tgt = config.get("target_column", "target")
        return f"# Feature Selection via {method.upper()}\nfrom sklearn.feature_selection import SelectKBest, f_classif\n# Keeps top {k} predictive features relative to '{tgt}'\n# Preserves target column for downstream train/test splitting"
