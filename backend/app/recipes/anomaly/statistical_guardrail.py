import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from backend.app.recipes.base.recipe import BaseRecipe


class StatisticalGuardrailRecipe(BaseRecipe):
    recipe_id = "statistical_guardrail"
    name = "Statistical Outlier Guardrail (Z-Score / IQR)"
    version = "1.0.0"
    category = "anomaly"
    description = "Detects or filters data quality outliers using Z-Score (standard deviation) or IQR thresholding."
    input_types = ["dataframe"]
    output_types = ["dataframe", "metrics"]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "title": "Detection Method",
                    "enum": ["z_score", "iqr"],
                    "default": "z_score",
                    "description": "Z-Score uses standard deviation (normal distributions), IQR uses interquartile range (skewed data)."
                },
                "threshold": {
                    "type": "number",
                    "title": "Threshold Multiplier",
                    "default": 3.0,
                    "minimum": 1.0,
                    "maximum": 6.0,
                    "description": "For Z-score: number of std devs (e.g. 3.0). For IQR: IQR multiplier (e.g. 1.5)."
                },
                "action": {
                    "type": "string",
                    "title": "Guardrail Action",
                    "enum": ["flag", "filter"],
                    "default": "flag",
                    "description": "'flag' adds an 'is_outlier' column; 'filter' removes outlier rows from the dataset."
                },
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "title": "Columns to Evaluate",
                    "description": "Specific numeric columns to inspect. If empty, all numeric features are evaluated."
                }
            }
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        df: pd.DataFrame = inputs.get("dataframe")
        if df is None:
            if context and isinstance(context, dict) and "dataframe" in context:
                df = context["dataframe"]
            else:
                raise ValueError("StatisticalGuardrail expects 'dataframe' in inputs.")

        df_out = df.copy()
        method = config.get("method", "z_score")
        threshold = float(config.get("threshold", 3.0 if method == "z_score" else 1.5))
        action = config.get("action", "flag")
        target_cols = config.get("columns", [])

        if isinstance(target_cols, str):
            if target_cols.strip():
                parsed = [c.strip() for c in target_cols.split(",") if c.strip() in df_out.columns]
                target_cols = parsed if parsed else ([target_cols] if target_cols in df_out.columns else [])
            else:
                target_cols = []
        elif isinstance(target_cols, (list, tuple)):
            target_cols = [c for c in target_cols if c in df_out.columns]
        else:
            target_cols = []

        # Detect numeric columns
        numeric_cols = [c for c in df_out.columns if pd.api.types.is_numeric_dtype(df_out[c]) and c not in ["is_anomaly", "is_outlier"]]
        if target_cols:
            numeric_cols = [c for c in target_cols if c in numeric_cols]

        if not numeric_cols:
            df_out["is_outlier"] = 0
            return {"dataframe": df_out, "metrics": {"total_records": len(df_out), "outlier_count": 0}}

        outlier_mask = pd.Series(False, index=df_out.index)
        column_outlier_counts = {}

        for col in numeric_cols:
            series = df_out[col].dropna()
            if len(series) == 0:
                continue

            if method == "z_score":
                mean = series.mean()
                std = series.std()
                if std > 0:
                    z_scores = (df_out[col] - mean).abs() / std
                    col_mask = z_scores > threshold
                else:
                    col_mask = pd.Series(False, index=df_out.index)

            elif method == "iqr":
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - (threshold * iqr)
                upper_bound = q3 + (threshold * iqr)
                col_mask = (df_out[col] < lower_bound) | (df_out[col] > upper_bound)

            col_outliers = int(col_mask.sum())
            column_outlier_counts[col] = col_outliers
            outlier_mask = outlier_mask | col_mask

        total_records = len(df_out)
        outlier_count = int(outlier_mask.sum())
        outlier_pct = float(round((outlier_count / total_records) * 100, 2)) if total_records > 0 else 0.0

        if action == "filter":
            filtered_df = df_out[~outlier_mask].copy()
            final_df = filtered_df
        else:
            df_out["is_outlier"] = np.where(outlier_mask, 1, 0)
            final_df = df_out

        metrics = {
            "task_type": "anomaly_detection",
            "algorithm": f"Statistical Guardrail ({method.upper()})",
            "action_taken": action,
            "total_records_before": total_records,
            "outliers_detected": outlier_count,
            "outlier_percentage": outlier_pct,
            "column_breakdown": column_outlier_counts
        }

        return {
            "dataframe": final_df,
            "metrics": metrics,
            "anomaly_summary": metrics
        }
