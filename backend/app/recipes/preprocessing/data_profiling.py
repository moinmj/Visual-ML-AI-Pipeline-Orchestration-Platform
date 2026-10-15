"""
Data Profiling Report Recipe.
Sub-group: Cleanup & Quality
Category: Data Preparation & Transformation (preprocessing)

Produces a comprehensive profile of every column: types, missing values, distributions, and quality issues.
"""
from typing import Dict, Any, List, Optional
import math
import pandas as pd
import numpy as np
from backend.app.recipes.base.recipe import BaseRecipe, RecipePort


class DataProfilingReportRecipe(BaseRecipe):
    recipe_id = "data_profiling_report"
    name = "Data Profiling Report"
    version = "1.0.0"
    category = "preprocessing"
    group = "Data Preparation & Transformation"
    subgroup = "Cleanup & Quality"
    description = "Produces a comprehensive profile of every column: inferred physical/semantic types, missing rates, value distributions, cardinality, and automated quality issue diagnostics."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    inputs = [
        RecipePort(
            id="input",
            label="Input Table",
            type="dataframe",
            required=True,
            max_connections=1,
            description="Dataset to analyze, profile, and audit"
        )
    ]

    outputs = [
        RecipePort(
            id="dataframe",
            label="Dataset / Profile Table",
            type="dataframe",
            description="Dataset passed through or transposed column profiling summary table"
        ),
        RecipePort(
            id="profile_table",
            label="Profiling Summary",
            type="dataframe",
            description="Dedicated tabular audit profile with column quality indicators"
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "output_mode": {
                    "type": "string",
                    "title": "Output Data Mode",
                    "enum": [
                        "passthrough_with_report",
                        "profile_summary_table"
                    ],
                    "default": "passthrough_with_report",
                    "description": "'passthrough_with_report' preserves the incoming dataset for downstream processors while attaching the full audit report; 'profile_summary_table' replaces the output with the column profiling metrics table."
                },
                "null_threshold_pct": {
                    "type": "number",
                    "title": "High Null Alert Threshold (%)",
                    "default": 20.0,
                    "minimum": 0.0,
                    "maximum": 100.0,
                    "description": "Missing values percentage threshold to trigger a quality warning alert."
                },
                "high_cardinality_threshold_pct": {
                    "type": "number",
                    "title": "High Cardinality Threshold (%)",
                    "default": 95.0,
                    "minimum": 50.0,
                    "maximum": 100.0,
                    "description": "Unique values percentage to flag potential surrogate keys or leaky ID columns."
                },
                "sample_rows": {
                    "type": "integer",
                    "title": "Max Rows to Profile (0 = All)",
                    "default": 0,
                    "minimum": 0,
                    "description": "Limit row profiling volume on massive tables for sub-second generation."
                }
            },
            "required": []
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        df = inputs.get("dataframe")
        if df is None:
            if context and isinstance(context, dict) and "dataframe" in context:
                df = context["dataframe"]
            else:
                raise ValueError("DataProfilingReportRecipe expects 'dataframe' in inputs.")

        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        orig_rows = len(df)
        orig_cols = len(df.columns)
        if orig_rows == 0 or orig_cols == 0:
            empty_df = pd.DataFrame(columns=["column", "type", "missing_count", "quality_alerts"])
            return {
                "dataframe": df,
                "profile_table": empty_df,
                "metrics": {"total_rows": 0, "total_columns": 0, "quality_score": 100.0},
                "output_summary": {"title": "Data Profiling Report", "message": "Input table is empty."}
            }

        output_mode = config.get("output_mode", "passthrough_with_report")
        null_thresh = float(config.get("null_threshold_pct", 20.0))
        cardinality_thresh = float(config.get("high_cardinality_threshold_pct", 95.0))
        sample_limit = int(config.get("sample_rows", 0))

        profile_df = df if sample_limit <= 0 or sample_limit >= orig_rows else df.sample(n=sample_limit, random_state=42)
        n_rows = len(profile_df)

        profile_rows: List[Dict[str, Any]] = []
        quality_alerts: List[Dict[str, Any]] = []

        total_cells = orig_rows * orig_cols
        total_missing_cells = int(df.isna().sum().sum())
        missing_rate_overall = round((total_missing_cells / total_cells) * 100, 2) if total_cells > 0 else 0.0

        for col in df.columns:
            s = df[col]
            dtype_str = str(s.dtype)
            missing_count = int(s.isna().sum())
            missing_pct = round((missing_count / orig_rows) * 100, 2) if orig_rows > 0 else 0.0
            n_unique = int(s.nunique(dropna=True))
            unique_pct = round((n_unique / orig_rows) * 100, 2) if orig_rows > 0 else 0.0

            col_alerts = []
            if missing_pct >= null_thresh:
                col_alerts.append(f"High missing rate ({missing_pct}%)")
                quality_alerts.append({"column": col, "issue": "High Missing Rate", "severity": "WARNING", "detail": f"{missing_pct}% values are null."})

            if n_unique <= 1 and orig_rows > 1:
                col_alerts.append("Zero variance (constant)")
                quality_alerts.append({"column": col, "issue": "Zero Variance", "severity": "WARNING", "detail": "Column contains only 1 distinct value; provides no predictive signal."})

            if unique_pct >= cardinality_thresh and (s.dtype == "object" or s.dtype.name == "string"):
                col_alerts.append(f"Near-unique identifier ({unique_pct}%)")
                quality_alerts.append({"column": col, "issue": "High Cardinality ID", "severity": "INFO", "detail": "Column appears to be a unique ID or hash key."})

            # Check if numeric
            is_numeric = pd.api.types.is_numeric_dtype(s)
            p_row: Dict[str, Any] = {
                "column": col,
                "data_type": dtype_str,
                "inferred_type": "Numeric" if is_numeric else ("Datetime" if pd.api.types.is_datetime64_any_dtype(s) else "Categorical"),
                "total_count": orig_rows,
                "missing_count": missing_count,
                "missing_pct": f"{missing_pct}%",
                "unique_count": n_unique,
                "unique_pct": f"{unique_pct}%",
            }

            if is_numeric and s.dropna().shape[0] > 0:
                clean_num = s.dropna()
                mean_val = round(float(clean_num.mean()), 3)
                std_val = round(float(clean_num.std()), 3) if len(clean_num) > 1 else 0.0
                min_val = round(float(clean_num.min()), 3)
                max_val = round(float(clean_num.max()), 3)
                p25 = round(float(clean_num.quantile(0.25)), 3)
                p50 = round(float(clean_num.median()), 3)
                p75 = round(float(clean_num.quantile(0.75)), 3)
                zeros = int((clean_num == 0).sum())
                zeros_pct = round((zeros / orig_rows) * 100, 2)

                p_row["min"] = min_val
                p_row["max"] = max_val
                p_row["mean"] = mean_val
                p_row["std"] = std_val
                p_row["median"] = p50
                p_row["q25"] = p25
                p_row["q75"] = p75
                p_row["zero_count"] = zeros
                p_row["zero_pct"] = f"{zeros_pct}%"
            else:
                top_vals = s.value_counts(dropna=True).head(3).to_dict()
                top_str = ", ".join(f"'{k}' ({v})" for k, v in top_vals.items())
                p_row["top_values"] = top_str

            p_row["quality_alerts"] = "; ".join(col_alerts) if col_alerts else "Clean"
            profile_rows.append(p_row)

        profile_table_df = pd.DataFrame(profile_rows)

        # Dataset Health Score (0 - 100)
        penalty_missing = min(40.0, missing_rate_overall * 1.5)
        penalty_alerts = min(40.0, len(quality_alerts) * 4.0)
        quality_score = max(0.0, round(100.0 - penalty_missing - penalty_alerts, 1))

        output_summary = {
            "title": "Data Profiling Report",
            "message": f"Profiled {orig_cols} columns across {orig_rows:,} rows. Health Quality Score: {quality_score}/100 with {len(quality_alerts)} alerts detected.",
            "metrics": {
                "total_rows": orig_rows,
                "total_columns": orig_cols,
                "missing_cells_count": total_missing_cells,
                "overall_missing_pct": missing_rate_overall,
                "quality_alerts_count": len(quality_alerts),
                "dataset_health_score": quality_score
            },
            "quality_alerts": quality_alerts[:8],
            "column_profiles": profile_rows
        }

        chosen_df = profile_table_df if output_mode == "profile_summary_table" else df

        return {
            "dataframe": chosen_df,
            "profile_table": profile_table_df,
            "metrics": output_summary["metrics"],
            "output_summary": output_summary
        }

    def to_code(self, config: Dict[str, Any]) -> str:
        return (
            "# Data Profiling Report\n"
            "profile_records = []\n"
            "for col in df.columns:\n"
            "    s = df[col]\n"
            "    profile_records.append({\n"
            "        'column': col,\n"
            "        'dtype': str(s.dtype),\n"
            "        'missing': s.isna().sum(),\n"
            "        'unique': s.nunique(),\n"
            "        'mean': s.mean() if pd.api.types.is_numeric_dtype(s) else None\n"
            "    })\n"
            "profile_df = pd.DataFrame(profile_records)\n"
            "print('Profiling summary generated for', len(df.columns), 'columns.')\n"
        )
