"""
Date Range Expander Recipe.
Sub-group: Domain-Specific Cleaning
Category: Data Preparation & Transformation (preprocessing)

Expands start/end date ranges into individual daily, weekly, or monthly rows with optional
metric proration (e.g. splitting total revenue evenly across stay dates or billing periods).
"""
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
from backend.app.recipes.base.recipe import BaseRecipe, RecipePort


class DateRangeExpanderRecipe(BaseRecipe):
    recipe_id = "date_range_expander"
    name = "Date Range Expander"
    version = "1.0.0"
    category = "preprocessing"
    group = "Data Preparation & Transformation"
    subgroup = "Domain-Specific Cleaning"
    description = "Expands start/end date ranges into one row per day, week, or month with optional metric proration."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    inputs = [
        RecipePort(
            id="input",
            label="Input Table",
            type="dataframe",
            required=True,
            max_connections=1,
            description="Dataset containing date intervals to expand"
        )
    ]

    outputs = [
        RecipePort(
            id="dataframe",
            label="Expanded Table",
            type="dataframe",
            description="Dataset expanded into individual calendar intervals"
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "start_date_column": {
                    "type": "string",
                    "title": "Start Date Column",
                    "default": "start_date",
                    "description": "Column containing the start of the date interval."
                },
                "end_date_column": {
                    "type": "string",
                    "title": "End Date Column",
                    "default": "end_date",
                    "description": "Column containing the end of the date interval."
                },
                "frequency": {
                    "type": "string",
                    "title": "Expansion Frequency",
                    "enum": ["day", "week", "month", "hour"],
                    "default": "day",
                    "description": "Calendar interval frequency: 'day' (1 row/day), 'week', 'month', or 'hour'."
                },
                "date_column_name": {
                    "type": "string",
                    "title": "Expanded Date Column Name",
                    "default": "date",
                    "description": "Name of the newly created date column."
                },
                "include_end_date": {
                    "type": "boolean",
                    "title": "Include End Date in Range",
                    "default": True,
                    "description": "Whether the end date is inclusive in the expansion."
                },
                "prorate_columns": {
                    "type": "array",
                    "title": "Columns to Prorate (Divide Evenly)",
                    "items": {"type": "string"},
                    "default": [],
                    "description": "Numeric columns (e.g. revenue, cost, budget) to divide evenly across expanded intervals."
                },
                "max_intervals_per_row": {
                    "type": "integer",
                    "title": "Max Intervals per Row",
                    "default": 1000,
                    "description": "Safety threshold to prevent memory explosion on huge date spans."
                }
            },
            "required": ["start_date_column", "end_date_column"]
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        df = inputs.get("dataframe") if "dataframe" in inputs else inputs.get("input")
        if df is None:
            df = inputs.get("df")

        if df is None:
            return {
                "dataframe": pd.DataFrame(),
                "metrics": {"original_rows": 0, "expanded_rows": 0},
                "output_summary": {"title": "Date Range Expander", "message": "No input table provided.", "metrics": {}}
            }

        df = df.copy()
        if len(df) == 0:
            return {
                "dataframe": df,
                "metrics": {"original_rows": 0, "expanded_rows": 0},
                "output_summary": {"title": "Date Range Expander", "message": "Input table is empty.", "metrics": {}}
            }

        start_col = config.get("start_date_column", "start_date")
        end_col = config.get("end_date_column", "end_date")
        freq_str = config.get("frequency", "day")
        date_col = config.get("date_column_name", "date")
        inclusive = config.get("include_end_date", True)
        prorate_cols = config.get("prorate_columns", [])
        max_intervals = config.get("max_intervals_per_row", 1000)

        if start_col not in df.columns or end_col not in df.columns:
            return {
                "dataframe": df,
                "metrics": {"original_rows": len(df), "expanded_rows": len(df)},
                "output_summary": {"title": "Date Range Expander", "message": f"Columns '{start_col}' or '{end_col}' not found.", "metrics": {}}
            }

        # Frequency mapping to pandas offset aliases
        freq_map = {
            "day": "D",
            "week": "W-MON",
            "month": "MS",
            "hour": "h"
        }
        pd_freq = freq_map.get(freq_str.lower(), "D")

        active_prorate = [c for c in prorate_cols if c in df.columns]

        expanded_rows: List[Dict[str, Any]] = []

        start_dt = pd.to_datetime(df[start_col], errors="coerce")
        end_dt = pd.to_datetime(df[end_col], errors="coerce")

        for idx in range(len(df)):
            row = df.iloc[idx].to_dict()
            s_val = start_dt.iloc[idx]
            e_val = end_dt.iloc[idx]

            if pd.isna(s_val) or pd.isna(e_val):
                row[date_col] = s_val if pd.notna(s_val) else e_val
                row["interval_index"] = 0
                row["total_intervals"] = 1
                expanded_rows.append(row)
                continue

            if s_val > e_val:
                s_val, e_val = e_val, s_val

            # Generate range
            date_range = pd.date_range(start=s_val, end=e_val, freq=pd_freq, inclusive="both" if inclusive else "left")
            if len(date_range) == 0:
                date_range = pd.DatetimeIndex([s_val])

            # Safety cap
            if len(date_range) > max_intervals:
                date_range = date_range[:max_intervals]

            n_intervals = len(date_range)

            for step_idx, dt_pt in enumerate(date_range):
                new_row = dict(row)
                new_row[date_col] = dt_pt
                new_row["interval_index"] = step_idx
                new_row["total_intervals"] = n_intervals

                # Prorate numerical columns
                for p_col in active_prorate:
                    orig_val = row.get(p_col)
                    if pd.notna(orig_val):
                        try:
                            new_row[p_col] = round(float(orig_val) / n_intervals, 4)
                        except (ValueError, TypeError):
                            pass

                expanded_rows.append(new_row)

        out_df = pd.DataFrame(expanded_rows)

        output_summary = {
            "title": "Date Range Expander",
            "message": f"Expanded {len(df):,} records into {len(out_df):,} interval rows at frequency '{freq_str}'.",
            "metrics": {
                "original_rows": len(df),
                "expanded_rows": len(out_df),
                "expansion_factor": round(len(out_df) / max(1, len(df)), 2),
                "frequency": freq_str
            }
        }

        return {
            "dataframe": out_df,
            "metrics": output_summary["metrics"],
            "output_summary": output_summary
        }

    def to_code(self, config: Dict[str, Any]) -> str:
        s_col = repr(config.get("start_date_column", "start_date"))
        e_col = repr(config.get("end_date_column", "end_date"))
        freq = repr(config.get("frequency", "day"))
        return (
            f"# Recipe: {self.name}\n"
            f"# ID: {self.recipe_id}\n"
            f"# Start: {s_col}, End: {e_col}, Freq: {freq}\n"
            f"# Expands date ranges into granular interval rows\n"
        )
