"""
Parse Date Recipe.
Sub-group: Custom Logic
Category: Data Preparation & Transformation (preprocessing)

Parses text into proper dates using given or auto-detected formats with
support for part extraction (year, month, day_name, is_weekend).
"""
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
from backend.app.recipes.base.recipe import BaseRecipe, RecipePort


class ParseDateRecipe(BaseRecipe):
    recipe_id = "parse_date"
    name = "Parse Date"
    version = "1.0.0"
    category = "preprocessing"
    group = "Data Preparation & Transformation"
    subgroup = "Custom Logic"
    description = "Parses text into proper dates using a given or auto-detected format, with options to extract calendar features (year, month, day name, weekend)."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    inputs = [
        RecipePort(
            id="input",
            label="Input Table",
            type="dataframe",
            required=True,
            max_connections=1,
            description="Dataset containing date/timestamp strings to parse"
        )
    ]

    outputs = [
        RecipePort(
            id="dataframe",
            label="Parsed Table",
            type="dataframe",
            description="Dataset with parsed datetime columns and optional extracted calendar parts"
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "columns": {
                    "type": "array",
                    "title": "Date Columns to Parse",
                    "input_type": "multi_column_select",
                    "items": {"type": "string"},
                    "default": [],
                    "description": "List of columns containing dates or timestamps to parse."
                },
                "column": {
                    "type": "string",
                    "title": "Single Date Column",
                    "input_type": "column_select",
                    "description": "Alternative single column to parse."
                },
                "date_format": {
                    "type": "string",
                    "title": "Date Format String",
                    "input_type": "text",
                    "default": "auto",
                    "description": "Explicit strftime format string (e.g. '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y %H:%M:%S', '%d-%b-%Y') or 'auto' for intelligent heuristic detection."
                },
                "dayfirst": {
                    "type": "boolean",
                    "title": "Day First (DD/MM/YYYY)",
                    "input_type": "checkbox",
                    "default": False,
                    "description": "Whether to interpret the first value in an ambiguous 3-integer date (e.g. 01/05/09) as the day."
                },
                "yearfirst": {
                    "type": "boolean",
                    "title": "Year First (YYYY/MM/DD)",
                    "input_type": "checkbox",
                    "default": False,
                    "description": "Whether to interpret the first value in an ambiguous 3-integer date as the year."
                },
                "errors": {
                    "type": "string",
                    "title": "Error Handling Action",
                    "input_type": "dropdown",
                    "enum": ["coerce", "raise", "ignore"],
                    "default": "coerce",
                    "description": "'coerce' sets invalid/unparseable values to NaT (null); 'raise' throws an exception; 'ignore' leaves invalid values as original strings."
                },
                "output_mode": {
                    "type": "string",
                    "title": "Output Mode",
                    "input_type": "dropdown",
                    "enum": ["in_place", "new_column"],
                    "default": "in_place",
                    "description": "Whether to overwrite the original column in-place or generate a new column with a suffix."
                },
                "new_column_suffix": {
                    "type": "string",
                    "title": "New Column Suffix",
                    "input_type": "text",
                    "default": "_parsed",
                    "description": "Suffix for new column when output_mode is 'new_column'."
                },
                "output_format": {
                    "type": "string",
                    "title": "Output Representation",
                    "input_type": "dropdown",
                    "enum": ["datetime64", "date_only", "iso8601", "custom"],
                    "default": "datetime64",
                    "description": "'datetime64' keeps native pandas datetime; 'date_only' formats as YYYY-MM-DD; 'iso8601' produces standard ISO timestamps; 'custom' uses target_date_format."
                },
                "target_date_format": {
                    "type": "string",
                    "title": "Target Date Format (for custom format)",
                    "input_type": "text",
                    "default": "%Y-%m-%d",
                    "description": "Format string if output_format is set to 'custom'."
                },
                "extract_parts": {
                    "type": "array",
                    "title": "Extract Calendar Components",
                    "input_type": "multi_select",
                    "items": {
                        "type": "string",
                        "enum": ["year", "month", "day", "day_name", "day_of_week", "hour", "quarter", "is_weekend"]
                    },
                    "default": [],
                    "description": "Calendar features to extract into dedicated columns (e.g. [col]_year, [col]_month, [col]_is_weekend)."
                }
            },
            "required": []
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        df = inputs.get("dataframe") if "dataframe" in inputs else inputs.get("input")
        if df is None:
            df = inputs.get("df")

        if df is None:
            return {
                "dataframe": pd.DataFrame(),
                "metrics": {"columns_parsed": 0, "rows_processed": 0},
                "output_summary": {"title": "Parse Date", "message": "No input table provided.", "metrics": {}}
            }

        df = df.copy()
        if len(df) == 0:
            return {
                "dataframe": df,
                "metrics": {"columns_parsed": 0, "rows_processed": 0},
                "output_summary": {"title": "Parse Date", "message": "Input table is empty.", "metrics": {}}
            }

        cols_config = config.get("columns", [])
        single_col = config.get("column")
        date_format = config.get("date_format", "auto")
        dayfirst = config.get("dayfirst", False)
        yearfirst = config.get("yearfirst", False)
        errors_mode = config.get("errors", "coerce")
        output_mode = config.get("output_mode", "in_place")
        suffix = config.get("new_column_suffix", "_parsed")
        output_format = config.get("output_format", "datetime64")
        target_date_format = config.get("target_date_format", "%Y-%m-%d")
        extract_parts = config.get("extract_parts", [])

        target_cols = []
        if single_col and single_col in df.columns:
            target_cols.append(single_col)
        for c in cols_config:
            if c in df.columns and c not in target_cols:
                target_cols.append(c)

        if not target_cols:
            # Auto-detect potential date columns
            for c in df.columns:
                c_lower = str(c).lower()
                if any(k in c_lower for k in ["date", "time", "timestamp", "created_at", "updated_at", "dob", "birth"]):
                    target_cols.append(c)

        if not target_cols:
            return {
                "dataframe": df,
                "metrics": {"columns_parsed": 0, "rows_processed": len(df)},
                "output_summary": {"title": "Parse Date", "message": "No date columns specified or detected.", "metrics": {}}
            }

        format_arg = "mixed" if (not date_format or date_format == "auto") else date_format
        cols_summary = []
        total_nat_coerced = 0

        for col in target_cols:
            dest_col = col if output_mode == "in_place" else f"{col}{suffix}"
            col_series = df[col]

            # Execute date parsing
            try:
                parsed_dt = pd.to_datetime(
                    col_series,
                    format=format_arg,
                    dayfirst=dayfirst,
                    yearfirst=yearfirst,
                    errors="coerce" if errors_mode in ["coerce", "ignore"] else "raise"
                )
            except Exception as e:
                if errors_mode == "raise":
                    raise ValueError(f"Failed to parse dates in column '{col}': {str(e)}")
                parsed_dt = pd.Series([pd.NaT] * len(df), index=df.index)

            if errors_mode == "ignore":
                # Keep original values where parsing resulted in NaT and original was not null
                mask_failed = parsed_dt.isna() & col_series.notna()
                out_series = parsed_dt.copy()
                out_series[mask_failed] = col_series[mask_failed]
            else:
                out_series = parsed_dt

            nat_count = int(parsed_dt.isna().sum()) - int(col_series.isna().sum())
            total_nat_coerced += max(0, nat_count)

            # Format representation
            if output_format == "date_only":
                df[dest_col] = parsed_dt.dt.strftime("%Y-%m-%d")
            elif output_format == "iso8601":
                df[dest_col] = parsed_dt.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            elif output_format == "custom" and target_date_format:
                df[dest_col] = parsed_dt.dt.strftime(target_date_format)
            else:
                df[dest_col] = parsed_dt

            # Calendar parts extraction
            if extract_parts and isinstance(extract_parts, list):
                dt_accessor = parsed_dt.dt
                for part in extract_parts:
                    p = str(part).lower()
                    if p == "year":
                        df[f"{dest_col}_year"] = dt_accessor.year
                    elif p == "month":
                        df[f"{dest_col}_month"] = dt_accessor.month
                    elif p == "day":
                        df[f"{dest_col}_day"] = dt_accessor.day
                    elif p == "day_name":
                        df[f"{dest_col}_day_name"] = dt_accessor.day_name()
                    elif p == "day_of_week":
                        df[f"{dest_col}_day_of_week"] = dt_accessor.dayofweek
                    elif p == "hour":
                        df[f"{dest_col}_hour"] = dt_accessor.hour
                    elif p == "quarter":
                        df[f"{dest_col}_quarter"] = dt_accessor.quarter
                    elif p == "is_weekend":
                        df[f"{dest_col}_is_weekend"] = dt_accessor.dayofweek >= 5

            cols_summary.append(dest_col)

        output_summary = {
            "title": "Parse Date",
            "message": f"Successfully parsed {len(cols_summary)} date column(s): {', '.join(cols_summary)}.",
            "metrics": {
                "columns_parsed": len(cols_summary),
                "parsed_columns": cols_summary,
                "unparseable_coerced_nulls": total_nat_coerced,
                "date_format_used": date_format,
                "rows_processed": len(df)
            }
        }

        return {
            "dataframe": df,
            "metrics": output_summary["metrics"],
            "output_summary": output_summary
        }

    def to_code(self, config: Dict[str, Any]) -> str:
        cols = repr(config.get("columns", []))
        fmt = repr(config.get("date_format", "auto"))
        return (
            f"# Recipe: {self.name}\n"
            f"# ID: {self.recipe_id}\n"
            f"# Target Columns: {cols}, Format: {fmt}\n"
            f"# Parses string columns into proper datetime\n"
        )
