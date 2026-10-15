"""
Timezone Converter Recipe.
Sub-group: Domain-Specific Cleaning
Category: Data Preparation & Transformation (preprocessing)

Converts timestamp columns between time zones with support for naive timestamp
localization, Daylight Saving Time (DST) shifts, and ISO-8601 / wall-clock output formatting.
"""
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
from backend.app.recipes.base.recipe import BaseRecipe, RecipePort


COMMON_TIMEZONES = [
    "UTC",
    "America/New_York",
    "America/Chicago",
    "America/Denver",
    "America/Los_Angeles",
    "Europe/London",
    "Europe/Paris",
    "Europe/Berlin",
    "Asia/Tokyo",
    "Asia/Hong_Kong",
    "Asia/Singapore",
    "Asia/Dubai",
    "Asia/Kolkata",
    "Australia/Sydney",
    "Pacific/Auckland"
]


class TimezoneConverterRecipe(BaseRecipe):
    recipe_id = "timezone_converter"
    name = "Timezone Converter"
    version = "1.0.0"
    category = "preprocessing"
    group = "Data Preparation & Transformation"
    subgroup = "Domain-Specific Cleaning"
    description = "Converts timestamp columns between time zones with automatic DST handling, naive timestamp localization, and ISO-8601 formatting."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    inputs = [
        RecipePort(
            id="input",
            label="Input Table",
            type="dataframe",
            required=True,
            max_connections=1,
            description="Dataset containing timestamp columns to localize or convert"
        )
    ]

    outputs = [
        RecipePort(
            id="dataframe",
            label="Converted Table",
            type="dataframe",
            description="Dataset with converted timestamps in target time zone"
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "columns": {
                    "type": "array",
                    "title": "Timestamp Columns to Convert",
                    "items": {"type": "string"},
                    "default": [],
                    "description": "List of timestamp or datetime columns to convert."
                },
                "column": {
                    "type": "string",
                    "title": "Single Timestamp Column",
                    "default": "",
                    "description": "Alternative single column to convert."
                },
                "source_timezone": {
                    "type": "string",
                    "title": "Source Timezone",
                    "default": "UTC",
                    "description": "Timezone of incoming timestamps if naive (e.g. 'UTC', 'America/New_York', 'Europe/London', 'Asia/Kolkata') or 'auto'."
                },
                "target_timezone": {
                    "type": "string",
                    "title": "Target Timezone",
                    "default": "UTC",
                    "description": "Desired destination timezone (e.g. 'America/New_York', 'UTC', 'Asia/Tokyo', 'Europe/Paris')."
                },
                "naive_handling": {
                    "type": "string",
                    "title": "Naive Timestamp Handling",
                    "enum": ["assume_source_tz", "raise_error", "drop_na"],
                    "default": "assume_source_tz",
                    "description": "How to handle timestamps without timezone metadata: 'assume_source_tz' localizes with source timezone; 'raise_error' throws; 'drop_na' sets to null."
                },
                "output_mode": {
                    "type": "string",
                    "title": "Output Column Mode",
                    "enum": ["in_place", "new_column"],
                    "default": "in_place",
                    "description": "Whether to replace timestamps in-place or generate new columns with a suffix."
                },
                "new_column_suffix": {
                    "type": "string",
                    "title": "New Column Suffix",
                    "default": "_converted",
                    "description": "Suffix for new column when output_mode is 'new_column'."
                },
                "output_format": {
                    "type": "string",
                    "title": "Output Format",
                    "enum": ["datetime64", "iso8601", "naive_local_wallclock", "custom"],
                    "default": "datetime64",
                    "description": "'datetime64' keeps timezone-aware pandas datetime; 'iso8601' produces ISO strings with offset; 'naive_local_wallclock' strips timezone offset leaving local wall-clock time."
                },
                "target_date_format": {
                    "type": "string",
                    "title": "Custom Format String",
                    "default": "%Y-%m-%d %H:%M:%S",
                    "description": "Format string if output_format is 'custom'."
                }
            },
            "required": ["target_timezone"]
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        df = inputs.get("dataframe") if "dataframe" in inputs else inputs.get("input")
        if df is None:
            df = inputs.get("df")

        if df is None:
            return {
                "dataframe": pd.DataFrame(),
                "metrics": {"columns_converted": 0, "rows_processed": 0},
                "output_summary": {"title": "Timezone Converter", "message": "No input table provided.", "metrics": {}}
            }

        df = df.copy()
        if len(df) == 0:
            return {
                "dataframe": df,
                "metrics": {"columns_converted": 0, "rows_processed": 0},
                "output_summary": {"title": "Timezone Converter", "message": "Input table is empty.", "metrics": {}}
            }

        cols_config = config.get("columns", [])
        single_col = config.get("column")
        source_tz = config.get("source_timezone", "UTC")
        target_tz = config.get("target_timezone", "UTC")
        naive_handling = config.get("naive_handling", "assume_source_tz")
        output_mode = config.get("output_mode", "in_place")
        suffix = config.get("new_column_suffix", "_converted")
        output_format = config.get("output_format", "datetime64")
        custom_fmt = config.get("target_date_format", "%Y-%m-%d %H:%M:%S")

        # Resolve target columns
        target_cols = []
        if single_col and single_col in df.columns:
            target_cols.append(single_col)
        for c in cols_config:
            if c in df.columns and c not in target_cols:
                target_cols.append(c)

        if not target_cols:
            # Auto-detect datetime columns
            for c in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[c]):
                    target_cols.append(c)

        if not target_cols:
            return {
                "dataframe": df,
                "metrics": {"columns_converted": 0, "rows_processed": len(df)},
                "output_summary": {"title": "Timezone Converter", "message": "No timestamp columns found to convert.", "metrics": {}}
            }

        columns_modified = []

        for col in target_cols:
            dest_col = col if output_mode == "in_place" else f"{col}{suffix}"
            s = df[col]

            # Ensure datetime dtype
            if not pd.api.types.is_datetime64_any_dtype(s):
                s = pd.to_datetime(s, errors="coerce")

            # Check timezone awareness
            tz_info = getattr(s.dt, "tz", None)

            if tz_info is None:
                # Naive datetime
                if naive_handling == "raise_error":
                    raise ValueError(f"Column '{col}' contains naive timestamps without timezone information.")
                elif naive_handling == "drop_na":
                    s_converted = pd.Series([pd.NaT] * len(df), index=df.index)
                else:
                    # Assume source timezone
                    active_src_tz = "UTC" if (not source_tz or source_tz == "auto") else source_tz
                    s_localized = s.dt.tz_localize(active_src_tz, ambiguous="NaT", nonexistent="NaT")
                    s_converted = s_localized.dt.tz_convert(target_tz)
            else:
                # Already timezone-aware, convert directly
                s_converted = s.dt.tz_convert(target_tz)

            # Apply output formatting
            if output_format == "iso8601":
                df[dest_col] = s_converted.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
            elif output_format == "naive_local_wallclock":
                df[dest_col] = s_converted.dt.tz_localize(None)
            elif output_format == "custom":
                df[dest_col] = s_converted.dt.strftime(custom_fmt)
            else:
                df[dest_col] = s_converted

            columns_modified.append(dest_col)

        output_summary = {
            "title": "Timezone Converter",
            "message": f"Converted {len(columns_modified)} column(s) to timezone '{target_tz}'.",
            "metrics": {
                "columns_converted": len(columns_modified),
                "target_timezone": target_tz,
                "source_timezone": source_tz,
                "rows_processed": len(df)
            }
        }

        return {
            "dataframe": df,
            "metrics": output_summary["metrics"],
            "output_summary": output_summary
        }

    def to_code(self, config: Dict[str, Any]) -> str:
        tz = repr(config.get("target_timezone", "UTC"))
        cols = repr(config.get("columns", []))
        return (
            f"# Recipe: {self.name}\n"
            f"# ID: {self.recipe_id}\n"
            f"# Target Timezone: {tz}, Columns: {cols}\n"
            f"# Converts timestamp columns between timezones\n"
        )
