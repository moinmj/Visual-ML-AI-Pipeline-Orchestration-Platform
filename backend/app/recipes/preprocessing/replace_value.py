"""
Replace Value Recipe.
Sub-group: Custom Logic
Category: Data Preparation & Transformation (preprocessing)

Replaces specific values, substrings, nulls, or mappings in selected columns.
"""
import re
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
from backend.app.recipes.base.recipe import BaseRecipe, RecipePort


class ReplaceValueRecipe(BaseRecipe):
    recipe_id = "replace_value"
    name = "Replace Value"
    version = "1.0.0"
    category = "preprocessing"
    group = "Data Preparation & Transformation"
    subgroup = "Custom Logic"
    description = "Replaces specific values, substrings, nulls, or value mappings across designated columns."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    inputs = [
        RecipePort(
            id="input",
            label="Input Table",
            type="dataframe",
            required=True,
            max_connections=1,
            description="Dataset containing values to replace"
        )
    ]

    outputs = [
        RecipePort(
            id="dataframe",
            label="Cleaned Table",
            type="dataframe",
            description="Dataset with requested values replaced"
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "columns": {
                    "type": "array",
                    "title": "Target Columns",
                    "items": {"type": "string"},
                    "default": [],
                    "description": "Columns to apply replacement to. If empty, can apply to all matching columns if 'apply_to_all' is true."
                },
                "apply_to_all": {
                    "type": "boolean",
                    "title": "Apply to All Columns",
                    "default": False,
                    "description": "If true, checks and applies replacements across all columns in the dataset."
                },
                "find_value": {
                    "type": "string",
                    "title": "Value to Find",
                    "description": "Specific value or string to search for (e.g. 'N/A', '?', '0', 'Unknown').",
                    "default": ""
                },
                "replace_with": {
                    "type": "string",
                    "title": "Replacement Value",
                    "description": "Value to replace with. Can be text, numbers, or special keywords '__NULL__' or '__EMPTY__'.",
                    "default": ""
                },
                "mapping": {
                    "type": "object",
                    "title": "Value Mapping Dictionary",
                    "description": "Dictionary of multiple find-replace pairs (e.g. {'M': 'Male', 'F': 'Female', 'Y': 'Yes', 'N': 'No'}).",
                    "default": {},
                    "additionalProperties": {"type": "string"}
                },
                "match_type": {
                    "type": "string",
                    "title": "Match Type",
                    "enum": ["exact", "contains", "starts_with", "ends_with", "regex"],
                    "default": "exact",
                    "description": "How values are matched ('exact', 'contains' substring, 'starts_with', 'ends_with', or 'regex')."
                },
                "case_sensitive": {
                    "type": "boolean",
                    "title": "Case Sensitive",
                    "default": True,
                    "description": "Whether string matching should respect letter case."
                },
                "replace_nulls": {
                    "type": "boolean",
                    "title": "Replace Null/NaN Values",
                    "default": False,
                    "description": "If true, replaces null/missing values with 'null_replacement_value'."
                },
                "null_replacement_value": {
                    "type": "string",
                    "title": "Null Replacement Value",
                    "default": "Missing",
                    "description": "Value to replace nulls with when 'replace_nulls' is true."
                },
                "replace_empty_strings": {
                    "type": "boolean",
                    "title": "Replace Empty Strings ('')",
                    "default": False,
                    "description": "If true, replaces blank or whitespace-only strings with a chosen replacement."
                },
                "empty_replacement_value": {
                    "type": "string",
                    "title": "Empty String Replacement",
                    "default": "None",
                    "description": "Value to replace empty strings with."
                },
                "output_mode": {
                    "type": "string",
                    "title": "Output Mode",
                    "enum": ["in_place", "new_column"],
                    "default": "in_place",
                    "description": "Replace in-place or generate new columns with a suffix."
                },
                "new_column_suffix": {
                    "type": "string",
                    "title": "New Column Suffix",
                    "default": "_replaced",
                    "description": "Suffix for new columns if output_mode is 'new_column'."
                }
            },
            "required": []
        }

    def _convert_special_val(self, val: Any) -> Any:
        if val == "__NULL__" or val == "__NAN__":
            return np.nan
        if val == "__EMPTY__":
            return ""
        # Try numeric conversion
        if isinstance(val, str):
            if val.lower() == "true":
                return True
            if val.lower() == "false":
                return False
            if val.isdigit():
                return int(val)
            try:
                return float(val)
            except ValueError:
                pass
        return val

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        df = inputs.get("dataframe") if "dataframe" in inputs else inputs.get("input")
        if df is None:
            df = inputs.get("df")

        if df is None:
            return {
                "dataframe": pd.DataFrame(),
                "metrics": {"total_replacements": 0, "columns_modified": 0},
                "output_summary": {"title": "Replace Value", "message": "No input table provided.", "metrics": {}}
            }

        df = df.copy()
        if len(df) == 0:
            return {
                "dataframe": df,
                "metrics": {"total_replacements": 0, "columns_modified": 0},
                "output_summary": {"title": "Replace Value", "message": "Input table is empty.", "metrics": {}}
            }

        cols_config = config.get("columns", [])
        apply_to_all = config.get("apply_to_all", False)
        find_value = config.get("find_value")
        replace_with = config.get("replace_with", "")
        mapping = config.get("mapping", {})
        match_type = config.get("match_type", "exact")
        case_sensitive = config.get("case_sensitive", True)
        replace_nulls = config.get("replace_nulls", False)
        null_replacement_value = config.get("null_replacement_value", "Missing")
        replace_empty_strings = config.get("replace_empty_strings", False)
        empty_replacement_value = config.get("empty_replacement_value", "None")
        output_mode = config.get("output_mode", "in_place")
        suffix = config.get("new_column_suffix", "_replaced")

        # Determine target columns
        if apply_to_all or not cols_config:
            target_cols = [c for c in df.columns] if apply_to_all else []
        else:
            target_cols = [c for c in cols_config if c in df.columns]

        if not target_cols and cols_config:
            target_cols = [c for c in cols_config if c in df.columns]

        total_replacements = 0
        columns_modified = 0

        # Build combined replacement dictionary for exact matching
        active_mapping: Dict[Any, Any] = {}
        if mapping:
            for k, v in mapping.items():
                active_mapping[self._convert_special_val(k)] = self._convert_special_val(v)
                # Also include string versions if user input is string
                if str(k) not in active_mapping:
                    active_mapping[str(k)] = self._convert_special_val(v)

        if find_value is not None and str(find_value) != "":
            parsed_find = self._convert_special_val(find_value)
            parsed_replace = self._convert_special_val(replace_with)
            active_mapping[parsed_find] = parsed_replace
            if str(find_value) not in active_mapping:
                active_mapping[str(find_value)] = parsed_replace

        for col in target_cols:
            col_series = df[col].copy()
            changes_in_col = 0
            dest_col = col if output_mode == "in_place" else f"{col}{suffix}"

            # 1. Null Replacement
            if replace_nulls:
                null_mask = col_series.isna()
                n_nulls = int(null_mask.sum())
                if n_nulls > 0:
                    col_series[null_mask] = self._convert_special_val(null_replacement_value)
                    changes_in_col += n_nulls

            # 2. Empty String Replacement
            is_str_col = pd.api.types.is_string_dtype(col_series) or col_series.dtype == "object" or col_series.dtype.name in ["str", "string"]
            if replace_empty_strings:
                if is_str_col:
                    empty_mask = col_series.astype(str).str.strip() == ""
                    n_empties = int(empty_mask.sum())
                    if n_empties > 0:
                        col_series[empty_mask] = self._convert_special_val(empty_replacement_value)
                        changes_in_col += n_empties

            # 3. Value or Mapping Replacement
            if active_mapping:
                if match_type == "exact":
                    before = col_series.copy()
                    if not case_sensitive and is_str_col:
                        # Case insensitive exact match mapping
                        lower_map = {str(k).lower(): v for k, v in active_mapping.items()}
                        col_series = col_series.apply(lambda x: lower_map.get(str(x).lower(), x) if pd.notna(x) else x)
                    else:
                        col_series = col_series.replace(active_mapping)
                    diff_mask = (before != col_series) & ~(before.isna() & col_series.isna())
                    changes_in_col += int(diff_mask.sum())

                elif match_type in ["contains", "starts_with", "ends_with", "regex"]:
                    for f_val, r_val in active_mapping.items():
                        f_str = str(f_val)
                        r_str = "" if r_val is None or pd.isna(r_val) else str(r_val)
                        if is_str_col:
                            str_s = col_series.astype(str)
                            if match_type == "contains":
                                if case_sensitive:
                                    mask = str_s.str.contains(re.escape(f_str), regex=True, na=False)
                                    col_series = col_series.str.replace(f_str, r_str, regex=False)
                                else:
                                    mask = str_s.str.contains(re.escape(f_str), case=False, regex=True, na=False)
                                    col_series = col_series.str.replace(re.compile(re.escape(f_str), re.IGNORECASE), r_str, regex=True)
                            elif match_type == "starts_with":
                                pattern = f"^{re.escape(f_str)}"
                                flags = 0 if case_sensitive else re.IGNORECASE
                                mask = str_s.str.contains(pattern, flags=flags, regex=True, na=False)
                                col_series = col_series.str.replace(re.compile(pattern, flags), r_str, regex=True)
                            elif match_type == "ends_with":
                                pattern = f"{re.escape(f_str)}$"
                                flags = 0 if case_sensitive else re.IGNORECASE
                                mask = str_s.str.contains(pattern, flags=flags, regex=True, na=False)
                                col_series = col_series.str.replace(re.compile(pattern, flags), r_str, regex=True)
                            elif match_type == "regex":
                                flags = 0 if case_sensitive else re.IGNORECASE
                                mask = str_s.str.contains(f_str, flags=flags, regex=True, na=False)
                                col_series = col_series.str.replace(re.compile(f_str, flags), r_str, regex=True)
                            changes_in_col += int(mask.sum())

            df[dest_col] = col_series
            if changes_in_col > 0:
                columns_modified += 1
                total_replacements += changes_in_col

        output_summary = {
            "title": "Replace Value",
            "message": f"Replaced {total_replacements:,} value occurrences across {columns_modified} column(s).",
            "metrics": {
                "total_replacements": total_replacements,
                "columns_modified": columns_modified,
                "target_columns": target_cols
            }
        }

        return {
            "dataframe": df,
            "metrics": output_summary["metrics"],
            "output_summary": output_summary
        }

    def to_code(self, config: Dict[str, Any]) -> str:
        columns = repr(config.get("columns", []))
        mapping = repr(config.get("mapping", {}))
        return (
            f"# Recipe: {self.name}\n"
            f"# ID: {self.recipe_id}\n"
            f"# Target Columns: {columns}, Mapping: {mapping}\n"
            f"# Replaces values matching specifications\n"
        )
