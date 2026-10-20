"""
Regex Replace / Extract Recipe.
Sub-group: Custom Logic
Category: Data Preparation & Transformation (preprocessing)

Replaces, extracts, or flags text in dataset columns using regular expressions.
"""
import re
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
from backend.app.recipes.base.recipe import BaseRecipe, RecipePort


class RegexReplaceExtractRecipe(BaseRecipe):
    recipe_id = "regex_replace_extract"
    name = "Regex Replace / Extract"
    version = "1.0.0"
    category = "preprocessing"
    group = "Data Preparation & Transformation"
    subgroup = "Custom Logic"
    description = "Replaces or extracts text in columns using regular expressions with support for named capture groups, backreferences, and pattern matching."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    inputs = [
        RecipePort(
            id="input",
            label="Input Table",
            type="dataframe",
            required=True,
            max_connections=1,
            description="Dataset containing text columns to transform via regex"
        )
    ]

    outputs = [
        RecipePort(
            id="dataframe",
            label="Transformed Table",
            type="dataframe",
            description="Dataset with regex replacements or extracted columns"
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "source_column": {
                    "type": "string",
                    "title": "Source Column",
                    "input_type": "column_select",
                    "description": "Column to extract from or perform regex replacement on."
                },
                "columns": {
                    "type": "array",
                    "title": "Multiple Source Columns (for Replace mode)",
                    "input_type": "multi_column_select",
                    "items": {"type": "string"},
                    "default": [],
                    "description": "Optional list of columns to apply regex replacement across simultaneously."
                },
                "mode": {
                    "type": "string",
                    "title": "Operation Mode",
                    "input_type": "dropdown",
                    "enum": ["replace", "extract", "extract_all", "match_flag"],
                    "default": "replace",
                    "description": "'replace' substitutes matched patterns; 'extract' captures regex groups into new columns; 'extract_all' collects all matches joined by delimiter; 'match_flag' adds boolean match column."
                },
                "regex_pattern": {
                    "type": "string",
                    "title": "Regular Expression Pattern",
                    "input_type": "text",
                    "description": "Regex pattern. E.g. '\\d{4}' (digits), '(?P<user>[^@]+)@(?P<domain>[^\\s]+)' (email groups), '\\$|,|\\s' (currency symbols).",
                    "default": ""
                },
                "replacement": {
                    "type": "string",
                    "title": "Replacement String (Replace mode)",
                    "input_type": "text",
                    "description": "Replacement string or backreference (e.g. '', '_', '\\1'). Used when mode is 'replace'.",
                    "default": ""
                },
                "target_column": {
                    "type": "string",
                    "title": "Target / Output Column",
                    "input_type": "text",
                    "description": "Name of output column. In 'replace' mode, if blank, replaces in-place. In 'extract' mode with single match, name of extracted column."
                },
                "target_columns": {
                    "type": "array",
                    "title": "Extracted Column Names",
                    "input_type": "tags",
                    "items": {"type": "string"},
                    "default": [],
                    "description": "Explicit list of column names for multiple regex capture groups in 'extract' mode. Overrides auto-detected group names."
                },
                "delimiter": {
                    "type": "string",
                    "title": "Extract All Delimiter",
                    "input_type": "text",
                    "default": ", ",
                    "description": "Delimiter used to join multiple matches in 'extract_all' mode."
                },
                "ignore_case": {
                    "type": "boolean",
                    "title": "Ignore Case (re.IGNORECASE)",
                    "input_type": "checkbox",
                    "default": True,
                    "description": "Case-insensitive matching."
                },
                "multiline": {
                    "type": "boolean",
                    "title": "Multiline Matching (re.MULTILINE)",
                    "input_type": "checkbox",
                    "default": False,
                    "description": "Makes '^' and '$' match the beginning and end of each line."
                },
                "dotall": {
                    "type": "boolean",
                    "title": "Dotall (re.DOTALL)",
                    "input_type": "checkbox",
                    "default": False,
                    "description": "Makes '.' match any character including newlines."
                }
            },
            "required": ["regex_pattern"]
        }

    def _compile_flags(self, ignore_case: bool, multiline: bool, dotall: bool) -> int:
        flags = 0
        if ignore_case:
            flags |= re.IGNORECASE
        if multiline:
            flags |= re.MULTILINE
        if dotall:
            flags |= re.DOTALL
        return flags

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        df = inputs.get("dataframe") if "dataframe" in inputs else inputs.get("input")
        if df is None:
            df = inputs.get("df")

        if df is None:
            return {
                "dataframe": pd.DataFrame(),
                "metrics": {"rows_processed": 0, "columns_created": 0},
                "output_summary": {"title": "Regex Replace / Extract", "message": "No input table provided.", "metrics": {}}
            }

        df = df.copy()
        if len(df) == 0:
            return {
                "dataframe": df,
                "metrics": {"rows_processed": 0, "columns_created": 0},
                "output_summary": {"title": "Regex Replace / Extract", "message": "Input table is empty.", "metrics": {}}
            }

        source_col = config.get("source_column")
        multiple_cols = config.get("columns", [])
        mode = config.get("mode", "replace")
        pattern_str = config.get("regex_pattern", "")
        replacement = config.get("replacement", "")
        target_col = config.get("target_column")
        target_cols = config.get("target_columns", [])
        delimiter = config.get("delimiter", ", ")
        ignore_case = config.get("ignore_case", True)
        multiline = config.get("multiline", False)
        dotall = config.get("dotall", False)

        if not pattern_str:
            return {
                "dataframe": df,
                "metrics": {"rows_processed": len(df), "columns_created": 0},
                "output_summary": {"title": "Regex Replace / Extract", "message": "No regex pattern provided.", "metrics": {}}
            }

        flags = self._compile_flags(ignore_case, multiline, dotall)
        compiled_regex = re.compile(pattern_str, flags=flags)

        # Resolve columns to process
        cols_to_process = []
        if source_col and source_col in df.columns:
            cols_to_process.append(source_col)
        for c in multiple_cols:
            if c in df.columns and c not in cols_to_process:
                cols_to_process.append(c)

        if not cols_to_process:
            # Fallback to first string/object column
            string_cols = [c for c in df.columns if df[c].dtype == "object" or df[c].dtype.name == "string"]
            if string_cols:
                cols_to_process.append(string_cols[0])

        if not cols_to_process:
            return {
                "dataframe": df,
                "metrics": {"rows_processed": len(df), "columns_created": 0},
                "output_summary": {"title": "Regex Replace / Extract", "message": "No matching string columns found.", "metrics": {}}
            }

        columns_created = []
        replacements_count = 0

        # MODE 1: REPLACE
        if mode == "replace":
            for col in cols_to_process:
                dest = target_col if (target_col and len(cols_to_process) == 1) else col
                s = df[col].astype(str)
                # Count matches
                mask = s.str.contains(pattern_str, flags=flags, regex=True, na=False)
                replacements_count += int(mask.sum())
                df[dest] = s.str.replace(pattern_str, replacement, flags=flags, regex=True)
                if dest not in cols_to_process:
                    columns_created.append(dest)

        # MODE 2: EXTRACT
        elif mode == "extract":
            primary_col = cols_to_process[0]
            s = df[primary_col].fillna("").astype(str)

            # Check if regex contains named groups
            groupindex = compiled_regex.groupindex
            num_groups = compiled_regex.groups

            if num_groups == 0:
                # No capture groups, extract entire match
                dest = target_col or f"{primary_col}_extracted"
                df[dest] = s.str.extract(f"({pattern_str})", flags=flags, expand=False)
                columns_created.append(dest)
            elif groupindex:
                # Named groups
                extracted_df = s.str.extract(pattern_str, flags=flags, expand=True)
                for grp_name in extracted_df.columns:
                    col_dest = grp_name
                    df[col_dest] = extracted_df[grp_name]
                    columns_created.append(col_dest)
            else:
                # Unnamed groups
                extracted_df = s.str.extract(pattern_str, flags=flags, expand=True)
                for idx in range(num_groups):
                    if idx < len(target_cols) and target_cols[idx]:
                        col_dest = target_cols[idx]
                    else:
                        col_dest = f"{primary_col}_group_{idx + 1}"
                    df[col_dest] = extracted_df.iloc[:, idx]
                    columns_created.append(col_dest)

        # MODE 3: EXTRACT ALL
        elif mode == "extract_all":
            primary_col = cols_to_process[0]
            dest = target_col or f"{primary_col}_all_matches"
            s = df[primary_col].fillna("").astype(str)

            def find_all_matches(val):
                matches = compiled_regex.findall(val)
                if not matches:
                    return None
                if isinstance(matches[0], tuple):
                    # Tuple from groups: join each tuple then join all
                    return delimiter.join("-".join(m) for m in matches)
                return delimiter.join(str(m) for m in matches)

            df[dest] = s.apply(find_all_matches)
            columns_created.append(dest)

        # MODE 4: MATCH FLAG
        elif mode == "match_flag":
            primary_col = cols_to_process[0]
            dest = target_col or f"{primary_col}_matches"
            s = df[primary_col].astype(str)
            df[dest] = s.str.contains(pattern_str, flags=flags, regex=True, na=False)
            columns_created.append(dest)

        output_summary = {
            "title": "Regex Replace / Extract",
            "message": f"Processed regex '{pattern_str}' in mode '{mode}'. Created/updated: {', '.join(columns_created) if columns_created else cols_to_process[0]}.",
            "metrics": {
                "mode": mode,
                "pattern": pattern_str,
                "replacements_count": replacements_count,
                "columns_created": columns_created,
                "rows_processed": len(df)
            }
        }

        return {
            "dataframe": df,
            "metrics": output_summary["metrics"],
            "output_summary": output_summary
        }

    def to_code(self, config: Dict[str, Any]) -> str:
        mode = repr(config.get("mode", "replace"))
        pattern = repr(config.get("regex_pattern", ""))
        return (
            f"# Recipe: {self.name}\n"
            f"# ID: {self.recipe_id}\n"
            f"# Mode: {mode}, Pattern: {pattern}\n"
            f"# Executes regular expression transformation\n"
        )
