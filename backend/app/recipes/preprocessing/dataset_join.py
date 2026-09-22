import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from backend.app.recipes.base.recipe import BaseRecipe
from backend.app.core.logging import logger


class DatasetJoinRecipe(BaseRecipe):
    recipe_id = "dataset_join"
    name = "Dataset Join / Merge"
    version = "1.0.0"
    category = "preprocessing"
    description = "Merges two tabular datasets using SQL-like joins (Inner, Left, Right, Outer) on key columns."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "join_type": {
                    "type": "string",
                    "title": "Join Type",
                    "enum": ["inner", "left", "right", "outer"],
                    "default": "inner",
                    "description": "'inner' keeps only matching keys; 'left' keeps all left rows; 'right' keeps all right rows; 'outer' keeps all rows from both datasets."
                },
                "on": {
                    "type": "string",
                    "title": "Join Key (Common Column)",
                    "description": "Column name present in BOTH datasets to join on (e.g. 'customer_id' or 'store_id')."
                },
                "left_on": {
                    "type": "string",
                    "title": "Left Join Key",
                    "description": "Column name in the Left dataset (use when column names differ between datasets)."
                },
                "right_on": {
                    "type": "string",
                    "title": "Right Join Key",
                    "description": "Column name in the Right dataset (use when column names differ between datasets)."
                },
                "suffixes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "title": "Column Overlap Suffixes",
                    "default": ["_left", "_right"],
                    "description": "Suffixes to append to overlapping non-key column names to prevent name collision."
                },
                "indicator": {
                    "type": "boolean",
                    "title": "Add Merge Indicator Column",
                    "default": False,
                    "description": "If enabled, appends a '_merge' column indicating whether each row came from 'left_only', 'right_only', or 'both'."
                },
                "left_parent_id": {
                    "type": "string",
                    "title": "Left Dataset Node ID",
                    "description": "Explicit node ID of the upstream parent representing the Left dataset."
                },
                "right_parent_id": {
                    "type": "string",
                    "title": "Right Dataset Node ID",
                    "description": "Explicit node ID of the upstream parent representing the Right dataset."
                }
            },
            "required": ["join_type"]
        }

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        errors = []
        on = config.get("on")
        left_on = config.get("left_on")
        right_on = config.get("right_on")

        if not on and not (left_on and right_on):
            errors.append("Dataset Join requires either a common 'on' key or both 'left_on' and 'right_on' keys.")

        join_type = config.get("join_type", "inner")
        if join_type not in ["inner", "left", "right", "outer"]:
            errors.append(f"Invalid join_type '{join_type}'. Must be one of: inner, left, right, outer.")

        return errors

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        # 1. Resolve Left and Right DataFrames from inputs
        left_df: Optional[pd.DataFrame] = None
        right_df: Optional[pd.DataFrame] = None

        parent_outputs = inputs.get("parent_outputs", {})
        left_parent_id = config.get("left_parent_id")
        right_parent_id = config.get("right_parent_id")

        # A. Explicit Parent IDs in Config
        if left_parent_id and left_parent_id in parent_outputs:
            left_df = parent_outputs[left_parent_id].get("dataframe")
        if right_parent_id and right_parent_id in parent_outputs:
            right_df = parent_outputs[right_parent_id].get("dataframe")

        # B. Handle-Aware Inputs from DAGExecutor
        if left_df is None and "left_dataframe" in inputs and isinstance(inputs["left_dataframe"], pd.DataFrame):
            left_df = inputs["left_dataframe"]
        if right_df is None and "right_dataframe" in inputs and isinstance(inputs["right_dataframe"], pd.DataFrame):
            right_df = inputs["right_dataframe"]

        # C. Ordered Parent DataFrames List
        if (left_df is None or right_df is None) and "parent_dataframes" in inputs:
            parent_dfs = inputs["parent_dataframes"]
            if len(parent_dfs) >= 2:
                if left_df is None:
                    left_df = parent_dfs[0]
                if right_df is None:
                    right_df = parent_dfs[1]

        # Validation: Verify two datasets are present
        if left_df is None or right_df is None:
            raise ValueError(
                "Dataset Join processor requires two incoming datasets (Left and Right). "
                f"Found left_df={'Present' if left_df is not None else 'Missing'}, "
                f"right_df={'Present' if right_df is not None else 'Missing'}. "
                "Please connect two dataset nodes to this Join processor."
            )

        # 2. Resolve Join Parameters
        join_type = str(config.get("join_type") or "inner").lower()
        on = config.get("on")
        left_on = config.get("left_on")
        right_on = config.get("right_on")
        suffixes = config.get("suffixes") or ["_left", "_right"]
        if isinstance(suffixes, list) and len(suffixes) == 2:
            suffix_tuple: Tuple[str, str] = (str(suffixes[0]), str(suffixes[1]))
        else:
            suffix_tuple = ("_left", "_right")

        indicator = bool(config.get("indicator", False))

        # Helper to convert comma-separated string to list if multiple keys provided
        def _parse_keys(k: Union[str, List[str], None]) -> Optional[Union[str, List[str]]]:
            if not k:
                return None
            if isinstance(k, str) and "," in k:
                return [s.strip() for s in k.split(",") if s.strip()]
            return k

        on_keys = _parse_keys(on)
        left_keys = _parse_keys(left_on)
        right_keys = _parse_keys(right_on)

        # If only 'on' is provided, both left and right use it
        if on_keys and not left_keys and not right_keys:
            # Check presence in both
            keys_to_check = on_keys if isinstance(on_keys, list) else [on_keys]
            for k in keys_to_check:
                if k not in left_df.columns:
                    raise KeyError(f"Join key '{k}' was not found in Left dataset columns: {list(left_df.columns)}")
                if k not in right_df.columns:
                    raise KeyError(f"Join key '{k}' was not found in Right dataset columns: {list(right_df.columns)}")
        else:
            # Check left_keys in left_df
            if not left_keys:
                raise ValueError("Dataset Join: 'left_on' must be specified if 'on' is not provided.")
            l_check = left_keys if isinstance(left_keys, list) else [left_keys]
            for k in l_check:
                if k not in left_df.columns:
                    raise KeyError(f"Left join key '{k}' was not found in Left dataset columns: {list(left_df.columns)}")

            # Check right_keys in right_df
            if not right_keys:
                raise ValueError("Dataset Join: 'right_on' must be specified if 'on' is not provided.")
            r_check = right_keys if isinstance(right_keys, list) else [right_keys]
            for k in r_check:
                if k not in right_df.columns:
                    raise KeyError(f"Right join key '{k}' was not found in Right dataset columns: {list(right_df.columns)}")

        # 3. Perform the Merge
        try:
            if on_keys:
                merged_df = pd.merge(
                    left_df,
                    right_df,
                    how=join_type,
                    on=on_keys,
                    suffixes=suffix_tuple,
                    indicator=indicator
                )
            else:
                merged_df = pd.merge(
                    left_df,
                    right_df,
                    how=join_type,
                    left_on=left_keys,
                    right_on=right_keys,
                    suffixes=suffix_tuple,
                    indicator=indicator
                )
        except Exception as e:
            logger.error(f"pd.merge execution failed: {str(e)}")
            raise ValueError(f"Failed to join datasets with join_type='{join_type}': {str(e)}")

        # 4. Compute Match Statistics & Diagnostics
        left_rows = len(left_df)
        right_rows = len(right_df)
        merged_rows = len(merged_df)

        # Match rate calculation
        left_key_name = (on_keys if isinstance(on_keys, str) else on_keys[0]) if on_keys else (left_keys if isinstance(left_keys, str) else left_keys[0])
        right_key_name = (on_keys if isinstance(on_keys, str) else on_keys[0]) if on_keys else (right_keys if isinstance(right_keys, str) else right_keys[0])

        try:
            left_unique = set(left_df[left_key_name].dropna().unique())
            right_unique = set(right_df[right_key_name].dropna().unique())
            common_keys = left_unique.intersection(right_unique)
            left_match_pct = round((len(common_keys) / max(len(left_unique), 1)) * 100.0, 1)
            right_match_pct = round((len(common_keys) / max(len(right_unique), 1)) * 100.0, 1)
        except Exception:
            left_match_pct = 100.0
            right_match_pct = 100.0

        join_metrics = {
            "join_type": join_type.upper(),
            "left_rows": left_rows,
            "right_rows": right_rows,
            "merged_rows": merged_rows,
            "left_columns_count": len(left_df.columns),
            "right_columns_count": len(right_df.columns),
            "merged_columns_count": len(merged_df.columns),
            "left_match_rate_pct": left_match_pct,
            "right_match_rate_pct": right_match_pct,
            "join_keys": left_key_name if left_key_name == right_key_name else f"{left_key_name} = {right_key_name}"
        }

        output_summary = {
            "title": f"{join_type.upper()} Join Result",
            "message": f"Successfully joined {left_rows:,} left rows with {right_rows:,} right rows into {merged_rows:,} merged rows ({len(merged_df.columns)} total columns).",
            "match_rate": f"Left Match Rate: {left_match_pct}%, Right Match Rate: {right_match_pct}%"
        }

        return {
            "dataframe": merged_df,
            "metrics": join_metrics,
            "output_summary": output_summary
        }
