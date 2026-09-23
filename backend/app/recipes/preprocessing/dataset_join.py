import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from backend.app.recipes.base.recipe import BaseRecipe
from backend.app.core.logging import logger


class DatasetJoinRecipe(BaseRecipe):
    recipe_id = "dataset_join"
    name = "Dataset Join / Merge"
    version = "1.1.0"
    category = "preprocessing"
    description = "Enterprise visual multi-dataset join matching Databricks visual data prep (Inner, Left, Right, Full, and Split joins with column selection and inline renaming)."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "join_type": {
                    "type": "string",
                    "title": "Join Type",
                    "enum": ["inner", "left", "right", "outer", "split"],
                    "default": "inner",
                    "description": "'inner' (matched records only); 'left' (all left rows); 'right' (all right rows); 'outer' (full join of all records); 'split' (produces matched, left-only, and right-only partitions)."
                },
                "conditions": {
                    "type": "array",
                    "title": "Join Conditions (Column Pairs)",
                    "items": {
                        "type": "object",
                        "properties": {
                            "left": {"type": "string", "title": "Left Column"},
                            "right": {"type": "string", "title": "Right Column"}
                        },
                        "required": ["left", "right"]
                    },
                    "description": "Pairs of matching columns from Left and Right datasets (supports multiple composite conditions, e.g. Store = Store AND Date = Date)."
                },
                "on": {
                    "type": "string",
                    "title": "Common Key (Single or Comma-Separated)",
                    "description": "Column name present in BOTH datasets (e.g. 'customer_id' or 'Store, Date')."
                },
                "left_on": {
                    "type": "string",
                    "title": "Left Join Key",
                    "description": "Column name(s) in Left dataset (used when column names differ)."
                },
                "right_on": {
                    "type": "string",
                    "title": "Right Join Key",
                    "description": "Column name(s) in Right dataset (used when column names differ)."
                },
                "selected_columns_left": {
                    "type": "array",
                    "items": {"type": "string"},
                    "title": "Choose Columns (Left Table)",
                    "description": "Specific columns to keep from Left table. Leave empty to keep all."
                },
                "selected_columns_right": {
                    "type": "array",
                    "items": {"type": "string"},
                    "title": "Choose Columns (Right Table)",
                    "description": "Specific columns to keep from Right table. Leave empty to keep all."
                },
                "rename_columns_left": {
                    "type": "object",
                    "title": "Rename Columns (Left Table)",
                    "description": "Key-value dictionary of {old_column_name: new_column_name} for Left table."
                },
                "rename_columns_right": {
                    "type": "object",
                    "title": "Rename Columns (Right Table)",
                    "description": "Key-value dictionary of {old_column_name: new_column_name} for Right table."
                },
                "suffixes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "title": "Column Overlap Suffixes",
                    "default": ["_left", "_right"],
                    "description": "Suffixes appended to overlapping non-key column names to avoid collision."
                },
                "indicator": {
                    "type": "boolean",
                    "title": "Add Merge Indicator Column (_merge)",
                    "default": False,
                    "description": "If enabled, appends a '_merge' column ('left_only', 'right_only', 'both')."
                },
                "left_parent_id": {
                    "type": "string",
                    "title": "Left Dataset Node ID",
                    "description": "Explicit node ID of the upstream parent representing Left dataset."
                },
                "right_parent_id": {
                    "type": "string",
                    "title": "Right Dataset Node ID",
                    "description": "Explicit node ID of the upstream parent representing Right dataset."
                }
            },
            "required": ["join_type"]
        }

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        errors = []
        conditions = config.get("conditions")
        on = config.get("on")
        left_on = config.get("left_on")
        right_on = config.get("right_on")

        if not conditions and not on and not (left_on and right_on):
            errors.append("Dataset Join requires join criteria: configure 'conditions' (column pairs), 'on', or both 'left_on' and 'right_on'.")

        if conditions and isinstance(conditions, list):
            for idx, c in enumerate(conditions):
                if not isinstance(c, dict) or not (c.get("left") or c.get("left_column")) or not (c.get("right") or c.get("right_column")):
                    errors.append(f"Join condition #{idx+1} is missing 'left' or 'right' column mapping.")

        join_type = str(config.get("join_type", "inner")).lower()
        if join_type not in ["inner", "left", "right", "outer", "split"]:
            errors.append(f"Invalid join_type '{join_type}'. Must be one of: inner, left, right, outer, split.")

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

        # Make local copies so we do not mutate upstream in-memory DataFrames
        left_df = left_df.copy()
        right_df = right_df.copy()

        # 2. Resolve Conditions & Keys
        conditions = config.get("conditions")
        on = config.get("on")
        left_on = config.get("left_on")
        right_on = config.get("right_on")

        def _parse_keys(k: Union[str, List[str], None]) -> Optional[List[str]]:
            if not k:
                return None
            if isinstance(k, str):
                return [s.strip() for s in k.split(",") if s.strip()]
            if isinstance(k, list):
                return [str(s).strip() for s in k if str(s).strip()]
            return None

        left_keys: List[str] = []
        right_keys: List[str] = []

        if conditions and isinstance(conditions, list):
            for c in conditions:
                if isinstance(c, dict):
                    l_col = c.get("left") or c.get("left_column")
                    r_col = c.get("right") or c.get("right_column")
                    if l_col and r_col:
                        left_keys.append(str(l_col).strip())
                        right_keys.append(str(r_col).strip())
        elif on:
            parsed_on = _parse_keys(on) or []
            left_keys = list(parsed_on)
            right_keys = list(parsed_on)
        else:
            left_keys = _parse_keys(left_on) or []
            right_keys = _parse_keys(right_on) or []

        if not left_keys or not right_keys:
            raise ValueError("Dataset Join: Missing join keys. Please specify 'conditions', 'on', or 'left_on'/'right_on'.")

        if len(left_keys) != len(right_keys):
            raise ValueError(f"Dataset Join: Number of left keys ({len(left_keys)}) does not match right keys ({len(right_keys)}).")

        # Verify key presence before any transformation
        for k in left_keys:
            if k not in left_df.columns:
                raise KeyError(f"Join key '{k}' was not found in Left dataset columns: {list(left_df.columns)}")
        for k in right_keys:
            if k not in right_df.columns:
                raise KeyError(f"Join key '{k}' was not found in Right dataset columns: {list(right_df.columns)}")

        # 3. Column Selection & Renaming (Databricks Visual Data Prep Style)
        rename_left = config.get("rename_columns_left") or {}
        rename_right = config.get("rename_columns_right") or {}
        selected_left = config.get("selected_columns_left")
        selected_right = config.get("selected_columns_right")

        # A. Apply Column Selection on Left (always keep left join keys!)
        if selected_left and isinstance(selected_left, list) and len(selected_left) > 0:
            left_cols_needed = set(selected_left).union(set(left_keys))
            left_df = left_df[[c for c in left_df.columns if c in left_cols_needed]]

        # B. Apply Column Selection on Right (always keep right join keys!)
        if selected_right and isinstance(selected_right, list) and len(selected_right) > 0:
            right_cols_needed = set(selected_right).union(set(right_keys))
            right_df = right_df[[c for c in right_df.columns if c in right_cols_needed]]

        # C. Apply Renaming
        if rename_left and isinstance(rename_left, dict):
            left_df = left_df.rename(columns=rename_left)
            # Update left_keys if any join key was renamed
            left_keys = [rename_left.get(k, k) for k in left_keys]

        if rename_right and isinstance(rename_right, dict):
            right_df = right_df.rename(columns=rename_right)
            # Update right_keys if any join key was renamed
            right_keys = [rename_right.get(k, k) for k in right_keys]

        # 4. Perform the Merge
        join_type = str(config.get("join_type") or "inner").lower()
        suffixes = config.get("suffixes") or ["_left", "_right"]
        if isinstance(suffixes, list) and len(suffixes) == 2:
            suffix_tuple: Tuple[str, str] = (str(suffixes[0]), str(suffixes[1]))
        else:
            suffix_tuple = ("_left", "_right")

        indicator_requested = bool(config.get("indicator", False))
        # If split join or match statistics need indicator, force indicator=True
        use_indicator = True

        how_method = "outer" if join_type in ["split", "outer"] else join_type

        # Check if identical keys on both sides
        is_same_keys = (left_keys == right_keys)

        try:
            if is_same_keys:
                merged_full = pd.merge(
                    left_df,
                    right_df,
                    how=how_method,
                    on=left_keys if len(left_keys) > 1 else left_keys[0],
                    suffixes=suffix_tuple,
                    indicator=use_indicator
                )
            else:
                merged_full = pd.merge(
                    left_df,
                    right_df,
                    how=how_method,
                    left_on=left_keys if len(left_keys) > 1 else left_keys[0],
                    right_on=right_keys if len(right_keys) > 1 else right_keys[0],
                    suffixes=suffix_tuple,
                    indicator=use_indicator
                )
        except Exception as e:
            logger.error(f"pd.merge execution failed: {str(e)}")
            raise ValueError(f"Failed to join datasets with join_type='{join_type}': {str(e)}")

        # 5. Partition Matching (Split Join Support)
        matched_mask = merged_full["_merge"] == "both"
        left_only_mask = merged_full["_merge"] == "left_only"
        right_only_mask = merged_full["_merge"] == "right_only"

        matched_count = int(matched_mask.sum())
        left_unmatched_count = int(left_only_mask.sum())
        right_unmatched_count = int(right_only_mask.sum())

        left_rows = len(left_df)
        right_rows = len(right_df)

        left_match_pct = round((matched_count / max(left_rows, 1)) * 100.0, 1)
        right_match_pct = round((matched_count / max(right_rows, 1)) * 100.0, 1)

        # Build output DataFrames
        matched_df = merged_full[matched_mask].copy()
        left_unmatched_df = merged_full[left_only_mask].copy()
        right_unmatched_df = merged_full[right_only_mask].copy()

        # Resolve primary returned dataframe
        if join_type == "split" or join_type == "inner":
            final_df = matched_df
        elif join_type == "left":
            final_df = merged_full[matched_mask | left_only_mask].copy()
        elif join_type == "right":
            final_df = merged_full[matched_mask | right_only_mask].copy()
        else:  # outer
            final_df = merged_full.copy()

        # Clean _merge indicator if user didn't explicitly request it
        if not indicator_requested:
            if "_merge" in final_df.columns:
                final_df.drop(columns=["_merge"], inplace=True)
            if "_merge" in matched_df.columns:
                matched_df.drop(columns=["_merge"], inplace=True)
            if "_merge" in left_unmatched_df.columns:
                left_unmatched_df.drop(columns=["_merge"], inplace=True)
            if "_merge" in right_unmatched_df.columns:
                right_unmatched_df.drop(columns=["_merge"], inplace=True)

        key_repr = ", ".join(left_keys) if is_same_keys else f"[{', '.join(left_keys)}] = [{', '.join(right_keys)}]"

        join_metrics = {
            "join_type": join_type.upper(),
            "left_rows": left_rows,
            "right_rows": right_rows,
            "merged_rows": len(final_df),
            "matched_rows": matched_count,
            "left_unmatched_rows": left_unmatched_count,
            "right_unmatched_rows": right_unmatched_count,
            "left_columns_count": len(left_df.columns),
            "right_columns_count": len(right_df.columns),
            "merged_columns_count": len(final_df.columns),
            "left_match_rate_pct": left_match_pct,
            "right_match_rate_pct": right_match_pct,
            "join_keys": key_repr,
            "selected_columns_left": list(left_df.columns),
            "selected_columns_right": list(right_df.columns)
        }

        output_summary = {
            "title": f"{join_type.upper()} Join Result",
            "message": f"Successfully merged {left_rows:,} left rows with {right_rows:,} right rows into {len(final_df):,} output rows.",
            "match_rate": f"Matched: {matched_count:,} ({left_match_pct}% Left, {right_match_pct}% Right) | Left Unmatched: {left_unmatched_count:,} | Right Unmatched: {right_unmatched_count:,}"
        }

        return {
            "dataframe": final_df,
            "matched_dataframe": matched_df,
            "left_unmatched_dataframe": left_unmatched_df,
            "right_unmatched_dataframe": right_unmatched_df,
            "metrics": join_metrics,
            "output_summary": output_summary
        }
