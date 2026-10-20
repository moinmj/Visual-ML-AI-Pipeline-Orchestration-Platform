import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from backend.app.recipes.base.recipe import BaseRecipe, RecipePort
from backend.app.core.logging import logger


class DatasetJoinRecipe(BaseRecipe):
    recipe_id = "dataset_join"
    name = "Dataset Join / Merge"
    version = "1.2.0"
    category = "preprocessing"
    description = "Enterprise visual multi-dataset join matching Databricks visual data prep (Inner, Left, Right, Full, and Split joins with column selection and inline renaming)."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    # Structured multi-port definitions for DAG canvas handles
    inputs = [
        RecipePort(
            id="left",
            label="Left Table (Primary)",
            type="dataframe",
            required=True,
            max_connections=1,
            description="Primary base table"
        ),
        RecipePort(
            id="right",
            label="Right Table (Lookup)",
            type="dataframe",
            required=True,
            max_connections=1,
            description="Secondary table to join"
        )
    ]
    outputs = [
        RecipePort(
            id="joined",
            label="Joined Output",
            type="dataframe",
            description="Combined dataset after join"
        ),
        RecipePort(
            id="unmatched_left",
            label="Unmatched Left (Split Only)",
            type="dataframe",
            description="Rows from left table that had no match"
        ),
        RecipePort(
            id="unmatched_right",
            label="Unmatched Right (Split Only)",
            type="dataframe",
            description="Rows from right table that had no match"
        )
    ]

    @staticmethod
    def _parse_rename_mapping(raw: Any) -> Dict[str, str]:
        """Robustly parses rename mappings from dict, JSON string, comma-separated string, or list of mappings."""
        if not raw:
            return {}
        if isinstance(raw, dict):
            return {str(k).strip(): str(v).strip() for k, v in raw.items() if str(k).strip() and str(v).strip()}
        if isinstance(raw, str):
            raw = raw.strip()
            if not raw:
                return {}
            import json
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    return {str(k).strip(): str(v).strip() for k, v in parsed.items() if str(k).strip() and str(v).strip()}
                if isinstance(parsed, list):
                    return DatasetJoinRecipe._parse_rename_mapping(parsed)
            except Exception:
                pass
            res = {}
            for pair in raw.split(","):
                if ":" in pair:
                    k, v = pair.split(":", 1)
                    if k.strip() and v.strip():
                        res[k.strip()] = v.strip()
                elif "=" in pair:
                    k, v = pair.split("=", 1)
                    if k.strip() and v.strip():
                        res[k.strip()] = v.strip()
            return res
        if isinstance(raw, list):
            res = {}
            for item in raw:
                if isinstance(item, dict):
                    old_name = item.get("old") or item.get("from") or item.get("source") or item.get("old_name") or item.get("column") or item.get("left")
                    new_name = item.get("new") or item.get("to") or item.get("target") or item.get("new_name") or item.get("renamed") or item.get("right")
                    if old_name and new_name:
                        res[str(old_name).strip()] = str(new_name).strip()
            return res
        return {}

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "join_type": {
                    "type": "string",
                    "title": "Join Type",
                    "input_type": "dropdown",
                    "enum": ["inner", "left", "right", "outer", "split", "cross"],
                    "default": "inner",
                    "description": "'inner' (matched records only); 'left' (all left rows); 'right' (all right rows); 'outer' (full join of all records); 'split' (produces matched, left-only, and right-only partitions); 'cross' (cartesian product of all rows)."
                },
                "conditions": {
                    "type": "array",
                    "title": "Join Conditions",
                    "input_type": "conditions",
                    "items": {
                        "type": "object",
                        "properties": {
                            "left": {
                                "type": "string",
                                "title": "Left Column",
                                "input_type": "column_select"
                            },
                            "right": {
                                "type": "string",
                                "title": "Right Column",
                                "input_type": "column_select"
                            },
                            "operator": {
                                "type": "string",
                                "title": "Operator",
                                "input_type": "dropdown",
                                "enum": ["=", "!=", ">", "<", ">=", "<="],
                                "default": "="
                            }
                        },
                        "required": ["left", "right"]
                    },
                    "default": [{"left": "", "right": "", "operator": "="}],
                    "description": "Pairs of matching columns from Left and Right datasets (supports single or composite keys, e.g. store_id = store_id AND date = date)."
                },
                "selected_columns_left": {
                    "type": "array",
                    "items": {"type": "string"},
                    "title": "Choose Columns (Left Table)",
                    "input_type": "multi_column_select",
                    "description": "Specific columns to keep from Left table. Leave empty to keep all."
                },
                "selected_columns_right": {
                    "type": "array",
                    "items": {"type": "string"},
                    "title": "Choose Columns (Right Table)",
                    "input_type": "multi_column_select",
                    "description": "Specific columns to keep from Right table. Leave empty to keep all."
                },
                "rename_columns_left": {
                    "type": "object",
                    "title": "Rename Columns (Left Table)",
                    "input_type": "key_value",
                    "description": "Key-value dictionary of {old_column_name: new_column_name} for Left table."
                },
                "rename_columns_right": {
                    "type": "object",
                    "title": "Rename Columns (Right Table)",
                    "input_type": "key_value",
                    "description": "Key-value dictionary of {old_column_name: new_column_name} for Right table."
                },
                "suffixes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "title": "Column Overlap Suffixes",
                    "input_type": "tags",
                    "default": ["_left", "_right"],
                    "description": "Suffixes appended to overlapping non-key column names to avoid collision."
                },
                "indicator": {
                    "type": "boolean",
                    "title": "Add Merge Indicator Column (_merge)",
                    "input_type": "checkbox",
                    "default": False,
                    "description": "If enabled, appends a '_merge' column ('left_only', 'right_only', 'both')."
                },
                "save_as_dataset": {
                    "type": "boolean",
                    "title": "Save Output as New Dataset",
                    "input_type": "checkbox",
                    "default": False,
                    "description": "If enabled, automatically persists the merged result to your Datasets library."
                },
                "output_dataset_name": {
                    "type": "string",
                    "title": "New Dataset Name",
                    "input_type": "text",
                    "description": "Custom name for the newly created dataset (e.g. 'Customers_Orders_Joined')."
                },
                "output_dataset_description": {
                    "type": "string",
                    "title": "New Dataset Description",
                    "input_type": "text",
                    "description": "Optional description for the newly created dataset."
                }
            },
            "required": ["join_type"]
        }

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        errors = []
        join_type = str(config.get("join_type", "inner")).lower()
        if join_type not in ["inner", "left", "right", "outer", "split", "cross"]:
            errors.append(f"Invalid join_type '{join_type}'. Must be one of: inner, left, right, outer, split, cross.")

        if join_type != "cross":
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

        return errors

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        # 1. Resolve Left and Right DataFrames from inputs
        left_df: Optional[pd.DataFrame] = None
        right_df: Optional[pd.DataFrame] = None

        parent_outputs = inputs.get("parent_outputs", {})
        left_parent_id = config.get("left_parent_id")
        right_parent_id = config.get("right_parent_id")

        # A. Handle-Aware Inputs from DAGExecutor (target_handle="left" / "right")
        if "left_dataframe" in inputs and isinstance(inputs["left_dataframe"], pd.DataFrame):
            left_df = inputs["left_dataframe"]
        if "right_dataframe" in inputs and isinstance(inputs["right_dataframe"], pd.DataFrame):
            right_df = inputs["right_dataframe"]

        # B. Fallback: Explicit Parent IDs in Config (legacy support)
        if left_df is None and left_parent_id and left_parent_id in parent_outputs:
            left_df = parent_outputs[left_parent_id].get("dataframe")
        if right_df is None and right_parent_id and right_parent_id in parent_outputs:
            right_df = parent_outputs[right_parent_id].get("dataframe")

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

        # 2. Extract Options & Join Type
        join_type = str(config.get("join_type") or "inner").lower()
        suffixes = config.get("suffixes") or ["_left", "_right"]
        if isinstance(suffixes, list) and len(suffixes) == 2:
            suffix_tuple: Tuple[str, str] = (str(suffixes[0]), str(suffixes[1]))
        else:
            suffix_tuple = ("_left", "_right")

        raw_indicator = config.get("indicator", False)
        if isinstance(raw_indicator, str):
            indicator_requested = raw_indicator.strip().lower() in ("true", "1", "yes")
        else:
            indicator_requested = bool(raw_indicator)

        rename_left = self._parse_rename_mapping(config.get("rename_columns_left"))
        rename_right = self._parse_rename_mapping(config.get("rename_columns_right"))
        selected_left = config.get("selected_columns_left")
        selected_right = config.get("selected_columns_right")

        if join_type == "cross":
            if selected_left and isinstance(selected_left, list) and len(selected_left) > 0:
                left_df = left_df[[c for c in left_df.columns if c in selected_left]]
            if selected_right and isinstance(selected_right, list) and len(selected_right) > 0:
                right_df = right_df[[c for c in right_df.columns if c in selected_right]]

            if rename_left:
                left_df = left_df.rename(columns=rename_left)
            if rename_right:
                right_df = right_df.rename(columns=rename_right)

            try:
                merged_full = pd.merge(
                    left_df,
                    right_df,
                    how="cross",
                    suffixes=suffix_tuple
                )
            except Exception as e:
                logger.error(f"pd.merge cross failed: {str(e)}")
                raise ValueError(f"Failed to join datasets with join_type='cross': {str(e)}")

            final_df = merged_full.copy()
            matched_df = merged_full.copy()
            left_unmatched_df = pd.DataFrame(columns=left_df.columns)
            right_unmatched_df = pd.DataFrame(columns=right_df.columns)
            matched_count = len(final_df)
            left_unmatched_count = 0
            right_unmatched_count = 0
            left_rows = len(left_df)
            right_rows = len(right_df)
            left_match_pct = 100.0 if left_rows > 0 else 0.0
            right_match_pct = 100.0 if right_rows > 0 else 0.0
            key_repr = "CROSS (Cartesian Product)"
        else:
            # 2b. Resolve Conditions & Keys
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
            # A. Apply Column Selection on Left (always keep left join keys!)
            if selected_left and isinstance(selected_left, list) and len(selected_left) > 0:
                left_cols_needed = set(selected_left).union(set(left_keys))
                left_df = left_df[[c for c in left_df.columns if c in left_cols_needed]]

            # B. Apply Column Selection on Right (always keep right join keys!)
            if selected_right and isinstance(selected_right, list) and len(selected_right) > 0:
                right_cols_needed = set(selected_right).union(set(right_keys))
                right_df = right_df[[c for c in right_df.columns if c in right_cols_needed]]

            # C. Apply Renaming
            if rename_left:
                left_df = left_df.rename(columns=rename_left)
                # Update left_keys if any join key was renamed
                left_keys = [rename_left.get(k, k) for k in left_keys]

            if rename_right:
                right_df = right_df.rename(columns=rename_right)
                # Update right_keys if any join key was renamed
                right_keys = [rename_right.get(k, k) for k in right_keys]

            # 4. Perform the Merge
            use_indicator = True
            how_method = "outer" if join_type in ["split", "outer"] else join_type

            # Align mismatched join key data types & temporal granularities automatically
            left_join_keys = list(left_keys)
            right_join_keys = list(right_keys)
            temp_cols_to_drop: List[str] = []

            for i, (l_k, r_k) in enumerate(zip(left_keys, right_keys)):
                l_s = left_df[l_k]
                r_s = right_df[r_k]
                l_is_dt = pd.api.types.is_datetime64_any_dtype(l_s)
                r_is_dt = pd.api.types.is_datetime64_any_dtype(r_s)

                # Case 1: Temporal join (one or both are datetime, or string date vs datetime)
                if l_is_dt or r_is_dt:
                    l_dt = l_s if l_is_dt else pd.to_datetime(l_s, format="mixed", dayfirst=True, errors="coerce")
                    r_dt = r_s if r_is_dt else pd.to_datetime(r_s, format="mixed", dayfirst=True, errors="coerce")

                    if pd.api.types.is_datetime64_any_dtype(l_dt) and pd.api.types.is_datetime64_any_dtype(r_dt):
                        l_has_time = bool((l_dt.dt.hour != 0).any() or (l_dt.dt.minute != 0).any() or (l_dt.dt.second != 0).any())
                        r_has_time = bool((r_dt.dt.hour != 0).any() or (r_dt.dt.minute != 0).any() or (r_dt.dt.second != 0).any())

                        # If one has time component and the other does not (date-only), match at day-level
                        if l_has_time != r_has_time:
                            temp_l = f"__temp_join_{l_k}_{i}"
                            temp_r = f"__temp_join_{r_k}_{i}"
                            left_df[temp_l] = l_dt.dt.normalize()
                            right_df[temp_r] = r_dt.dt.normalize()
                            left_join_keys[i] = temp_l
                            right_join_keys[i] = temp_r
                            temp_cols_to_drop.extend([temp_l, temp_r])
                        else:
                            if not l_is_dt:
                                temp_l = f"__temp_join_{l_k}_{i}"
                                left_df[temp_l] = l_dt
                                left_join_keys[i] = temp_l
                                temp_cols_to_drop.append(temp_l)
                            if not r_is_dt:
                                temp_r = f"__temp_join_{r_k}_{i}"
                                right_df[temp_r] = r_dt
                                right_join_keys[i] = temp_r
                                temp_cols_to_drop.append(temp_r)
                    else:
                        # Fallback to string-based match if datetime conversion fails
                        temp_l = f"__temp_join_{l_k}_{i}"
                        temp_r = f"__temp_join_{r_k}_{i}"
                        left_df[temp_l] = l_s.astype(str)
                        right_df[temp_r] = r_s.astype(str)
                        left_join_keys[i] = temp_l
                        right_join_keys[i] = temp_r
                        temp_cols_to_drop.extend([temp_l, temp_r])

                # Case 2: Numeric vs String
                elif pd.api.types.is_numeric_dtype(l_s) != pd.api.types.is_numeric_dtype(r_s):
                    try:
                        if pd.api.types.is_numeric_dtype(l_s):
                            temp_r = f"__temp_join_{r_k}_{i}"
                            right_df[temp_r] = pd.to_numeric(r_s, errors="coerce")
                            right_join_keys[i] = temp_r
                            temp_cols_to_drop.append(temp_r)
                        else:
                            temp_l = f"__temp_join_{l_k}_{i}"
                            left_df[temp_l] = pd.to_numeric(l_s, errors="coerce")
                            left_join_keys[i] = temp_l
                            temp_cols_to_drop.append(temp_l)
                    except Exception:
                        temp_l = f"__temp_join_{l_k}_{i}"
                        temp_r = f"__temp_join_{r_k}_{i}"
                        left_df[temp_l] = l_s.astype(str)
                        right_df[temp_r] = r_s.astype(str)
                        left_join_keys[i] = temp_l
                        right_join_keys[i] = temp_r
                        temp_cols_to_drop.extend([temp_l, temp_r])

            # Check if identical keys on both sides
            is_same_keys = (left_join_keys == right_join_keys)

            try:
                if is_same_keys:
                    merged_full = pd.merge(
                        left_df,
                        right_df,
                        how=how_method,
                        on=left_join_keys if len(left_join_keys) > 1 else left_join_keys[0],
                        suffixes=suffix_tuple,
                        indicator=use_indicator
                    )
                else:
                    merged_full = pd.merge(
                        left_df,
                        right_df,
                        how=how_method,
                        left_on=left_join_keys if len(left_join_keys) > 1 else left_join_keys[0],
                        right_on=right_join_keys if len(right_join_keys) > 1 else right_join_keys[0],
                        suffixes=suffix_tuple,
                        indicator=use_indicator
                    )
            except Exception as e:
                logger.error(f"pd.merge execution failed: {str(e)}")
                raise ValueError(f"Failed to join datasets with join_type='{join_type}': {str(e)}")

            if temp_cols_to_drop:
                drop_targets = [c for c in merged_full.columns if any(c == tc or c.startswith(f"{tc}_") for tc in temp_cols_to_drop)]
                merged_full = merged_full.drop(columns=drop_targets, errors="ignore")
                left_df = left_df.drop(columns=[c for c in temp_cols_to_drop if c in left_df.columns], errors="ignore")
                right_df = right_df.drop(columns=[c for c in temp_cols_to_drop if c in right_df.columns], errors="ignore")

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
                final_df = final_df.drop(columns=["_merge"], errors="ignore")
                matched_df = matched_df.drop(columns=["_merge"], errors="ignore")
                left_unmatched_df = left_unmatched_df.drop(columns=["_merge"], errors="ignore")
                right_unmatched_df = right_unmatched_df.drop(columns=["_merge"], errors="ignore")

            key_repr = ", ".join(left_keys) if is_same_keys else f"[{', '.join(left_keys)}] = [{', '.join(right_keys)}]"

        # 6. Optional: Persist Merged Dataset into Datasets Library
        raw_save = config.get("save_as_dataset", False)
        save_as_dataset = (raw_save.strip().lower() in ("true", "1", "yes")) if isinstance(raw_save, str) else bool(raw_save)
        saved_dataset_info = None

        # The tenant comes from the workflow run (injected by the executor), never
        # from a default: without it the joined dataset can't be placed in the
        # right ml-workflow/t<tenant_id>/ folder, so saving is skipped.
        join_tenant_id = config.get("tenant_id")
        if save_as_dataset and len(final_df) > 0 and join_tenant_id is None:
            logger.error("save_as_dataset requested but no tenant_id in node config; joined dataset was NOT saved.")

        if save_as_dataset and len(final_df) > 0 and join_tenant_id is not None:
            try:
                join_tenant_id = int(join_tenant_id)
                import uuid
                import time
                import asyncio
                import concurrent.futures
                from backend.app.infrastructure.storage.storage_manager import storage_manager
                from backend.app.infrastructure.database.session import AsyncSessionLocal
                from backend.app.datasets.models import Dataset
                from backend.app.profiling.profiler import DataProfiler

                new_ds_id = str(uuid.uuid4())
                custom_name = str(config.get("output_dataset_name") or "").strip()
                new_ds_name = custom_name if custom_name else f"Joined_Dataset_{int(time.time())}"
                if not new_ds_name.lower().endswith(".csv"):
                    file_name = f"{new_ds_name}.csv"
                else:
                    file_name = new_ds_name
                    new_ds_name = new_ds_name[:-4]

                rel_storage_path = f"datasets/{new_ds_id}_{file_name}"
                csv_bytes = final_df.to_csv(index=False).encode("utf-8")
                file_size_bytes = len(csv_bytes)

                async def _upload():
                    return await storage_manager.save_file(csv_bytes, rel_storage_path, join_tenant_id)

                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                            storage_path = pool.submit(lambda: asyncio.run(_upload())).result()
                    else:
                        storage_path = loop.run_until_complete(_upload())
                except Exception:
                    storage_path = asyncio.run(_upload())

                profile_data = DataProfiler.profile_dataframe(final_df)

                async def _save_record():
                    async with AsyncSessionLocal() as session:
                        ds_rec = Dataset(
                            id=new_ds_id,
                            tenant_id=join_tenant_id,
                            name=new_ds_name,
                            description=config.get("output_dataset_description") or f"Automatically generated via dataset join on {key_repr}",
                            file_name=file_name,
                            file_format="csv",
                            file_size_bytes=file_size_bytes,
                            storage_path=storage_path,
                            row_count=len(final_df),
                            column_count=len(final_df.columns),
                            quality_score=float(profile_data.get("quality_score", 100.0)),
                            profile=profile_data
                        )
                        session.add(ds_rec)
                        await session.commit()

                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                            pool.submit(lambda: asyncio.run(_save_record())).result()
                    else:
                        loop.run_until_complete(_save_record())
                except Exception:
                    asyncio.run(_save_record())

                saved_dataset_info = {
                    "id": new_ds_id,
                    "name": new_ds_name,
                    "file_name": file_name,
                    "row_count": len(final_df),
                    "column_count": len(final_df.columns)
                }
                logger.info(f"Successfully saved joined dataset as '{new_ds_name}' ({new_ds_id}) in Datasets library.")
            except Exception as e:
                logger.exception(f"Failed to auto-save joined dataset: {e}")

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
        if saved_dataset_info:
            join_metrics["saved_dataset"] = saved_dataset_info

        output_summary = {
            "title": f"{join_type.upper()} Join Result",
            "message": f"Successfully merged {left_rows:,} left rows with {right_rows:,} right rows into {len(final_df):,} output rows.",
            "match_rate": f"Matched: {matched_count:,} ({left_match_pct}% Left, {right_match_pct}% Right) | Left Unmatched: {left_unmatched_count:,} | Right Unmatched: {right_unmatched_count:,}"
        }
        if saved_dataset_info:
            output_summary["saved_dataset"] = f"Saved into Datasets library as '{saved_dataset_info['name']}' (ID: {saved_dataset_info['id']})."

        return {
            "dataframe": final_df,
            "matched_dataframe": matched_df,
            "left_unmatched_dataframe": left_unmatched_df,
            "right_unmatched_dataframe": right_unmatched_df,
            "metrics": join_metrics,
            "output_summary": output_summary
        }