"""
Select (Rename / Reorder Columns) Recipe.
Sub-group: Combine/Structure
Category: Data Preparation & Transformation (preprocessing)

Renames and reorders dataset columns in one unified step with strict or flexible ordering modes.
"""
from typing import Dict, Any, List, Optional, Union
import pandas as pd
from backend.app.recipes.base.recipe import BaseRecipe, RecipePort


class SelectRenameReorderRecipe(BaseRecipe):
    recipe_id = "select_rename_reorder"
    name = "Select (Rename / Reorder Columns)"
    version = "1.0.0"
    category = "preprocessing"
    group = "Data Preparation & Transformation"
    subgroup = "Combine/Structure"
    description = "Renames and reorders columns in one step with flexible ordering, column pruning, and schema restructuring."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    inputs = [
        RecipePort(
            id="input",
            label="Input Table",
            type="dataframe",
            required=True,
            max_connections=1,
            description="Incoming tabular dataset to rename or reorder"
        )
    ]

    outputs = [
        RecipePort(
            id="dataframe",
            label="Restructured Table",
            type="dataframe",
            description="Transformed dataset with renamed and reordered column structure"
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "rename_mapping": {
                    "type": "object",
                    "title": "Rename Mapping (Old -> New)",
                    "description": "Key-value dictionary of existing column names to their new column names (e.g. {'old_col': 'new_col'}).",
                    "default": {},
                    "additionalProperties": {"type": "string"}
                },
                "column_order": {
                    "type": "array",
                    "title": "Desired Column Order",
                    "items": {"type": "string"},
                    "default": [],
                    "description": "Explicit order of columns in the final table. Columns can be referenced by their new or existing names."
                },
                "mode": {
                    "type": "string",
                    "title": "Column Inclusion Mode",
                    "enum": ["keep_all", "strict"],
                    "default": "keep_all",
                    "description": "'keep_all' places reordered columns first and retains any remaining unlisted columns; 'strict' retains only the columns listed in column_order."
                },
                "drop_columns": {
                    "type": "array",
                    "title": "Explicit Columns to Drop",
                    "items": {"type": "string"},
                    "default": [],
                    "description": "Columns to explicitly prune from the final dataset."
                }
            },
            "required": []
        }

    def _parse_rename_mapping(self, raw: Any) -> Dict[str, str]:
        if not raw:
            return {}
        if isinstance(raw, dict):
            return {str(k).strip(): str(v).strip() for k, v in raw.items() if str(k).strip() and str(v).strip()}
        if isinstance(raw, list):
            res = {}
            for item in raw:
                if isinstance(item, dict):
                    old_name = item.get("old") or item.get("from") or item.get("source") or item.get("old_name") or item.get("column")
                    new_name = item.get("new") or item.get("to") or item.get("target") or item.get("new_name")
                    if old_name and new_name:
                        res[str(old_name).strip()] = str(new_name).strip()
            return res
        return {}

    def _parse_list(self, raw: Any) -> List[str]:
        if not raw:
            return []
        if isinstance(raw, str):
            return [c.strip() for c in raw.split(",") if c.strip()]
        if isinstance(raw, (list, tuple)):
            res = []
            for item in raw:
                if item and str(item).strip():
                    parts = str(item).split(",")
                    res.extend([p.strip() for p in parts if p.strip()])
            return res
        return []

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        df = inputs.get("dataframe")
        if df is None:
            if context and isinstance(context, dict) and "dataframe" in context:
                df = context["dataframe"]
            else:
                raise ValueError("SelectRenameReorderRecipe expects 'dataframe' in inputs.")

        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        df_out = df.copy()
        orig_columns = list(df.columns)
        orig_count = len(orig_columns)

        # 1. Parse configs
        rename_map = self._parse_rename_mapping(config.get("rename_mapping"))
        desired_order = self._parse_list(config.get("column_order"))
        mode = str(config.get("mode", "keep_all")).lower().strip()
        drop_cols = set(self._parse_list(config.get("drop_columns")))

        # 2. Apply rename mapping
        active_renames = {}
        for old_col, new_col in rename_map.items():
            if old_col in df_out.columns:
                active_renames[old_col] = new_col

        if active_renames:
            df_out = df_out.rename(columns=active_renames)

        # 3. Apply explicit column drops
        valid_drops = [c for c in drop_cols if c in df_out.columns]
        if valid_drops:
            df_out = df_out.drop(columns=valid_drops)

        # 4. Resolve column ordering
        curr_cols = list(df_out.columns)
        if desired_order:
            # Map old names in desired_order to new names if they were renamed
            mapped_order = [active_renames.get(c, c) for c in desired_order]
            ordered_present = [c for c in mapped_order if c in curr_cols]
            
            # De-duplicate while preserving order
            seen = set()
            clean_ordered = []
            for c in ordered_present:
                if c not in seen:
                    clean_ordered.append(c)
                    seen.add(c)

            if mode == "strict":
                if clean_ordered:
                    df_out = df_out[clean_ordered]
            else:  # "keep_all"
                remaining = [c for c in curr_cols if c not in seen]
                df_out = df_out[clean_ordered + remaining]

        final_columns = list(df_out.columns)

        # Output Summary & Metrics
        rename_diff = [
            {"original": old, "renamed": new}
            for old, new in active_renames.items()
        ]

        output_summary = {
            "title": "Select (Rename / Reorder Columns)",
            "message": f"Successfully renamed {len(active_renames)} columns and organized {len(final_columns)} columns in {mode} mode.",
            "metrics": {
                "original_columns_count": orig_count,
                "final_columns_count": len(final_columns),
                "renamed_columns_count": len(active_renames),
                "dropped_columns_count": len(valid_drops),
                "mode": mode
            },
            "rename_details": rename_diff,
            "final_column_order": final_columns[:15]
        }

        return {
            "dataframe": df_out,
            "metrics": output_summary["metrics"],
            "output_summary": output_summary
        }

    def to_code(self, config: Dict[str, Any]) -> str:
        rename_map = self._parse_rename_mapping(config.get("rename_mapping"))
        order = self._parse_list(config.get("column_order"))
        mode = config.get("mode", "keep_all")
        drop_cols = self._parse_list(config.get("drop_columns"))

        lines = ["# Select (Rename / Reorder Columns)"]
        if rename_map:
            lines.append(f"rename_map = {rename_map}")
            lines.append("df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})")
        if drop_cols:
            lines.append(f"drop_cols = {drop_cols}")
            lines.append("df = df.drop(columns=[c for c in drop_cols if c in df.columns])")
        if order:
            lines.append(f"desired_order = {order}")
            if mode == "strict":
                lines.append("df = df[[c for c in desired_order if c in df.columns]]")
            else:
                lines.append("ordered = [c for c in desired_order if c in df.columns]")
                lines.append("remaining = [c for c in df.columns if c not in ordered]")
                lines.append("df = df[ordered + remaining]")
        return "\n".join(lines)
