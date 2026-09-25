import pandas as pd
from typing import Dict, Any, List, Optional
from backend.app.recipes.base.recipe import BaseRecipe, RecipePort


class CombineRecipe(BaseRecipe):
    recipe_id = "combine"
    name = "Combine"
    version = "1.1.0"
    category = "preprocessing"
    description = "Stacks (unions) or sets (intersect/except) rows from two tables with column alignment controls."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    # Structured multi-port definitions for DAG canvas handles
    inputs = [
        RecipePort(id="left", label="Table A (Primary)", type="dataframe", required=True, max_connections=1, description="Primary incoming dataset"),
        RecipePort(id="right", label="Table B (Secondary)", type="dataframe", required=True, max_connections=1, description="Secondary incoming dataset")
    ]
    outputs = [
        RecipePort(id="combined", label="Combined Output", type="dataframe", description="Resulting dataset after combine operation")
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "title": "Operation Mode",
                    "enum": ["union", "union_distinct", "intersect", "except"],
                    "default": "union",
                    "description": "union (all rows), union_distinct (deduplicated), intersect (matching rows), except (Table A minus Table B)."
                },
                "column_alignment": {
                    "type": "string",
                    "title": "Column Alignment",
                    "enum": ["common_columns", "all_columns"],
                    "default": "common_columns",
                    "description": "common_columns keeps only overlapping columns; all_columns preserves all columns (fills NaNs)."
                }
            },
            "required": ["mode"]
        }

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        errors = []
        mode = str(config.get("mode", "union")).lower().strip()
        if mode not in ["union", "union_distinct", "intersect", "except"]:
            errors.append(f"Invalid mode '{mode}'. Must be one of: 'union', 'union_distinct', 'intersect', 'except'.")

        alignment = str(config.get("column_alignment", "common_columns")).lower().strip()
        if alignment not in ["common_columns", "all_columns"]:
            errors.append(f"Invalid column_alignment '{alignment}'. Must be 'common_columns' or 'all_columns'.")

        return errors

    def _resolve_inputs(self, inputs: Dict[str, Any]) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        left: Optional[pd.DataFrame] = None
        right: Optional[pd.DataFrame] = None

        # 1. Handle-Aware inputs from DAGExecutor
        if "left_dataframe" in inputs and isinstance(inputs["left_dataframe"], pd.DataFrame):
            left = inputs["left_dataframe"]
        elif "left" in inputs and isinstance(inputs["left"], pd.DataFrame):
            left = inputs["left"]
        elif "dataframe" in inputs and isinstance(inputs["dataframe"], pd.DataFrame):
            left = inputs["dataframe"]

        if "right_dataframe" in inputs and isinstance(inputs["right_dataframe"], pd.DataFrame):
            right = inputs["right_dataframe"]
        elif "right" in inputs and isinstance(inputs["right"], pd.DataFrame):
            right = inputs["right"]

        # 2. Ordered parent_dataframes list fallback from DAGExecutor
        if (left is None or right is None) and "parent_dataframes" in inputs:
            parent_dfs = inputs.get("parent_dataframes", [])
            if len(parent_dfs) >= 2:
                if left is None:
                    left = parent_dfs[0]
                if right is None:
                    right = parent_dfs[1]

        # 3. Fallback to parent_outputs
        if (left is None or right is None) and "parent_outputs" in inputs:
            p_outs = inputs.get("parent_outputs", {})
            dfs = [
                v["dataframe"] for v in p_outs.values()
                if isinstance(v, dict) and isinstance(v.get("dataframe"), pd.DataFrame)
            ]
            if len(dfs) >= 2:
                if left is None:
                    left = dfs[0]
                if right is None:
                    right = dfs[1]

        return left, right

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        left, right = self._resolve_inputs(inputs)

        if left is None or right is None:
            raise ValueError(
                "CombineRecipe requires two incoming datasets (Table A and Table B). "
                f"Found Table A={'Present' if left is not None else 'Missing'}, "
                f"Table B={'Present' if right is not None else 'Missing'}. "
                "Please connect two dataset nodes to this Combine processor."
            )

        left = left.copy()
        right = right.copy()

        mode = str(config.get("mode", "union")).lower().strip()
        alignment = str(config.get("column_alignment", "common_columns")).lower().strip()

        # Determine common vs all columns
        common = [c for c in left.columns if c in right.columns]

        if mode in ["union", "union_distinct"]:
            if alignment == "all_columns":
                out = pd.concat([left, right], ignore_index=True)
            else:
                if not common and (len(left.columns) > 0 or len(right.columns) > 0):
                    raise ValueError(
                        f"Tables A and B have 0 common columns under 'common_columns' alignment. "
                        f"Table A columns: {list(left.columns)}, Table B columns: {list(right.columns)}. "
                        "Switch 'column_alignment' to 'all_columns' to union with NaN filling."
                    )
                out = pd.concat([left[common], right[common]], ignore_index=True)

            if mode == "union_distinct":
                out = out.drop_duplicates().reset_index(drop=True)

        elif mode == "intersect":
            if not common:
                # Disjoint schemas have an empty intersection
                out = pd.DataFrame(columns=[])
            else:
                out = left[common].merge(right[common].drop_duplicates(), how="inner").drop_duplicates().reset_index(drop=True)

        else:  # except (Table A minus Table B)
            if not common:
                # If no columns in common, nothing can be subtracted
                out = left.copy()
            else:
                indicator_col = "__combine_merge_indicator__"
                out = left[common].merge(right[common].drop_duplicates(), how="left", indicator=indicator_col)
                out = out[out[indicator_col] == "left_only"].drop(columns=[indicator_col])
                # Preserve all original columns from left if common was a subset
                if len(common) < len(left.columns):
                    out = left.loc[out.index]
                out = out.drop_duplicates().reset_index(drop=True)

        out = out.reset_index(drop=True)
        return {
            "dataframe": out,
            "feature_names": list(out.columns),
            "output_summary": {"row_count": len(out), "columns": list(out.columns)}
        }

    def to_code(self, config: Dict[str, Any]) -> str:
        mode = config.get("mode", "union")
        align = config.get("column_alignment", "common_columns")
        return f"# Combine Table A and Table B (mode='{mode}', alignment='{align}')\n" \
               f"df = pd.concat([left, right], ignore_index=True)"