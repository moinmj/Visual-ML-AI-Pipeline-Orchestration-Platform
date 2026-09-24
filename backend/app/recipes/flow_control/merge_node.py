import pandas as pd
from typing import Dict, Any, List, Optional
from backend.app.recipes.base.recipe import BaseRecipe, RecipePort


class MergeDatasetsRecipe(BaseRecipe):
    recipe_id = "merge"
    name = "Merge / Union"
    version = "1.0.0"
    category = "flow_control"
    description = "Merges or unions two incoming datasets together (vertical row stacking / Union, or horizontal column concatenation)."
    input_types = ["dataframe", "dataframe"]
    output_types = ["dataframe"]

    inputs = [
        RecipePort(
            id="input_1",
            label="Dataset 1",
            type="dataframe",
            required=True,
            max_connections=1,
            description="First incoming dataset"
        ),
        RecipePort(
            id="input_2",
            label="Dataset 2",
            type="dataframe",
            required=True,
            max_connections=1,
            description="Second incoming dataset"
        )
    ]

    outputs = [
        RecipePort(
            id="output",
            label="Merged Dataset",
            type="dataframe",
            description="Unified merged dataset"
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "title": "Merge Mode",
                    "enum": ["union", "concat_columns"],
                    "default": "union",
                    "description": "'union' stacks rows vertically (appends records); 'concat_columns' places columns side-by-side horizontally."
                },
                "ignore_index": {
                    "type": "boolean",
                    "title": "Reset Row Index",
                    "default": True,
                    "description": "If enabled, creates a fresh continuous integer index."
                }
            }
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        # Collect candidate DataFrames from handle inputs or parent_dataframes
        dfs: List[pd.DataFrame] = []

        if "left_dataframe" in inputs and isinstance(inputs["left_dataframe"], pd.DataFrame):
            dfs.append(inputs["left_dataframe"])
        if "right_dataframe" in inputs and isinstance(inputs["right_dataframe"], pd.DataFrame):
            dfs.append(inputs["right_dataframe"])

        if len(dfs) < 2 and "parent_dataframes" in inputs:
            for p_df in inputs["parent_dataframes"]:
                if isinstance(p_df, pd.DataFrame) and p_df not in dfs:
                    dfs.append(p_df)

        if not dfs:
            if "dataframe" in inputs and isinstance(inputs["dataframe"], pd.DataFrame):
                dfs.append(inputs["dataframe"])

        if len(dfs) < 2:
            raise ValueError(f"Merge / Union node requires at least 2 incoming datasets. Found {len(dfs)}.")

        mode = str(config.get("mode", "union")).lower().strip()
        ignore_index = bool(config.get("ignore_index", True))

        if mode == "concat_columns":
            merged_df = pd.concat(dfs, axis=1)
        else:  # union (rows)
            merged_df = pd.concat(dfs, axis=0, ignore_index=ignore_index)

        summary = {
            "title": f"{mode.upper()} Merge Complete",
            "message": f"Successfully merged {len(dfs)} datasets into {len(merged_df):,} rows and {len(merged_df.columns):,} columns.",
            "total_rows": len(merged_df),
            "total_columns": len(merged_df.columns)
        }

        return {
            "dataframe": merged_df,
            "output": merged_df,
            "output_summary": summary,
            "metrics": {
                "input_datasets_count": len(dfs),
                "merged_rows": len(merged_df),
                "merged_columns": len(merged_df.columns)
            }
        }
