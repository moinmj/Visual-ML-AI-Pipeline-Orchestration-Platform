import pandas as pd
from typing import Dict, Any, List, Optional
from backend.app.recipes.base.recipe import BaseRecipe


class ColumnSelectorRecipe(BaseRecipe):
    recipe_id = "column_selector"
    name = "Column Selector & Drop"
    version = "1.0.0"
    category = "preprocessing"
    description = "Selects, keeps, or drops specific columns from a tabular dataset to prune unwanted features."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "title": "Selection Mode",
                    "enum": ["keep", "drop"],
                    "default": "keep",
                    "description": "'keep' retains only the specified columns; 'drop' removes the specified columns and retains all others."
                },
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "title": "Target Columns",
                    "description": "Columns to retain (if mode='keep') or eliminate (if mode='drop')."
                }
            },
            "required": ["mode"]
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        df: pd.DataFrame = inputs.get("dataframe")
        if df is None:
            if context and isinstance(context, dict) and "dataframe" in context:
                df = context["dataframe"]
            else:
                raise ValueError("ColumnSelectorRecipe expects 'dataframe' in inputs.")

        df_out = df.copy()
        mode = config.get("mode", "keep")
        cols_cfg = config.get("columns", [])

        # Parse columns list from array or comma-separated string
        if isinstance(cols_cfg, str):
            target_cols = [c.strip() for c in cols_cfg.split(",") if c.strip()]
        elif isinstance(cols_cfg, (list, tuple)):
            target_cols = [str(c).strip() for c in cols_cfg if str(c).strip()]
        else:
            target_cols = []

        if mode == "keep":
            valid_cols = [c for c in target_cols if c in df_out.columns]
            if valid_cols:
                df_out = df_out[valid_cols]
        elif mode == "drop":
            drop_cols = [c for c in target_cols if c in df_out.columns]
            if drop_cols:
                df_out = df_out.drop(columns=drop_cols)

        return {"dataframe": df_out}

    def to_code(self, config: Dict[str, Any]) -> str:
        mode = config.get("mode", "keep")
        cols = config.get("columns", [])
        if isinstance(cols, str):
            cols = [c.strip() for c in cols.split(",") if c.strip()]
        if mode == "drop":
            return f"# Drop Columns\ncols_to_drop = {cols}\ndf = df.drop(columns=[c for c in cols_to_drop if c in df.columns])"
        return f"# Retain Specified Columns\ncols_to_keep = {cols}\ndf = df[[c for c in cols_to_keep if c in df.columns]]"
