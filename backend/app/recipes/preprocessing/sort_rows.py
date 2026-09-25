import pandas as pd
from typing import Dict, Any, List, Optional
from backend.app.recipes.base.recipe import BaseRecipe


class SortRecipe(BaseRecipe):
    recipe_id = "sort"
    name = "Sort"
    version = "1.0.0"
    category = "preprocessing"
    description = "Orders rows by one or more columns."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "columns": {"type": "array", "title": "Sort By Columns", "items": {"type": "string"}},
                "ascending": {"type": "boolean", "title": "Ascending", "default": True}
            },
            "required": ["columns"]
        }

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        return ["At least one sort column is required."] if not config.get("columns") else []

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        df: pd.DataFrame = inputs.get("dataframe")
        if df is None:
            raise ValueError("SortRecipe expects 'dataframe' in inputs.")
        cols = [c for c in config.get("columns", []) if c in df.columns]
        if not cols:
            raise ValueError("None of the specified sort columns exist in the dataframe.")
        out = df.sort_values(by=cols, ascending=config.get("ascending", True)).reset_index(drop=True)
        return {"dataframe": out, "feature_names": list(out.columns), "output_summary": {"row_count": len(out)}}

    def to_code(self, config: Dict[str, Any]) -> str:
        return f"df = df.sort_values(by={config.get('columns', [])}, ascending={config.get('ascending', True)}).reset_index(drop=True)"