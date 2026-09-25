import pandas as pd
from typing import Dict, Any, List, Optional
from backend.app.recipes.base.recipe import BaseRecipe


class AggregateRecipe(BaseRecipe):
    recipe_id = "aggregate"
    name = "Aggregate"
    version = "1.0.0"
    category = "preprocessing"
    description = "Summarizes rows via group-by + aggregation functions."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "group_by": {"type": "array", "title": "Group By Columns", "items": {"type": "string"}},
                "aggregations": {
                    "type": "array", "title": "Aggregations",
                    "items": {"type": "object", "properties": {
                        "column": {"type": "string"},
                        "func": {"type": "string", "enum": ["sum", "mean", "count", "min", "max", "median", "std", "nunique"]}
                    }},
                    "default": []
                }
            },
            "required": ["group_by", "aggregations"]
        }

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        errors = []
        if not config.get("group_by"):
            errors.append("At least one group-by column is required.")
        if not config.get("aggregations"):
            errors.append("At least one aggregation is required.")
        return errors

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        df: pd.DataFrame = inputs.get("dataframe")
        if df is None:
            raise ValueError("AggregateRecipe expects 'dataframe' in inputs.")
        group_cols = [c for c in config.get("group_by", []) if c in df.columns]
        agg_map: Dict[str, list] = {}
        for a in config.get("aggregations", []):
            col, func = a.get("column"), a.get("func")
            if col in df.columns and func:
                agg_map.setdefault(col, []).append(func)
        if not group_cols or not agg_map:
            raise ValueError("Invalid group_by/aggregations against this dataframe's columns.")
        out = df.groupby(group_cols).agg(agg_map).reset_index()
        out.columns = ["_".join([c for c in tup if c]) if isinstance(tup, tuple) else tup for tup in out.columns]
        return {"dataframe": out, "feature_names": list(out.columns), "output_summary": {"row_count": len(out)}}

    def to_code(self, config: Dict[str, Any]) -> str:
        return f"df = df.groupby({config.get('group_by', [])}).agg(...).reset_index()"