import pandas as pd
from typing import Dict, Any, List, Optional
from backend.app.recipes.base.recipe import BaseRecipe


class PivotRecipe(BaseRecipe):
    recipe_id = "pivot"
    name = "Pivot"
    version = "1.0.0"
    category = "preprocessing"
    description = "Rotates columns/rows: long-to-wide (pivot) or wide-to-long (unpivot/melt)."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["pivot", "unpivot"], "default": "pivot"},
                "index": {"type": "array", "items": {"type": "string"}, "title": "Row Keys (pivot) / ID Vars (unpivot)"},
                "columns": {"type": "string", "title": "Pivot On Column (pivot only)"},
                "values": {"type": "array", "items": {"type": "string"}, "title": "Values Columns"},
                "agg_func": {"type": "string", "enum": ["sum", "mean", "count", "first"], "default": "sum"}
            },
            "required": ["mode"]
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        df: pd.DataFrame = inputs.get("dataframe")
        if df is None:
            raise ValueError("PivotRecipe expects 'dataframe' in inputs.")
        mode = config.get("mode", "pivot")
        if mode == "pivot":
            out = pd.pivot_table(
                df, index=config.get("index") or None, columns=config.get("columns"),
                values=config.get("values") or None, aggfunc=config.get("agg_func", "sum")
            ).reset_index()
            out.columns = ["_".join([str(c) for c in tup if c]) if isinstance(tup, tuple) else str(tup) for tup in out.columns]
        else:
            out = df.melt(id_vars=config.get("index") or [], value_vars=config.get("values") or None,
                           var_name="variable", value_name="value")
        return {"dataframe": out, "feature_names": list(out.columns), "output_summary": {"row_count": len(out)}}

    def to_code(self, config: Dict[str, Any]) -> str:
        return f"df = pd.pivot_table(df, ...)" if config.get("mode", "pivot") == "pivot" else "df = df.melt(...)"