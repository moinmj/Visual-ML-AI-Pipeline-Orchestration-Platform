import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from backend.app.recipes.base.recipe import BaseRecipe

SUPPORTED_FUNCS = {
    "sum": "sum",
    "mean": "mean",
    "avg": "mean",
    "average": "mean",
    "count": "count",
    "min": "min",
    "max": "max",
    "median": "median",
    "std": "std",
    "nunique": "nunique",
    "unique": "nunique",
    "count_distinct": "nunique"
}

NUMERIC_ONLY_FUNCS = {"mean", "median", "std"}


class AggregateRecipe(BaseRecipe):
    recipe_id = "aggregate"
    name = "Aggregate"
    version = "1.1.0"
    category = "preprocessing"
    description = "Summarizes rows via group-by + aggregation functions (sum, mean, min, max, count, nunique, median, std)."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "group_by": {
                    "type": "array",
                    "title": "Group By Columns",
                    "items": {"type": "string"},
                    "description": "Columns to group by (e.g. ['Store', 'Dept'] or 'Category')."
                },
                "aggregations": {
                    "type": "array",
                    "title": "Aggregations",
                    "items": {
                        "type": "object",
                        "properties": {
                            "column": {"type": "string", "title": "Column"},
                            "func": {
                                "type": "string",
                                "title": "Function",
                                "enum": ["sum", "mean", "count", "min", "max", "median", "std", "nunique"]
                            }
                        },
                        "required": ["column", "func"]
                    },
                    "description": "List of column aggregation definitions."
                }
            },
            "required": ["group_by", "aggregations"]
        }

    def _normalize_group_by(self, config: Dict[str, Any]) -> List[str]:
        raw = config.get("group_by")
        if raw is None:
            return []
        if isinstance(raw, str):
            clean = raw.strip()
            return [clean] if clean else []
        if isinstance(raw, (list, tuple)):
            return [str(c).strip() for c in raw if c is not None and str(c).strip()]
        return []

    def _normalize_aggregations(self, config: Dict[str, Any]) -> List[Dict[str, str]]:
        raw = config.get("aggregations")
        if not raw:
            return []
        norm_list = []
        if isinstance(raw, dict):
            # Format: {"col1": "sum", "col2": ["mean", "max"]}
            for col, funcs in raw.items():
                if isinstance(funcs, (list, tuple)):
                    for f in funcs:
                        norm_list.append({"column": str(col).strip(), "func": str(f).lower().strip()})
                else:
                    norm_list.append({"column": str(col).strip(), "func": str(funcs).lower().strip()})
        elif isinstance(raw, (list, tuple)):
            for item in raw:
                if isinstance(item, dict):
                    col = str(item.get("column", "")).strip()
                    func = str(item.get("func", "")).lower().strip()
                    if col and func:
                        norm_list.append({"column": col, "func": func})
        return norm_list

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        errors = []
        gb = self._normalize_group_by(config)
        if not gb:
            errors.append("At least one group-by column is required in 'group_by'.")

        aggs = self._normalize_aggregations(config)
        if not aggs:
            errors.append("At least one aggregation definition is required in 'aggregations'.")
        else:
            for idx, a in enumerate(aggs):
                func = a.get("func", "")
                if func not in SUPPORTED_FUNCS:
                    valid_opts = ", ".join(["sum", "mean", "count", "min", "max", "median", "std", "nunique"])
                    errors.append(f"Aggregation #{idx+1} specifies unsupported function '{func}'. Supported: {valid_opts}.")

        return errors

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        df: pd.DataFrame = inputs.get("dataframe")
        if df is None:
            raise ValueError("AggregateRecipe expects 'dataframe' in inputs.")

        group_cols = self._normalize_group_by(config)
        if not group_cols:
            raise ValueError("AggregateRecipe requires at least one 'group_by' column.")

        missing_group = [c for c in group_cols if c in df.columns]
        if not missing_group:
            raise ValueError(f"None of the specified group_by columns {group_cols} exist in dataframe. Available: {list(df.columns)}")
        valid_group_cols = missing_group

        raw_aggs = self._normalize_aggregations(config)
        if not raw_aggs:
            raise ValueError("AggregateRecipe requires at least one valid aggregation in 'aggregations'.")

        agg_map: Dict[str, List[str]] = {}
        for a in raw_aggs:
            col, raw_func = a["column"], a["func"]
            if col not in df.columns:
                continue
            canonical_func = SUPPORTED_FUNCS.get(raw_func, raw_func)

            # Prevent non-numeric type crashes for statistical aggregates
            if canonical_func in NUMERIC_ONLY_FUNCS and not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(
                    f"Cannot compute '{raw_func}' on non-numeric column '{col}' ({df[col].dtype}). "
                    "Use numeric columns for mean/median/std or use 'count'/'nunique'."
                )

            agg_map.setdefault(col, []).append(canonical_func)

        if not agg_map:
            agg_cols = [a['column'] for a in raw_aggs]
            raise ValueError(f"None of the aggregation columns {agg_cols} exist in the dataframe. Available: {list(df.columns)}")

        if len(df) == 0:
            # Construct empty output schema
            cols = list(valid_group_cols)
            for c, funcs in agg_map.items():
                for f in funcs:
                    cols.append(f"{c}_{f}")
            out = pd.DataFrame(columns=cols)
        else:
            try:
                out = df.groupby(valid_group_cols, observed=False).agg(agg_map).reset_index()
            except Exception as e:
                raise ValueError(f"Aggregation execution failed: {str(e)}") from e

            # Flatten MultiIndex tuple columns cleanly into 'col_func'
            flat_cols = []
            for tup in out.columns:
                if isinstance(tup, tuple):
                    parts = [str(c) for c in tup if c is not None and str(c).strip()]
                    flat_cols.append("_".join(parts))
                else:
                    flat_cols.append(str(tup))
            out.columns = flat_cols

        return {
            "dataframe": out,
            "feature_names": list(out.columns),
            "output_summary": {"row_count": len(out), "columns": list(out.columns)}
        }

    def to_code(self, config: Dict[str, Any]) -> str:
        gb = self._normalize_group_by(config)
        aggs = self._normalize_aggregations(config)
        agg_map = {}
        for a in aggs:
            agg_map.setdefault(a["column"], []).append(a["func"])
        return f"df = df.groupby({repr(gb)}).agg({repr(agg_map)}).reset_index()"