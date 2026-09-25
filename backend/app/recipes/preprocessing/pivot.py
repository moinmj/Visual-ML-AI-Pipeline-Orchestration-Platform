import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from backend.app.recipes.base.recipe import BaseRecipe

SUPPORTED_AGG_FUNCS = {
    "sum": "sum",
    "mean": "mean",
    "avg": "mean",
    "count": "count",
    "min": "min",
    "max": "max",
    "first": "first",
    "last": "last",
    "median": "median"
}

NUMERIC_ONLY_AGGS = {"mean", "median"}


class PivotRecipe(BaseRecipe):
    recipe_id = "pivot"
    name = "Pivot"
    version = "1.1.0"
    category = "preprocessing"
    description = "Rotates columns and rows: long-to-wide (pivot) or wide-to-long (unpivot/melt) with aggregation and fill controls."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "title": "Operation Mode",
                    "enum": ["pivot", "unpivot"],
                    "default": "pivot",
                    "description": "pivot (long-to-wide by grouping index and pivoting column headers) or unpivot (wide-to-long/melt)."
                },
                "index": {
                    "title": "Row Identifiers (Index / ID Vars)",
                    "description": "Columns to retain as row identifier keys.",
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}}
                    ]
                },
                "columns": {
                    "title": "Pivot Column(s) (pivot mode only)",
                    "description": "Column whose unique values become the new wide column headers.",
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}}
                    ]
                },
                "values": {
                    "title": "Value Column(s)",
                    "description": "Column(s) containing values to aggregate in cells (pivot) or unpivot into values (unpivot).",
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}}
                    ]
                },
                "agg_func": {
                    "type": "string",
                    "title": "Aggregation Function (pivot only)",
                    "enum": ["sum", "mean", "count", "first", "last", "min", "max", "median"],
                    "default": "sum"
                },
                "fill_value": {
                    "title": "Fill Value (pivot only)",
                    "description": "Value to replace missing/null cells in the pivoted table (e.g. 0).",
                    "oneOf": [
                        {"type": "number"},
                        {"type": "string"},
                        {"type": "null"}
                    ],
                    "default": 0
                },
                "var_name": {
                    "type": "string",
                    "title": "Variable Column Name (unpivot only)",
                    "default": "variable",
                    "description": "Name of the column for unpivoted headers."
                },
                "value_name": {
                    "type": "string",
                    "title": "Value Column Name (unpivot only)",
                    "default": "value",
                    "description": "Name of the column for unpivoted values."
                }
            },
            "required": ["mode"]
        }

    def _normalize_col_list(self, val: Any) -> List[str]:
        if val is None:
            return []
        if isinstance(val, str):
            clean = val.strip()
            return [clean] if clean else []
        if isinstance(val, (list, tuple)):
            return [str(c).strip() for c in val if c is not None and str(c).strip()]
        return []

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        errors = []
        mode = str(config.get("mode", "pivot")).lower().strip()
        if mode not in ["pivot", "unpivot"]:
            errors.append(f"Invalid mode '{mode}'. Must be 'pivot' or 'unpivot'.")
            return errors

        if mode == "pivot":
            cols = self._normalize_col_list(config.get("columns"))
            if not cols:
                errors.append("Pivot mode requires 'columns' (the column whose unique values become headers).")

            agg = str(config.get("agg_func", "sum")).lower().strip()
            if agg not in SUPPORTED_AGG_FUNCS:
                valid_aggs = ", ".join(SUPPORTED_AGG_FUNCS.keys())
                errors.append(f"Unsupported agg_func '{agg}'. Supported: {valid_aggs}.")

        return errors

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        df: pd.DataFrame = inputs.get("dataframe")
        if df is None:
            raise ValueError("PivotRecipe expects 'dataframe' in inputs.")

        mode = str(config.get("mode", "pivot")).lower().strip()
        index_cols = self._normalize_col_list(config.get("index"))
        pivot_cols = self._normalize_col_list(config.get("columns"))
        value_cols = self._normalize_col_list(config.get("values"))

        if mode == "pivot":
            if not pivot_cols:
                raise ValueError("PivotRecipe in 'pivot' mode requires at least one column specified in 'columns'.")

            # Validate pivot columns exist
            for c in pivot_cols:
                if c not in df.columns:
                    raise ValueError(f"Pivot column '{c}' does not exist in dataframe. Available: {list(df.columns)}")

            # Validate index columns
            valid_index = [c for c in index_cols if c in df.columns] if index_cols else None
            # Validate value columns
            valid_values = [c for c in value_cols if c in df.columns] if value_cols else None

            raw_agg = str(config.get("agg_func", "sum")).lower().strip()
            aggfunc = SUPPORTED_AGG_FUNCS.get(raw_agg, "sum")
            fill_val = config.get("fill_value", 0)

            # Prevent mean/median on non-numeric value columns
            if valid_values and aggfunc in NUMERIC_ONLY_AGGS:
                for vc in valid_values:
                    if not pd.api.types.is_numeric_dtype(df[vc]):
                        raise ValueError(
                            f"Cannot compute '{raw_agg}' on non-numeric value column '{vc}' ({df[vc].dtype}). "
                            "Please choose a numeric column or use 'first'/'count'."
                        )

            if len(df) == 0:
                out = pd.DataFrame(columns=(valid_index or []) + pivot_cols)
            else:
                try:
                    p_cols = pivot_cols[0] if len(pivot_cols) == 1 else pivot_cols
                    v_cols = valid_values[0] if valid_values and len(valid_values) == 1 else valid_values
                    out = pd.pivot_table(
                        df,
                        index=valid_index,
                        columns=p_cols,
                        values=v_cols,
                        aggfunc=aggfunc,
                        fill_value=fill_val,
                        observed=False
                    ).reset_index()
                except Exception as e:
                    raise ValueError(f"Pivot table execution failed: {str(e)}") from e

                # Clean MultiIndex column headers
                flat_cols = []
                for tup in out.columns:
                    if isinstance(tup, tuple):
                        parts = [str(c).strip() for c in tup if c is not None and str(c).strip()]
                        flat_cols.append("_".join(parts) if parts else "col")
                    else:
                        flat_cols.append(str(tup))
                out.columns = flat_cols

        else:  # unpivot (melt)
            valid_id_vars = [c for c in index_cols if c in df.columns] if index_cols else None
            valid_value_vars = [c for c in value_cols if c in df.columns] if value_cols else None
            var_name = str(config.get("var_name", "variable")).strip() or "variable"
            value_name = str(config.get("value_name", "value")).strip() or "value"

            if len(df) == 0:
                cols = (valid_id_vars or []) + [var_name, value_name]
                out = pd.DataFrame(columns=cols)
            else:
                try:
                    out = df.melt(
                        id_vars=valid_id_vars,
                        value_vars=valid_value_vars,
                        var_name=var_name,
                        value_name=value_name
                    )
                except Exception as e:
                    raise ValueError(f"Unpivot (melt) execution failed: {str(e)}") from e

        out = out.reset_index(drop=True)
        return {
            "dataframe": out,
            "feature_names": list(out.columns),
            "output_summary": {"row_count": len(out), "columns": list(out.columns)}
        }

    def to_code(self, config: Dict[str, Any]) -> str:
        mode = config.get("mode", "pivot")
        if mode == "pivot":
            index = self._normalize_col_list(config.get("index"))
            columns = self._normalize_col_list(config.get("columns"))
            values = self._normalize_col_list(config.get("values"))
            agg = config.get("agg_func", "sum")
            fill = config.get("fill_value", 0)
            return f"df = pd.pivot_table(df, index={repr(index)}, columns={repr(columns)}, " \
                   f"values={repr(values)}, aggfunc='{agg}', fill_value={repr(fill)}).reset_index()"
        else:
            id_vars = self._normalize_col_list(config.get("index"))
            val_vars = self._normalize_col_list(config.get("values"))
            return f"df = df.melt(id_vars={repr(id_vars)}, value_vars={repr(val_vars)}, " \
                   f"var_name='{config.get('var_name', 'variable')}', value_name='{config.get('value_name', 'value')}')"