import pandas as pd
from typing import Dict, Any, List, Optional, Union
from backend.app.recipes.base.recipe import BaseRecipe


class SortRecipe(BaseRecipe):
    recipe_id = "sort"
    name = "Sort"
    version = "1.1.0"
    category = "preprocessing"
    description = "Orders rows by one or more columns with ascending/descending and nulls placement controls."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "columns": {
                    "type": "array",
                    "title": "Sort By Columns",
                    "items": {"type": "string"},
                    "description": "List of columns (or single column name) to order by."
                },
                "ascending": {
                    "title": "Ascending",
                    "default": True,
                    "description": "True for ascending, False for descending (or a list of booleans per column).",
                    "oneOf": [
                        {"type": "boolean"},
                        {"type": "array", "items": {"type": "boolean"}},
                        {"type": "string", "enum": ["asc", "desc", "true", "false"]}
                    ]
                },
                "na_position": {
                    "type": "string",
                    "title": "Missing Values (Nulls) Position",
                    "enum": ["last", "first"],
                    "default": "last",
                    "description": "Where to place NaN / Null values in the sorted output."
                }
            },
            "required": ["columns"]
        }

    def _normalize_columns(self, config: Dict[str, Any]) -> List[str]:
        raw_cols = config.get("columns")
        if raw_cols is None:
            return []
        if isinstance(raw_cols, str):
            clean = raw_cols.strip()
            return [clean] if clean else []
        if isinstance(raw_cols, (list, tuple)):
            return [str(c).strip() for c in raw_cols if c is not None and str(c).strip()]
        return []

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        errors = []
        cols = self._normalize_columns(config)
        if not cols:
            errors.append("At least one sort column is required in 'columns'.")

        na_pos = str(config.get("na_position", "last")).lower().strip()
        if na_pos not in ["last", "first"]:
            errors.append(f"Invalid na_position '{na_pos}'. Must be 'first' or 'last'.")

        return errors

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        df: pd.DataFrame = inputs.get("dataframe")
        if df is None:
            raise ValueError("SortRecipe expects 'dataframe' in inputs.")

        req_cols = self._normalize_columns(config)
        if not req_cols:
            raise ValueError("SortRecipe requires at least one sort column specified in 'columns'.")

        valid_cols = [c for c in req_cols if c in df.columns]
        if not valid_cols:
            raise ValueError(f"None of the specified sort columns {req_cols} exist in the dataframe. Available columns: {list(df.columns)}")

        # Resolve ascending parameter (boolean, string, or list of booleans)
        raw_asc = config.get("ascending", True)
        if isinstance(raw_asc, bool):
            asc: Union[bool, List[bool]] = raw_asc
        elif isinstance(raw_asc, str):
            asc = raw_asc.lower().strip() in ["true", "asc", "ascending", "1"]
        elif isinstance(raw_asc, (list, tuple)):
            asc_list = [
                x if isinstance(x, bool) else (str(x).lower().strip() in ["true", "asc", "ascending", "1"])
                for x in raw_asc
            ]
            if len(asc_list) == len(valid_cols):
                asc = asc_list
            else:
                asc = asc_list[0] if asc_list else True
        else:
            asc = True

        na_position = str(config.get("na_position", "last")).lower().strip()
        if na_position not in ["first", "last"]:
            na_position = "last"

        if len(df) == 0:
            out = df.copy()
        else:
            out = df.sort_values(by=valid_cols, ascending=asc, na_position=na_position).reset_index(drop=True)

        return {
            "dataframe": out,
            "feature_names": list(out.columns),
            "output_summary": {"row_count": len(out), "columns": list(out.columns)}
        }

    def to_code(self, config: Dict[str, Any]) -> str:
        cols = self._normalize_columns(config)
        asc = config.get("ascending", True)
        na_pos = config.get("na_position", "last")
        return f"df = df.sort_values(by={repr(cols)}, ascending={repr(asc)}, na_position={repr(na_pos)}).reset_index(drop=True)"