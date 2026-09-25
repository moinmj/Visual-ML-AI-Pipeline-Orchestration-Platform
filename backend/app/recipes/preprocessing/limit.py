import pandas as pd
from typing import Dict, Any, List, Optional
from backend.app.recipes.base.recipe import BaseRecipe


class LimitRecipe(BaseRecipe):
    recipe_id = "limit"
    name = "Limit"
    version = "1.1.0"
    category = "preprocessing"
    description = "Restricts row count (head/tail/random sample) with optional row offset."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "rows": {"type": "integer", "title": "Row Count", "default": 1000, "minimum": 1},
                "mode": {"type": "string", "title": "Mode", "enum": ["head", "tail", "random"], "default": "head"},
                "offset": {"type": "integer", "title": "Offset (Skip Rows)", "default": 0, "minimum": 0},
                "random_state": {"type": "integer", "title": "Random Seed", "default": 42}
            },
            "required": ["rows"]
        }

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        errors = []
        raw_rows = config.get("rows")
        if raw_rows is None:
            errors.append("Row count ('rows') is required.")
        else:
            try:
                rows_val = int(raw_rows)
                if rows_val <= 0:
                    errors.append(f"Row count must be greater than 0. Received: {rows_val}.")
            except (ValueError, TypeError):
                errors.append(f"Row count must be a valid integer. Received: {repr(raw_rows)}.")

        mode = str(config.get("mode", "head")).lower().strip()
        if mode not in ["head", "tail", "random"]:
            errors.append(f"Invalid mode '{mode}'. Must be one of: 'head', 'tail', 'random'.")

        raw_offset = config.get("offset")
        if raw_offset is not None:
            try:
                offset_val = int(raw_offset)
                if offset_val < 0:
                    errors.append(f"Offset cannot be negative. Received: {offset_val}.")
            except (ValueError, TypeError):
                errors.append(f"Offset must be a valid integer. Received: {repr(raw_offset)}.")

        return errors

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        df: pd.DataFrame = inputs.get("dataframe")
        if df is None:
            raise ValueError("LimitRecipe expects 'dataframe' in inputs.")

        # Safe parameter resolution
        raw_rows = config.get("rows")
        try:
            n = 1000 if raw_rows is None else int(raw_rows)
            if n <= 0:
                raise ValueError(f"Row count must be positive (> 0). Received: {n}")
        except (ValueError, TypeError) as e:
            raise ValueError(f"LimitRecipe: invalid 'rows' value {repr(raw_rows)}") from e

        mode = str(config.get("mode", "head")).lower().strip()
        raw_offset = config.get("offset")
        offset = 0 if raw_offset is None else max(0, int(raw_offset))

        raw_seed = config.get("random_state")
        seed = 42 if raw_seed is None else int(raw_seed)

        if len(df) == 0:
            out = df.copy()
        elif mode == "tail":
            out = df.tail(n)
        elif mode == "random":
            out = df.sample(n=min(n, len(df)), random_state=seed)
        else:  # head (with optional offset)
            if offset > 0:
                out = df.iloc[offset : offset + n]
            else:
                out = df.head(n)

        out = out.reset_index(drop=True)
        return {
            "dataframe": out,
            "feature_names": list(out.columns),
            "output_summary": {"row_count": len(out), "columns": list(out.columns)}
        }

    def to_code(self, config: Dict[str, Any]) -> str:
        n = config.get("rows", 1000) or 1000
        mode = str(config.get("mode", "head")).lower().strip()
        offset = int(config.get("offset", 0) or 0)
        if mode == "tail":
            return f"df = df.tail({n}).reset_index(drop=True)"
        elif mode == "random":
            seed = config.get("random_state", 42)
            return f"df = df.sample(n=min({n}, len(df)), random_state={seed}).reset_index(drop=True)"
        elif offset > 0:
            return f"df = df.iloc[{offset}:{offset} + {n}].reset_index(drop=True)"
        else:
            return f"df = df.head({n}).reset_index(drop=True)"