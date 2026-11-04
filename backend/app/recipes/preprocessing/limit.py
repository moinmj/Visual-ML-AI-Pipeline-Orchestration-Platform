import re
from typing import Dict, Any, List, Optional
import pandas as pd
from backend.app.recipes.base.recipe import BaseRecipe


class LimitRecipe(BaseRecipe):
    recipe_id = "limit"
    name = "Limit"
    version = "1.2.0"
    category = "preprocessing"
    description = "Restricts row count (head/tail/random sample) with exact counts or percentage-based sampling, plus forward and reverse offset skipping."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "title": "Sampling Mode",
                    "enum": ["head", "tail", "random"],
                    "default": "head",
                    "description": "'head' selects top records, 'tail' selects bottom records (supporting reverse offset skipping), 'random' performs uniform random sampling."
                },
                "limit_type": {
                    "type": "string",
                    "title": "Limit By",
                    "enum": ["rows", "percentage"],
                    "default": "rows",
                    "description": "Choose whether to limit rows by an exact count ('rows') or by a proportion of the dataset ('percentage')."
                },
                "rows": {
                    "type": "integer",
                    "title": "Row Count",
                    "default": 1000,
                    "minimum": 1,
                    "description": "Exact number of rows to return when Limit By is 'rows'."
                },
                "percentage": {
                    "type": "number",
                    "title": "Sampling Percentage (%)",
                    "default": 10.0,
                    "minimum": 0.01,
                    "maximum": 100.0,
                    "description": "Percentage of total rows to retain when Limit By is 'percentage' (e.g., 10 for 10%, 25 for 25%)."
                },
                "offset": {
                    "type": "integer",
                    "title": "Offset (Skip Rows)",
                    "default": 0,
                    "minimum": 0,
                    "description": "Rows to skip: forward from the top in 'head' mode, or reverse from the bottom in 'tail' mode."
                },
                "random_state": {
                    "type": "integer",
                    "title": "Random Seed",
                    "default": 42,
                    "description": "Seed for reproducible sampling in 'random' mode."
                }
            },
            "required": []
        }

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        errors: List[str] = []

        mode = str(config.get("mode", "head")).lower().strip()
        if mode not in ["head", "tail", "random"]:
            errors.append(f"Invalid mode '{mode}'. Must be one of: 'head', 'tail', 'random'.")

        limit_type = str(config.get("limit_type", "")).lower().strip()
        if not limit_type:
            # Auto-infer limit_type if percentage is explicitly provided and rows is not
            if "percentage" in config and "rows" not in config:
                limit_type = "percentage"
            else:
                limit_type = "rows"

        if limit_type not in ["rows", "percentage"]:
            errors.append(f"Invalid limit_type '{limit_type}'. Must be 'rows' or 'percentage'.")

        if limit_type == "percentage":
            raw_pct = config.get("percentage")
            if raw_pct is None:
                errors.append("Sampling percentage ('percentage') is required when Limit By is 'percentage'.")
            else:
                try:
                    pct_val = self._parse_percentage_value(raw_pct)
                    if pct_val <= 0 or pct_val > 100.0:
                        errors.append(f"Sampling percentage must be > 0 and <= 100. Received: {pct_val}%.")
                except (ValueError, TypeError):
                    errors.append(f"Sampling percentage must be a valid numeric value. Received: {repr(raw_pct)}.")
        else:
            raw_rows = config.get("rows")
            if raw_rows is not None:
                try:
                    rows_val = int(raw_rows)
                    if rows_val <= 0:
                        errors.append(f"Row count must be greater than 0. Received: {rows_val}.")
                except (ValueError, TypeError):
                    errors.append(f"Row count must be a valid integer. Received: {repr(raw_rows)}.")

        raw_offset = config.get("offset")
        if raw_offset is not None:
            try:
                offset_val = int(raw_offset)
                if offset_val < 0:
                    errors.append(f"Offset cannot be negative. Received: {offset_val}.")
            except (ValueError, TypeError):
                errors.append(f"Offset must be a valid integer. Received: {repr(raw_offset)}.")

        raw_seed = config.get("random_state")
        if raw_seed is not None:
            try:
                int(raw_seed)
            except (ValueError, TypeError):
                errors.append(f"Random seed must be a valid integer. Received: {repr(raw_seed)}.")

        return errors

    @staticmethod
    def _parse_percentage_value(val: Any) -> float:
        """Parses a percentage value from float, int, or string like '10%' or '25.5'."""
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            clean = val.strip().rstrip("%").strip()
            return float(clean)
        raise ValueError(f"Cannot parse percentage from {repr(val)}")

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        df: pd.DataFrame = inputs.get("dataframe")
        if df is None:
            if context and isinstance(context, dict) and "dataframe" in context:
                df = context["dataframe"]
            else:
                raise ValueError("LimitRecipe expects 'dataframe' in inputs.")

        total_rows = len(df)
        mode = str(config.get("mode", "head")).lower().strip()

        # Determine limit_type
        limit_type = str(config.get("limit_type", "")).lower().strip()
        if not limit_type:
            if "percentage" in config and "rows" not in config:
                limit_type = "percentage"
            else:
                limit_type = "rows"

        pct_val: Optional[float] = None
        if limit_type == "percentage":
            raw_pct = config.get("percentage", 10.0)
            try:
                pct_val = self._parse_percentage_value(raw_pct if raw_pct is not None else 10.0)
            except Exception as e:
                raise ValueError(f"LimitRecipe: invalid 'percentage' value {repr(raw_pct)}") from e

            if total_rows == 0:
                n = 0
            else:
                fraction = pct_val / 100.0
                n = max(1, int(round(total_rows * fraction)))
                n = min(n, total_rows)
        else:
            raw_rows = config.get("rows")
            try:
                n = 1000 if raw_rows is None else int(raw_rows)
                if n <= 0:
                    raise ValueError(f"Row count must be positive (> 0). Received: {n}")
            except (ValueError, TypeError) as e:
                raise ValueError(f"LimitRecipe: invalid 'rows' value {repr(raw_rows)}") from e

        raw_offset = config.get("offset")
        offset = 0 if raw_offset is None else max(0, int(raw_offset))

        raw_seed = config.get("random_state")
        seed = 42 if raw_seed is None else int(raw_seed)

        if total_rows == 0 or n == 0:
            out = df.copy()
        elif mode == "random":
            # Uniform random sampling with reproducible seed
            sample_size = min(n, total_rows)
            out = df.sample(n=sample_size, random_state=seed)
        elif mode == "tail":
            # Tail mode supporting reverse offset skipping
            if offset > 0:
                if offset >= total_rows:
                    out = df.iloc[0:0].copy()
                else:
                    end_idx = total_rows - offset
                    start_idx = max(0, end_idx - n)
                    out = df.iloc[start_idx:end_idx]
            else:
                out = df.tail(n)
        else:
            # Head mode with forward offset skipping
            if offset > 0:
                if offset >= total_rows:
                    out = df.iloc[0:0].copy()
                else:
                    out = df.iloc[offset : offset + n]
            else:
                out = df.head(n)

        out = out.reset_index(drop=True)
        return {
            "dataframe": out,
            "feature_names": list(out.columns),
            "output_summary": {
                "row_count": len(out),
                "columns": list(out.columns),
                "mode": mode,
                "limit_type": limit_type,
                "input_row_count": total_rows,
                "requested_rows": n if limit_type == "rows" else None,
                "requested_percentage": pct_val if limit_type == "percentage" else None,
                "effective_rows": len(out),
                "offset": offset,
                "retention_rate": round(len(out) / max(1, total_rows), 4)
            }
        }

    def to_code(self, config: Dict[str, Any]) -> str:
        mode = str(config.get("mode", "head")).lower().strip()
        limit_type = str(config.get("limit_type", "")).lower().strip()
        if not limit_type:
            if "percentage" in config and "rows" not in config:
                limit_type = "percentage"
            else:
                limit_type = "rows"

        offset = int(config.get("offset", 0) or 0)
        seed = config.get("random_state", 42)

        code_lines = []
        if limit_type == "percentage":
            pct = config.get("percentage", 10.0)
            code_lines.append(f"# Percentage-based row limiting ({pct}%)")
            code_lines.append(f"n = max(1, int(round(len(df) * ({pct} / 100.0))))")
        else:
            n = config.get("rows", 1000) or 1000
            code_lines.append(f"n = {n}")

        if mode == "random":
            code_lines.append(f"df = df.sample(n=min(n, len(df)), random_state={seed}).reset_index(drop=True)")
        elif mode == "tail":
            if offset > 0:
                code_lines.append(f"# Tail mode with reverse offset skipping ({offset} rows from bottom)")
                code_lines.append(f"end_idx = max(0, len(df) - {offset})")
                code_lines.append(f"start_idx = max(0, end_idx - n)")
                code_lines.append(f"df = df.iloc[start_idx:end_idx].reset_index(drop=True)")
            else:
                code_lines.append(f"df = df.tail(n).reset_index(drop=True)")
        else:  # head
            if offset > 0:
                code_lines.append(f"# Head mode with forward offset ({offset} rows from top)")
                code_lines.append(f"df = df.iloc[{offset} : {offset} + n].reset_index(drop=True)")
            else:
                code_lines.append(f"df = df.head(n).reset_index(drop=True)")

        return "\n".join(code_lines)