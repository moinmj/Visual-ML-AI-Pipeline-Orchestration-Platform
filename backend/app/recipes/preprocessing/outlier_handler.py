import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from backend.app.recipes.base.recipe import BaseRecipe


class OutlierHandlerRecipe(BaseRecipe):
    recipe_id = "outlier_handler"
    name = "Outlier Handler & Winsorizer"
    version = "1.0.0"
    category = "preprocessing"
    description = "Detects, clips (winsorizes), filters, or flags numerical outliers using IQR, Z-Score, or Quantile bounds."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "title": "Detection Method",
                    "enum": ["iqr", "z_score", "quantile"],
                    "default": "iqr",
                    "description": "'iqr' uses interquartile range (best for skewed data); 'z_score' uses standard deviations (best for normal data); 'quantile' uses percentiles."
                },
                "action": {
                    "type": "string",
                    "title": "Handling Action",
                    "enum": ["clip", "filter", "flag"],
                    "default": "clip",
                    "description": "'clip' (Winsorization) caps values at boundary limits to avoid data loss; 'filter' removes outlier rows; 'flag' adds boolean indicator columns."
                },
                "threshold": {
                    "type": "number",
                    "title": "Threshold Multiplier",
                    "default": 1.5,
                    "description": "For IQR: multiplier (default 1.5). For Z-Score: number of standard deviations (e.g. 3.0)."
                },
                "lower_quantile": {
                    "type": "number",
                    "title": "Lower Quantile",
                    "default": 0.01,
                    "description": "Used only when method='quantile' (e.g. 0.01 for 1st percentile)."
                },
                "upper_quantile": {
                    "type": "number",
                    "title": "Upper Quantile",
                    "default": 0.99,
                    "description": "Used only when method='quantile' (e.g. 0.99 for 99th percentile)."
                },
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "title": "Target Columns",
                    "description": "Numeric columns to inspect. If empty, automatically evaluates all numeric features."
                }
            },
            "required": ["method", "action"]
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        df: pd.DataFrame = inputs.get("dataframe")
        if df is None:
            if context and isinstance(context, dict) and "dataframe" in context:
                df = context["dataframe"]
            else:
                raise ValueError("OutlierHandlerRecipe expects 'dataframe' in inputs.")

        df_out = df.copy()
        method = config.get("method", "iqr")
        action = config.get("action", "clip")
        threshold = float(config.get("threshold", 1.5 if method == "iqr" else 3.0))
        lq = float(config.get("lower_quantile", 0.01))
        uq = float(config.get("upper_quantile", 0.99))
        cols_cfg = config.get("columns", [])

        # Parse column list
        if isinstance(cols_cfg, str):
            specified_cols = [c.strip() for c in cols_cfg.split(",") if c.strip()]
        elif isinstance(cols_cfg, (list, tuple)):
            specified_cols = [str(c).strip() for c in cols_cfg if str(c).strip()]
        else:
            specified_cols = []

        # Auto-detect numeric columns if none or empty
        if specified_cols:
            num_cols = [c for c in specified_cols if c in df_out.columns and pd.api.types.is_numeric_dtype(df_out[c])]
        else:
            num_cols = [c for c in df_out.columns if pd.api.types.is_numeric_dtype(df_out[c])]

        if not num_cols:
            return {"dataframe": df_out}

        # Track row masks for filter action
        valid_row_mask = pd.Series(True, index=df_out.index)

        for col in num_cols:
            series = df_out[col].dropna()
            if len(series) < 3:
                continue

            if method == "iqr":
                q25 = series.quantile(0.25)
                q75 = series.quantile(0.75)
                iqr = q75 - q25
                lower = q25 - (threshold * iqr)
                upper = q75 + (threshold * iqr)

            elif method == "z_score":
                mean = series.mean()
                std = series.std()
                if std == 0 or np.isnan(std):
                    continue
                lower = mean - (threshold * std)
                upper = mean + (threshold * std)

            elif method == "quantile":
                lower = series.quantile(lq)
                upper = series.quantile(uq)

            else:
                continue

            if action == "clip":
                df_out[col] = df_out[col].clip(lower=lower, upper=upper)

            elif action == "filter":
                col_mask = (df_out[col].isna()) | ((df_out[col] >= lower) & (df_out[col] <= upper))
                valid_row_mask = valid_row_mask & col_mask

            elif action == "flag":
                is_outlier = ((df_out[col] < lower) | (df_out[col] > upper)).astype(int)
                df_out[f"{col}_is_outlier"] = is_outlier

        if action == "filter":
            df_out = df_out[valid_row_mask].reset_index(drop=True)

        return {"dataframe": df_out}

    def to_code(self, config: Dict[str, Any]) -> str:
        method = config.get("method", "iqr")
        action = config.get("action", "clip")
        thresh = config.get("threshold", 1.5 if method == "iqr" else 3.0)
        return f"# Outlier Handling ({method.upper()} - {action.title()})\n# Target columns evaluated against threshold: {thresh}\n# Action '{action}' applied across numeric features"
