import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from backend.app.recipes.base.recipe import BaseRecipe


class MissingValueImputerRecipe(BaseRecipe):
    recipe_id = "missing_value_imputer"
    name = "Missing Value Imputer"
    version = "1.0.0"
    category = "preprocessing"
    description = "Handles missing values via Mean, Median, Mode, Forward Fill, or Constant value."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "strategy": {
                    "type": "string",
                    "title": "Imputation Strategy",
                    "enum": ["mean", "median", "mode", "drop_rows", "constant", "ffill", "bfill"],
                    "default": "median"
                },
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "title": "Target Columns",
                    "description": "Columns to impute. If left empty, strategy will apply to all compatible columns."
                },
                "fill_value": {
                    "type": "string",
                    "title": "Constant Fill Value",
                    "description": "Used only when strategy is 'constant'."
                }
            },
            "required": ["strategy"]
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        df: pd.DataFrame = inputs.get("dataframe")
        if df is None:
            raise ValueError("MissingValueImputer expects 'dataframe' in inputs.")

        df = df.copy()
        strategy = config.get("strategy", "median")
        target_cols = config.get("columns", [])
        fill_val = config.get("fill_value", 0)

        cols_to_process = target_cols if target_cols else list(df.columns)

        if strategy == "drop_rows":
            df = df.dropna(subset=cols_to_process)
        elif strategy == "ffill":
            df[cols_to_process] = df[cols_to_process].ffill()
        elif strategy == "bfill":
            df[cols_to_process] = df[cols_to_process].bfill()
        elif strategy in ["mean", "median"]:
            for col in cols_to_process:
                if pd.api.types.is_numeric_dtype(df[col]):
                    val = df[col].median() if strategy == "median" else df[col].mean()
                    if pd.isna(val):
                        val = 0
                    df[col] = df[col].fillna(val)
                else:
                    mode_val = df[col].mode()
                    fill = mode_val[0] if not mode_val.empty else "Missing"
                    df[col] = df[col].fillna(fill)
        elif strategy == "mode":
            for col in cols_to_process:
                mode_val = df[col].mode()
                fill = mode_val[0] if not mode_val.empty else (0 if pd.api.types.is_numeric_dtype(df[col]) else "Missing")
                df[col] = df[col].fillna(fill)
        elif strategy == "constant":
            for col in cols_to_process:
                df[col] = df[col].fillna(fill_val)

        return {"dataframe": df}

    def to_code(self, config: Dict[str, Any]) -> str:
        strat = config.get("strategy", "median")
        if strat == "median":
            return "# Impute missing values with column medians\ndf = df.fillna(df.median(numeric_only=True))"
        elif strat == "mean":
            return "# Impute missing values with column means\ndf = df.fillna(df.mean(numeric_only=True))"
        elif strat == "mode":
            return "# Impute categorical missing values with column mode\ndf = df.fillna(df.mode().iloc[0])"
        else:
            return f"# Handle missing values using strategy: {strat}\ndf = df.dropna() if '{strat}' == 'drop_rows' else df.ffill()"
