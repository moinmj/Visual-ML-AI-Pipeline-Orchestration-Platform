import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from backend.app.recipes.base.recipe import BaseRecipe


class FeatureScalerRecipe(BaseRecipe):
    recipe_id = "feature_scaler"
    name = "Feature Scaler"
    version = "1.0.0"
    category = "preprocessing"
    description = "Scales numeric feature columns using StandardScaler, MinMaxScaler, or RobustScaler while preserving the target column."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "title": "Scaling Method",
                    "enum": ["standard", "minmax", "robust"],
                    "default": "standard"
                },
                "exclude_target": {
                    "type": "boolean",
                    "title": "Exclude Target / Label Column from Scaling",
                    "default": True
                },
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "title": "Columns to Scale",
                    "description": "Specific numeric columns to scale. If empty, all continuous numeric features are scaled."
                }
            },
            "required": ["method"]
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        df: pd.DataFrame = inputs.get("dataframe")
        if df is None:
            raise ValueError("FeatureScaler expects 'dataframe' in inputs.")

        df = df.copy()
        method = config.get("method", "standard")
        target_cols = config.get("columns", [])
        exclude_target = config.get("exclude_target", True)

        if isinstance(target_cols, str):
            if target_cols.strip():
                parsed = [c.strip() for c in target_cols.split(",") if c.strip() in df.columns]
                target_cols = parsed if parsed else ([target_cols] if target_cols in df.columns else [])
            else:
                target_cols = []
        elif isinstance(target_cols, (list, tuple)):
            target_cols = [c for c in target_cols if c in df.columns]
        else:
            target_cols = []

        if not target_cols:
            # Auto-detect numeric columns
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

            target_var = config.get("target_column")
            if not target_var and isinstance(context, dict):
                target_var = context.get("target_column")
            if not target_var and isinstance(inputs, dict):
                target_var = inputs.get("target_column")

            if exclude_target:
                # Identify candidate targets to exclude
                filtered_cols = []
                for c in numeric_cols:
                    if target_var and c == target_var:
                        continue
                    is_candidate_target = (df[c].nunique() <= 2) or (c.lower() in ["target", "churn", "survived", "label", "class", "y"])
                    if not is_candidate_target:
                        filtered_cols.append(c)
                target_cols = filtered_cols if filtered_cols else [c for c in numeric_cols if c != target_var]
            else:
                target_cols = numeric_cols

        if target_var and exclude_target and target_var in target_cols:
            target_cols = [c for c in target_cols if c != target_var]

        if not target_cols:
            return {"dataframe": df}

        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        elif method == "robust":
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")

        df[target_cols] = scaler.fit_transform(df[target_cols].fillna(0))
        return {"dataframe": df, "scaler": scaler}

    def to_code(self, config: Dict[str, Any]) -> str:
        method = config.get("method", "standard")
        scaler_cls = "StandardScaler" if method == "standard" else ("MinMaxScaler" if method == "minmax" else "RobustScaler")
        return f"from sklearn.preprocessing import {scaler_cls}\n\nscaler = {scaler_cls}()\ndf[numeric_cols] = scaler.fit_transform(df[numeric_cols])"
