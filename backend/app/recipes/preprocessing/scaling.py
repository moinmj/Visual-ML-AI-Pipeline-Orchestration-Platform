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
        exclude_target_cfg = config.get("exclude_target", True)

        # Resolve target variable from config, exclude_target string, context, or inputs
        target_var = config.get("target_column")
        if not target_var and isinstance(exclude_target_cfg, str) and exclude_target_cfg.strip():
            target_var = exclude_target_cfg.strip()
        if not target_var and isinstance(context, dict):
            target_var = context.get("target_column")
        if not target_var and isinstance(inputs, dict):
            target_var = inputs.get("target_column")

        exclude_target = bool(exclude_target_cfg) if not isinstance(exclude_target_cfg, str) else True

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

        date_kws = ["date", "year", "ds", "timestamp", "time", "month", "period"]
        domain_target_kws = ["target", "churn", "survived", "label", "class", "y", "sales", "weekly_sales", "revenue", "demand", "price", "amount", "score", "value", "unemployment"]

        if not target_cols:
            # Auto-detect numeric columns
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

            # Always exclude date/year/time columns from z-score scaling
            numeric_cols = [
                c for c in numeric_cols
                if not any(_kw == c.lower().strip() or c.lower().strip().startswith(_kw + "_") or c.lower().strip().endswith("_" + _kw) for _kw in date_kws)
            ]

            if exclude_target:
                filtered_cols = []
                for c in numeric_cols:
                    if target_var and c.lower().strip() == target_var.lower().strip():
                        continue
                    is_candidate_target = (df[c].nunique() <= 2) or any(_kw in c.lower() for _kw in domain_target_kws)
                    if not is_candidate_target:
                        filtered_cols.append(c)
                target_cols = filtered_cols if filtered_cols else [c for c in numeric_cols if not (target_var and c.lower().strip() == target_var.lower().strip())]
            else:
                target_cols = numeric_cols

        # Final safety filter: remove target_var and date columns from target_cols
        if target_var and exclude_target:
            target_cols = [c for c in target_cols if c.lower().strip() != target_var.lower().strip()]
        
        target_cols = [
            c for c in target_cols
            if not any(_kw == c.lower().strip() or c.lower().strip().startswith(_kw + "_") or c.lower().strip().endswith("_" + _kw) for _kw in date_kws)
        ]

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
