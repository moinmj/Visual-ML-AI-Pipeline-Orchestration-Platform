import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.preprocessing import LabelEncoder
from backend.app.recipes.base.recipe import BaseRecipe


class CategoricalEncoderRecipe(BaseRecipe):
    recipe_id = "categorical_encoder"
    name = "Categorical Encoder"
    version = "1.0.0"
    category = "preprocessing"
    description = "Encodes categorical features using One-Hot Encoding (pd.get_dummies) or Label Encoding."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "title": "Encoding Method",
                    "enum": ["one_hot", "label"],
                    "default": "one_hot"
                },
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "title": "Columns to Encode",
                    "description": "Specific categorical columns to encode. If empty, all string/categorical features are processed."
                },
                "drop_first": {
                    "type": "boolean",
                    "title": "Drop First Dummy",
                    "default": False,
                    "description": "Used only with One-Hot encoding to avoid multicollinearity."
                }
            },
            "required": ["method"]
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        df: pd.DataFrame = inputs.get("dataframe")
        if df is None:
            raise ValueError("CategoricalEncoder expects 'dataframe' in inputs.")

        df = df.copy()
        method = config.get("method", "one_hot")
        target_cols = config.get("columns", [])
        drop_first = config.get("drop_first", False)

        target_col_name = config.get("target_column")
        if not target_col_name and context and isinstance(context, dict):
            target_col_name = context.get("target_column")

        if not target_cols:
            # Pick non-numeric columns
            non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
            
            # Avoid splitting target column into one-hot dummies
            feature_cols = []
            for c in non_numeric:
                is_target_name = (target_col_name and c == target_col_name) or (c.lower() in ["target", "churn", "survived", "label", "class", "species", "y"])
                if is_target_name:
                    # Label encode the target column so it stays a single 1D column (0, 1, 2...)
                    le = LabelEncoder()
                    df[c] = le.fit_transform(df[c].astype(str))
                else:
                    feature_cols.append(c)
            target_cols = feature_cols

        if not target_cols:
            return {"dataframe": df}

        if method == "one_hot":
            df = pd.get_dummies(df, columns=target_cols, drop_first=drop_first, dtype=int)
        elif method == "label":
            for col in target_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))

        return {"dataframe": df}
