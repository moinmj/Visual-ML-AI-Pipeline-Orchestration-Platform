import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from backend.app.recipes.base.recipe import BaseRecipe


class CategoricalEncoderRecipe(BaseRecipe):
    recipe_id = "categorical_encoder"
    name = "Categorical Encoder"
    version = "1.1.0"
    category = "preprocessing"
    description = "Encodes categorical features using 7 advanced techniques: One-Hot, Label, Ordinal, Target/Mean, Frequency/Count, Binary, or Weight of Evidence (WoE)."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "title": "Encoding Method",
                    "enum": ["one_hot", "label", "ordinal", "target", "frequency", "binary", "woe"],
                    "default": "one_hot",
                    "description": "Select from 7 encoding techniques."
                },
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "title": "Columns to Encode",
                    "description": "Specific categorical columns to encode. If empty, all string/categorical features are processed."
                },
                "target_column": {
                    "type": "string",
                    "title": "Target Column",
                    "description": "Required for Target and WoE encodings. If blank, auto-detected from context."
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
            # Pick non-numeric columns
            non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
            
            # Avoid encoding the target column itself if it exists
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

        # Resolve Target Column for Target/WoE encodings
        target_series = None
        if target_col_name and target_col_name in df.columns:
            target_series = df[target_col_name]
        else:
            # Find candidate numeric target
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in target_cols]
            if num_cols:
                target_series = df[num_cols[-1]]

        # -------------------------------------------------------------
        # ENCODING METHOD IMPLEMENTATIONS
        # -------------------------------------------------------------
        if method == "one_hot":
            low_card_cols = [c for c in target_cols if df[c].nunique() <= 50]
            high_card_cols = [c for c in target_cols if df[c].nunique() > 50]

            if low_card_cols:
                df = pd.get_dummies(df, columns=low_card_cols, drop_first=drop_first, dtype=int)
            if high_card_cols:
                for col in high_card_cols:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))

        elif method in ["label", "ordinal"]:
            for col in target_cols:
                oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                encoded_arr = oe.fit_transform(df[[col]].astype(str))
                df[col] = encoded_arr.flatten().astype(int)

        elif method == "frequency":
            for col in target_cols:
                freq_map = df[col].value_counts(normalize=True).to_dict()
                df[col] = df[col].map(freq_map).fillna(0.0).astype(float)

        elif method == "target":
            if target_series is not None:
                numeric_target = pd.to_numeric(target_series, errors="coerce").fillna(0)
                global_mean = numeric_target.mean()
                for col in target_cols:
                    means = numeric_target.groupby(df[col]).mean().to_dict()
                    df[col] = df[col].map(means).fillna(global_mean).astype(float)
            else:
                # Fallback to frequency if no target column present
                for col in target_cols:
                    freq_map = df[col].value_counts(normalize=True).to_dict()
                    df[col] = df[col].map(freq_map).fillna(0.0).astype(float)

        elif method == "binary":
            for col in target_cols:
                le = LabelEncoder()
                int_codes = le.fit_transform(df[col].astype(str))
                n_bits = int(np.ceil(np.log2(max(len(le.classes_), 2))))
                for b in range(n_bits):
                    bit_col_name = f"{col}_bin_{b}"
                    df[bit_col_name] = ((int_codes >> b) & 1).astype(int)
                df.drop(columns=[col], inplace=True)

        elif method == "woe":
            if target_series is not None:
                numeric_target = pd.to_numeric(target_series, errors="coerce").fillna(0)
                bin_target = (numeric_target > numeric_target.median()).astype(int)
                total_pos = (bin_target == 1).sum() or 1
                total_neg = (bin_target == 0).sum() or 1

                for col in target_cols:
                    woe_map = {}
                    grouped = bin_target.groupby(df[col])
                    for val, group in grouped:
                        pos = (group == 1).sum()
                        neg = (group == 0).sum()
                        pos_ratio = (pos + 0.5) / total_pos
                        neg_ratio = (neg + 0.5) / total_neg
                        woe_map[val] = np.log(pos_ratio / neg_ratio)

                    df[col] = df[col].map(woe_map).fillna(0.0).astype(float)
            else:
                # Fallback to frequency if no binary target available
                for col in target_cols:
                    freq_map = df[col].value_counts(normalize=True).to_dict()
                    df[col] = df[col].map(freq_map).fillna(0.0).astype(float)

        return {
            "dataframe": df,
            "metrics": {
                "method_applied": method,
                "encoded_columns": target_cols,
                "final_column_count": len(df.columns)
            }
        }
