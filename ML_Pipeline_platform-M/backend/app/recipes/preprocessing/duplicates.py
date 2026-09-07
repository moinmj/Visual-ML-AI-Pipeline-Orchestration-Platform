import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from backend.app.recipes.base.recipe import BaseRecipe


class DuplicateRemoverRecipe(BaseRecipe):
    recipe_id = "duplicate_remover"
    name = "Duplicate Row Handler & Remover"
    version = "1.0.0"
    category = "preprocessing"
    description = "Detects and eliminates duplicate data rows (keep first, keep last, drop all, or flag only) to prevent train-test data leakage."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "strategy": {
                    "type": "string",
                    "title": "Deduplication Strategy",
                    "enum": ["drop_first", "drop_last", "drop_all", "flag_only"],
                    "default": "drop_first",
                    "description": "'drop_first' keeps first occurrence, 'drop_last' keeps last, 'drop_all' removes all duplicates, 'flag_only' adds an is_duplicate column."
                },
                "subset_columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "title": "Subset Columns",
                    "description": "Columns to inspect for duplicates. If empty, inspects all columns."
                }
            },
            "required": ["strategy"]
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        df: pd.DataFrame = inputs.get("dataframe")
        if df is None:
            raise ValueError("DuplicateRemoverRecipe expects 'dataframe' in inputs.")

        df = df.copy()
        strategy = config.get("strategy", "drop_first")
        subset = config.get("subset_columns", [])

        # Parse subset
        if isinstance(subset, str):
            parsed = [c.strip() for c in subset.split(",") if c.strip() in df.columns]
            subset = parsed if parsed else None
        elif isinstance(subset, (list, tuple)):
            parsed = [c for c in subset if c in df.columns]
            subset = parsed if parsed else None
        else:
            subset = None

        orig_rows = len(df)
        if strategy == "flag_only":
            dup_mask = df.duplicated(subset=subset, keep=False)
            df["is_duplicate"] = dup_mask.astype(int)
            removed = 0
        elif strategy == "drop_first":
            df = df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
            removed = orig_rows - len(df)
        elif strategy == "drop_last":
            df = df.drop_duplicates(subset=subset, keep="last").reset_index(drop=True)
            removed = orig_rows - len(df)
        elif strategy == "drop_all":
            df = df.drop_duplicates(subset=subset, keep=False).reset_index(drop=True)
            removed = orig_rows - len(df)
        else:
            df = df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
            removed = orig_rows - len(df)

        return {
            "dataframe": df,
            "metrics": {
                "original_rows": orig_rows,
                "final_rows": len(df),
                "rows_removed": removed,
                "strategy": strategy
            }
        }

    def to_code(self, config: Dict[str, Any]) -> str:
        strategy = config.get("strategy", "drop_first")
        return f"# Duplicate Row Handler\nif '{strategy}' == 'flag_only':\n    df['is_duplicate'] = df.duplicated().astype(int)\nelse:\n    keep_val = 'first' if '{strategy}' == 'drop_first' else ('last' if '{strategy}' == 'drop_last' else False)\n    df = df.drop_duplicates(keep=keep_val).reset_index(drop=True)"


class CategorySanitizerRecipe(BaseRecipe):
    recipe_id = "category_sanitizer"
    name = "Categorical & Text Label Standardizer"
    version = "1.0.0"
    category = "preprocessing"
    description = "Normalizes text and categorical columns (lowercase, strip whitespace, remove special chars) to merge duplicate label variations like 'USA' vs 'usa'."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "lowercase": {
                    "type": "boolean",
                    "title": "Lowercase Text",
                    "default": True
                },
                "strip_whitespace": {
                    "type": "boolean",
                    "title": "Strip Extra Whitespace",
                    "default": True
                },
                "remove_punctuation": {
                    "type": "boolean",
                    "title": "Remove Special Characters",
                    "default": False
                },
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "title": "Target Categorical Columns",
                    "description": "If empty, automatically cleans all text/string columns."
                }
            }
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        df: pd.DataFrame = inputs.get("dataframe")
        if df is None:
            raise ValueError("CategorySanitizerRecipe expects 'dataframe' in inputs.")

        df = df.copy()
        do_lower = config.get("lowercase", True)
        do_strip = config.get("strip_whitespace", True)
        do_punct = config.get("remove_punctuation", False)
        target_cols = config.get("columns", [])

        if not target_cols:
            target_cols = [c for c in df.columns if df[c].dtype == "object" or isinstance(df[c].dtype, pd.CategoricalDtype)]
        elif isinstance(target_cols, str):
            target_cols = [c.strip() for c in target_cols.split(",") if c.strip() in df.columns]

        modified_cols = []
        for col in target_cols:
            if col in df.columns and (df[col].dtype == "object" or str(df[col].dtype) == "category"):
                s = df[col].astype(str)
                if do_strip:
                    s = s.str.strip()
                if do_lower:
                    s = s.str.lower()
                if do_punct:
                    s = s.str.replace(r"[^\w\s]", "", regex=True)
                df[col] = s
                modified_cols.append(col)

        return {
            "dataframe": df,
            "metrics": {
                "columns_sanitized": modified_cols,
                "sanitized_count": len(modified_cols)
            }
        }


class CorrelationFilterRecipe(BaseRecipe):
    recipe_id = "correlation_filter"
    name = "Redundant Column & Multicollinearity Filter"
    version = "1.0.0"
    category = "preprocessing"
    description = "Computes correlation matrix and automatically drops duplicate or highly correlated redundant columns above a threshold."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "threshold": {
                    "type": "number",
                    "title": "Correlation Threshold",
                    "default": 0.95,
                    "minimum": 0.5,
                    "maximum": 1.0,
                    "description": "Columns with correlation greater than this threshold will be dropped as redundant."
                },
                "method": {
                    "type": "string",
                    "title": "Correlation Method",
                    "enum": ["pearson", "spearman"],
                    "default": "pearson"
                }
            },
            "required": ["threshold"]
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        df: pd.DataFrame = inputs.get("dataframe")
        if df is None:
            raise ValueError("CorrelationFilterRecipe expects 'dataframe' in inputs.")

        df = df.copy()
        thresh = float(config.get("threshold", 0.95))
        method = config.get("method", "pearson")

        num_df = df.select_dtypes(include=[np.number])
        dropped_cols = []

        if len(num_df.columns) > 1:
            corr_matrix = num_df.corr(method=method).abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            dropped_cols = [column for column in upper_tri.columns if any(upper_tri[column] > thresh)]
            df = df.drop(columns=dropped_cols)

        return {
            "dataframe": df,
            "metrics": {
                "threshold": thresh,
                "dropped_redundant_columns": dropped_cols,
                "columns_dropped_count": len(dropped_cols)
            }
        }


class VarianceFilterRecipe(BaseRecipe):
    recipe_id = "variance_filter"
    name = "Constant & Low Variance Feature Filter"
    version = "1.0.0"
    category = "preprocessing"
    description = "Detects and drops constant columns where 99%+ of rows contain the exact same duplicate value."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "max_dominant_percentage": {
                    "type": "number",
                    "title": "Max Dominant Value %",
                    "default": 0.99,
                    "minimum": 0.5,
                    "maximum": 1.0,
                    "description": "If a single duplicate value represents more than this percentage of rows, drop the column."
                }
            }
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        df: pd.DataFrame = inputs.get("dataframe")
        if df is None:
            raise ValueError("VarianceFilterRecipe expects 'dataframe' in inputs.")

        df = df.copy()
        max_pct = float(config.get("max_dominant_percentage", 0.99))
        n_rows = len(df)
        dropped_cols = []

        if n_rows > 0:
            for col in df.columns:
                top_freq = df[col].value_counts(dropna=False).max()
                if (top_freq / n_rows) >= max_pct:
                    dropped_cols.append(col)

            if dropped_cols:
                df = df.drop(columns=dropped_cols)

        return {
            "dataframe": df,
            "metrics": {
                "max_dominant_percentage": max_pct,
                "dropped_constant_columns": dropped_cols,
                "columns_dropped_count": len(dropped_cols)
            }
        }
