import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import math


class DataProfiler:
    """
    Automated statistical profiling engine for tabular datasets.
    Computes schema, distributions, missingness, cardinality, and quality scores.
    """

    @staticmethod
    def _infer_column_type(series: pd.Series) -> str:
        if pd.api.types.is_bool_dtype(series):
            return "boolean"
        elif pd.api.types.is_numeric_dtype(series):
            return "numeric"
        elif pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        else:
            # Check if strings could be parsed as datetime or categorical
            # If unique count is small compared to length, it's categorical
            non_null_count = series.dropna().count()
            if non_null_count == 0:
                return "empty"
            unique_count = series.nunique()
            if unique_count <= 20 or (unique_count / non_null_count < 0.05):
                return "categorical"
            return "text"

    @classmethod
    def profile_dataframe(cls, df: pd.DataFrame) -> Dict[str, Any]:
        row_count = int(len(df))
        col_count = int(len(df.columns))

        if row_count == 0:
            return {
                "row_count": 0,
                "column_count": col_count,
                "memory_bytes": 0,
                "duplicate_rows": 0,
                "duplicate_percentage": 0.0,
                "total_missing_cells": 0,
                "missing_cells_percentage": 0.0,
                "quality_score": 100.0,
                "columns": {}
            }

        # Global dataset metrics
        duplicate_rows = int(df.duplicated().sum())
        duplicate_percentage = round((duplicate_rows / row_count) * 100.0, 2)
        total_cells = row_count * col_count
        total_missing = int(df.isna().sum().sum())
        missing_percentage = round((total_missing / total_cells) * 100.0, 2) if total_cells > 0 else 0.0
        memory_bytes = int(df.memory_usage(deep=True).sum())

        # Column-level profiles
        columns_profile: Dict[str, Any] = {}

        for col_name in df.columns:
            series = df[col_name]
            col_type = cls._infer_column_type(series)
            null_count = int(series.isna().sum())
            null_percentage = round((null_count / row_count) * 100.0, 2)
            unique_count = int(series.nunique())
            cardinality_ratio = round(unique_count / row_count, 4)

            col_meta: Dict[str, Any] = {
                "name": str(col_name),
                "inferred_type": col_type,
                "raw_dtype": str(series.dtype),
                "null_count": null_count,
                "null_percentage": null_percentage,
                "unique_count": unique_count,
                "cardinality_ratio": cardinality_ratio,
                "stats": {}
            }

            # Numeric metrics
            if col_type == "numeric":
                clean_series = series.dropna()
                if not clean_series.empty:
                    stats: Dict[str, Any] = {
                        "mean": float(round(clean_series.mean(), 4)),
                        "std": float(round(clean_series.std(), 4)) if len(clean_series) > 1 else 0.0,
                        "min": float(clean_series.min()),
                        "p25": float(round(clean_series.quantile(0.25), 4)),
                        "median": float(round(clean_series.median(), 4)),
                        "p75": float(round(clean_series.quantile(0.75), 4)),
                        "max": float(clean_series.max()),
                        "zeros_count": int((clean_series == 0).sum()),
                        "negative_count": int((clean_series < 0).sum())
                    }
                    col_meta["stats"] = stats

            # Categorical & Text metrics
            elif col_type in ["categorical", "text", "boolean"]:
                clean_series = series.dropna().astype(str)
                if not clean_series.empty:
                    top_counts = clean_series.value_counts().head(5)
                    top_values = [
                        {"value": val, "count": int(count), "percentage": round((count / row_count) * 100.0, 2)}
                        for val, count in top_counts.items()
                    ]
                    col_meta["stats"] = {
                        "top_values": top_values,
                        "mode": str(clean_series.mode()[0]) if not clean_series.mode().empty else None
                    }

            columns_profile[str(col_name)] = col_meta

        # Calculate Data Quality Health Score (0 - 100)
        # Deduct penalties for missing values, duplicates, and empty columns
        missing_penalty = min(missing_percentage * 1.5, 50.0)
        duplicate_penalty = min(duplicate_percentage * 1.0, 30.0)
        quality_score = max(round(100.0 - missing_penalty - duplicate_penalty, 1), 0.0)

        return {
            "row_count": row_count,
            "column_count": col_count,
            "memory_bytes": memory_bytes,
            "duplicate_rows": duplicate_rows,
            "duplicate_percentage": duplicate_percentage,
            "total_missing_cells": total_missing,
            "missing_cells_percentage": missing_percentage,
            "quality_score": quality_score,
            "columns": columns_profile
        }
