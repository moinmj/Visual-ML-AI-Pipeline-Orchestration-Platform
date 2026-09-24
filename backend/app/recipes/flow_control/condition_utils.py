import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union


def evaluate_condition_mask(df: pd.DataFrame, condition: Dict[str, Any]) -> pd.Series:
    """
    Evaluates a single condition rule against a DataFrame column
    and returns a boolean Series mask.
    """
    col = condition.get("column") or condition.get("field")
    if not col or col not in df.columns:
        # If column does not exist, default to False for all rows
        return pd.Series(False, index=df.index)

    op = str(condition.get("operator", "==")).lower().strip()
    val = condition.get("value")
    series = df[col]

    try:
        if op in ["is_null", "is_empty", "null", "empty"]:
            return series.isna() | (series.astype(str).str.strip() == "")
        elif op in ["not_null", "not_empty", "not null", "not empty"]:
            return series.notna() & (series.astype(str).str.strip() != "")

        # Numeric comparisons
        if pd.api.types.is_numeric_dtype(series) or (isinstance(val, (int, float)) and str(val).replace(".", "", 1).isdigit()):
            num_series = pd.to_numeric(series, errors="coerce")
            num_val = float(val) if val is not None and str(val).strip() != "" else 0.0

            if op in ["=", "==", "eq", "equals"]:
                return num_series == num_val
            elif op in ["!=", "<>", "ne"]:
                return num_series != num_val
            elif op in [">", "gt"]:
                return num_series > num_val
            elif op in [">=", "gte"]:
                return num_series >= num_val
            elif op in ["<", "lt"]:
                return num_series < num_val
            elif op in ["<=", "lte"]:
                return num_series <= num_val

        # Text comparisons
        str_series = series.astype(str)
        str_val = str(val) if val is not None else ""

        if op in ["=", "==", "eq", "equals"]:
            return str_series.str.lower() == str_val.lower()
        elif op in ["!=", "<>", "ne"]:
            return str_series.str.lower() != str_val.lower()
        elif op in ["contains", "has"]:
            return str_series.str.contains(str_val, case=False, na=False)
        elif op in ["not_contains", "not contains", "does_not_contain"]:
            return ~str_series.str.contains(str_val, case=False, na=False)
        elif op in ["starts_with", "startswith"]:
            return str_series.str.startswith(str_val, na=False)
        elif op in ["ends_with", "endswith"]:
            return str_series.str.endswith(str_val, na=False)
        elif op in ["in", "is_in"]:
            allowed = [x.strip().lower() for x in str_val.split(",") if x.strip()]
            return str_series.str.lower().isin(allowed)
        elif op in ["not_in", "is_not_in"]:
            allowed = [x.strip().lower() for x in str_val.split(",") if x.strip()]
            return ~str_series.str.lower().isin(allowed)
        elif op in [">", "gt"]:
            return str_series > str_val
        elif op in [">=", "gte"]:
            return str_series >= str_val
        elif op in ["<", "lt"]:
            return str_series < str_val
        elif op in ["<=", "lte"]:
            return str_series <= str_val

        return pd.Series(False, index=df.index)
    except Exception:
        return pd.Series(False, index=df.index)


def evaluate_composite_conditions(
    df: pd.DataFrame,
    conditions: List[Dict[str, Any]],
    combine_with: str = "AND"
) -> pd.Series:
    """
    Evaluates a list of conditions joined by 'AND' or 'OR' logic.
    """
    if not conditions:
        return pd.Series(True, index=df.index)

    combine = str(combine_with).upper().strip()
    is_and = (combine != "OR")

    overall_mask = pd.Series(True if is_and else False, index=df.index)

    for cond in conditions:
        if not isinstance(cond, dict):
            continue
        mask = evaluate_condition_mask(df, cond)
        if is_and:
            overall_mask = overall_mask & mask
        else:
            overall_mask = overall_mask | mask

    return overall_mask
