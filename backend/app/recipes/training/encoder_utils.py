import pandas as pd
import numpy as np
from typing import Optional, Tuple, Any, Dict, List
from sklearn.preprocessing import LabelEncoder


def _is_valid_calendar_datetime_series(series: Optional[pd.Series]) -> Tuple[bool, Optional[pd.Series]]:
    """
    Production-grade validation for temporal calendar series.
    Detects native datetime64 dtypes, ISO string dates, and valid epoch timestamps.
    Rejects degenerate numeric nanosecond Epoch artifacts (e.g. small integers mapped to 1970-01-01).

    Rules:
    - Native datetime64 columns: always accepted.
    - Numeric columns: ONLY accepted when values >= 1e8 (valid Unix timestamps in seconds).
      Year integers like 2010..2025 are < 1e8 → REJECTED.
    - String columns: parsed, then checked that distinct strings don't all collapse to
      a single constant date (the epoch collapse guard).
    """
    if series is None or len(series) == 0:
        return False, None

    # Case 1: Native datetime64 dtype
    if pd.api.types.is_datetime64_any_dtype(series):
        return True, series

    # Case 2: Numeric dtype (integers or floats)
    if pd.api.types.is_numeric_dtype(series):
        s_clean = series.dropna()
        if s_clean.empty:
            return False, None
        v_min = float(s_clean.min())
        # Only values >= 1e8 can represent valid Unix timestamps in seconds (1.6e9 → year 2020+)
        # Year integers 1970..2099 are all < 1e8 → rejected to prevent epoch collapse
        if v_min < 1e8:
            return False, None
        try:
            parsed = pd.to_datetime(s_clean, unit="s", errors="coerce")
            if parsed.notna().sum() > 0.5 * len(s_clean):
                return True, parsed
        except Exception:
            return False, None

    # Case 3: Object / String / Categorical series
    try:
        s_str = series.astype(str).str.strip()

        # Priority 1: Try strict ISO format (YYYY-MM-DD) — unambiguous, most common in data pipelines
        # This must come BEFORE dayfirst=True, otherwise '2010-02-05' becomes May 2nd (wrong!)
        parsed = pd.to_datetime(s_str, format="%Y-%m-%d", errors="coerce")
        if parsed.notna().sum() <= 0.5 * len(s_str):
            # Priority 2: Try mixed format (handles ISO with time component, various separators)
            parsed = pd.to_datetime(s_str, format="mixed", errors="coerce")
        if parsed.notna().sum() <= 0.5 * len(s_str):
            # Priority 3: Fully automatic inference (no dayfirst to avoid ambiguity)
            parsed = pd.to_datetime(s_str, errors="coerce")

        valid_count = parsed.notna().sum()
        if valid_count > 0.5 * len(s_str):
            # Production Guard: Detect degenerate conversions where distinct input strings
            # all collapse to one constant date (e.g. small integers → "1970-01-01")
            formatted_dates = parsed.dt.strftime("%Y-%m-%d").dropna()
            if len(s_str) > 1 and s_str.nunique() > 1 and formatted_dates.nunique() == 1:
                return False, None
            return True, parsed
    except Exception:
        pass

    return False, None


def extract_test_dates_from_column(series: pd.Series) -> Optional[List[str]]:
    """
    Extract a list of ISO date strings from a column that might be:
    - A native datetime64 series
    - A string column containing ISO dates
    - A Unix timestamp series (>= 1e8)

    Returns None if the column is not a valid calendar date column
    (e.g. year integers 2010..2025, dayofweek integers 0..6, etc.)
    Never returns '1970-01-01' artifacts from nanosecond epoch conversion.
    """
    is_valid, parsed = _is_valid_calendar_datetime_series(series)
    if not is_valid or parsed is None:
        return None
    raw = series.reset_index(drop=True)
    return [
        v.strftime("%Y-%m-%d") if pd.notna(v) else str(raw.iloc[i])
        for i, v in enumerate(parsed)
    ]


def find_date_column(df: pd.DataFrame, exclude_cols: Optional[List[str]] = None) -> Optional[str]:
    """
    Find the most likely real-date column in a dataframe.
    Priority: datetime64 dtype > strict keyword match ("date", "timestamp", "period", "ds") > content validation.
    Explicitly skips integer-only columns like Date_year / Date_dayofweek.

    Args:
        df: Input dataframe.
        exclude_cols: Column names to skip (e.g. the target column).

    Returns:
        Column name of the best date column, or None.
    """
    exclude = set(exclude_cols or [])
    # Priority 1: native datetime64 columns
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c

    # Priority 2: strict keyword match (exact, prefix, or suffix) — avoids "Weekly_Sales_lag_1"
    strict_kws = ["date", "timestamp", "period", "ds"]
    for c in df.columns:
        if c in exclude:
            continue
        cl = c.lower().strip()
        if any(cl == kw or cl.startswith(kw + "_") or cl.endswith("_" + kw) for kw in strict_kws):
            is_v, _ = _is_valid_calendar_datetime_series(df[c])
            if is_v:
                return c

    # Priority 3: any non-numeric column that passes the calendar datetime validator
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            continue  # skip integer/float columns — year integers, dayofweek, etc.
        is_v, _ = _is_valid_calendar_datetime_series(df[c])
        if is_v:
            return c

    return None


def safe_prepare_training_data(
    X_train: pd.DataFrame,
    X_test: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Safely encodes all non-numeric features (dates, text, IDs, high & low-cardinality categoricals)
    without blowing up memory. Low cardinality (<=30 unique) uses one-hot dummy encoding;
    high cardinality (>30 unique or date/ID strings) uses robust label/integer encoding.
    """
    X_tr = X_train.copy()
    X_te = X_test.copy() if X_test is not None else None

    non_numeric = [c for c in X_tr.columns if not pd.api.types.is_numeric_dtype(X_tr[c])]

    for col in non_numeric:
        is_date_col_name = any(kw in col.lower() for kw in ["date", "time", "year", "period", "timestamp", "ds", "month"])
        # Check if date by name or content
        try:
            dt_s = pd.to_datetime(X_tr[col], errors="coerce", format="mixed")
            if dt_s.notna().sum() > 0.3 * len(X_tr) or is_date_col_name:
                dt_s_valid = dt_s.fillna(pd.Timestamp("2020-01-01"))
                X_tr[f"{col}_year"] = dt_s_valid.dt.year.astype(int)
                X_tr[f"{col}_month"] = dt_s_valid.dt.month.astype(int)
                X_tr[f"{col}_day"] = dt_s_valid.dt.day.astype(int)
                X_tr[f"{col}_dayofweek"] = dt_s_valid.dt.dayofweek.astype(int)
                X_tr = X_tr.drop(columns=[col])

                if X_te is not None and col in X_te.columns:
                    dt_te = pd.to_datetime(X_te[col], errors="coerce", format="mixed").fillna(pd.Timestamp("2020-01-01"))
                    X_te[f"{col}_year"] = dt_te.dt.year.astype(int)
                    X_te[f"{col}_month"] = dt_te.dt.month.astype(int)
                    X_te[f"{col}_day"] = dt_te.dt.day.astype(int)
                    X_te[f"{col}_dayofweek"] = dt_te.dt.dayofweek.astype(int)
                    X_te = X_te.drop(columns=[col])
                continue
        except Exception:
            if is_date_col_name:
                X_tr = X_tr.drop(columns=[col])
                if X_te is not None and col in X_te.columns:
                    X_te = X_te.drop(columns=[col])
                continue

        nunique = X_tr[col].nunique()
        if nunique <= 30 and nunique < (len(X_tr) * 0.3):
            dummies_tr = pd.get_dummies(X_tr[[col]], columns=[col], drop_first=False, dtype=int)
            X_tr = pd.concat([X_tr.drop(columns=[col]), dummies_tr], axis=1)
            if X_te is not None and col in X_te.columns:
                dummies_te = pd.get_dummies(X_te[[col]], columns=[col], drop_first=False, dtype=int)
                dummies_te = dummies_te.reindex(columns=dummies_tr.columns, fill_value=0)
                X_te = pd.concat([X_te.drop(columns=[col]), dummies_te], axis=1)
        else:
            le = LabelEncoder()
            X_tr[col] = le.fit_transform(X_tr[col].astype(str))
            if X_te is not None and col in X_te.columns:
                # Handle unseen labels gracefully
                mapping = {v: i for i, v in enumerate(le.classes_)}
                X_te[col] = X_te[col].astype(str).map(mapping).fillna(-1).astype(int)

    if X_te is not None:
        # Align columns
        X_te = X_te.reindex(columns=X_tr.columns, fill_value=0)

    return X_tr, X_te


def extract_train_test_data(inputs: dict) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series], Optional[pd.DataFrame], Optional[pd.Series]]:
    """
    Extracts (X_train, y_train, X_test, y_test) from inputs dictionary,
    supporting both flat keys ('X_train', 'y_train', etc.) and nested dicts
    ('train_data' -> {'X_train', 'y_train'}, 'test_data' -> {'X_test', 'y_test'}).
    """
    X_train = inputs.get("X_train")
    y_train = inputs.get("y_train")
    X_test = inputs.get("X_test")
    y_test = inputs.get("y_test")

    if (X_train is None or y_train is None) and "train_data" in inputs and isinstance(inputs["train_data"], dict):
        X_train = inputs["train_data"].get("X_train", X_train)
        y_train = inputs["train_data"].get("y_train", y_train)

    if (X_test is None or y_test is None) and "test_data" in inputs and isinstance(inputs["test_data"], dict):
        X_test = inputs["test_data"].get("X_test", X_test)
        y_test = inputs["test_data"].get("y_test", y_test)

    return X_train, y_train, X_test, y_test


def pass_through_metadata(inputs: dict, output: dict, context: Optional[Any] = None) -> dict:
    """
    Preserves and passes through pipeline metadata (such as test_dates, date_column_name,
    split_mode, time_column) from inputs/context to the output dictionary.
    This ensures downstream recipes like ModelEvaluator receive explicit date sidecars
    without having to rely on heuristic column scanning or guessing.
    """
    keys_to_pass = ["test_dates", "date_column_name", "split_mode", "time_column"]
    ctx_dict = context if isinstance(context, dict) else {}

    for key in keys_to_pass:
        val = inputs.get(key)
        if val is None and isinstance(inputs.get("test_data"), dict):
            val = inputs["test_data"].get(key)
        if val is None and isinstance(inputs.get("train_data"), dict):
            val = inputs["train_data"].get(key)
        if val is None:
            val = ctx_dict.get(key)

        if val is not None and key not in output:
            output[key] = val

    return output
