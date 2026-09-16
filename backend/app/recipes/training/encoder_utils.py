import pandas as pd
from typing import Optional, Tuple
from sklearn.preprocessing import LabelEncoder


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

