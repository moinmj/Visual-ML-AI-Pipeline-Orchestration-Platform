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
        # Check if date
        try:
            dt_s = pd.to_datetime(X_tr[col], errors="coerce")
            if dt_s.notna().sum() > 0.5 * len(X_tr):
                X_tr[f"{col}_year"] = dt_s.dt.year.fillna(0).astype(int)
                X_tr[f"{col}_month"] = dt_s.dt.month.fillna(0).astype(int)
                X_tr[f"{col}_day"] = dt_s.dt.day.fillna(0).astype(int)
                X_tr = X_tr.drop(columns=[col])
                if X_te is not None and col in X_te.columns:
                    dt_te = pd.to_datetime(X_te[col], errors="coerce")
                    X_te[f"{col}_year"] = dt_te.dt.year.fillna(0).astype(int)
                    X_te[f"{col}_month"] = dt_te.dt.month.fillna(0).astype(int)
                    X_te[f"{col}_day"] = dt_te.dt.day.fillna(0).astype(int)
                    X_te = X_te.drop(columns=[col])
                continue
        except Exception:
            pass

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
