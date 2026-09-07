import pytest
import pandas as pd
import numpy as np
from backend.app.profiling.profiler import DataProfiler


def test_profiler_basic_stats():
    df = pd.DataFrame({
        "age": [20, 30, 40, np.nan],
        "category": ["A", "B", "A", "B"],
        "is_active": [True, False, True, True]
    })

    profile = DataProfiler.profile_dataframe(df)

    assert profile["row_count"] == 4
    assert profile["column_count"] == 3
    assert profile["total_missing_cells"] == 1
    assert profile["columns"]["age"]["inferred_type"] == "numeric"
    assert profile["columns"]["age"]["null_count"] == 1
    assert profile["columns"]["category"]["inferred_type"] == "categorical"
    assert profile["columns"]["is_active"]["inferred_type"] == "boolean"
    assert 0 <= profile["quality_score"] <= 100


def test_profiler_empty_dataframe():
    df = pd.DataFrame()
    profile = DataProfiler.profile_dataframe(df)
    assert profile["row_count"] == 0
    assert profile["quality_score"] == 100.0
