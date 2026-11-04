import pytest
import pandas as pd
from backend.app.recipes.preprocessing.limit import LimitRecipe


def test_limit_head_exact_rows():
    recipe = LimitRecipe()
    df = pd.DataFrame({"id": range(100), "val": range(100, 200)})

    res = recipe.execute({"dataframe": df}, {"limit_type": "rows", "rows": 10, "mode": "head"})
    out_df = res["dataframe"]
    assert len(out_df) == 10
    assert list(out_df["id"]) == list(range(10))
    assert res["output_summary"]["row_count"] == 10
    assert res["output_summary"]["mode"] == "head"


def test_limit_head_forward_offset():
    recipe = LimitRecipe()
    df = pd.DataFrame({"id": range(100)})

    res = recipe.execute({"dataframe": df}, {"rows": 5, "mode": "head", "offset": 20})
    out_df = res["dataframe"]
    assert len(out_df) == 5
    assert list(out_df["id"]) == [20, 21, 22, 23, 24]

    # Offset exceeds dataset size
    res_exceed = recipe.execute({"dataframe": df}, {"rows": 5, "mode": "head", "offset": 150})
    assert len(res_exceed["dataframe"]) == 0


def test_limit_tail_exact_rows():
    recipe = LimitRecipe()
    df = pd.DataFrame({"id": range(100)})

    res = recipe.execute({"dataframe": df}, {"rows": 5, "mode": "tail", "offset": 0})
    out_df = res["dataframe"]
    assert len(out_df) == 5
    assert list(out_df["id"]) == [95, 96, 97, 98, 99]


def test_limit_tail_reverse_offset():
    """Verify reverse offset in tail mode skips rows from the bottom."""
    recipe = LimitRecipe()
    df = pd.DataFrame({"id": range(100)})

    # Take 5 rows, skipping the bottom 10 rows (rows 90..99 skipped -> returns 85..89)
    res = recipe.execute({"dataframe": df}, {"rows": 5, "mode": "tail", "offset": 10})
    out_df = res["dataframe"]
    assert len(out_df) == 5
    assert list(out_df["id"]) == [85, 86, 87, 88, 89]

    # Reverse offset skipping where requested rows exceed available remaining rows
    res_overlap = recipe.execute({"dataframe": df}, {"rows": 15, "mode": "tail", "offset": 90})
    # Remaining rows before offset 90 are rows 0..9 (10 rows)
    out_overlap = res_overlap["dataframe"]
    assert len(out_overlap) == 10
    assert list(out_overlap["id"]) == list(range(10))

    # Offset equals or exceeds length
    res_empty = recipe.execute({"dataframe": df}, {"rows": 5, "mode": "tail", "offset": 100})
    assert len(res_empty["dataframe"]) == 0


def test_limit_random_sampling():
    recipe = LimitRecipe()
    df = pd.DataFrame({"id": range(100)})

    res1 = recipe.execute({"dataframe": df}, {"rows": 20, "mode": "random", "random_state": 123})
    res2 = recipe.execute({"dataframe": df}, {"rows": 20, "mode": "random", "random_state": 123})
    assert len(res1["dataframe"]) == 20
    assert res1["dataframe"]["id"].tolist() == res2["dataframe"]["id"].tolist()


def test_limit_percentage_sampling():
    recipe = LimitRecipe()
    df = pd.DataFrame({"id": range(200)})

    # 10% of 200 rows = 20 rows
    res_10pct = recipe.execute({"dataframe": df}, {"limit_type": "percentage", "percentage": 10.0, "mode": "head"})
    assert len(res_10pct["dataframe"]) == 20
    assert res_10pct["output_summary"]["requested_percentage"] == 10.0

    # 25% of 200 rows = 50 rows
    res_25pct = recipe.execute({"dataframe": df}, {"limit_type": "percentage", "percentage": 25, "mode": "head"})
    assert len(res_25pct["dataframe"]) == 50

    # Percentage as string with % sign e.g. '15%'
    res_str = recipe.execute({"dataframe": df}, {"limit_type": "percentage", "percentage": "15%", "mode": "head"})
    assert len(res_str["dataframe"]) == 30

    # Percentage in tail mode with reverse offset
    res_tail_pct = recipe.execute(
        {"dataframe": df},
        {"limit_type": "percentage", "percentage": 10.0, "mode": "tail", "offset": 20}
    )
    # 10% = 20 rows. Ending index is 200 - 20 = 180. Starting index is 180 - 20 = 160.
    out_tail = res_tail_pct["dataframe"]
    assert len(out_tail) == 20
    assert list(out_tail["id"]) == list(range(160, 180))

    # Percentage in random mode
    res_rand_pct = recipe.execute(
        {"dataframe": df},
        {"limit_type": "percentage", "percentage": 15.0, "mode": "random", "random_state": 42}
    )
    assert len(res_rand_pct["dataframe"]) == 30


def test_limit_edge_cases():
    recipe = LimitRecipe()

    # Empty DataFrame
    empty_df = pd.DataFrame({"a": [], "b": []})
    res_empty = recipe.execute({"dataframe": empty_df}, {"rows": 10})
    assert len(res_empty["dataframe"]) == 0
    assert list(res_empty["dataframe"].columns) == ["a", "b"]

    # Small DataFrame with percentage guaranteeing at least 1 row
    small_df = pd.DataFrame({"x": [1, 2, 3]})
    res_small = recipe.execute({"dataframe": small_df}, {"limit_type": "percentage", "percentage": 10.0})
    assert len(res_small["dataframe"]) == 1

    # Backward compatibility: old format without limit_type
    compat_res = recipe.execute({"dataframe": pd.DataFrame({"x": range(50)})}, {"rows": 7})
    assert len(compat_res["dataframe"]) == 7


def test_limit_validation():
    recipe = LimitRecipe()

    # Valid config
    assert recipe.validate_config({"limit_type": "rows", "rows": 10, "mode": "head"}) == []
    assert recipe.validate_config({"limit_type": "percentage", "percentage": 25.0, "mode": "tail"}) == []

    # Invalid mode
    errs_mode = recipe.validate_config({"mode": "invalid_mode", "rows": 10})
    assert any("Invalid mode" in e for e in errs_mode)

    # Invalid percentage values
    assert any("Sampling percentage must be > 0 and <= 100" in e for e in recipe.validate_config({"limit_type": "percentage", "percentage": 0}))
    assert any("Sampling percentage must be > 0 and <= 100" in e for e in recipe.validate_config({"limit_type": "percentage", "percentage": 150}))
    assert any("valid numeric value" in e for e in recipe.validate_config({"limit_type": "percentage", "percentage": "abc"}))

    # Invalid rows
    assert any("Row count must be greater than 0" in e for e in recipe.validate_config({"rows": 0}))
    assert any("Row count must be a valid integer" in e for e in recipe.validate_config({"rows": "not_an_int"}))

    # Invalid offset
    assert any("Offset cannot be negative" in e for e in recipe.validate_config({"rows": 10, "offset": -5}))


def test_limit_to_code():
    recipe = LimitRecipe()

    code_head = recipe.to_code({"rows": 50, "mode": "head", "offset": 10})
    assert "df.iloc[10 : 10 + n]" in code_head

    code_tail = recipe.to_code({"rows": 20, "mode": "tail", "offset": 5})
    assert "end_idx = max(0, len(df) - 5)" in code_tail
    assert "start_idx = max(0, end_idx - n)" in code_tail

    code_pct = recipe.to_code({"limit_type": "percentage", "percentage": 15.0, "mode": "random", "random_state": 99})
    assert "round(len(df) * (15.0 / 100.0))" in code_pct
    assert "random_state=99" in code_pct
