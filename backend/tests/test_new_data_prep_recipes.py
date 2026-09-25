import pytest
import pandas as pd
import numpy as np
from backend.app.recipes.base.registry import recipe_registry
from backend.app.recipes.preprocessing.sql_step import SQLStepRecipe
from backend.app.recipes.preprocessing.limit import LimitRecipe
from backend.app.recipes.preprocessing.sort_rows import SortRecipe
from backend.app.recipes.preprocessing.aggregate import AggregateRecipe
from backend.app.recipes.preprocessing.combine import CombineRecipe
from backend.app.recipes.preprocessing.pivot import PivotRecipe


def test_new_recipes_registered():
    """Verify all 6 recipes are registered and export valid metadata and code."""
    for rid in ["sql_step", "limit", "sort", "aggregate", "combine", "pivot"]:
        r = recipe_registry.get(rid)
        assert r is not None
        schema = r.get_schema()
        assert schema["type"] == "object"
        assert "properties" in schema
        assert len(r.to_code({})) > 0


def test_sql_step_recipe_comprehensive():
    recipe = SQLStepRecipe()
    df = pd.DataFrame({"category": ["A", "B", "A"], "amount": [10, 20, 30], "year": [2023, 2023, 2024]})

    # 1. SELECT query
    res1 = recipe.execute({"dataframe": df}, {"query": "SELECT category, SUM(amount) AS total FROM df GROUP BY category ORDER BY total DESC"})
    assert len(res1["dataframe"]) == 2
    assert list(res1["dataframe"].columns) == ["category", "total"]

    # 2. CTE WITH query
    res2 = recipe.execute({"dataframe": df}, {"query": "WITH filtered AS (SELECT * FROM df WHERE amount > 10) SELECT category, AVG(amount) as avg_amt FROM filtered GROUP BY category"})
    assert len(res2["dataframe"]) == 2

    # 3. PIVOT query
    res3 = recipe.execute({"dataframe": df}, {"query": "PIVOT df ON category USING SUM(amount)"})
    assert "A" in res3["dataframe"].columns and "B" in res3["dataframe"].columns

    # 4. Comments and table alias 'data'
    res4 = recipe.execute({"dataframe": df}, {"query": "-- Filter data\nSELECT * FROM data WHERE year = 2023"})
    assert len(res4["dataframe"]) == 2

    # 5. DDL blocking in validation
    errs = recipe.validate_config({"query": "DROP TABLE df"})
    assert len(errs) > 0


def test_limit_recipe_comprehensive():
    recipe = LimitRecipe()
    df = pd.DataFrame({"id": range(100), "val": range(100, 200)})

    # 1. Head mode
    res1 = recipe.execute({"dataframe": df}, {"rows": 10, "mode": "head"})
    assert len(res1["dataframe"]) == 10
    assert res1["dataframe"]["id"].iloc[0] == 0

    # 2. Tail mode
    res2 = recipe.execute({"dataframe": df}, {"rows": 5, "mode": "tail"})
    assert len(res2["dataframe"]) == 5
    assert res2["dataframe"]["id"].iloc[-1] == 99

    # 3. Offset support
    res3 = recipe.execute({"dataframe": df}, {"rows": 10, "mode": "head", "offset": 20})
    assert len(res3["dataframe"]) == 10
    assert res3["dataframe"]["id"].iloc[0] == 20

    # 4. Random sample reproducibility
    res4a = recipe.execute({"dataframe": df}, {"rows": 15, "mode": "random", "random_state": 123})
    res4b = recipe.execute({"dataframe": df}, {"rows": 15, "mode": "random", "random_state": 123})
    assert (res4a["dataframe"]["id"] == res4b["dataframe"]["id"]).all()

    # 5. Safe None handling
    res5 = recipe.execute({"dataframe": df}, {"rows": None})
    assert len(res5["dataframe"]) == 100


def test_sort_recipe_comprehensive():
    recipe = SortRecipe()
    df = pd.DataFrame({
        "cat": ["B", "A", "B", "A", "C"],
        "val": [10, 20, 15, 30, np.nan]
    })

    # 1. Single string column
    res1 = recipe.execute({"dataframe": df}, {"columns": "val", "ascending": True})
    assert res1["dataframe"]["val"].iloc[0] == 10
    assert np.isnan(res1["dataframe"]["val"].iloc[-1])

    # 2. Multi-column with [True, False]
    res2 = recipe.execute({"dataframe": df}, {"columns": ["cat", "val"], "ascending": [True, False]})
    assert res2["dataframe"]["cat"].tolist()[:2] == ["A", "A"]
    assert res2["dataframe"]["val"].iloc[0] == 30

    # 3. na_position first
    res3 = recipe.execute({"dataframe": df}, {"columns": ["val"], "ascending": True, "na_position": "first"})
    assert np.isnan(res3["dataframe"]["val"].iloc[0])


def test_aggregate_recipe_comprehensive():
    recipe = AggregateRecipe()
    df = pd.DataFrame({
        "store": ["S1", "S1", "S2", "S2"],
        "dept": ["D1", "D2", "D1", "D2"],
        "sales": [100.0, 150.0, 200.0, 250.0],
        "item_name": ["apple", "banana", "orange", "grape"]
    })

    # 1. Single string group_by & list aggregations
    res1 = recipe.execute({"dataframe": df}, {
        "group_by": "store",
        "aggregations": [{"column": "sales", "func": "sum"}, {"column": "sales", "func": "mean"}]
    })
    assert len(res1["dataframe"]) == 2
    assert "sales_sum" in res1["dataframe"].columns
    assert "sales_mean" in res1["dataframe"].columns

    # 2. Dict format aggregations
    res2 = recipe.execute({"dataframe": df}, {
        "group_by": ["store", "dept"],
        "aggregations": {"sales": ["min", "max"]}
    })
    assert len(res2["dataframe"]) == 4
    assert "sales_min" in res2["dataframe"].columns

    # 3. Categorical non-numeric error on mean
    with pytest.raises(ValueError):
        recipe.execute({"dataframe": df}, {
            "group_by": "store",
            "aggregations": [{"column": "item_name", "func": "mean"}]
        })


def test_combine_recipe_comprehensive():
    recipe = CombineRecipe()
    df_a = pd.DataFrame({"id": [1, 2, 3], "name": ["alice", "bob", "charlie"], "_merge": ["old", "old", "old"]})
    df_b = pd.DataFrame({"id": [2, 3, 4], "name": ["bob", "charlie", "dave"], "salary": [50, 60, 70]})

    # 1. DAGExecutor input keys: left_dataframe and right_dataframe
    res1 = recipe.execute({"left_dataframe": df_a, "right_dataframe": df_b}, {"mode": "union", "column_alignment": "common_columns"})
    assert len(res1["dataframe"]) == 6
    assert list(res1["dataframe"].columns) == ["id", "name"]

    # 2. parent_dataframes list & all_columns
    res2 = recipe.execute({"parent_dataframes": [df_a, df_b]}, {"mode": "union", "column_alignment": "all_columns"})
    assert len(res2["dataframe"]) == 6
    assert "salary" in res2["dataframe"].columns

    # 3. union_distinct
    res3 = recipe.execute({"left": df_a[["id", "name"]], "right": df_b[["id", "name"]]}, {"mode": "union_distinct"})
    assert len(res3["dataframe"]) == 4

    # 4. intersect
    res4 = recipe.execute({"left": df_a[["id", "name"]], "right": df_b[["id", "name"]]}, {"mode": "intersect"})
    assert len(res4["dataframe"]) == 2

    # 5. except with _merge collision
    res5 = recipe.execute({"left": df_a, "right": df_b}, {"mode": "except"})
    assert len(res5["dataframe"]) == 1
    assert res5["dataframe"]["name"].iloc[0] == "alice"


def test_pivot_recipe_comprehensive():
    recipe = PivotRecipe()
    df = pd.DataFrame({
        "date": ["2023-01", "2023-01", "2023-02", "2023-02"],
        "store": ["S1", "S2", "S1", "S2"],
        "sales": [100, 200, 150, 250],
        "item_name": ["apple", "banana", "orange", "grape"]
    })

    # 1. Pivot long to wide
    res1 = recipe.execute({"dataframe": df}, {
        "mode": "pivot",
        "index": "date",
        "columns": "store",
        "values": "sales",
        "agg_func": "sum",
        "fill_value": 0
    })
    assert len(res1["dataframe"]) == 2
    assert "S1" in res1["dataframe"].columns
    assert "S2" in res1["dataframe"].columns

    # 2. Unpivot wide to long
    res2 = recipe.execute({"dataframe": res1["dataframe"]}, {
        "mode": "unpivot",
        "index": "date",
        "values": ["S1", "S2"],
        "var_name": "store_id",
        "value_name": "total_sales"
    })
    assert len(res2["dataframe"]) == 4
    assert "store_id" in res2["dataframe"].columns
    assert "total_sales" in res2["dataframe"].columns

    # 3. Non-numeric value column error on mean
    with pytest.raises(ValueError):
        recipe.execute({"dataframe": df}, {
            "mode": "pivot",
            "index": "date",
            "columns": "store",
            "values": "item_name",
            "agg_func": "mean"
        })
