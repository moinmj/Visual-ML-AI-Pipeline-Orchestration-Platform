import pytest
import pandas as pd
import numpy as np
from backend.app.engine.dag.graph import WorkflowGraph, WorkflowNode, WorkflowEdge
from backend.app.engine.execution.executor import DAGExecutor
from backend.app.recipes.preprocessing.sql_step import SQLStepRecipe
from backend.app.recipes.preprocessing.limit import LimitRecipe
from backend.app.recipes.preprocessing.sort_rows import SortRecipe
from backend.app.recipes.preprocessing.aggregate import AggregateRecipe
from backend.app.recipes.preprocessing.combine import CombineRecipe
from backend.app.recipes.preprocessing.pivot import PivotRecipe


# =====================================================================
# SCENARIO 1: ISOLATED SINGLE-NODE EXECUTION ACROSS ALL 6 NEW RECIPES
# =====================================================================

def test_scenario_1_single_node_sql_step():
    """Test SQLStepRecipe with various queries, CTEs, filters, and aggregations."""
    recipe = SQLStepRecipe()
    df = pd.DataFrame({
        "employee_id": [1, 2, 3, 4, 5],
        "department": ["Engineering", "Engineering", "Marketing", "Marketing", "Sales"],
        "salary": [90000, 110000, 75000, 85000, 95000],
        "performance_score": [4.5, 4.8, 3.9, 4.2, 4.0]
    })

    # Test 1A: Simple SELECT with filter and order
    res = recipe.execute({"dataframe": df}, {"query": "SELECT employee_id, salary FROM df WHERE salary > 80000 ORDER BY salary DESC"})
    assert len(res["dataframe"]) == 4
    assert list(res["dataframe"].columns) == ["employee_id", "salary"]
    assert res["dataframe"]["salary"].iloc[0] == 110000

    # Test 1B: Aggregate query with GROUP BY and HAVING
    res_agg = recipe.execute({"dataframe": df}, {
        "query": "SELECT department, AVG(salary) as avg_sal, COUNT(*) as cnt FROM df GROUP BY department HAVING cnt > 1"
    })
    assert len(res_agg["dataframe"]) == 2
    assert "avg_sal" in res_agg["dataframe"].columns

    # Test 1C: CTE (Common Table Expression)
    res_cte = recipe.execute({"dataframe": df}, {
        "query": "WITH high_perf AS (SELECT * FROM data WHERE performance_score >= 4.0) SELECT department, MAX(salary) as max_s FROM high_perf GROUP BY department"
    })
    assert len(res_cte["dataframe"]) == 3


def test_scenario_1_single_node_limit():
    """Test LimitRecipe across head, tail, offset, and random modes."""
    recipe = LimitRecipe()
    df = pd.DataFrame({"rank": list(range(1, 101)), "score": [x * 10 for x in range(1, 101)]})

    # Head mode
    res_head = recipe.execute({"dataframe": df}, {"rows": 5, "mode": "head"})
    assert len(res_head["dataframe"]) == 5
    assert res_head["dataframe"]["rank"].iloc[-1] == 5

    # Tail mode
    res_tail = recipe.execute({"dataframe": df}, {"rows": 5, "mode": "tail"})
    assert len(res_tail["dataframe"]) == 5
    assert res_tail["dataframe"]["rank"].iloc[0] == 96

    # Offset mode
    res_offset = recipe.execute({"dataframe": df}, {"rows": 10, "mode": "head", "offset": 50})
    assert len(res_offset["dataframe"]) == 10
    assert res_offset["dataframe"]["rank"].iloc[0] == 51

    # Random mode
    res_rand = recipe.execute({"dataframe": df}, {"rows": 8, "mode": "random", "random_state": 42})
    assert len(res_rand["dataframe"]) == 8


def test_scenario_1_single_node_sort():
    """Test SortRecipe with single, multi-column, mixed directions, and null placement."""
    recipe = SortRecipe()
    df = pd.DataFrame({
        "country": ["USA", "Germany", "USA", "Germany", "France"],
        "city": ["Austin", "Berlin", "Boston", "Munich", "Paris"],
        "metric": [10.5, 45.2, np.nan, 32.1, 15.0]
    })

    # Multi-column sort: country asc, metric desc
    res = recipe.execute({"dataframe": df}, {
        "columns": ["country", "metric"],
        "ascending": [True, False],
        "na_position": "last"
    })
    assert res["dataframe"]["country"].iloc[0] == "France"
    germany_rows = res["dataframe"][res["dataframe"]["country"] == "Germany"]
    assert germany_rows["metric"].iloc[0] == 45.2
    assert germany_rows["metric"].iloc[1] == 32.1

    # Nulls first
    res_nulls_first = recipe.execute({"dataframe": df}, {
        "columns": ["metric"],
        "ascending": True,
        "na_position": "first"
    })
    assert np.isnan(res_nulls_first["dataframe"]["metric"].iloc[0])


def test_scenario_1_single_node_aggregate():
    """Test AggregateRecipe across all supported aggregation functions."""
    recipe = AggregateRecipe()
    df = pd.DataFrame({
        "group": ["A", "A", "A", "B", "B", "C"],
        "val_int": [10, 20, 30, 40, 50, 60],
        "val_float": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
    })

    res = recipe.execute({"dataframe": df}, {
        "group_by": ["group"],
        "aggregations": [
            {"column": "val_int", "func": "sum"},
            {"column": "val_int", "func": "mean"},
            {"column": "val_int", "func": "min"},
            {"column": "val_int", "func": "max"},
            {"column": "val_int", "func": "count"},
            {"column": "val_float", "func": "std"}
        ]
    })
    out = res["dataframe"]
    assert len(out) == 3
    assert "val_int_sum" in out.columns
    assert "val_int_mean" in out.columns
    assert "val_int_min" in out.columns
    assert "val_int_max" in out.columns
    assert "val_int_count" in out.columns
    assert "val_float_std" in out.columns
    assert out[out["group"] == "A"]["val_int_sum"].iloc[0] == 60


def test_scenario_1_single_node_combine():
    """Test CombineRecipe across union, union_distinct, intersect, and except."""
    recipe = CombineRecipe()
    df1 = pd.DataFrame({"id": [1, 2, 3], "value": ["a", "b", "c"]})
    df2 = pd.DataFrame({"id": [2, 3, 4], "value": ["b", "c", "d"]})

    # Union
    res_union = recipe.execute({"left_dataframe": df1, "right_dataframe": df2}, {"mode": "union"})
    assert len(res_union["dataframe"]) == 6

    # Union Distinct
    res_distinct = recipe.execute({"left_dataframe": df1, "right_dataframe": df2}, {"mode": "union_distinct"})
    assert len(res_distinct["dataframe"]) == 4

    # Intersect
    res_inter = recipe.execute({"left_dataframe": df1, "right_dataframe": df2}, {"mode": "intersect"})
    assert len(res_inter["dataframe"]) == 2
    assert set(res_inter["dataframe"]["id"]) == {2, 3}

    # Except
    res_exc = recipe.execute({"left_dataframe": df1, "right_dataframe": df2}, {"mode": "except"})
    assert len(res_exc["dataframe"]) == 1
    assert res_exc["dataframe"]["id"].iloc[0] == 1


def test_scenario_1_single_node_pivot_and_unpivot():
    """Test PivotRecipe for both pivot (wide) and unpivot (melt) operations."""
    recipe = PivotRecipe()

    # Pivot mode
    df_long = pd.DataFrame({
        "date": ["2023-01", "2023-01", "2023-02", "2023-02"],
        "store": ["Store_A", "Store_B", "Store_A", "Store_B"],
        "sales": [100, 200, 150, 250]
    })
    res_pivot = recipe.execute({"dataframe": df_long}, {
        "mode": "pivot",
        "index": "date",
        "columns": "store",
        "values": "sales",
        "agg_func": "sum"
    })
    assert len(res_pivot["dataframe"]) == 2
    assert "Store_A" in res_pivot["dataframe"].columns
    assert "Store_B" in res_pivot["dataframe"].columns

    # Unpivot mode
    df_wide = pd.DataFrame({
        "item": ["Shirt", "Pants"],
        "Q1_sales": [10, 20],
        "Q2_sales": [15, 25]
    })
    res_melt = recipe.execute({"dataframe": df_wide}, {
        "mode": "unpivot",
        "id_vars": ["item"],
        "value_vars": ["Q1_sales", "Q2_sales"],
        "var_name": "quarter",
        "value_name": "units"
    })
    assert len(res_melt["dataframe"]) == 4
    assert set(res_melt["dataframe"].columns) == {"item", "quarter", "units"}


# =====================================================================
# SCENARIO 2: SEQUENTIAL MULTI-NODE DATA PREP PIPELINE
# csv_loader -> sql_step -> aggregate -> sort -> limit
# =====================================================================

def test_scenario_2_chained_sequential_pipeline():
    """Verify end-to-end execution of a 5-node data transformation chain."""
    raw_data = pd.DataFrame({
        "order_id": range(1, 21),
        "region": ["North", "South", "East", "West"] * 5,
        "amount": [15.0, 25.0, 50.0, 100.0, 200.0, 30.0, 45.0, 60.0, 80.0, 120.0,
                   150.0, 180.0, 210.0, 240.0, 270.0, 300.0, 330.0, 360.0, 390.0, 420.0],
        "status": ["Completed", "Pending", "Completed", "Cancelled"] * 5
    })

    workflow = WorkflowGraph(
        nodes=[
            WorkflowNode(id="n_raw", recipe_id="csv_loader", config={"dataframe": raw_data}),
            WorkflowNode(id="n_sql", recipe_id="sql_step", config={
                "query": "SELECT region, amount, status FROM df WHERE status = 'Completed' AND amount >= 50"
            }),
            WorkflowNode(id="n_agg", recipe_id="aggregate", config={
                "group_by": "region",
                "aggregations": [
                    {"column": "amount", "func": "sum"},
                    {"column": "amount", "func": "mean"},
                    {"column": "amount", "func": "count"}
                ]
            }),
            WorkflowNode(id="n_sort", recipe_id="sort", config={
                "columns": "amount_sum",
                "ascending": False
            }),
            WorkflowNode(id="n_limit", recipe_id="limit", config={
                "rows": 3,
                "mode": "head"
            })
        ],
        edges=[
            WorkflowEdge(source="n_raw", target="n_sql"),
            WorkflowEdge(source="n_sql", target="n_agg"),
            WorkflowEdge(source="n_agg", target="n_sort"),
            WorkflowEdge(source="n_sort", target="n_limit")
        ]
    )

    res = DAGExecutor.execute_workflow(
        execution_id="exec_test_chained_pipeline",
        workflow=workflow
    )

    assert res.status == "SUCCESS"
    assert len(res.node_results) == 5

    # Check the final output from n_limit in step_snapshots
    assert res.step_snapshots["n_limit"]["row_count"] <= 3
    assert "amount_sum" in res.step_snapshots["n_limit"]["columns"]


# =====================================================================
# SCENARIO 3: MULTI-PARENT BRANCHING & MERGING WITH COMBINE INTO ML
# Branch 1 (raw_a -> sql_step)
# Branch 2 (raw_b -> limit)
# Merge -> combine (union_distinct) -> train_test_split -> random_forest_trainer -> model_evaluator
# =====================================================================

def test_scenario_3_multi_parent_branching_combine_into_ml():
    """Verify multi-parent DAG where two parallel branches merge via combine and train a model."""
    df_cohort_1 = pd.DataFrame({
        "age": [25, 30, 45, 50, 22, 38, 41, 55, 60, 28],
        "income": [50000, 65000, 95000, 110000, 45000, 75000, 80000, 120000, 130000, 58000],
        "churn": [0, 0, 1, 1, 0, 0, 1, 1, 1, 0]
    })
    df_cohort_2 = pd.DataFrame({
        "age": [24, 31, 46, 52, 23, 39, 42, 56, 61, 29],
        "income": [49000, 67000, 93000, 115000, 44000, 77000, 81000, 118000, 128000, 60000],
        "churn": [0, 0, 1, 1, 0, 0, 1, 1, 1, 0]
    })

    workflow = WorkflowGraph(
        nodes=[
            WorkflowNode(id="n_cohort1", recipe_id="csv_loader", config={"dataframe": df_cohort_1}),
            WorkflowNode(id="n_cohort2", recipe_id="csv_loader", config={"dataframe": df_cohort_2}),
            WorkflowNode(id="n_sql_filter", recipe_id="sql_step", config={
                "query": "SELECT age, income, churn FROM df WHERE income >= 45000"
            }),
            WorkflowNode(id="n_limit_sample", recipe_id="limit", config={
                "rows": 8,
                "mode": "head"
            }),
            WorkflowNode(id="n_combine", recipe_id="combine", config={
                "mode": "union"
            }),
            WorkflowNode(id="n_split", recipe_id="train_test_split", config={
                "target_column": "churn",
                "test_size": 0.3,
                "random_state": 42
            }),
            WorkflowNode(id="n_rf", recipe_id="random_forest_trainer", config={
                "target_column": "churn",
                "n_estimators": 10,
                "max_depth": 3
            }),
            WorkflowNode(id="n_eval", recipe_id="model_evaluator", config={})
        ],
        edges=[
            WorkflowEdge(source="n_cohort1", target="n_sql_filter"),
            WorkflowEdge(source="n_cohort2", target="n_limit_sample"),
            WorkflowEdge(source="n_sql_filter", target="n_combine", target_handle="left"),
            WorkflowEdge(source="n_limit_sample", target="n_combine", target_handle="right"),
            WorkflowEdge(source="n_combine", target="n_split"),
            WorkflowEdge(source="n_split", target="n_rf"),
            WorkflowEdge(source="n_split", target="n_eval"),
            WorkflowEdge(source="n_rf", target="n_eval")
        ]
    )

    res = DAGExecutor.execute_workflow(
        execution_id="exec_test_combine_branching",
        workflow=workflow
    )

    assert res.status == "SUCCESS"
    assert res.final_metrics is not None
    combine_node_res = next(r for r in res.node_results if r.node_id == "n_combine")
    assert combine_node_res.status == "SUCCESS"
    assert res.step_snapshots["n_combine"]["row_count"] == 18  # 10 from cohort1 + 8 from cohort2
    assert "churn" in res.step_snapshots["n_combine"]["columns"]


# =====================================================================
# SCENARIO 4: RESHAPING PIPELINE WITH PIVOT AND SORT
# loader -> pivot -> sort -> limit
# =====================================================================

def test_scenario_4_reshaping_pivot_in_dag():
    """Verify pivot recipe inside DAG execution pipeline."""
    df_sales = pd.DataFrame({
        "department": ["Electronics", "Electronics", "Clothing", "Clothing", "Home", "Home"],
        "quarter": ["Q1", "Q2", "Q1", "Q2", "Q1", "Q2"],
        "revenue": [50000, 65000, 20000, 28000, 35000, 42000]
    })

    workflow = WorkflowGraph(
        nodes=[
            WorkflowNode(id="n_sales", recipe_id="csv_loader", config={"dataframe": df_sales}),
            WorkflowNode(id="n_pivot", recipe_id="pivot", config={
                "mode": "pivot",
                "index": "department",
                "columns": "quarter",
                "values": "revenue",
                "agg_func": "sum"
            }),
            WorkflowNode(id="n_sort", recipe_id="sort", config={
                "columns": "Q1",
                "ascending": False
            }),
            WorkflowNode(id="n_limit", recipe_id="limit", config={
                "rows": 2,
                "mode": "head"
            })
        ],
        edges=[
            WorkflowEdge(source="n_sales", target="n_pivot"),
            WorkflowEdge(source="n_pivot", target="n_sort"),
            WorkflowEdge(source="n_sort", target="n_limit")
        ]
    )

    res = DAGExecutor.execute_workflow(
        execution_id="exec_test_pivot_pipeline",
        workflow=workflow
    )

    assert res.status == "SUCCESS"
    limit_res = next(r for r in res.node_results if r.node_id == "n_limit")
    assert limit_res.status == "SUCCESS"
    assert res.step_snapshots["n_limit"]["row_count"] == 2
    assert "Q1" in res.step_snapshots["n_limit"]["columns"]
    assert "Q2" in res.step_snapshots["n_limit"]["columns"]


# =====================================================================
# SCENARIO 5: EDGE CASES, EMPTY DATAFRAMES, AND GRACEFUL ERROR HANDLING
# =====================================================================

def test_scenario_5_empty_dataframe_handling():
    """Verify all 6 recipes gracefully handle empty DataFrames without unhandled crashes."""
    empty_df = pd.DataFrame(columns=["id", "val", "cat"])

    # 1. SQL Step on empty DataFrame
    sql_rec = SQLStepRecipe()
    res_sql = sql_rec.execute({"dataframe": empty_df}, {"query": "SELECT * FROM df"})
    assert len(res_sql["dataframe"]) == 0

    # 2. Limit on empty DataFrame
    lim_rec = LimitRecipe()
    res_lim = lim_rec.execute({"dataframe": empty_df}, {"rows": 10})
    assert len(res_lim["dataframe"]) == 0

    # 3. Sort on empty DataFrame
    sort_rec = SortRecipe()
    res_sort = sort_rec.execute({"dataframe": empty_df}, {"columns": "val"})
    assert len(res_sort["dataframe"]) == 0

    # 4. Aggregate on empty DataFrame
    agg_rec = AggregateRecipe()
    res_agg = agg_rec.execute({"dataframe": empty_df}, {
        "group_by": "cat",
        "aggregations": [{"column": "val", "func": "count"}]
    })
    assert len(res_agg["dataframe"]) == 0

    # 5. Combine on empty DataFrames
    comb_rec = CombineRecipe()
    res_comb = comb_rec.execute({"left_dataframe": empty_df, "right_dataframe": empty_df}, {"mode": "union"})
    assert len(res_comb["dataframe"]) == 0

    # 6. Pivot on empty DataFrame
    piv_rec = PivotRecipe()
    res_piv = piv_rec.execute({"dataframe": empty_df}, {
        "mode": "pivot",
        "index": "id",
        "columns": "cat",
        "values": "val"
    })
    assert len(res_piv["dataframe"]) == 0
