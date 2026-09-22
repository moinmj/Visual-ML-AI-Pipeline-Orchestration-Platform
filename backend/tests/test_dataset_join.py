import pytest
import pandas as pd
import numpy as np
from backend.app.recipes.base.registry import recipe_registry
from backend.app.recipes.preprocessing.dataset_join import DatasetJoinRecipe
from backend.app.engine.dag.graph import WorkflowGraph, WorkflowNode, WorkflowEdge
from backend.app.engine.execution.executor import DAGExecutor
from backend.app.templates.router import get_template


def test_dataset_join_recipe_direct_inner():
    recipe = DatasetJoinRecipe()
    
    # Left: Customers
    df_customers = pd.DataFrame({
        "customer_id": [1, 2, 3, 4],
        "name": ["Alice", "Bob", "Charlie", "David"],
        "city": ["New York", "London", "Paris", "Tokyo"]
    })
    
    # Right: Orders
    df_orders = pd.DataFrame({
        "customer_id": [2, 3, 4, 5],
        "order_id": [101, 102, 103, 104],
        "amount": [250.0, 450.0, 150.0, 800.0]
    })
    
    inputs = {
        "left_dataframe": df_customers,
        "right_dataframe": df_orders
    }
    config = {
        "join_type": "inner",
        "on": "customer_id"
    }
    
    res = recipe.execute(inputs=inputs, config=config)
    assert "dataframe" in res
    merged = res["dataframe"]
    
    # Inner join on customer_id: keys [2, 3, 4] should match -> 3 rows
    assert len(merged) == 3
    assert set(merged["customer_id"]) == {2, 3, 4}
    assert "city" in merged.columns
    assert "amount" in merged.columns
    
    metrics = res["metrics"]
    assert metrics["join_type"] == "INNER"
    assert metrics["left_rows"] == 4
    assert metrics["right_rows"] == 4
    assert metrics["merged_rows"] == 3
    assert metrics["left_match_rate_pct"] == 75.0
    assert metrics["right_match_rate_pct"] == 75.0


def test_dataset_join_recipe_left_and_outer_with_differing_keys():
    recipe = DatasetJoinRecipe()
    
    # Left uses 'cust_id'
    df_left = pd.DataFrame({
        "cust_id": [10, 20, 30],
        "tier": ["Gold", "Silver", "Bronze"],
        "score": [90, 80, 70]
    })
    
    # Right uses 'user_id'
    df_right = pd.DataFrame({
        "user_id": [20, 30, 40],
        "balance": [5000, 3000, 1000],
        "score": [85, 75, 65]  # Overlapping column
    })
    
    inputs = {
        "left_dataframe": df_left,
        "right_dataframe": df_right
    }
    
    # Left Join
    config_left = {
        "join_type": "left",
        "left_on": "cust_id",
        "right_on": "user_id",
        "suffixes": ["_left", "_right"]
    }
    res_left = recipe.execute(inputs=inputs, config=config_left)
    merged_left = res_left["dataframe"]
    assert len(merged_left) == 3
    assert "score_left" in merged_left.columns
    assert "score_right" in merged_left.columns
    assert merged_left[merged_left["cust_id"] == 10]["balance"].isna().values[0]

    # Outer Join
    config_outer = {
        "join_type": "outer",
        "left_on": "cust_id",
        "right_on": "user_id"
    }
    res_outer = recipe.execute(inputs=inputs, config=config_outer)
    merged_outer = res_outer["dataframe"]
    # Outer join keys: [10, 20, 30, 40] -> 4 rows
    assert len(merged_outer) == 4


def test_dataset_join_missing_key_validation():
    recipe = DatasetJoinRecipe()
    
    df_a = pd.DataFrame({"id": [1, 2]})
    df_b = pd.DataFrame({"other_id": [1, 2]})
    
    # 1. Config validation fails if neither on nor left_on/right_on
    errs = recipe.validate_config({"join_type": "inner"})
    assert len(errs) > 0

    # 2. Execution fails with clear KeyError when on key does not exist
    with pytest.raises(KeyError) as exc_info:
        recipe.execute(
            inputs={"left_dataframe": df_a, "right_dataframe": df_b},
            config={"join_type": "inner", "on": "non_existent_key"}
        )
    assert "Join key 'non_existent_key' was not found" in str(exc_info.value)


def test_dataset_join_in_workflow_dag_end_to_end():
    # Construct complete DAG: 2 Loaders -> Join -> Split -> Trainer -> Evaluator
    df_users = pd.DataFrame({
        "user_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "age": [25, 32, 45, 29, 50, 22, 38, 41, 55, 30],
        "churn": [0, 1, 0, 1, 0, 0, 1, 0, 1, 0]
    })
    
    df_usage = pd.DataFrame({
        "user_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "monthly_calls": [120, 20, 300, 15, 450, 90, 35, 210, 10, 180],
        "data_gb": [10.5, 1.2, 25.0, 0.5, 40.0, 8.0, 2.0, 15.0, 0.8, 12.0]
    })
    
    workflow = WorkflowGraph(
        nodes=[
            WorkflowNode(
                id="n_users",
                recipe_id="csv_loader",
                config={"dataframe": df_users}
            ),
            WorkflowNode(
                id="n_usage",
                recipe_id="csv_loader",
                config={"dataframe": df_usage}
            ),
            WorkflowNode(
                id="n_join",
                recipe_id="dataset_join",
                config={"join_type": "inner", "on": "user_id"}
            ),
            WorkflowNode(
                id="n_split",
                recipe_id="train_test_split",
                config={"target_column": "churn", "test_size": 0.3, "random_state": 42}
            ),
            WorkflowNode(
                id="n_rf",
                recipe_id="random_forest_trainer",
                config={"target_column": "churn", "n_estimators": 10, "max_depth": 4}
            ),
            WorkflowNode(
                id="n_eval",
                recipe_id="model_evaluator",
                config={}
            )
        ],
        edges=[
            WorkflowEdge(source="n_users", target="n_join", target_handle="left"),
            WorkflowEdge(source="n_usage", target="n_join", target_handle="right"),
            WorkflowEdge(source="n_join", target="n_split"),
            WorkflowEdge(source="n_split", target="n_rf"),
            WorkflowEdge(source="n_split", target="n_eval"),
            WorkflowEdge(source="n_rf", target="n_eval")
        ]
    )
    
    res = DAGExecutor.execute_workflow(
        execution_id="exec_test_join_dag",
        workflow=workflow
    )
    
    assert res.status == "SUCCESS"
    assert res.final_metrics is not None
    assert "accuracy" in res.final_metrics or "roc_auc" in res.final_metrics or "f1_score" in res.final_metrics
    
    # Check that join node executed and reported join metrics
    join_node_res = next(r for r in res.node_results if r.node_id == "n_join")
    assert join_node_res.status == "SUCCESS"


def test_dataset_join_graph_validation_diagnostics():
    # Only 1 parent connected to dataset_join
    workflow = WorkflowGraph(
        nodes=[
            WorkflowNode(id="n_csv", recipe_id="csv_loader", config={"dataset_id": "test_ds"}),
            WorkflowNode(id="n_join", recipe_id="dataset_join", config={"join_type": "inner", "on": "id"})
        ],
        edges=[
            WorkflowEdge(source="n_csv", target="n_join")
        ]
    )
    
    diagnostics = workflow.get_diagnostics()
    # Should flag incomplete connections warning
    assert any("Incomplete Connections" in w["message"] for w in diagnostics["warnings"])


@pytest.mark.asyncio
async def test_template_multi_dataset_join():
    tpl = await get_template("multi_dataset_join")
    assert tpl["id"] == "multi_dataset_join"
    assert tpl["node_count"] == 6
    assert any(n["recipe_id"] == "dataset_join" for n in tpl["dag"]["nodes"])


@pytest.mark.asyncio
async def test_autowire_with_dataset_join():
    from backend.app.recommendation.router import autowire_nodes, AutoWireRequest

    req = AutoWireRequest(
        nodes=[
            {"id": "node_loader_a", "recipe_id": "csv_loader", "label": "Customers", "position": {"x": 40, "y": 50}},
            {"id": "node_loader_b", "recipe_id": "csv_loader", "label": "Transactions", "position": {"x": 40, "y": 200}},
            {"id": "node_join", "recipe_id": "dataset_join", "label": "Join Customers and Transactions", "position": {"x": 300, "y": 120}},
            {"id": "node_split", "recipe_id": "train_test_split", "label": "Splitter", "position": {"x": 550, "y": 120}},
            {"id": "node_model", "recipe_id": "xgboost_trainer", "label": "XGBoost", "position": {"x": 800, "y": 120}},
            {"id": "node_eval", "recipe_id": "model_evaluator", "label": "Evaluator", "position": {"x": 1050, "y": 120}},
        ]
    )

    res = await autowire_nodes(req)
    assert res["status"] == "AUTOWIRED"
    edges = res["edges"]

    # Verify both loader_a and loader_b connect to node_join with handles
    edge_a = next((e for e in edges if e["source"] == "node_loader_a" and e["target"] == "node_join"), None)
    assert edge_a is not None
    assert edge_a["target_handle"] == "left"

    edge_b = next((e for e in edges if e["source"] == "node_loader_b" and e["target"] == "node_join"), None)
    assert edge_b is not None
    assert edge_b["target_handle"] == "right"

    # Verify loaders do NOT connect to each other!
    assert not any(e["source"] == "node_loader_a" and e["target"] == "node_loader_b" for e in edges)
    assert not any(e["source"] == "node_loader_b" and e["target"] == "node_loader_a" for e in edges)

    # Verify join connects downstream to splitter
    assert any(e["source"] == "node_join" and e["target"] == "node_split" for e in edges)

    # Verify splitter connects to model and evaluator
    assert any(e["source"] == "node_split" and e["target"] == "node_model" for e in edges)
    assert any(e["source"] == "node_split" and e["target"] == "node_eval" for e in edges)
