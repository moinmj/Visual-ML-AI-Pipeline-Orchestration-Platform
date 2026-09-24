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


def test_dataset_join_databricks_conditions_selection_and_renaming():
    recipe = DatasetJoinRecipe()

    df_sales = pd.DataFrame({
        "Store": [1, 1, 2, 3],
        "Date": ["2012-01-01", "2012-01-08", "2012-01-01", "2012-01-01"],
        "Weekly_Sales": [24000, 26000, 15000, 18000],
        "Temperature": [45.0, 47.0, 50.0, 52.0],
        "Fuel_Price": [3.5, 3.6, 3.5, 3.7]
    })

    df_weather = pd.DataFrame({
        "Store_ID": [1, 1, 2, 4],
        "Date_Val": ["2012-01-01", "2012-01-08", "2012-01-01", "2012-01-01"],
        "CPI": [211.0, 211.5, 190.0, 220.0],
        "Unemployment": [8.1, 8.0, 7.5, 6.9],
        "Extra_Col_To_Drop": [1, 2, 3, 4]
    })

    config = {
        "join_type": "inner",
        # 1. Composite Databricks conditions: Store = Store_ID AND Date = Date_Val
        "conditions": [
            {"left": "Store", "right": "Store_ID"},
            {"left": "Date", "right": "Date_Val"}
        ],
        # 2. Databricks Column Selection: Keep only Weekly_Sales from left, and CPI + Unemployment from right
        "selected_columns_left": ["Weekly_Sales"],
        "selected_columns_right": ["CPI", "Unemployment"],
        # 3. Databricks Inline Renaming
        "rename_columns_left": {"Weekly_Sales": "target_sales"},
        "rename_columns_right": {"CPI": "consumer_price_index"}
    }

    res = recipe.execute(
        inputs={"left_dataframe": df_sales, "right_dataframe": df_weather},
        config=config
    )

    df_res = res["dataframe"]
    # Verify Store 1 (2 dates) + Store 2 (1 date) = 3 rows matched
    assert len(df_res) == 3
    # Check that selected and renamed columns are present
    assert "target_sales" in df_res.columns
    assert "consumer_price_index" in df_res.columns
    assert "Unemployment" in df_res.columns
    # Check that unselected columns were omitted (e.g. Fuel_Price, Temperature, Extra_Col_To_Drop)
    assert "Fuel_Price" not in df_res.columns
    assert "Temperature" not in df_res.columns
    assert "Extra_Col_To_Drop" not in df_res.columns


def test_dataset_join_split_join_outputs():
    recipe = DatasetJoinRecipe()

    df_left = pd.DataFrame({"id": [1, 2, 3], "val_a": ["A1", "A2", "A3"]})
    df_right = pd.DataFrame({"id": [2, 3, 4], "val_b": ["B2", "B3", "B4"]})

    config = {
        "join_type": "split",
        "on": "id"
    }

    res = recipe.execute(
        inputs={"left_dataframe": df_left, "right_dataframe": df_right},
        config=config
    )

    assert "matched_dataframe" in res
    assert "left_unmatched_dataframe" in res
    assert "right_unmatched_dataframe" in res

    # Matched rows: id=2, 3 (2 rows)
    assert len(res["matched_dataframe"]) == 2
    # Left unmatched: id=1 (1 row)
    assert len(res["left_unmatched_dataframe"]) == 1
    assert res["left_unmatched_dataframe"]["id"].values[0] == 1
    # Right unmatched: id=4 (1 row)
    assert len(res["right_unmatched_dataframe"]) == 1
    assert res["right_unmatched_dataframe"]["id"].values[0] == 4

    metrics = res["metrics"]
    assert metrics["matched_rows"] == 2
    assert metrics["left_unmatched_rows"] == 1
    assert metrics["right_unmatched_rows"] == 1


@pytest.mark.asyncio
async def test_workflow_multi_dataset_endpoints():
    from httpx import AsyncClient, ASGITransport
    from backend.app.main import app
    from backend.app.infrastructure.database.session import init_db
    from backend.app.core.security import get_current_user, TokenData, create_access_token
    import uuid

    app.dependency_overrides[get_current_user] = lambda: TokenData("test", 1, ["Tenant Admin", "Data Scientist", "ML Engineer"], ["*"])

    def auth_headers():
        token = create_access_token(sub="test-user", tenant_id=1, roles=["Tenant Admin"])
        return {"Authorization": f"Bearer {token}"}

    await init_db()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test", headers=auth_headers()) as client:
        # 1. Upload two datasets: Customers.csv and Orders.csv
        csv_customers = b"customer_id,name,age\n1,Alice,25\n2,Bob,30\n"
        up_c = await client.post(
            "/api/v1/datasets/upload",
            files={"file": ("Customers.csv", csv_customers, "text/csv")},
            data={"name": "Customers.csv"}
        )
        assert up_c.status_code == 201
        cust_id = up_c.json()["id"]

        csv_orders = b"customer_id,order_id,amount\n1,101,50.0\n2,102,120.0\n"
        up_o = await client.post(
            "/api/v1/datasets/upload",
            files={"file": ("Orders.csv", csv_orders, "text/csv")},
            data={"name": "Orders.csv"}
        )
        assert up_o.status_code == 201
        ord_id = up_o.json()["id"]

        # 2. Create a join workflow
        wf_id = str(uuid.uuid4())
        payload = {
            "workflow_id": wf_id,
            "name": "Customer Churn Prediction",
            "dataset_id": cust_id,
            "dataset_name": "Customers.csv",
            "nodes": [
                {"id": "node_left", "recipe_id": "csv_loader", "config": {"dataset_id": cust_id, "dataset_name": "Customers.csv"}},
                {"id": "node_right", "recipe_id": "csv_loader", "config": {"dataset_id": ord_id, "dataset_name": "Orders.csv"}},
                {"id": "node_join", "recipe_id": "dataset_join", "config": {"join_type": "inner", "on": "customer_id"}}
            ],
            "edges": [
                {"source": "node_left", "target": "node_join", "target_handle": "left"},
                {"source": "node_right", "target": "node_join", "target_handle": "right"}
            ],
            "node_configs": {
                "node_left": {"recipe_id": "csv_loader", "config": {"dataset_id": cust_id, "dataset_name": "Customers.csv"}},
                "node_right": {"recipe_id": "csv_loader", "config": {"dataset_id": ord_id, "dataset_name": "Orders.csv"}},
                "node_join": {"recipe_id": "dataset_join", "config": {"join_type": "inner", "on": "customer_id"}}
            }
        }

        # 3. Test POST /api/v1/workflows/
        save_resp = await client.post("/api/v1/workflows/", json=payload)
        assert save_resp.status_code == 201
        data = save_resp.json()
        assert data["workflow_id"] == wf_id
        assert data["id"] == wf_id
        assert data["name"] == "Customer Churn Prediction"
        assert data["dataset_id"] == cust_id
        assert data["dataset_name"] == "Customers.csv"
        assert len(data["datasets"]) == 2

        ds_left = next(d for d in data["datasets"] if d["id"] == cust_id)
        assert ds_left["name"] == "Customers.csv"
        assert ds_left["role"] == "Primary (Left)"

        ds_right = next(d for d in data["datasets"] if d["id"] == ord_id)
        assert ds_right["name"] == "Orders.csv"
        assert ds_right["role"] == "Secondary (Right)"

        # 4. Test GET /api/v1/workflows/{id}
        get_resp = await client.get(f"/api/v1/workflows/{wf_id}")
        assert get_resp.status_code == 200
        get_data = get_resp.json()
        assert get_data["workflow_id"] == wf_id
        assert len(get_data["datasets"]) == 2
        assert get_data["datasets"][0]["role"] == "Primary (Left)"
        assert get_data["datasets"][1]["role"] == "Secondary (Right)"

        # 5. Test GET /api/v1/workflows/ (list)
        list_resp = await client.get("/api/v1/workflows/?search=Customer+Churn")
        assert list_resp.status_code == 200
        list_data = list_resp.json()
        matching = [w for w in list_data["data"] if w["id"] == wf_id]
        assert len(matching) == 1
        item = matching[0]
        assert item["workflow_id"] == wf_id
        assert len(item["datasets"]) == 2
        assert item["datasets"][0]["role"] == "Primary (Left)"
        assert item["datasets"][1]["role"] == "Secondary (Right)"

        # 6. Execute workflow to generate node output table
        exec_resp = await client.post(f"/api/v1/workflows/{wf_id}/execute")
        assert exec_resp.status_code == 200
        exec_data = exec_resp.json()
        exec_id = exec_data["execution_id"]

        # 7. Test GET /{execution_id}/nodes/{node_id}/data (Paginated Table Grid for UI)
        grid_resp = await client.get(f"/api/v1/workflows/{exec_id}/nodes/node_join/data?page=1&limit=10")
        assert grid_resp.status_code == 200
        grid_data = grid_resp.json()
        assert grid_data["execution_id"] == exec_id
        assert grid_data["node_id"] == "node_join"
        assert grid_data["total_rows"] == 2
        assert "_merge" not in grid_data["columns"]
        assert "customer_id" in grid_data["columns"]
        assert "amount" in grid_data["columns"]
        assert len(grid_data["data"]) == 2

        # 8. Test GET /{execution_id}/nodes/{node_id}/export-csv (Direct CSV File Download)
        csv_resp = await client.get(f"/api/v1/workflows/{exec_id}/nodes/node_join/export-csv")
        assert csv_resp.status_code == 200
        assert "text/csv" in csv_resp.headers["content-type"]
        assert "attachment" in csv_resp.headers["content-disposition"]
        csv_text = csv_resp.text
        assert "customer_id" in csv_text
        assert "Alice" in csv_text
        assert "_merge" not in csv_text

        # 9. Test POST /export-dataset (Save to Datasets Library)
        export_resp = await client.post("/api/v1/workflows/export-dataset", json={
            "execution_id": exec_id,
            "node_id": "node_join",
            "dataset_name": "Exported_Joined_Table"
        })
        assert export_resp.status_code == 201
        exp_res = export_resp.json()
        assert exp_res["status"] == "SUCCESS"
        assert exp_res["dataset"]["name"] == "Exported_Joined_Table"
        assert exp_res["dataset"]["row_count"] == 2
