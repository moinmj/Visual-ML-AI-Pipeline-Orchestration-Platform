import pytest
import pandas as pd
import numpy as np
import uuid
import os
from httpx import AsyncClient, ASGITransport

from backend.app.main import app
from backend.app.engine.execution.step_executor import NodeStepExecutor
from backend.app.infrastructure.storage.storage_manager import storage_manager


@pytest.fixture(autouse=True)
def cleanup_test_sessions():
    """Cleanup test sessions directory after tests."""
    yield
    test_sessions_dir = storage_manager.get_absolute_path("test_sessions")
    if test_sessions_dir.exists():
        import shutil
        for s in test_sessions_dir.glob("test_*"):
            try:
                shutil.rmtree(s)
            except Exception:
                pass


@pytest.mark.asyncio
async def test_single_node_execution_and_caching():
    """Test executing a single data transformer node and caching its output."""
    session_id = f"test_session_{uuid.uuid4().hex[:8]}"

    # 1. Preload a mock dataset in storage
    df_raw = pd.DataFrame({
        "Store": [1, 2, 3],
        "Sales": [100.5, 200.0, 300.2],
        "Dept": ["A", "B", "A"]
    })
    rel_path = f"datasets/{session_id}_raw.parquet"
    abs_path = storage_manager.get_absolute_path(rel_path)
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    df_raw.to_parquet(abs_path, index=False)

    nodes = [
        {"id": "node_loader", "recipe_id": "csv_loader", "config": {}},
        {"id": "node_filter", "recipe_id": "column_selector", "config": {"columns": ["Store", "Sales"]}}
    ]
    edges = [
        {"source": "node_loader", "target": "node_filter", "source_handle": "output", "target_handle": "input"}
    ]

    # Pre-seed node_loader output into session cache
    NodeStepExecutor.save_node_output(
        session_id=session_id,
        node_id="node_loader",
        recipe_id="csv_loader",
        outputs={"dataframe": df_raw}
    )
    assert NodeStepExecutor.is_node_cached(session_id, "node_loader") is True

    # 2. Test Step on node_filter
    resp = await NodeStepExecutor.execute_node_step(
        target_node_id="node_filter",
        nodes=nodes,
        edges=edges,
        node_configs={"node_filter": {"columns": ["Store", "Sales"]}},
        session_id=session_id,
        auto_run_ancestors=False
    )

    assert resp.success is True
    assert resp.status == "SUCCESS"
    assert resp.node_id == "node_filter"
    assert resp.columns == ["Store", "Sales"]
    assert resp.row_count == 3
    assert len(resp.preview_rows) == 3

    # Check cache persistence
    assert NodeStepExecutor.is_node_cached(session_id, "node_filter") is True
    loaded_out = NodeStepExecutor.load_node_output(session_id, "node_filter")
    assert "dataframe" in loaded_out
    assert list(loaded_out["dataframe"].columns) == ["Store", "Sales"]


@pytest.mark.asyncio
async def test_chained_dataset_join_step_by_step():
    """
    Test the exact chained dataset join scenario:
    Walmart_Sales (8 cols) + stores (3 cols) -> Join 1 (XYz with 10 cols)
    Then Join 2 joins online_retail (4 cols) with XYz!
    """
    session_id = f"test_session_{uuid.uuid4().hex[:8]}"

    # Create 3 test tables
    df_walmart = pd.DataFrame({
        "Store": [1, 2, 3],
        "Date": ["2026-01-01", "2026-01-02", "2026-01-03"],
        "Weekly_Sales": [15000, 22000, 18000],
        "Holiday_Flag": [0, 1, 0],
        "Temperature": [45.0, 52.0, 48.0],
        "Fuel_Price": [3.5, 3.6, 3.4],
        "CPI": [211.0, 212.0, 211.5],
        "Unemployment": [7.8, 7.5, 7.9]
    })

    df_stores = pd.DataFrame({
        "Store": [1, 2, 3],
        "Type": ["A", "B", "A"],
        "Size": [150000, 120000, 160000]
    })

    df_retail = pd.DataFrame({
        "Store": [1, 2, 3],
        "InvoiceNo": [101, 102, 103],
        "Quantity": [5, 10, 2],
        "UnitPrice": [12.5, 8.0, 25.0]
    })

    nodes = [
        {"id": "node_walmart", "recipe_id": "csv_loader"},
        {"id": "node_stores", "recipe_id": "csv_loader"},
        {"id": "node_join_1", "recipe_id": "dataset_join"},
        {"id": "node_retail", "recipe_id": "csv_loader"},
        {"id": "node_join_2", "recipe_id": "dataset_join"}
    ]

    edges = [
        {"source": "node_walmart", "target": "node_join_1", "source_handle": "output", "target_handle": "left"},
        {"source": "node_stores", "target": "node_join_1", "source_handle": "output", "target_handle": "right"},
        {"source": "node_retail", "target": "node_join_2", "source_handle": "output", "target_handle": "left"},
        {"source": "node_join_1", "target": "node_join_2", "source_handle": "joined", "target_handle": "right"}
    ]

    node_configs = {
        "node_join_1": {
            "join_type": "inner",
            "conditions": [{"left": "Store", "right": "Store"}],
            "new_dataset_name": "XYz"
        },
        "node_join_2": {
            "join_type": "left",
            "conditions": [{"left": "Store", "right": "Store"}],
            "new_dataset_name": "Master_Joined"
        }
    }

    # Pre-seed loaders into session cache
    NodeStepExecutor.save_node_output(session_id, "node_walmart", "csv_loader", {"dataframe": df_walmart})
    NodeStepExecutor.save_node_output(session_id, "node_stores", "csv_loader", {"dataframe": df_stores})
    NodeStepExecutor.save_node_output(session_id, "node_retail", "csv_loader", {"dataframe": df_retail})

    # Step A: Run Join 1 Test Step
    resp_join1 = await NodeStepExecutor.execute_node_step(
        target_node_id="node_join_1",
        nodes=nodes,
        edges=edges,
        node_configs=node_configs,
        session_id=session_id
    )

    assert resp_join1.success is True
    assert resp_join1.status == "SUCCESS"
    assert resp_join1.row_count == 3
    # Check that all 10 columns are present in Join 1's output
    for col in ["Store", "Date", "Weekly_Sales", "Holiday_Flag", "Temperature", "Fuel_Price", "CPI", "Unemployment", "Type", "Size"]:
        assert col in resp_join1.columns

    # Step B: Run Join 2 Test Step (using Join 1's cached result)
    resp_join2 = await NodeStepExecutor.execute_node_step(
        target_node_id="node_join_2",
        nodes=nodes,
        edges=edges,
        node_configs=node_configs,
        session_id=session_id
    )

    assert resp_join2.success is True
    assert resp_join2.status == "SUCCESS"
    assert resp_join2.row_count == 3
    # Left table had 4 cols (Store, InvoiceNo, Quantity, UnitPrice)
    # Right table XYz had 10 cols (Store, Date, Weekly_Sales, Holiday_Flag, Temperature, Fuel_Price, CPI, Unemployment, Type, Size)
    # Merged on Store -> 4 + 9 = 13 columns
    assert "InvoiceNo" in resp_join2.columns
    assert "Type" in resp_join2.columns
    assert "Size" in resp_join2.columns
    assert "Weekly_Sales" in resp_join2.columns
    assert len(resp_join2.columns) == 13


@pytest.mark.asyncio
async def test_auto_run_ancestors():
    """
    Test that calling test-step on a downstream node automatically executes
    and caches all unrun ancestors in topological order.
    """
    session_id = f"test_session_{uuid.uuid4().hex[:8]}"

    df = pd.DataFrame({
        "A": [10, 20, 30, 40, 50],
        "B": [1.5, 2.5, 3.5, 4.5, 5.5],
        "Target": [0, 1, 0, 1, 0]
    })

    nodes = [
        {"id": "loader", "recipe_id": "csv_loader"},
        {"id": "imputer", "recipe_id": "missing_value_imputer"},
        {"id": "scaler", "recipe_id": "feature_scaler", "config": {"method": "standard", "features": ["A", "B"]}}
    ]

    edges = [
        {"source": "loader", "target": "imputer", "source_handle": "output", "target_handle": "input"},
        {"source": "imputer", "target": "scaler", "source_handle": "output", "target_handle": "input"}
    ]

    # Pre-seed only the root loader
    NodeStepExecutor.save_node_output(session_id, "loader", "csv_loader", {"dataframe": df})

    # Call test-step directly on "scaler" (skipping "imputer") with auto_run_ancestors=True
    resp = await NodeStepExecutor.execute_node_step(
        target_node_id="scaler",
        nodes=nodes,
        edges=edges,
        node_configs={},
        session_id=session_id,
        auto_run_ancestors=True
    )

    assert resp.success is True
    assert resp.status == "SUCCESS"
    assert "imputer" in resp.ancestors_executed
    assert NodeStepExecutor.is_node_cached(session_id, "imputer") is True
    assert NodeStepExecutor.is_node_cached(session_id, "scaler") is True


@pytest.mark.asyncio
async def test_ml_step_execution_with_model_artifact_caching():
    """
    Test ML training and evaluation step-by-step:
    Verifies that the trained scikit-learn/xgboost model and train/test splits
    are serialized via joblib and restored when testing model_evaluator in a separate call!
    """
    session_id = f"test_session_{uuid.uuid4().hex[:8]}"

    df_ml = pd.DataFrame({
        "Feature1": np.random.randn(60),
        "Feature2": np.random.randn(60),
        "target": np.random.choice([0, 1], size=60)
    })

    nodes = [
        {"id": "node_loader", "recipe_id": "csv_loader"},
        {"id": "node_split", "recipe_id": "train_test_split", "config": {"target_column": "target", "test_size": 0.25}},
        {"id": "node_rf", "recipe_id": "random_forest_trainer", "config": {"target_column": "target", "n_estimators": 5}},
        {"id": "node_eval", "recipe_id": "model_evaluator"}
    ]

    edges = [
        {"source": "node_loader", "target": "node_split", "source_handle": "output", "target_handle": "input"},
        {"source": "node_split", "target": "node_rf", "source_handle": "output_1", "target_handle": "input"},
        {"source": "node_rf", "target": "node_eval", "source_handle": "output", "target_handle": "input_1"},
        {"source": "node_split", "target": "node_eval", "source_handle": "output_2", "target_handle": "input_2"}
    ]

    # Pre-seed loader
    NodeStepExecutor.save_node_output(session_id, "node_loader", "csv_loader", {"dataframe": df_ml})

    # Step 1: Run Splitter
    resp_split = await NodeStepExecutor.execute_node_step(
        target_node_id="node_split",
        nodes=nodes,
        edges=edges,
        node_configs={"node_split": {"target_column": "target", "test_size": 0.25}},
        session_id=session_id
    )
    assert resp_split.success is True
    assert resp_split.status == "SUCCESS"

    # Step 2: Run Random Forest Trainer
    resp_rf = await NodeStepExecutor.execute_node_step(
        target_node_id="node_rf",
        nodes=nodes,
        edges=edges,
        node_configs={"node_rf": {"target_column": "target", "n_estimators": 5}},
        session_id=session_id
    )
    assert resp_rf.success is True
    assert resp_rf.status == "SUCCESS"

    # Step 3: Run Model Evaluator (must load model from RF and test data from Splitter)
    resp_eval = await NodeStepExecutor.execute_node_step(
        target_node_id="node_eval",
        nodes=nodes,
        edges=edges,
        node_configs={},
        session_id=session_id
    )
    assert resp_eval.success is True
    assert resp_eval.status == "SUCCESS"
    assert resp_eval.metrics is not None
    assert "accuracy" in resp_eval.metrics or "f1" in resp_eval.metrics or "precision" in resp_eval.metrics


@pytest.mark.asyncio
async def test_session_clear():
    """Test clearing the session cache directory."""
    session_id = f"test_session_{uuid.uuid4().hex[:8]}"

    df = pd.DataFrame({"x": [1, 2]})
    NodeStepExecutor.save_node_output(session_id, "n1", "csv_loader", {"dataframe": df})
    assert NodeStepExecutor.is_node_cached(session_id, "n1") is True

    clear_resp = NodeStepExecutor.clear_session(session_id)
    assert clear_resp.success is True
    assert NodeStepExecutor.is_node_cached(session_id, "n1") is False


from backend.app.core.security import create_access_token


def auth_headers():
    token = create_access_token(sub="test-user", tenant_id=1, roles=["Tenant Admin"])
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_api_test_step_endpoint():
    """Test calling the FastAPI test-step endpoint via HTTP."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test", headers=auth_headers()) as client:
        session_id = f"test_api_session_{uuid.uuid4().hex[:8]}"

        df_source = pd.DataFrame({
            "col_a": [1, 2, 3],
            "col_b": ["x", "y", "z"]
        })
        NodeStepExecutor.save_node_output(session_id, "loader_api", "csv_loader", {"dataframe": df_source})

        payload = {
            "session_id": session_id,
            "nodes": [
                {"id": "loader_api", "recipe_id": "csv_loader"},
                {"id": "selector_api", "recipe_id": "column_selector", "config": {"columns": ["col_a"]}}
            ],
            "edges": [
                {"source": "loader_api", "target": "selector_api", "source_handle": "output", "target_handle": "input"}
            ],
            "node_configs": {
                "selector_api": {"columns": ["col_a"]}
            }
        }

        resp = await client.post(
            "/api/v1/workflows/nodes/selector_api/test-step",
            json=payload
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["status"] == "SUCCESS"
        assert data["columns"] == ["col_a"]
        assert data["row_count"] == 3

        # Test preview endpoint
        preview_resp = await client.get(
            f"/api/v1/workflows/sessions/{session_id}/nodes/selector_api/preview?page=1&limit=10"
        )
        assert preview_resp.status_code == 200
        preview_data = preview_resp.json()
        assert preview_data["total_rows"] == 3
        assert len(preview_data["data"]) == 3
        assert preview_data["columns"] == ["col_a"]

        # Test clear session endpoint
        del_resp = await client.delete(
            f"/api/v1/workflows/sessions/{session_id}"
        )
        assert del_resp.status_code == 200
        assert del_resp.json()["success"] is True


@pytest.mark.asyncio
async def test_api_test_step_datetime_serialization():
    """Test executing DataTypeConverter to datetime and verifying Timestamp objects are JSON serialized without error."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test", headers=auth_headers()) as client:
        session_id = f"test_dt_session_{uuid.uuid4().hex[:8]}"

        df_source = pd.DataFrame({
            "InvoiceDate": ["12/1/2010 8:26", "12/1/2010 8:34", "12/2/2010 11:20"],
            "Price": [2.55, 3.39, 4.25]
        })
        NodeStepExecutor.save_node_output(session_id, "loader_dt", "csv_loader", {"dataframe": df_source})

        payload = {
            "session_id": session_id,
            "nodes": [
                {"id": "loader_dt", "recipe_id": "csv_loader"},
                {
                    "id": "converter_dt",
                    "recipe_id": "data_type_converter",
                    "config": {
                        "conversions": {"InvoiceDate": "datetime"},
                        "datetime_format": "%m/%d/%Y %H:%M",
                        "errors": "coerce"
                    }
                }
            ],
            "edges": [
                {"source": "loader_dt", "target": "converter_dt", "source_handle": "output", "target_handle": "input"}
            ],
            "node_configs": {
                "converter_dt": {
                    "conversions": {"InvoiceDate": "datetime"},
                    "datetime_format": "%m/%d/%Y %H:%M",
                    "errors": "coerce"
                }
            }
        }

        resp = await client.post(
            "/api/v1/workflows/nodes/converter_dt/test-step",
            json=payload
        )
        assert resp.status_code == 200, f"Response error: {resp.text}"
        data = resp.json()
        assert data["success"] is True
        assert data["status"] == "SUCCESS"
        assert "InvoiceDate" in data["columns"]
        assert len(data["preview_rows"]) == 3
        # Ensure timestamp was serialized as a string in preview rows
        assert isinstance(data["preview_rows"][0]["InvoiceDate"], str)

        # Test preview table endpoint
        preview_resp = await client.get(
            f"/api/v1/workflows/sessions/{session_id}/nodes/converter_dt/preview?page=1&limit=10"
        )
        assert preview_resp.status_code == 200, f"Preview error: {preview_resp.text}"
        preview_data = preview_resp.json()
        assert preview_data["total_rows"] == 3
        assert isinstance(preview_data["data"][0]["InvoiceDate"], str)


@pytest.mark.asyncio
async def test_join_with_reactflow_camelcase_handles():
    """Verify join assigns left and right dataframes accurately when ReactFlow sends camelCase targetHandle."""
    session_id = f"test_camelcase_{uuid.uuid4().hex[:8]}"

    df_retail = pd.DataFrame({
        "InvoiceDate": ["2010-12-01", "2010-12-02"],
        "UnitPrice": [2.5, 3.5]
    })
    df_walmart = pd.DataFrame({
        "Date": ["2010-12-01", "2010-12-02"],
        "Weekly_Sales": [10000.0, 20000.0]
    })

    NodeStepExecutor.save_node_output(session_id, "node_retail", "csv_loader", {"dataframe": df_retail})
    NodeStepExecutor.save_node_output(session_id, "node_walmart", "csv_loader", {"dataframe": df_walmart})

    nodes = [
        {"id": "node_retail", "recipe_id": "csv_loader"},
        {"id": "node_walmart", "recipe_id": "csv_loader"},
        {
            "id": "node_join",
            "recipe_id": "dataset_join",
            "config": {
                "join_type": "inner",
                "conditions": [{"left": "InvoiceDate", "right": "Date", "operator": "="}]
            }
        }
    ]

    # Note reverse order in edges and upstreamIds: walmart is first in list, but retail targets handle "left"
    edges = [
        {"source": "node_walmart", "target": "node_join", "sourceHandle": "joined", "targetHandle": "right"},
        {"source": "node_retail", "target": "node_join", "sourceHandle": "output", "targetHandle": "left"}
    ]

    resp = await NodeStepExecutor.execute_node_step(
        target_node_id="node_join",
        nodes=nodes,
        edges=edges,
        session_id=session_id
    )

    assert resp.success is True, f"Failed: {resp.error_message}"
    assert resp.status == "SUCCESS"
    assert resp.row_count == 2
    assert "InvoiceDate" in resp.columns
    assert "Weekly_Sales" in resp.columns


@pytest.mark.asyncio
async def test_join_datetime_with_object_string_date():
    """Verify join succeeds when left key is datetime64[ns] (with hour:min) and right key is object string."""
    session_id = f"test_dt_obj_{uuid.uuid4().hex[:8]}"

    df_retail = pd.DataFrame({
        "InvoiceDate": pd.to_datetime(["12/1/2010 8:26", "12/2/2010 11:00"]),
        "Sales": [10.0, 20.0]
    })
    df_walmart = pd.DataFrame({
        "Date": ["01-12-2010", "02-12-2010"],
        "Weekly_Sales": [1000.0, 2000.0]
    })

    NodeStepExecutor.save_node_output(session_id, "node_retail", "csv_loader", {"dataframe": df_retail})
    NodeStepExecutor.save_node_output(session_id, "node_walmart", "csv_loader", {"dataframe": df_walmart})

    nodes = [
        {"id": "node_retail", "recipe_id": "csv_loader"},
        {"id": "node_walmart", "recipe_id": "csv_loader"},
        {
            "id": "node_join",
            "recipe_id": "dataset_join",
            "config": {
                "join_type": "left",
                "conditions": [{"left": "InvoiceDate", "right": "Date", "operator": "="}]
            }
        }
    ]

    edges = [
        {"source": "node_retail", "target": "node_join", "sourceHandle": "output", "targetHandle": "left"},
        {"source": "node_walmart", "target": "node_join", "sourceHandle": "output", "targetHandle": "right"}
    ]

    resp = await NodeStepExecutor.execute_node_step(
        target_node_id="node_join",
        nodes=nodes,
        edges=edges,
        session_id=session_id
    )

    assert resp.success is True, f"Failed: {resp.error_message}"
    assert resp.status == "SUCCESS"
    assert resp.row_count == 2
    assert "Weekly_Sales" in resp.columns
    # Ensure temporary columns were dropped cleanly
    assert not any(c.startswith("__temp_join_") for c in resp.columns)


@pytest.mark.asyncio
async def test_mlflow_tracker_preview():
    """Verify that MLflow Tracker produces a tabular preview row and does not 404."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test", headers=auth_headers()) as client:
        session_id = f"test_mlflow_{uuid.uuid4().hex[:8]}"
        payload = {
            "session_id": session_id,
            "nodes": [
                {"id": "node_mlflow", "recipe_id": "mlflow_tracker", "config": {"registered_model_name": "TestModel"}}
            ],
            "edges": [],
            "node_configs": {
                "node_mlflow": {"registered_model_name": "TestModel"}
            }
        }
        resp = await client.post(
            "/api/v1/workflows/nodes/node_mlflow/test-step",
            json=payload
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True

        preview_resp = await client.get(
            f"/api/v1/workflows/sessions/{session_id}/nodes/node_mlflow/preview?page=1&limit=10"
        )
        assert preview_resp.status_code == 200
        preview_data = preview_resp.json()
        assert preview_data["total_rows"] == 1
        assert "mlflow_run_id" in preview_data["columns"]
        assert len(preview_data["data"]) == 1
        assert preview_data["data"][0]["registered_model_name"] == "TestModel"


@pytest.mark.asyncio
async def test_preview_from_full_workflow_execution():
    """Verify preview endpoint resolves nodes from full DAG execution without 404."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test", headers=auth_headers()) as client:
        # 1. Execute a full workflow DAG via /workflows/execute
        wf_payload = {
            "name": "Full DAG Test",
            "nodes": [
                {"id": "node_csv", "recipe_id": "csv_loader", "config": {}},
                {"id": "node_imputer", "recipe_id": "missing_value_imputer", "config": {}}
            ],
            "edges": [
                {"source": "node_csv", "target": "node_imputer", "source_handle": "output", "target_handle": "input"}
            ],
            "node_configs": {}
        }
        exec_resp = await client.post("/api/v1/workflows/execute", json=wf_payload)
        assert exec_resp.status_code == 200
        exec_data = exec_resp.json()
        assert exec_data["status"] == "SUCCESS"
        exec_id = exec_data["execution_id"]

        # 2. Preview the imputer node using the execution_id on the session preview endpoint
        preview_resp = await client.get(
            f"/api/v1/workflows/sessions/{exec_id}/nodes/node_imputer/preview?page=1&limit=10"
        )
        assert preview_resp.status_code == 200
        pdata = preview_resp.json()
        assert pdata["total_rows"] > 0
        assert len(pdata["data"]) > 0

        # 3. Preview via canvas node card endpoint /{execution_id}/nodes/{node_id}/data
        data_resp = await client.get(
            f"/api/v1/workflows/{exec_id}/nodes/node_imputer/data?page=1&limit=10"
        )
        assert data_resp.status_code == 200, f"Node table data returned {data_resp.status_code}: {data_resp.text}"
        node_data = data_resp.json()
        assert node_data["total_rows"] > 0
        assert len(node_data["data"]) > 0
        assert "execution_id" in node_data
        assert node_data["node_id"] == "node_imputer"


@pytest.mark.asyncio
async def test_safe_to_parquet_with_mixed_struct_and_preview_data():
    """Verify safe_to_parquet handles mixed struct/scalar values and get_node_table_data resolves it."""
    from backend.app.infrastructure.storage.storage_manager import storage_manager
    from backend.app.workflows.router import _RECENT_EXECUTION_RESULTS

    # DataFrame with mixed dict and scalar values (like the user's age column)
    df_mixed = pd.DataFrame([
        {"user_id": "U1", "age": {"source": "25.0", "parsedValue": 25}, "city": "NYC"},
        {"user_id": "U2", "age": "11", "city": "LA"},
        {"user_id": "U3", "age": 30, "city": "CHI"}
    ])

    test_path = storage_manager.get_absolute_path("test_mixed_safe.parquet")
    try:
        success = storage_manager.safe_to_parquet(df_mixed, test_path)
        assert success is True
        assert test_path.exists()
        loaded_df = pd.read_parquet(test_path)
        assert len(loaded_df) == 3
        assert "age" in loaded_df.columns
    finally:
        if test_path.exists():
            test_path.unlink()

    # Verify /{execution_id}/nodes/{node_id}/data resolves from in-memory _RECENT_EXECUTION_RESULTS with step_snapshots
    mock_exec_id = f"test_mock_exec_{uuid.uuid4().hex[:8]}"
    _RECENT_EXECUTION_RESULTS[mock_exec_id] = {
        "execution_id": mock_exec_id,
        "node_outputs": {},  # intentionally empty, like real execution
        "step_snapshots": {
            "node_test": {
                "node_id": "node_test",
                "columns": ["user_id", "age", "city"],
                "preview_rows": [
                    {"user_id": "U1", "age": {"source": "25.0", "parsedValue": 25}, "city": "NYC"},
                    {"user_id": "U2", "age": "11", "city": "LA"}
                ]
            }
        }
    }

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test", headers=auth_headers()) as client:
        resp = await client.get(f"/api/v1/workflows/{mock_exec_id}/nodes/node_test/data?page=1&limit=10")
        assert resp.status_code == 200, f"Failed to get table data: {resp.text}"
        body = resp.json()
        assert body["total_rows"] == 2
        assert len(body["data"]) == 2
        assert body["data"][0]["user_id"] == "U1"
        assert body["columns"] == ["user_id", "age", "city"]






