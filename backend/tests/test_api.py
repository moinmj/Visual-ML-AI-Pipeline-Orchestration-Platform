import pytest
from httpx import AsyncClient, ASGITransport
from backend.app.main import app
from backend.app.infrastructure.database.session import init_db


@pytest.mark.asyncio
async def test_health_and_recipes_api():
    await init_db()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Health check
        resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

        # List recipes
        resp = await client.get("/api/v1/recipes/")
        assert resp.status_code == 200
        recipes = resp.json()
        assert len(recipes) >= 7

        # Get recipe schema
        resp = await client.get("/api/v1/recipes/missing_value_imputer/schema")
        assert resp.status_code == 200
        schema = resp.json()
        assert "parameters_schema" in schema
        assert schema["recipe_id"] == "missing_value_imputer"


@pytest.mark.asyncio
async def test_workflow_validation_api():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "nodes": [
                {"id": "n1", "recipe_id": "missing_value_imputer", "config": {}},
                {"id": "n2", "recipe_id": "feature_scaler", "config": {}}
            ],
            "edges": [
                {"source": "n1", "target": "n2"}
            ]
        }
        resp = await client.post("/api/v1/workflows/validate", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is True
        assert len(data["errors"]) == 0


@pytest.mark.asyncio
async def test_workflow_execution_api():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "nodes": [
                {"id": "n1", "recipe_id": "csv_loader", "config": {}},
                {"id": "n2", "recipe_id": "feature_scaler", "config": {"method": "standard"}}
            ],
            "edges": [
                {"source": "n1", "target": "n2"}
            ]
        }
        resp = await client.post("/api/v1/workflows/execute", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "SUCCESS"
        assert "node_outputs" in data
        assert "step_snapshots" in data


@pytest.mark.asyncio
async def test_recommendation_and_templates_api():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Test AI Recommendation
        rec_resp = await client.post("/api/v1/recommend/pipeline", json={"preset": "balanced"})
        assert rec_resp.status_code == 200
        rec_data = rec_resp.json()
        assert "task_type" in rec_data
        assert "model_rankings" in rec_data
        assert "recommended_dag" in rec_data
        assert len(rec_data["recommended_dag"]["nodes"]) >= 3

        # 2. Test Auto-Wire
        wire_resp = await client.post("/api/v1/recommend/autowire", json={
            "nodes": [
                {"id": "node_1", "recipe_id": "csv_loader", "label": "CSV Ingest"},
                {"id": "node_2", "recipe_id": "feature_scaler", "label": "Scaler"},
                {"id": "node_3", "recipe_id": "xgboost_trainer", "label": "Model"}
            ]
        })
        assert wire_resp.status_code == 200
        wire_data = wire_resp.json()
        assert wire_data["status"] == "AUTOWIRED"
        assert len(wire_data["edges"]) >= 2

        # 3. Test Templates List
        tmpl_list = await client.get("/api/v1/templates/")
        assert tmpl_list.status_code == 200
        templates = tmpl_list.json()
        assert len(templates) >= 3

        # 4. Test Get Template DAG
        ml_tmpl = await client.get("/api/v1/templates/ml_supervised")
        assert ml_tmpl.status_code == 200
        ml_dag = ml_tmpl.json()["dag"]
        assert len(ml_dag["nodes"]) == 6
        assert len(ml_dag["edges"]) == 6


@pytest.mark.asyncio
async def test_upload_dataset_and_execute_custom_pipeline():
    await init_db()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Upload a custom CSV with unique identifiable column names
        csv_bytes = b"custom_col_x,custom_col_y,custom_target\n10.5,20.1,1\n30.2,40.5,0\n50.1,60.8,1\n70.3,80.9,0\n90.0,100.2,1\n"
        files = {"file": ("custom_experiment.csv", csv_bytes, "text/csv")}
        upload_resp = await client.post("/api/v1/datasets/upload", files=files, data={"name": "Custom Experiment Dataset"})
        assert upload_resp.status_code == 201
        ds_data = upload_resp.json()
        dataset_id = ds_data["id"]
        assert dataset_id is not None
        assert ds_data["row_count"] == 5
        assert ds_data["column_count"] == 3

        # 2. Execute a workflow DAG pointing explicitly to this dataset_id
        dag_payload = {
            "nodes": [
                {
                    "id": "node_csv",
                    "recipe_id": "csv_loader",
                    "label": "Custom CSV Loader",
                    "config": {"dataset_id": dataset_id}
                },
                {
                    "id": "node_scaler",
                    "recipe_id": "feature_scaler",
                    "label": "Feature Scaler",
                    "config": {"method": "standard"}
                }
            ],
            "edges": [
                {"source": "node_csv", "target": "node_scaler"}
            ]
        }
        exec_resp = await client.post("/api/v1/workflows/execute", json=dag_payload)
        assert exec_resp.status_code == 200
        exec_data = exec_resp.json()
        assert exec_data["status"] == "SUCCESS"

        # 3. Verify step snapshots contains our unique uploaded columns
        csv_snapshot = exec_data["step_snapshots"]["node_csv"]
        assert "custom_col_x" in csv_snapshot["columns"]
        assert "custom_col_y" in csv_snapshot["columns"]
        assert "custom_target" in csv_snapshot["columns"]
        assert csv_snapshot["row_count"] == 5


@pytest.mark.asyncio
async def test_run_then_save_workflow_preserves_last_execution():
    await init_db()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. User runs an unsaved pipeline on the canvas
        dag_payload = {
            "nodes": [
                {"id": "node_csv", "recipe_id": "csv_loader", "config": {}},
                {"id": "node_scaler", "recipe_id": "feature_scaler", "config": {"method": "standard"}}
            ],
            "edges": [
                {"source": "node_csv", "target": "node_scaler"}
            ]
        }
        exec_resp = await client.post("/api/v1/workflows/execute", json=dag_payload)
        assert exec_resp.status_code == 200
        exec_data = exec_resp.json()
        assert exec_data["status"] == "SUCCESS"
        assert exec_data["execution_id"] is not None
        assert exec_data["workflow_id"] is not None

        # 2. User then clicks "Save Workflow", passing last_execution in the save payload
        save_payload = {
            "name": "My Saved Pipeline With Reports",
            "description": "Saved after successful run",
            "nodes": dag_payload["nodes"],
            "edges": dag_payload["edges"],
            "node_configs": {
                "node_csv": {"recipe_id": "csv_loader", "config": {}},
                "node_scaler": {"recipe_id": "feature_scaler", "config": {"method": "standard"}}
            },
            "last_execution": exec_data
        }
        save_resp = await client.post("/api/v1/workflows/", json=save_payload)
        assert save_resp.status_code == 201
        saved_wf = save_resp.json()
        saved_id = saved_wf["id"]
        assert saved_wf["name"] == "My Saved Pipeline With Reports"
        assert saved_wf["last_execution"] is not None
        assert saved_wf["last_execution"]["execution_id"] == exec_data["execution_id"]
        assert saved_wf["last_execution"]["status"] == "SUCCESS"

        # 3. Retrieve workflow by ID and verify last_execution is intact
        get_resp = await client.get(f"/api/v1/workflows/{saved_id}")
        assert get_resp.status_code == 200
        retrieved_wf = get_resp.json()
        assert retrieved_wf["last_execution"] is not None
        assert retrieved_wf["last_execution"]["execution_id"] == exec_data["execution_id"]

        # 4. User updates workflow using ID without passing last_execution (must NOT wipe existing reports)
        update_payload = {
            "id": saved_id,
            "name": "My Renamed Pipeline",
            "nodes": dag_payload["nodes"],
            "edges": dag_payload["edges"]
        }
        update_resp = await client.post("/api/v1/workflows/", json=update_payload)
        assert update_resp.status_code == 201
        updated_wf = update_resp.json()
        assert updated_wf["name"] == "My Renamed Pipeline"
        assert updated_wf["last_execution"] is not None
        assert updated_wf["last_execution"]["execution_id"] == exec_data["execution_id"]

        # 5. Test dedicated save-execution endpoint
        save_exec_resp = await client.post(
            f"/api/v1/workflows/{saved_id}/save-execution",
            json={
                "execution_id": "custom-exec-999",
                "status": "SUCCESS",
                "total_duration_ms": 250.5,
                "final_metrics": {"accuracy": 0.99, "task_type": "classification"}
            }
        )
        assert save_exec_resp.status_code == 200
        exec_saved_wf = save_exec_resp.json()
        assert exec_saved_wf["last_execution"]["execution_id"] == "custom-exec-999"
        assert exec_saved_wf["last_execution"]["final_metrics"]["accuracy"] == 0.99


@pytest.mark.asyncio
async def test_class_imbalance_resampler_pipeline():
    await init_db()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Verify recipe schema retrieval
        schema_resp = await client.get("/api/v1/recipes/class_imbalance_resampler/schema")
        assert schema_resp.status_code == 200
        schema = schema_resp.json()
        assert schema["recipe_id"] == "class_imbalance_resampler"
        assert "strategy" in schema["parameters_schema"]["properties"]

        # 2. Upload an imbalanced dataset (15 samples Class 0, 3 samples Class 1)
        rows = []
        for i in range(15):
            rows.append(f"{i*2.1},{i*1.5},0")
        for i in range(3):
            rows.append(f"{(i+20)*3.5},{(i+20)*2.1},1")
        csv_content = "feature1,feature2,fraud_label\n" + "\n".join(rows) + "\n"

        files = {"file": ("imbalanced_fraud.csv", csv_content.encode("utf-8"), "text/csv")}
        upload_resp = await client.post("/api/v1/datasets/upload", files=files, data={"name": "Imbalanced Fraud Dataset"})
        assert upload_resp.status_code == 201
        dataset_id = upload_resp.json()["id"]

        # 3. Build end-to-end DAG with SMOTE Resampler post-split
        dag_payload = {
            "nodes": [
                {
                    "id": "node_csv",
                    "recipe_id": "csv_loader",
                    "label": "CSV Ingest",
                    "config": {"dataset_id": dataset_id}
                },
                {
                    "id": "node_split",
                    "recipe_id": "train_test_split",
                    "label": "Train/Test Split",
                    "config": {"target_column": "fraud_label", "test_size": 0.25, "random_state": 42}
                },
                {
                    "id": "node_smote",
                    "recipe_id": "class_imbalance_resampler",
                    "label": "SMOTE Resampler",
                    "config": {"strategy": "smote", "sampling_ratio": 1.0, "k_neighbors": 2, "random_state": 42}
                },
                {
                    "id": "node_model",
                    "recipe_id": "random_forest_trainer",
                    "label": "Random Forest",
                    "config": {"task_type": "classification", "n_estimators": 10, "random_state": 42}
                },
                {
                    "id": "node_eval",
                    "recipe_id": "model_evaluator",
                    "label": "Evaluator",
                    "config": {}
                }
            ],
            "edges": [
                {"source": "node_csv", "target": "node_split"},
                {"source": "node_split", "target": "node_smote"},
                {"source": "node_smote", "target": "node_model"},
                {"source": "node_model", "target": "node_eval"}
            ]
        }

        exec_resp = await client.post("/api/v1/workflows/execute", json=dag_payload)
        assert exec_resp.status_code == 200
        data = exec_resp.json()
        assert data["status"] == "SUCCESS"

        # 4. Verify step snapshot diagnostics from SMOTE node
        smote_snap = data["step_snapshots"]["node_smote"]
        assert "output_summary" in smote_snap
        summary = smote_snap["output_summary"]
        assert summary["strategy"] == "smote"
        assert summary["samples_delta"] > 0
        assert summary["resampled_samples"] > summary["original_samples"]

        # 5. Verify final metrics produced by Model Evaluator
        assert data["final_metrics"] is not None
        assert "accuracy" in data["final_metrics"]
        assert data["final_metrics"]["task_type"] == "classification"

