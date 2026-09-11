import pytest
import uuid
from httpx import AsyncClient, ASGITransport
from backend.app.main import app
from backend.app.infrastructure.database.session import init_db


@pytest.mark.asyncio
async def test_workflow_history_and_rollback_full_suite():
    await init_db()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Upload small dataset for execution
        csv_bytes = b"feature_a,feature_b,target\n1.0,10.0,0\n2.0,20.0,0\n3.0,30.0,1\n4.0,40.0,1\n5.0,50.0,0\n6.0,60.0,1\n"
        files = {"file": ("history_test_data.csv", csv_bytes, "text/csv")}
        up_resp = await client.post("/api/v1/datasets/upload", files=files, data={"name": "History Test Dataset"})
        assert up_resp.status_code == 201
        dataset_id = up_resp.json()["id"]

        wf_id = f"wf_hist_{uuid.uuid4().hex[:8]}"

        # 2. Run #1 (v1): Random Forest with n_estimators=10
        dag_v1 = {
            "name": "Audit History Pipeline",
            "dataset_id": dataset_id,
            "nodes": [
                {"id": "n_csv", "recipe_id": "csv_loader", "config": {"dataset_id": dataset_id}},
                {"id": "n_split", "recipe_id": "train_test_split", "config": {"target_column": "target", "test_size": 0.33, "random_state": 42}},
                {"id": "n_model", "recipe_id": "random_forest_trainer", "config": {"target_column": "target", "n_estimators": 10}},
                {"id": "n_eval", "recipe_id": "model_evaluator", "config": {}}
            ],
            "edges": [
                {"source": "n_csv", "target": "n_split"},
                {"source": "n_split", "target": "n_model"},
                {"source": "n_split", "target": "n_eval"},
                {"source": "n_model", "target": "n_eval"}
            ]
        }

        exec1_resp = await client.post(
            f"/api/v1/workflows/execute?workflow_id={wf_id}&auto_save=true",
            json=dag_v1
        )
        assert exec1_resp.status_code == 200
        exec1_data = exec1_resp.json()
        assert exec1_data["status"] == "SUCCESS"
        exec1_id = exec1_data["execution_id"]

        # 3. Test Endpoint 1: GET /{id}/history (Verify v1 recorded)
        hist1_resp = await client.get(f"/api/v1/workflows/{wf_id}/history")
        assert hist1_resp.status_code == 200
        hist1_list = hist1_resp.json()
        assert len(hist1_list) == 1
        assert hist1_list[0]["version_number"] == 1
        assert hist1_list[0]["id"] == exec1_id
        assert hist1_list[0]["status"] == "SUCCESS"
        assert hist1_list[0]["nodes_count"] == 4

        # 4. Run #2 (v2): Add SMOTE Resampler and change Random Forest n_estimators=25
        dag_v2 = {
            "name": "Audit History Pipeline v2 (SMOTE)",
            "dataset_id": dataset_id,
            "nodes": [
                {"id": "n_csv", "recipe_id": "csv_loader", "config": {"dataset_id": dataset_id}},
                {"id": "n_split", "recipe_id": "train_test_split", "config": {"target_column": "target", "test_size": 0.33, "random_state": 42}},
                {"id": "n_smote", "recipe_id": "class_imbalance_resampler", "config": {"strategy": "random_oversample", "sampling_ratio": 1.0, "target_column": "target"}},
                {"id": "n_model", "recipe_id": "random_forest_trainer", "config": {"target_column": "target", "n_estimators": 25}},
                {"id": "n_eval", "recipe_id": "model_evaluator", "config": {}}
            ],
            "edges": [
                {"source": "n_csv", "target": "n_split"},
                {"source": "n_split", "target": "n_smote"},
                {"source": "n_split", "target": "n_eval"},
                {"source": "n_smote", "target": "n_model"},
                {"source": "n_model", "target": "n_eval"}
            ]
        }

        exec2_resp = await client.post(
            f"/api/v1/workflows/execute?workflow_id={wf_id}&auto_save=true",
            json=dag_v2
        )
        assert exec2_resp.status_code == 200
        exec2_data = exec2_resp.json()
        assert exec2_data["status"] == "SUCCESS"
        exec2_id = exec2_data["execution_id"]
        assert exec2_id != exec1_id

        # 5. Verify GET /{id}/history returns 2 runs sorted descending (v2 first, then v1)
        hist2_resp = await client.get(f"/api/v1/workflows/{wf_id}/history")
        assert hist2_resp.status_code == 200
        hist2_list = hist2_resp.json()
        assert len(hist2_list) == 2
        assert hist2_list[0]["version_number"] == 2
        assert hist2_list[0]["id"] == exec2_id
        assert hist2_list[0]["nodes_count"] == 5
        assert hist2_list[1]["version_number"] == 1
        assert hist2_list[1]["id"] == exec1_id
        assert hist2_list[1]["nodes_count"] == 4

        # 6. Test Endpoint 2: GET /{id}/history/{exec_id} (Full Deep-Dive)
        detail_resp = await client.get(f"/api/v1/workflows/{wf_id}/history/{exec1_id}")
        assert detail_resp.status_code == 200
        detail_data = detail_resp.json()
        assert detail_data["id"] == exec1_id
        assert detail_data["version_number"] == 1
        assert len(detail_data["snapshot_nodes"]) == 4
        assert "n_smote" not in [n["id"] for n in detail_data["snapshot_nodes"]]
        assert "metrics" in detail_data
        assert "reports" in detail_data
        assert "logs" in detail_data

        # 7. Test Endpoint 4: GET /{id}/history/compare (Side-by-Side Diff)
        comp_resp = await client.get(f"/api/v1/workflows/{wf_id}/history/compare?run_a={exec1_id}&run_b={exec2_id}")
        assert comp_resp.status_code == 200
        comp_data = comp_resp.json()
        assert comp_data["workflow_id"] == wf_id
        assert comp_data["run_a"]["version_number"] == 1
        assert comp_data["run_b"]["version_number"] == 2

        # Verify config_diff caught that SMOTE was added and n_estimators changed
        config_diff = comp_data["config_diff"]
        added_node_ids = [n["id"] for n in config_diff["nodes_added"]]
        assert "n_smote" in added_node_ids
        param_changes = config_diff["parameter_changes"]
        est_change = next((p for p in param_changes if p["node_id"] == "n_model" and p["parameter"] == "n_estimators"), None)
        assert est_change is not None
        assert est_change["val_run_a"] == 10
        assert est_change["val_run_b"] == 25

        # 8. Test Endpoint 3: POST /{id}/history/{exec_id}/rollback (Restore to v1)
        # Verify active workflow currently has 5 nodes (v2 with SMOTE)
        active_resp_before = await client.get(f"/api/v1/workflows/{wf_id}")
        assert active_resp_before.status_code == 200
        assert len(active_resp_before.json()["nodes"]) == 5

        # Perform rollback to Run #1
        rollback_resp = await client.post(f"/api/v1/workflows/{wf_id}/history/{exec1_id}/rollback")
        assert rollback_resp.status_code == 200
        rb_data = rollback_resp.json()
        assert rb_data["id"] == wf_id
        assert len(rb_data["nodes"]) == 4
        assert "n_smote" not in [n["id"] for n in rb_data["nodes"]]
        # Verify Random Forest parameter n_estimators is restored back to 10
        assert rb_data["node_configs"]["n_model"]["config"]["n_estimators"] == 10
        assert rb_data["last_execution"]["rolled_back_from_version"] == 1

        # Verify GET /{id} confirms the restored state in database
        active_resp_after = await client.get(f"/api/v1/workflows/{wf_id}")
        assert active_resp_after.status_code == 200
        assert len(active_resp_after.json()["nodes"]) == 4
        assert "n_smote" not in [n["id"] for n in active_resp_after.json()["nodes"]]


@pytest.mark.asyncio
async def test_workflow_history_backward_compatibility_auto_synthesis():
    """Verify that existing legacy workflows with last_execution automatically synthesize Run #1."""
    await init_db()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        legacy_id = f"wf_legacy_{uuid.uuid4().hex[:8]}"
        save_resp = await client.post("/api/v1/workflows/", json={
            "id": legacy_id,
            "name": "Legacy Workflow Pre-History",
            "nodes": [{"id": "n1", "recipe_id": "csv_loader"}],
            "edges": [],
            "node_configs": {},
            "last_execution": {
                "execution_id": f"exec_legacy_{uuid.uuid4().hex[:6]}",
                "status": "SUCCESS",
                "final_metrics": {"accuracy": 0.95},
                "execution_logs": ["Legacy run completed."]
            }
        })
        assert save_resp.status_code == 201

        # Query history for this legacy workflow
        hist_resp = await client.get(f"/api/v1/workflows/{legacy_id}/history")
        assert hist_resp.status_code == 200
        hist_list = hist_resp.json()
        assert len(hist_list) == 1
        assert hist_list[0]["version_number"] == 1
        assert hist_list[0]["metrics"]["accuracy"] == 0.95
        assert "Initial Execution" in hist_list[0]["run_label"]

