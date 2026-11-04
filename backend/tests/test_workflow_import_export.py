import pytest
import uuid
import json
from unittest.mock import MagicMock, AsyncMock
from fastapi import HTTPException
from fastapi.datastructures import UploadFile
import io

from backend.app.workflows.router import (
    sanitize_node_configs_credentials,
    _execute_workflow_import,
    export_workflow,
    import_workflow,
    import_workflow_file
)
from backend.app.workflows.models import Workflow
from backend.app.datasets.models import Dataset
from backend.app.core.security import TokenData


def test_sanitize_node_configs_credentials():
    dirty_configs = {
        "node_1": {
            "recipe_id": "zendesk_connector",
            "config": {
                "subdomain": "mycompany",
                "email": "agent@company.com",
                "api_key": "secret_zendesk_token_123",
                "token": "bearer_abc_789",
                "client_secret": "top_secret_xyz"
            }
        },
        "node_2": {
            "recipe_id": "tfidf_vectorizer",
            "config": {
                "max_features": 500,
                "token_pattern": r"\b\w+\b",
                "max_tokens": 100,
                "password": "super_secret_db_pass"
            }
        }
    }

    clean = sanitize_node_configs_credentials(dirty_configs)

    # Sensitive keys should be wiped
    assert clean["node_1"]["config"]["api_key"] == ""
    assert clean["node_1"]["config"]["token"] == ""
    assert clean["node_1"]["config"]["client_secret"] == ""
    assert clean["node_2"]["config"]["password"] == ""

    # Non-sensitive keys must remain unchanged
    assert clean["node_1"]["config"]["subdomain"] == "mycompany"
    assert clean["node_1"]["config"]["email"] == "agent@company.com"
    assert clean["node_2"]["config"]["max_features"] == 500
    assert clean["node_2"]["config"]["token_pattern"] == r"\b\w+\b"
    assert clean["node_2"]["config"]["max_tokens"] == 100


@pytest.mark.asyncio
async def test_import_workflow_remapping_and_persistence():
    mock_db = AsyncMock()
    
    mock_res = MagicMock()
    mock_res.first.return_value = None
    mock_res.scalar_one_or_none.return_value = None
    mock_res.scalars().all.return_value = []
    mock_db.execute = AsyncMock(return_value=mock_res)
    
    added_workflows = []
    mock_db.add = MagicMock(side_effect=lambda obj: added_workflows.append(obj))
    mock_db.commit = AsyncMock()
    mock_db.refresh = AsyncMock()

    sample_export = {
        "schema_version": "1.0.0",
        "exported_at": "2026-11-04T12:00:00Z",
        "metadata": {
            "name": "Customer Churn Predictor",
            "description": "End-to-end churn classification"
        },
        "pipeline": {
            "nodes": [
                {"id": "old_node_1", "recipe_id": "csv_loader", "position": {"x": 50, "y": 100}},
                {"id": "old_node_2", "recipe_id": "missing_value_imputer", "position": {"x": 250, "y": 100}}
            ],
            "edges": [
                {"id": "old_edge_1", "source": "old_node_1", "target": "old_node_2"}
            ],
            "node_configs": {
                "old_node_1": {"recipe_id": "csv_loader", "config": {"filename": "churn.csv"}},
                "old_node_2": {"recipe_id": "missing_value_imputer", "config": {"strategy": "mean"}}
            }
        }
    }

    imported_wf = await _execute_workflow_import(
        db=mock_db,
        raw_payload=sample_export,
        name_override="Imported Churn Test",
        generate_new_ids=True
    )

    assert len(added_workflows) == 1
    wf = added_workflows[0]

    assert wf.name == "Imported Churn Test"
    assert wf.description == "End-to-end churn classification"
    assert len(wf.nodes) == 2
    assert len(wf.edges) == 1

    # Check node ID remapping
    new_node_ids = [n["id"] for n in wf.nodes]
    assert "old_node_1" not in new_node_ids
    assert "old_node_2" not in new_node_ids
    assert new_node_ids[0].startswith("node_")
    assert new_node_ids[1].startswith("node_")

    # Check edge source and target match new node IDs
    edge = wf.edges[0]
    assert edge["source"] == new_node_ids[0]
    assert edge["target"] == new_node_ids[1]

    # Check node configs are keyed by new node IDs
    assert new_node_ids[0] in wf.node_configs
    assert new_node_ids[1] in wf.node_configs
    assert "old_node_1" not in wf.node_configs
    assert wf.node_configs[new_node_ids[1]]["config"]["strategy"] == "mean"


@pytest.mark.asyncio
async def test_import_workflow_rejects_unknown_recipes():
    mock_db = AsyncMock()

    sample_export = {
        "schema_version": "1.0.0",
        "metadata": {"name": "Bad Pipeline"},
        "pipeline": {
            "nodes": [
                {"id": "n1", "recipe_id": "unknown_fictitious_recipe_12345"}
            ],
            "edges": [],
            "node_configs": {}
        }
    }

    with pytest.raises(HTTPException) as exc_info:
        await _execute_workflow_import(
            db=mock_db,
            raw_payload=sample_export
        )

    assert exc_info.value.status_code == 400
    assert "unknown or unsupported recipe" in str(exc_info.value.detail)
    assert "unknown_fictitious_recipe_12345" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_import_workflow_with_target_dataset_mapping():
    mock_db = AsyncMock()
    
    mock_ds = Dataset(id="ds-target-999", name="target_clean.csv", tenant_id="tenant-1")
    mock_res = MagicMock()
    mock_res.first.return_value = None
    mock_res.scalar_one_or_none.return_value = mock_ds
    mock_res.scalars().all.return_value = [mock_ds]
    mock_db.execute = AsyncMock(return_value=mock_res)
    
    added_workflows = []
    mock_db.add = MagicMock(side_effect=lambda obj: added_workflows.append(obj))
    mock_db.commit = AsyncMock()
    mock_db.refresh = AsyncMock()

    sample_export = {
        "pipeline": {
            "nodes": [
                {"id": "n1", "recipe_id": "csv_loader", "config": {"dataset_id": "old-ds-1", "dataset_name": "old.csv"}},
                {"id": "n2", "recipe_id": "feature_scaler", "config": {"method": "standard"}}
            ],
            "edges": [{"id": "e1", "source": "n1", "target": "n2"}],
            "node_configs": {
                "n1": {"recipe_id": "csv_loader", "config": {"dataset_id": "old-ds-1", "dataset_name": "old.csv"}},
                "n2": {"recipe_id": "feature_scaler", "config": {"method": "standard"}}
            }
        }
    }

    await _execute_workflow_import(
        db=mock_db,
        raw_payload=sample_export,
        target_dataset_id="ds-target-999"
    )

    assert len(added_workflows) == 1
    wf = added_workflows[0]
    assert wf.dataset_id == "ds-target-999"
    assert wf.dataset_name == "target_clean.csv"

    # Loader node config should have been updated
    loader_node = [n for n in wf.nodes if n["recipe_id"] == "csv_loader"][0]
    assert loader_node["config"]["dataset_id"] == "ds-target-999"
    assert loader_node["config"]["dataset_name"] == "target_clean.csv"


@pytest.mark.asyncio
async def test_export_workflow_endpoint():
    mock_db = AsyncMock()
    
    # Mock workflow in DB with credentials
    wf = Workflow(
        id="wf-12345",
        name="Telecom Churn XGBoost",
        description="Predict churn",
        dataset_id="ds-1",
        dataset_name="churn.csv",
        nodes=[
            {"id": "n1", "recipe_id": "zendesk_connector"},
            {"id": "n2", "recipe_id": "xgboost_trainer"}
        ],
        edges=[{"id": "e1", "source": "n1", "target": "n2"}],
        node_configs={
            "n1": {
                "recipe_id": "zendesk_connector",
                "config": {"subdomain": "corp", "api_key": "secret_key_999"}
            },
            "n2": {"recipe_id": "xgboost_trainer", "config": {"n_estimators": 100}}
        },
        last_execution={
            "status": "SUCCESS",
            "execution_id": "exec-abc",
            "total_duration_ms": 1250.0,
            "final_metrics": {"accuracy": 0.88, "roc_auc": 0.91}
        },
        is_active=True
    )
    mock_res = MagicMock()
    mock_res.first.return_value = None
    mock_res.scalar_one_or_none.return_value = wf
    mock_res.scalars().all.return_value = []
    mock_db.execute = AsyncMock(return_value=mock_res)

    current_user = TokenData(sub="u-1", tenant_id=1, roles=["Data Scientist"], permissions=[], email="test_scientist@company.com")

    # 1. Standard JSON export
    export_result = await export_workflow(
        workflow_id="wf-12345",
        download=False,
        include_execution_summary=True,
        db=mock_db,
        current_user=current_user
    )

    assert export_result["schema_version"] == "1.0.0"
    assert export_result["metadata"]["name"] == "Telecom Churn XGBoost"
    assert export_result["metadata"]["exported_by"] == "test_scientist@company.com"
    assert export_result["pipeline"]["node_configs"]["n1"]["config"]["api_key"] == ""
    assert export_result["pipeline"]["node_configs"]["n2"]["config"]["n_estimators"] == 100
    assert export_result["last_execution_summary"]["status"] == "SUCCESS"
    assert export_result["last_execution_summary"]["final_metrics"]["accuracy"] == 0.88

    # 2. Downloadable file export
    download_res = await export_workflow(
        workflow_id="wf-12345",
        download=True,
        db=mock_db,
        current_user=current_user
    )

    assert download_res.headers["Content-Disposition"].startswith('attachment; filename="')
    assert "Telecom_Churn_XGBoost_export.json" in download_res.headers["Content-Disposition"]
    parsed_body = json.loads(download_res.body.decode("utf-8"))
    assert parsed_body["schema_version"] == "1.0.0"
