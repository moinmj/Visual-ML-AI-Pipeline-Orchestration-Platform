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
