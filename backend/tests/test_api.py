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
