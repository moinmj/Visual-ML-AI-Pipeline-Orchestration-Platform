import uuid

import pytest
from httpx import AsyncClient, ASGITransport
from jose import jwt

from backend.app.main import app
from backend.app.core.config import settings
from backend.app.core.security import get_current_user, TokenData
from backend.app.infrastructure.database.session import init_db

ROLES = ["Tenant Admin", "Data Scientist", "ML Engineer"]


def _as_tenant(tenant_id: int):
    app.dependency_overrides[get_current_user] = lambda: TokenData(f"user-t{tenant_id}", tenant_id, ROLES, ["*"])


@pytest.fixture(autouse=True)
def _restore_override():
    saved = app.dependency_overrides.get(get_current_user)
    yield
    if saved is not None:
        app.dependency_overrides[get_current_user] = saved
    else:
        app.dependency_overrides.pop(get_current_user, None)


@pytest.mark.asyncio
async def test_workflows_are_isolated_between_tenants():
    await init_db()
    wf_id = f"wf_iso_{uuid.uuid4().hex[:8]}"
    name_a = f"Tenant A Private {uuid.uuid4().hex[:6]}"
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Tenant 1 creates a workflow
        _as_tenant(1)
        resp = await client.post("/api/v1/workflows/", json={
            "id": wf_id, "name": name_a, "nodes": [{"id": "n1", "recipe_id": "csv_loader"}], "edges": [], "node_configs": {},
        })
        assert resp.status_code == 201

        # Tenant 1 sees it
        listed = (await client.get("/api/v1/workflows/?limit=200")).json()["data"]
        assert wf_id in [w["id"] for w in listed]

        # Tenant 2 must not see it anywhere
        _as_tenant(2)
        listed = (await client.get("/api/v1/workflows/?limit=200")).json()
        assert wf_id not in [w["id"] for w in listed["data"]]
        assert (await client.get("/api/v1/workflows/?search=Tenant A Private")).json()["total_records"] == 0
        assert (await client.get(f"/api/v1/workflows/{wf_id}")).status_code == 404
        assert (await client.get(f"/api/v1/workflows/{wf_id}/history")).status_code == 404
        assert (await client.post(f"/api/v1/workflows/{wf_id}/validate")).status_code == 404
        assert (await client.post(f"/api/v1/workflows/{wf_id}/execute")).status_code == 404
        assert (await client.post(f"/api/v1/workflows/{wf_id}/restore")).status_code == 404
        assert (await client.delete(f"/api/v1/workflows/{wf_id}")).status_code == 404
        assert (await client.delete(f"/api/v1/workflows/{wf_id}/history")).status_code == 404

        # Tenant 2 must not be able to overwrite it by reusing the ID (POST, PUT, or execute auto-save)
        body = {"id": wf_id, "name": "hijacked", "nodes": [], "edges": [], "node_configs": {}}
        assert (await client.post("/api/v1/workflows/", json=body)).status_code == 404
        assert (await client.put(f"/api/v1/workflows/{wf_id}", json=body)).status_code == 404
        exec_resp = await client.post(
            f"/api/v1/workflows/execute?workflow_id={wf_id}&auto_save=true",
            json={"name": "hijacked", "nodes": [], "edges": []},
        )
        assert exec_resp.status_code == 404

        # Tenant 2 can create its own workflow, invisible to tenant 1
        own_id = f"wf_iso_{uuid.uuid4().hex[:8]}"
        assert (await client.post("/api/v1/workflows/", json={
            "id": own_id, "name": "Tenant B Own", "nodes": [], "edges": [], "node_configs": {},
        })).status_code == 201
        _as_tenant(1)
        assert (await client.get(f"/api/v1/workflows/{own_id}")).status_code == 404

        # Tenant 1's workflow is untouched
        got = await client.get(f"/api/v1/workflows/{wf_id}")
        assert got.status_code == 200
        assert got.json()["name"] == name_a


@pytest.mark.asyncio
async def test_token_without_tenant_claim_is_rejected():
    app.dependency_overrides.pop(get_current_user, None)
    token = jwt.encode(
        {"sub": "no-tenant-user", "roles": ["Tenant Admin"], "type": "access"},
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/auth/whoami", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 401
        # Workflow list is no longer anonymous
        assert (await client.get("/api/v1/workflows/")).status_code == 401