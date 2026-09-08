import pytest
from httpx import AsyncClient, ASGITransport

from backend.app.main import app
from backend.app.infrastructure.database.session import init_db
from backend.app.infrastructure.database.tenant_session import get_tenant_db
from backend.app.tenant_data.models import (
    AggregationType,
    DimensionAttributeV3,
    DimensionV3,
    EnvironmentV3,
    ModelDimensionV3,
    ModelMeasureV3,
    ModelV3,
)


import uuid

async def _seed_environment(tenant_id: int):
    uid = uuid.uuid4().hex[:8]
    env_name = f"Test Env {uid}"
    model_name = f"model_{uid}"
    async for db in get_tenant_db():
        env = EnvironmentV3(tenant_id=tenant_id, name=env_name, description="test env")
        db.add(env)
        await db.flush()

        dimension = DimensionV3(environment_id=env.id, tenant_id=tenant_id, name=f"dim_{uid}", key_column="customer_id", text_column="customer_name")
        db.add(dimension)
        await db.flush()
        db.add(DimensionAttributeV3(dimension_id=dimension.id, column_name="customer_name"))

        model = ModelV3(
            environment_id=env.id,
            tenant_id=tenant_id,
            name=model_name,
            description="orders fact",
            druid_datasource_name=f"orders_table_{uid}",
        )
        db.add(model)
        await db.flush()

        db.add(ModelMeasureV3(model_id=model.id, name="revenue", display_name="Revenue", source_column="amount", aggregation_type=AggregationType.SUM))
        db.add(ModelDimensionV3(model_id=model.id, dimension_id=dimension.id, join_key_fact_column="customer_id"))

        await db.commit()
        return env.id, model.id, env_name, model_name


@pytest.mark.asyncio
async def test_tenant_data_requires_auth_and_scopes_by_tenant():
    await init_db()
    env_id, model_id, env_name, model_name = await _seed_environment(tenant_id=1)

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # No token -> 401 or 403 (confirms auth is enforced)
            resp = await client.get("/api/v1/tenant-data/environments")
            assert resp.status_code in (401, 403)

            # Get a dev token for tenant 1
            resp = await client.post(
                "/api/v1/auth/dev-token",
                json={"sub": "user-1", "tenant_id": 1, "roles": ["Tenant Admin"], "permissions": []},
            )
            assert resp.status_code == 200
            token = resp.json()["access_token"]
            headers = {"Authorization": f"Bearer {token}"}

            # Verify whoami
            whoami_resp = await client.get("/api/v1/auth/whoami", headers=headers)
            assert whoami_resp.status_code == 200
            assert whoami_resp.json()["tenant_id"] == 1

            # List environments for tenant 1
            resp = await client.get("/api/v1/tenant-data/environments", headers=headers)
            assert resp.status_code == 200
            envs = resp.json()
            assert any(e["environment_id"] == env_id for e in envs)
            matched = next(e for e in envs if e["environment_id"] == env_id)
            assert any(m["name"] == model_name for m in matched["models"])

            # A token for a different tenant must NOT see tenant 1's environment
            resp = await client.post(
                "/api/v1/auth/dev-token",
                json={"sub": "user-2", "tenant_id": 99999, "roles": [], "permissions": []},
            )
            other_token = resp.json()["access_token"]
            resp = await client.get(
                "/api/v1/tenant-data/environments",
                params={"environment_id": env_id},
                headers={"Authorization": f"Bearer {other_token}"},
            )
            assert resp.status_code == 200
            assert resp.json() == []

            # Model ingest is scoped through the environment too - wrong tenant -> 404
            resp = await client.post(
                f"/api/v1/tenant-data/environments/{env_id}/models/{model_id}/ingest",
                json={},
                headers={"Authorization": f"Bearer {other_token}"},
            )
            assert resp.status_code == 404
    finally:
        # Clean up seeded environment
        async for db in get_tenant_db():
            env_obj = await db.get(EnvironmentV3, env_id)
            if env_obj:
                await db.delete(env_obj)
                await db.commit()
            break
