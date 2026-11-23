"""Save/update responses must never return a secret — not plaintext, not ciphertext."""
import uuid

import pytest
from httpx import AsyncClient, ASGITransport

from backend.app.main import app
from backend.app.core.security import get_current_user, TokenData
from backend.app.infrastructure.database.session import init_db

ROLES = ["Tenant Admin", "Data Scientist", "ML Engineer"]
PLAIN = "rsue weop wtvo expq "
MASK = "••••••••"


@pytest.fixture(autouse=True)
def _as_tenant():
    saved = app.dependency_overrides.get(get_current_user)
    app.dependency_overrides[get_current_user] = lambda: TokenData("user-t1", 1, ROLES, ["*"])
    yield
    if saved is not None:
        app.dependency_overrides[get_current_user] = saved
    else:
        app.dependency_overrides.pop(get_current_user, None)


def _body(wf_id, password):
    cfg = {"smtp_host": "smtp.gmail.com", "smtp_user": "u@x.com", "smtp_password": password}
    return {"id": wf_id, "name": f"Mask test {wf_id[:8]}", "edges": [],
            "nodes": [{"id": "g1", "recipe_id": "gmail", "data": {"config": dict(cfg)}}],
            "node_configs": {"g1": dict(cfg)}}


def _assert_masked(resp):
    assert resp.status_code in (200, 201), resp.text
    data = resp.json()
    assert data["node_configs"]["g1"]["smtp_password"] == MASK
    assert data["nodes"][0]["data"]["config"]["smtp_password"] == MASK
    assert PLAIN not in resp.text


@pytest.mark.asyncio
async def test_create_and_update_responses_mask_secrets():
    await init_db()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/v1/workflows/", json=_body(str(uuid.uuid4()), PLAIN))
        _assert_masked(resp)
        wf_id = resp.json()["id"]

        _assert_masked(await client.put(f"/api/v1/workflows/{wf_id}", json=_body(wf_id, PLAIN)))
        _assert_masked(await client.put(f"/api/v1/workflows/{wf_id}", json=_body(wf_id, MASK)))

        await client.delete(f"/api/v1/workflows/{wf_id}")
