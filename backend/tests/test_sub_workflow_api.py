"""
End-to-end through the API on the configured database (PostgreSQL in this project):
the sub-workflow dropdown list and a parent run that uses a sub-process.
"""
import uuid

import pytest
from httpx import AsyncClient, ASGITransport

from backend.app.main import app
from backend.app.core.security import get_current_user, TokenData
from backend.app.core.tenant_context import set_current_tenant_id
from backend.app.infrastructure.database.session import engine, init_db

ROLES = ["Tenant Admin", "Data Scientist", "ML Engineer"]


def _as(tenant_id):
    async def _user():
        set_current_tenant_id(tenant_id)
        return TokenData(f"user-t{tenant_id}", tenant_id, ROLES, ["*"])
    return _user


@pytest.fixture(autouse=True)
def _restore_override():
    saved = app.dependency_overrides.get(get_current_user)
    yield
    if saved is not None:
        app.dependency_overrides[get_current_user] = saved
    else:
        app.dependency_overrides.pop(get_current_user, None)


async def _list_ids(client, query, name):
    resp = await client.get("/api/v1/workflows/", params={"limit": 200, "search": name, **dict([query.split("=")])})
    assert resp.status_code == 200, resp.text
    return [w["id"] for w in resp.json()["data"]]


@pytest.mark.asyncio
async def test_sub_process_listed_and_runs_inside_parent():
    await engine.dispose()  # pooled connections from earlier tests belong to a closed event loop
    await init_db()
    app.dependency_overrides[get_current_user] = _as(1)
    child_name = f"Sub-process {uuid.uuid4().hex[:6]}"
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/v1/workflows/", json={
            "id": str(uuid.uuid4()), "name": child_name,
            "nodes": [{"id": "s", "recipe_id": "workflow_start"}, {"id": "e", "recipe_id": "workflow_end"}],
            "edges": [{"source": "s", "target": "e"}],
            "node_configs": {"s": {"mock_record_count": 3}, "e": {}},
            "last_execution": {"status": "SUCCESS"},
        })
        assert resp.status_code in (200, 201), resp.text
        child_id = resp.json()["id"]
        plain = await client.post("/api/v1/workflows/", json={
            "id": str(uuid.uuid4()), "name": f"{child_name} plain",
            "nodes": [{"id": "x", "recipe_id": "workflow_start"}], "edges": [], "node_configs": {},
            "last_execution": {"status": "SUCCESS"},
        })
        plain_id = plain.json()["id"]
        try:
            # Dropdown list and status filter work on this database (json_extract was SQLite-only)
            assert child_id in await _list_ids(client, "sub_workflow_eligible=true", child_name)
            assert plain_id not in await _list_ids(client, "sub_workflow_eligible=true", child_name)
            assert child_id in await _list_ids(client, "status=success", child_name)

            # Batch 4: flag + latest version in the list, versions endpoint, pinned run
            listed = (await client.get("/api/v1/workflows/", params={
                "sub_workflow_eligible": "true", "search": child_name})).json()["data"]
            item = next(w for w in listed if w["id"] == child_id)
            assert item["is_sub_workflow"] is True and item["latest_version"] == 1
            versions = (await client.get(f"/api/v1/workflows/{child_id}/sub-workflow-versions")).json()
            assert versions["latest_version"] == 1 and versions["versions"][0]["is_latest"] is True
            pinned_run = await client.post("/api/v1/workflows/execute", json={
                "name": "Parent pinned",
                "nodes": [{"id": "sub", "recipe_id": "add_workflow",
                           "config": {"workflow_id": child_id, "workflow_version": "1"}}],
                "edges": [],
            })
            pinned_node = next(n for n in pinned_run.json()["node_results"] if n["node_id"] == "sub")
            assert pinned_node["status"] == "SUCCESS", pinned_node
            assert pinned_node["output_summary"]["output_summary"]["sub_workflow_version"] == 1

            # Parent: Add Workflow -> runs the child prepared by the router
            run = await client.post("/api/v1/workflows/execute", json={
                "name": "Parent with sub-process",
                "nodes": [{"id": "sub", "recipe_id": "add_workflow", "config": {"workflow_id": child_id}}],
                "edges": [],
            })
            assert run.status_code == 200, run.text
            result = run.json()
            node = next(n for n in result["node_results"] if n["node_id"] == "sub")
            assert node["status"] == "SUCCESS", node
            summary = node["output_summary"]["output_summary"]
            assert summary["sub_workflow_id"] == child_id
            # Nothing upstream of the Add Workflow node: the child gets an empty table,
            # never the Start Anchor's standalone sample rows
            assert summary["output_records_emitted"] == 0

            # Another tenant can neither list nor run it
            app.dependency_overrides[get_current_user] = _as(2)
            assert child_id not in await _list_ids(client, "sub_workflow_eligible=true", child_name)
            run2 = await client.post("/api/v1/workflows/execute", json={
                "name": "Other tenant",
                "nodes": [{"id": "sub", "recipe_id": "add_workflow", "config": {"workflow_id": child_id}}],
                "edges": [],
            })
            assert run2.status_code == 200, run2.text
            node2 = next(n for n in run2.json()["node_results"] if n["node_id"] == "sub")
            assert node2["status"] == "FAILED"
            assert "could not be found" in (node2["error_message"] or "")
        finally:
            app.dependency_overrides[get_current_user] = _as(1)
            await client.delete(f"/api/v1/workflows/{child_id}")
            await client.delete(f"/api/v1/workflows/{plain_id}")
