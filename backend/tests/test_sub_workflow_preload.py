"""
Batch 1 of the sub-process fixes: the child workflow is loaded in the API request through the
tenant-guarded ORM (never across tenants), its saved credentials are preloaded, and the
Add Workflow recipe runs that prepared graph. Uses a real (in-memory SQLite) database.
"""
import uuid

import pandas as pd
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

import backend.app.infrastructure.database.tenant_guard  # noqa: F401  (registers the tenant filter)
import backend.app.credentials.models  # noqa: F401
from backend.app.core.tenant_context import set_current_tenant_id
from backend.app.credentials.service import CredentialService
from backend.app.engine.dag.graph import WorkflowGraph, WorkflowNode
from backend.app.infrastructure.database.session import Base
from backend.app.recipes.flow_control.add_workflow import AddWorkflowRecipe
from backend.app.workflows.models import Workflow
from backend.app.workflows.router import _storable_config
from backend.app.workflows.sub_workflows import (
    SUB_WORKFLOW_DEFINITION_KEY,
    SUB_WORKFLOW_ERROR_KEY,
    preload_sub_workflows,
)


@pytest_asyncio.fixture
async def db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)()
    yield session
    await session.close()
    await engine.dispose()
    set_current_tenant_id(None)


async def _save_child(db, tenant_id, nodes, edges, node_configs, name="Child"):
    set_current_tenant_id(tenant_id)
    wf = Workflow(id=str(uuid.uuid4()), name=name, nodes=nodes, edges=edges,
                  node_configs=node_configs, is_active=True,
                  last_execution={"status": "SUCCESS"})
    db.add(wf)
    await db.commit()
    return wf.id


def _parent(child_id):
    return WorkflowGraph(nodes=[
        WorkflowNode(id="sub", recipe_id="add_workflow", config={"workflow_id": child_id}),
    ], edges=[])


async def _anchor_child(db, tenant_id=1, extra_node=None):
    """Start -> End child, saved in the nested {"recipe_id", "config": {...}} format the API writes."""
    nodes = [{"id": "s", "recipe_id": "workflow_start"}, {"id": "e", "recipe_id": "workflow_end"}]
    configs = {"s": {"recipe_id": "workflow_start", "config": {"mock_record_count": 2}},
               "e": {"recipe_id": "workflow_end", "config": {}}}
    if extra_node:
        nodes.append(extra_node[0])
        configs[extra_node[0]["id"]] = extra_node[1]
    return await _save_child(db, tenant_id, nodes, [{"source": "s", "target": "e"}], configs)


@pytest.mark.asyncio
async def test_child_is_prepared_with_nested_configs_and_credentials(db):
    set_current_tenant_id(1)
    cred = await CredentialService.create_credential(
        db=db, provider="gmail", name="mail", data={"smtp_password": "app-pass"}, tenant_id=1)
    child_id = await _anchor_child(db, extra_node=(
        {"id": "mail", "recipe_id": "gmail"},
        {"recipe_id": "gmail", "config": {"credential_id": cred.id, "smtp_user": "u@x.com"}},
    ))

    parent = _parent(child_id)
    await preload_sub_workflows(parent, db, tenant_id=1)

    cfg = parent.nodes[0].config
    assert SUB_WORKFLOW_ERROR_KEY not in cfg
    child = cfg[SUB_WORKFLOW_DEFINITION_KEY]
    by_id = {n.id: n for n in child.nodes}
    assert by_id["s"].config == {"mock_record_count": 2}          # nested config unwrapped
    assert by_id["mail"].config["smtp_password"] == "app-pass"    # saved credential preloaded
    assert child.name == "Child"


@pytest.mark.asyncio
async def test_child_of_another_tenant_is_not_loaded(db):
    child_id = await _anchor_child(db, tenant_id=1)
    parent = _parent(child_id)

    set_current_tenant_id(2)
    await preload_sub_workflows(parent, db, tenant_id=2)

    cfg = parent.nodes[0].config
    assert SUB_WORKFLOW_DEFINITION_KEY not in cfg
    assert "could not be found" in cfg[SUB_WORKFLOW_ERROR_KEY]

    # The recipe reports it instead of reaching the database itself
    out = AddWorkflowRecipe().execute(
        inputs={"dataframe": pd.DataFrame({"a": [1]})},
        config={**cfg, "propagate_errors": False}, context={})
    assert out["output_summary"]["status"] == "NOT_FOUND"


@pytest.mark.asyncio
async def test_self_referencing_child_is_rejected(db):
    set_current_tenant_id(1)
    child_id = str(uuid.uuid4())
    db.add(Workflow(
        id=child_id, name="Loop", is_active=True, edges=[],
        nodes=[{"id": "s", "recipe_id": "workflow_start"}, {"id": "e", "recipe_id": "workflow_end"},
               {"id": "again", "recipe_id": "add_workflow"}],
        node_configs={"again": {"workflow_id": child_id}},
    ))
    await db.commit()

    parent = _parent(child_id)
    await preload_sub_workflows(parent, db, tenant_id=1, call_stack=["parent-wf"])

    child = parent.nodes[0].config[SUB_WORKFLOW_DEFINITION_KEY]
    nested = next(n for n in child.nodes if n.id == "again")
    assert "Circular" in nested.config[SUB_WORKFLOW_ERROR_KEY]


@pytest.mark.asyncio
async def test_prepared_child_runs_and_is_never_saved(db):
    child_id = await _anchor_child(db)
    parent = _parent(child_id)
    await preload_sub_workflows(parent, db, tenant_id=1)
    cfg = parent.nodes[0].config

    in_df = pd.DataFrame({"id": [1, 2, 3], "amount": [10.0, 20.0, 30.0]})
    out = AddWorkflowRecipe().execute(inputs={"dataframe": in_df}, config=cfg, context={})
    assert out["output_summary"]["status"] == "SUCCESS"
    pd.testing.assert_frame_equal(out["dataframe"].reset_index(drop=True), in_df)

    # Runtime-only key never reaches the DB payload
    assert SUB_WORKFLOW_DEFINITION_KEY not in _storable_config(cfg)
    assert _storable_config(cfg)["workflow_id"] == child_id
