"""
Batch 4 of the sub-process work: versions and pinning, the is_sub_workflow flag, the input
contract, parameters, an isolated child context and the child's step list.
"""
import uuid

import pandas as pd
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

import backend.app.infrastructure.database.tenant_guard  # noqa: F401  (registers the tenant filter)
from backend.app.core.exceptions import ExecutionException
from backend.app.core.tenant_context import set_current_tenant_id
from backend.app.engine.dag.graph import WorkflowGraph, WorkflowNode
from backend.app.infrastructure.database.session import Base
from backend.app.recipes.base.recipe import BaseRecipe
from backend.app.recipes.base.registry import recipe_registry
from backend.app.recipes.flow_control.add_workflow import AddWorkflowRecipe
from backend.app.workflows.models import Workflow
from backend.app.workflows.sub_workflows import (
    SUB_WORKFLOW_DEFINITION_KEY,
    SUB_WORKFLOW_ERROR_KEY,
    SUB_WORKFLOW_VERSION_INFO_KEY,
    list_sub_workflow_versions,
    preload_sub_workflows,
    record_sub_workflow_version,
    resolve_parameters,
    substitute_params,
)

SEEN = []


class _Spy(BaseRecipe):
    recipe_id = "test_b4_spy"
    name = "Spy"
    category = "test"
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    def get_schema(self):
        return {"type": "object", "properties": {}}

    def execute(self, inputs, config, context=None):
        SEEN.append({"config": dict(config), "context_keys": set(context or {}),
                     "params": dict((context or {}).get("sub_workflow_params") or {})})
        return {"dataframe": inputs["dataframe"]}


recipe_registry.register(_Spy())


def _child(start_cfg=None, spy_cfg=None, wf_id="wf_b4"):
    return {
        "id": wf_id, "name": "B4 child",
        "nodes": [{"id": "s", "recipe_id": "workflow_start"}, {"id": "spy", "recipe_id": "test_b4_spy"},
                  {"id": "e", "recipe_id": "workflow_end"}],
        "edges": [{"source": "s", "target": "spy"}, {"source": "spy", "target": "e"}],
        "node_configs": {"s": start_cfg or {}, "spy": spy_cfg or {}},
    }


def _run(child, in_df, context=None, **cfg):
    ctx = {"_cached_sub_workflows": {child["id"]: child}, "execution_id": "parent-b4", **(context or {})}
    return AddWorkflowRecipe().execute(inputs={"dataframe": in_df},
                                       config={"workflow_id": child["id"], **cfg}, context=ctx)


# ── Input contract ────────────────────────────────────────────────────────

CONTRACT = {"input_columns": [
    {"name": "customer_id", "type": "string", "required": True},
    {"name": "amount", "type": "number", "required": True},
    {"name": "note", "type": "string", "required": False},
]}


def test_contract_accepts_matching_table():
    res = _run(_child(CONTRACT), pd.DataFrame({"customer_id": ["c1"], "amount": [10]}))
    assert res["output_summary"]["status"] == "SUCCESS"


def test_contract_rejects_missing_and_mistyped_columns():
    bad = pd.DataFrame({"amount": ["ten"]})
    with pytest.raises(ExecutionException) as exc:
        _run(_child(CONTRACT), bad)
    assert "missing column 'customer_id'" in str(exc.value)
    assert "column 'amount' should be number" in str(exc.value)
    res = _run(_child(CONTRACT), bad, propagate_errors=False)
    assert res["output_summary"]["status"] == "CONTRACT_MISMATCH"
    assert res["_node_status"] == "WARNING"


def test_contract_on_empty_table_checks_columns_not_types():
    empty = pd.DataFrame({"customer_id": pd.Series([], dtype="object"),
                          "amount": pd.Series([], dtype="object")})
    assert _run(_child(CONTRACT), empty)["output_summary"]["status"] == "SUCCESS"


# ── Parameters ────────────────────────────────────────────────────────────

PARAMS = {"parameters": [
    {"name": "threshold", "type": "integer", "required": True},
    {"name": "region", "type": "string", "default": "EU"},
    {"name": "strict", "type": "boolean", "default": False},
]}
SPY_CFG = {"limit": "{{params.threshold}}", "title": "Report for {{params.region}}",
           "nested": {"flags": ["{{params.strict}}"]}, "untouched": "{{params.unknown}}"}


def test_parameters_are_typed_and_substituted_into_child_steps():
    SEEN.clear()
    res = _run(_child(PARAMS, SPY_CFG), pd.DataFrame({"v": [1]}), parameters={"threshold": "50"})
    assert res["output_summary"]["status"] == "SUCCESS"
    cfg = SEEN[0]["config"]
    assert cfg["limit"] == 50                          # whole reference keeps the type
    assert cfg["title"] == "Report for EU"             # default used, embedded as text
    assert cfg["nested"]["flags"] == [False]
    assert cfg["untouched"] == "{{params.unknown}}"     # undeclared reference left as written
    assert SEEN[0]["params"] == {"threshold": 50, "region": "EU", "strict": False}
    assert res["output_summary"]["parameters_used"] == ["region", "strict", "threshold"]


@pytest.mark.parametrize("given, expected", [
    ({}, "parameter 'threshold' is required"),
    ({"threshold": "abc"}, "parameter 'threshold' should be integer"),
    ({"threshold": 1, "colour": "red"}, "unknown parameter 'colour'"),
])
def test_invalid_parameters_are_rejected(given, expected):
    with pytest.raises(ExecutionException, match=expected):
        _run(_child(PARAMS), pd.DataFrame({"v": [1]}), parameters=given)
    res = _run(_child(PARAMS), pd.DataFrame({"v": [1]}), parameters=given, propagate_errors=False)
    assert res["output_summary"]["status"] == "INVALID_PARAMETERS"


def test_parameter_helpers():
    values, problems = resolve_parameters([{"name": "n", "type": "number"}], '{"n": "2.5"}')
    assert values == {"n": 2.5} and problems == []
    assert substitute_params("{{ params.n }}", {"n": 2.5}) == 2.5
    assert resolve_parameters([{"name": "bad-name"}], {})[1]


# ── Isolated context and child step list ──────────────────────────────────

def test_child_context_is_isolated_and_steps_are_reported():
    SEEN.clear()
    parent_ctx = {"model": object(), "target_column": "churn", "dataframe": pd.DataFrame({"x": [9]}),
                  "session_id": "sess-1"}
    res = _run(_child(), pd.DataFrame({"v": [1, 2]}), context=parent_ctx)
    keys = SEEN[0]["context_keys"]
    assert "target_column" not in keys
    assert "model" not in keys
    assert "session_id" in keys
    steps = res["output_summary"]["sub_steps"]
    assert [s["node_id"] for s in steps] == ["s", "spy", "e"]
    assert all(s["status"] == "SUCCESS" for s in steps)
    assert steps[-1]["row_count"] == 2


# ── Versions (real DB, tenant guard active) ───────────────────────────────

@pytest_asyncio.fixture
async def db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)()
    set_current_tenant_id(1)
    yield session
    await session.close()
    await engine.dispose()
    set_current_tenant_id(None)


def _anchor_nodes(limit):
    return ([{"id": "s", "recipe_id": "workflow_start"}, {"id": "spy", "recipe_id": "test_b4_spy"},
             {"id": "e", "recipe_id": "workflow_end"}],
            [{"source": "s", "target": "spy"}, {"source": "spy", "target": "e"}],
            {"spy": {"limit": limit}})


async def _save(db, wf=None, limit=1, anchors=True):
    nodes, edges, configs = _anchor_nodes(limit)
    if not anchors:
        nodes = [{"id": "x", "recipe_id": "test_b4_spy"}]
        edges, configs = [], {}
    if wf is None:
        wf = Workflow(id=str(uuid.uuid4()), name="Versioned", is_active=True)
        db.add(wf)
    wf.nodes, wf.edges, wf.node_configs = nodes, edges, configs
    await db.commit()
    return wf, await record_sub_workflow_version(db, wf)


@pytest.mark.asyncio
async def test_versions_are_recorded_only_when_the_graph_changes(db):
    wf, v = await _save(db, limit=1)
    assert v == 1 and wf.is_sub_workflow is True
    _, v = await _save(db, wf, limit=1)
    assert v == 1                                       # unchanged graph: no new version
    _, v = await _save(db, wf, limit=2)
    assert v == 2
    versions = await list_sub_workflow_versions(db, wf)
    assert [x["version"] for x in versions] == [2, 1]
    assert versions[0]["is_latest"] and versions[0]["matches_current_definition"]

    plain, v = await _save(db, anchors=False)
    assert v is None and plain.is_sub_workflow is False


@pytest.mark.asyncio
async def test_pinned_version_runs_the_old_graph(db):
    wf, _ = await _save(db, limit=1)
    await _save(db, wf, limit=2)

    parent = WorkflowGraph(nodes=[
        WorkflowNode(id="pinned", recipe_id="add_workflow", config={"workflow_id": wf.id, "workflow_version": "1"}),
        WorkflowNode(id="latest", recipe_id="add_workflow", config={"workflow_id": wf.id}),
        WorkflowNode(id="missing", recipe_id="add_workflow", config={"workflow_id": wf.id, "workflow_version": 7}),
    ], edges=[])
    await preload_sub_workflows(parent, db, tenant_id=1)
    pinned, latest, missing = (n.config for n in parent.nodes)

    spy_limit = lambda cfg: next(n for n in cfg[SUB_WORKFLOW_DEFINITION_KEY].nodes if n.id == "spy").config["limit"]
    assert spy_limit(pinned) == 1
    assert pinned[SUB_WORKFLOW_VERSION_INFO_KEY] == {"version": 1, "latest_version": 2, "update_available": True}
    assert spy_limit(latest) == 2
    assert latest[SUB_WORKFLOW_VERSION_INFO_KEY]["version"] == "latest"
    assert "Version 7" in missing[SUB_WORKFLOW_ERROR_KEY] and "latest is 2" in missing[SUB_WORKFLOW_ERROR_KEY]

    SEEN.clear()
    res = AddWorkflowRecipe().execute(inputs={"dataframe": pd.DataFrame({"v": [1]})}, config=pinned,
                                      context={"execution_id": "p"})
    assert SEEN[0]["config"]["limit"] == 1
    assert res["output_summary"]["update_available"] is True
    assert res["output_summary"]["sub_workflow_version"] == 1


@pytest.mark.asyncio
async def test_versions_of_another_tenant_are_invisible(db):
    wf, _ = await _save(db, limit=1)
    set_current_tenant_id(2)
    parent = WorkflowGraph(nodes=[WorkflowNode(
        id="sub", recipe_id="add_workflow", config={"workflow_id": wf.id, "workflow_version": 1})], edges=[])
    await preload_sub_workflows(parent, db, tenant_id=2)
    assert "could not be found" in parent.nodes[0].config[SUB_WORKFLOW_ERROR_KEY]
