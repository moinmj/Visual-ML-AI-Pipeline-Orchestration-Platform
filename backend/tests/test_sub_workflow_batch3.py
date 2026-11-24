"""
Batch 3 of the sub-process fixes: with 'Halt Pipeline on Sub-Workflow Error' off, a failed
sub-process is a WARNING and the steps after it are SKIPPED (other branches keep running);
the End Anchor's no-op 'pass_through' option is gone.
"""
import pandas as pd

from backend.app.engine.dag.graph import WorkflowEdge, WorkflowGraph, WorkflowNode
from backend.app.engine.execution.executor import DAGExecutor
from backend.app.recipes.base.recipe import BaseRecipe
from backend.app.recipes.base.registry import recipe_registry
from backend.app.recipes.flow_control.workflow_end import WorkflowEndRecipe

RAN = []


class _TestRecipe(BaseRecipe):
    category = "test"
    output_types = ["dataframe"]

    def get_schema(self):
        return {"type": "object", "properties": {}}


class _Source(_TestRecipe):
    recipe_id = "test_b3_source"
    name = "Source"
    input_types = []

    def execute(self, inputs, config, context=None):
        RAN.append(config.get("step_name"))
        return {"dataframe": pd.DataFrame({"v": [1, 2, 3]})}


class _Step(_TestRecipe):
    recipe_id = "test_b3_step"
    name = "Step"
    input_types = ["dataframe"]

    def execute(self, inputs, config, context=None):
        RAN.append(config.get("step_name"))
        return {"dataframe": inputs["dataframe"]}


recipe_registry.register(_Source())
recipe_registry.register(_Step())

BROKEN_CHILD = {  # End Anchor not connected to the Start Anchor
    "id": "wf_broken", "name": "Broken child", "edges": [],
    "nodes": [{"id": "s", "recipe_id": "workflow_start"}, {"id": "e", "recipe_id": "workflow_end"}],
}


def _parent(propagate_errors: bool) -> WorkflowGraph:
    """src -> sub (Add Workflow) -> after1 -> after2, plus an independent branch src2 -> other."""
    def n(node_id, recipe_id, **cfg):
        return WorkflowNode(id=node_id, recipe_id=recipe_id, config={"step_name": node_id, **cfg})
    return WorkflowGraph(
        nodes=[
            n("src", "test_b3_source"),
            n("sub", "add_workflow", workflow_id="wf_broken", propagate_errors=propagate_errors),
            n("after1", "test_b3_step"),
            n("after2", "test_b3_step"),
            n("src2", "test_b3_source"),
            n("other", "test_b3_step"),
        ],
        edges=[WorkflowEdge(source=a, target=b) for a, b in
               [("src", "sub"), ("sub", "after1"), ("after1", "after2"), ("src2", "other")]],
    )


def _statuses(result):
    return {r.node_id: r for r in result.node_results}


def test_failed_sub_process_is_warning_and_downstream_is_skipped():
    RAN.clear()
    result = DAGExecutor.execute_workflow(
        execution_id="b3-warn", workflow=_parent(propagate_errors=False),
        context={"_cached_sub_workflows": {"wf_broken": BROKEN_CHILD}})
    nodes = _statuses(result)

    assert nodes["sub"].status == "WARNING"
    assert "not connected to its Start Anchor" in nodes["sub"].error_message
    assert nodes["after1"].status == "SKIPPED"
    assert nodes["after2"].status == "SKIPPED"                 # skipped transitively
    assert nodes["after2"].output_summary["skipped_because_of"] == "sub"
    assert nodes["other"].status == "SUCCESS"                  # independent branch still runs
    assert "after1" not in RAN and "after2" not in RAN and "other" in RAN
    assert result.status == "SUCCESS"
    assert any("Warning" in line and "sub" in line for line in result.logs)
    assert result.step_snapshots["sub"]["status"] == "WARNING"


def test_halt_on_error_still_fails_the_whole_run():
    RAN.clear()
    result = DAGExecutor.execute_workflow(
        execution_id="b3-halt", workflow=_parent(propagate_errors=True),
        context={"_cached_sub_workflows": {"wf_broken": BROKEN_CHILD}})
    nodes = _statuses(result)
    assert result.status == "FAILED"
    assert nodes["sub"].status == "FAILED"
    assert "after1" not in RAN


def test_unconfigured_add_workflow_does_not_pass_data_on():
    RAN.clear()
    graph = _parent(propagate_errors=False)
    graph.nodes[1].config = {"step_name": "sub"}               # no workflow selected
    nodes = _statuses(DAGExecutor.execute_workflow(execution_id="b3-unconf", workflow=graph))
    assert nodes["sub"].status == "WARNING"
    assert "No sub-workflow selected" in nodes["sub"].error_message
    assert nodes["after1"].status == "SKIPPED"


def test_live_progress_counts_warning_and_skipped_nodes():
    from backend.app.engine.execution.job_manager import job_manager
    job = {"node_status": {}, "progress": {"completed": 0, "total": 3}, "current_node": None, "logs": []}
    events = [{"type": "node_finished", "node_id": i, "status": s}
              for i, s in (("a", "SUCCESS"), ("b", "WARNING"), ("c", "SKIPPED"))]
    with job_manager.lock:
        job_manager.jobs["b3-progress"] = job
    try:
        for ev in events:
            job_manager._on_progress("b3-progress", ev)
        assert job["progress"]["completed"] == 3
        assert job["node_status"] == {"a": "SUCCESS", "b": "WARNING", "c": "SKIPPED"}
    finally:
        job_manager.jobs.pop("b3-progress", None)


def test_end_anchor_has_no_pass_through_option():
    props = WorkflowEndRecipe().get_schema()["properties"]
    assert "pass_through" not in props
    out = WorkflowEndRecipe().execute(
        inputs={"dataframe": pd.DataFrame({"v": [1]})}, config={"pass_through": False}, context={})
    assert len(out["dataframe"]) == 1                          # old saved value is ignored
