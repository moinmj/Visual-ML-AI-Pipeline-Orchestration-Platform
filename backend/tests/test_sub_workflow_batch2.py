"""
Batch 2 of the sub-process fixes: no sample data under a parent, no silent pass-through when
the End Anchor produced nothing, one isolated run per call, timeout enforced, anchors checked.
"""
import threading
import time
from typing import Any, Dict, Optional

import pandas as pd
import pytest

from backend.app.core.exceptions import ExecutionException
from backend.app.recipes.base.recipe import BaseRecipe
from backend.app.recipes.base.registry import recipe_registry
from backend.app.recipes.flow_control.add_workflow import AddWorkflowRecipe

STEPS_RUN = []


class _TestRecipe(BaseRecipe):
    def get_schema(self):
        return {"type": "object", "properties": {}}


class _SlowStep(_TestRecipe):
    recipe_id = "test_slow_step"
    name = "Slow test step"
    category = "test"
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None):
        STEPS_RUN.append(config.get("step_name"))
        time.sleep(float(config.get("seconds", 0)))
        return {"dataframe": inputs.get("dataframe")}


class _DropAll(_TestRecipe):
    recipe_id = "test_drop_all_rows"
    name = "Drop all rows"
    category = "test"
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    def execute(self, inputs, config, context=None):
        return {"dataframe": inputs["dataframe"].iloc[0:0]}


recipe_registry.register(_SlowStep())
recipe_registry.register(_DropAll())


def _run(child: dict, in_df: pd.DataFrame, **cfg):
    ctx = {"_cached_sub_workflows": {child["id"]: child}, "execution_id": "parent-run-1"}
    return AddWorkflowRecipe().execute(
        inputs={"dataframe": in_df}, config={"workflow_id": child["id"], **cfg}, context=ctx)


def _chain(*middle, wf_id="wf_chain", configs=None):
    ids = ["s", *[m[0] for m in middle], "e"]
    nodes = ([{"id": "s", "recipe_id": "workflow_start"}]
             + [{"id": i, "recipe_id": r} for i, r in middle]
             + [{"id": "e", "recipe_id": "workflow_end"}])
    edges = [{"source": a, "target": b} for a, b in zip(ids, ids[1:])]
    return {"id": wf_id, "name": "Chain", "nodes": nodes, "edges": edges, "node_configs": configs or {}}


def test_empty_parent_table_stays_empty_instead_of_sample_rows():
    """Fix 4: 0 rows from the parent must not turn into the Start Anchor's sample customers."""
    empty = pd.DataFrame({"customer_id": pd.Series([], dtype="object")})
    res = _run(_chain(), empty)
    assert res["output_summary"]["status"] == "SUCCESS"
    assert len(res["dataframe"]) == 0
    assert "Alpha Corp" not in res["dataframe"].to_string()


def test_output_is_this_runs_table_with_types_intact():
    """Fixes 5/8/11: the End Anchor's DataFrame comes back as-is, per call (no shared cache)."""
    df1 = pd.DataFrame({"when": pd.to_datetime(["2026-01-01", "2026-01-02"]), "v": [1, 2]})
    df2 = pd.DataFrame({"when": pd.to_datetime(["2026-03-05"]), "v": [99]})
    child = _chain()
    out1, out2 = _run(child, df1)["dataframe"], _run(child, df2)["dataframe"]   # same child, same second
    pd.testing.assert_frame_equal(out1.reset_index(drop=True), df1)
    pd.testing.assert_frame_equal(out2.reset_index(drop=True), df2)
    assert str(out1["when"].dtype).startswith("datetime64")


def test_filter_to_zero_rows_is_a_real_empty_result():
    """A sub-process that filters everything out returns 0 rows, not the parent's input."""
    res = _run(_chain(("drop", "test_drop_all_rows")), pd.DataFrame({"v": [1, 2, 3]}))
    assert res["output_summary"]["status"] == "SUCCESS"
    assert len(res["dataframe"]) == 0


def test_timeout_is_enforced_and_child_stops_before_next_step():
    """Fix 9: timeout_seconds stops the parent node and the child's remaining steps."""
    STEPS_RUN.clear()
    child = _chain(("slow1", "test_slow_step"), ("slow2", "test_slow_step"), configs={
        "slow1": {"seconds": 2, "step_name": "slow1"}, "slow2": {"seconds": 0, "step_name": "slow2"}})

    t0 = time.time()
    res = _run(child, pd.DataFrame({"v": [1]}), timeout_seconds=1, propagate_errors=False)
    assert time.time() - t0 < 1.9
    assert res["output_summary"]["status"] == "TIMEOUT"
    assert "timed out after 1 seconds" in res["output_summary"]["error"]

    time.sleep(1.5)                     # slow1 finishes in the background...
    assert STEPS_RUN == ["slow1"]       # ...and slow2 never starts

    with pytest.raises(ExecutionException, match="timed out"):
        _run(child, pd.DataFrame({"v": [1]}), timeout_seconds=1, propagate_errors=True)


@pytest.mark.parametrize("child, expected", [
    ({"id": "w", "name": "Two starts", "edges": [{"source": "s1", "target": "e"}],
      "nodes": [{"id": "s1", "recipe_id": "workflow_start"}, {"id": "s2", "recipe_id": "workflow_start"},
                {"id": "e", "recipe_id": "workflow_end"}]}, "2 Start Anchors"),
    ({"id": "w", "name": "Two ends", "edges": [{"source": "s", "target": "e1"}, {"source": "s", "target": "e2"}],
      "nodes": [{"id": "s", "recipe_id": "workflow_start"}, {"id": "e1", "recipe_id": "workflow_end"},
                {"id": "e2", "recipe_id": "workflow_end"}]}, "2 End Anchors"),
    ({"id": "w", "name": "Disconnected", "edges": [],
      "nodes": [{"id": "s", "recipe_id": "workflow_start"}, {"id": "e", "recipe_id": "workflow_end"}]},
     "not connected to its Start Anchor"),
])
def test_anchor_structure_is_checked_before_running(child, expected):
    """Fix 12: invalid anchor layouts are rejected with a message naming the problem."""
    with pytest.raises(ExecutionException, match=expected):
        _run(child, pd.DataFrame({"v": [1]}))
    res = _run(child, pd.DataFrame({"v": [1]}), propagate_errors=False)
    assert res["output_summary"]["status"] == "INVALID_ANCHORS"


def test_child_runs_under_its_own_execution_id():
    """Fix 8: the child gets a unique run ID derived from the parent's, not the parent's own."""
    seen = []

    class _Spy(_TestRecipe):
        recipe_id = "test_ctx_spy"
        name = "ctx spy"
        category = "test"
        input_types = ["dataframe"]
        output_types = ["dataframe"]

        def execute(self, inputs, config, context=None):
            seen.append((context["execution_id"], context["workflow_id"]))
            return {"dataframe": inputs["dataframe"]}

    recipe_registry.register(_Spy())
    child = _chain(("spy", "test_ctx_spy"), wf_id="wf_spy")
    _run(child, pd.DataFrame({"v": [1]}))
    _run(child, pd.DataFrame({"v": [1]}))
    (id1, wf1), (id2, wf2) = seen
    assert id1.startswith("parent-run-1_sub_") and id2.startswith("parent-run-1_sub_")
    assert id1 != id2
    assert wf1 == wf2 == "wf_spy"
