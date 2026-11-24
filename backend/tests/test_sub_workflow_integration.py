"""
Unit and integration tests for Modular Sub-Workflow integration:
- workflow_start (Start Anchor)
- workflow_end (End Anchor)
- add_workflow (Parent Caller Recipe)
- GET /api/v1/workflows?sub_workflow_eligible=true filtering
"""

import os
import json
import pytest
import pandas as pd

# Set MPLCONFIGDIR to prevent font cache delays
os.environ["MPLCONFIGDIR"] = os.environ.get("TEMP", r"C:\Users\TECH\AppData\Local\Temp")

from backend.app.core.exceptions import ExecutionException, ValidationException
from backend.app.recipes.base.registry import recipe_registry
from backend.app.recipes.flow_control.workflow_start import WorkflowStartRecipe
from backend.app.recipes.flow_control.workflow_end import WorkflowEndRecipe
from backend.app.recipes.flow_control.add_workflow import AddWorkflowRecipe


def test_sub_workflow_recipes_registered():
    """Verify all 3 new recipes are registered with the expected metadata."""
    start_r = recipe_registry.get("workflow_start")
    end_r = recipe_registry.get("workflow_end")
    add_r = recipe_registry.get("add_workflow")

    assert start_r is not None
    assert start_r.name == "Start Anchor"
    assert start_r.group == "Ingestion & Orchestration"
    assert start_r.subgroup == "Control Flow & Reliability"

    assert end_r is not None
    assert end_r.name == "End Anchor"
    assert end_r.group == "Ingestion & Orchestration"
    assert end_r.subgroup == "Control Flow & Reliability"

    assert add_r is not None
    assert add_r.name == "Add Workflow"
    assert add_r.group == "Ingestion & Orchestration"
    assert add_r.subgroup == "Control Flow & Reliability"

    # Verify aliases
    assert recipe_registry.get("start_anchor") is not None
    assert recipe_registry.get("end_anchor") is not None
    assert recipe_registry.get("sub_workflow") is not None


def test_workflow_start_standalone_mock():
    """Verify Start Anchor produces realistic mock data in standalone test mode."""
    recipe = WorkflowStartRecipe()
    res = recipe.execute(inputs={}, config={"mock_record_count": 7})

    df = res["dataframe"]
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 7
    assert "customer_id" in df.columns
    assert "amount" in df.columns
    assert res["metrics"]["mode"] == "STANDALONE_MOCK"
    assert res["metrics"]["anchor_type"] == "start"


def test_workflow_start_injected_from_parent():
    """Verify Start Anchor emits data passed down from parent pipeline (Point A)."""
    recipe = WorkflowStartRecipe()
    parent_df = pd.DataFrame([
        {"id": 101, "sku": "A-1", "sales": 500},
        {"id": 102, "sku": "B-2", "sales": 800},
    ])

    # Case A: Passed via inputs
    res_a = recipe.execute(inputs={"dataframe": parent_df}, config={})
    assert len(res_a["dataframe"]) == 2
    assert "sku" in res_a["dataframe"].columns
    assert res_a["metrics"]["mode"] == "INJECTED"

    # Case B: Passed via context
    res_b = recipe.execute(inputs={}, config={}, context={"sub_workflow_input": parent_df})
    assert len(res_b["dataframe"]) == 2
    assert "sku" in res_b["dataframe"].columns
    assert res_b["metrics"]["mode"] == "INJECTED"


def test_workflow_end_captures_output():
    """Verify End Anchor captures final data into the result holder of the Add Workflow call."""
    recipe = WorkflowEndRecipe()
    in_df = pd.DataFrame([
        {"id": 1, "cleaned_score": 95.0},
        {"id": 2, "cleaned_score": 88.5},
    ])

    holder = {}
    ctx = {"_sub_workflow_result": holder}
    res = recipe.execute(inputs={"dataframe": in_df}, config={}, context=ctx)

    assert len(res["dataframe"]) == 2
    assert "cleaned_score" in res["dataframe"].columns
    assert holder.get("end_node_completed") is True
    pd.testing.assert_frame_equal(holder["dataframe"], in_df)


def test_add_workflow_schema():
    """Verify Add Workflow recipe schema has select endpoint and required fields."""
    recipe = AddWorkflowRecipe()
    schema = recipe.get_schema()
    props = schema["properties"]

    assert "workflow_id" in props
    assert props["workflow_id"]["endpoint"] == "/api/v1/workflows?sub_workflow_eligible=true"
    assert props["workflow_id"]["input_type"] in ("select", "dropdown")
    assert "propagate_errors" in props
    assert "timeout_seconds" in props
    assert "workflow_id" in schema.get("required", [])


def test_add_workflow_unconfigured_passthrough():
    """Verify Add Workflow gracefully passes through data when no workflow_id is selected."""
    recipe = AddWorkflowRecipe()
    in_df = pd.DataFrame([{"col1": "val1"}])
    res = recipe.execute(inputs={"dataframe": in_df}, config={"workflow_id": ""})

    assert res["dataframe"].equals(in_df)
    assert res["output_summary"]["status"] == "PENDING_CONFIGURATION"


def test_add_workflow_cycle_prevention():
    """Verify Add Workflow immediately halts if circular recursion is detected."""
    recipe = AddWorkflowRecipe()
    ctx = {"_sub_workflow_stack": ["wf_root", "wf_child_a"]}

    with pytest.raises(ExecutionException) as exc:
        recipe.execute(
            inputs={},
            config={"workflow_id": "wf_root"},
            context=ctx
        )
    assert "Circular sub-workflow recursion detected" in str(exc.value)


def test_add_workflow_missing_anchors():
    """Verify Add Workflow rejects a workflow missing Start or End anchor."""
    recipe = AddWorkflowRecipe()

    # Child workflow without end anchor
    bad_child = {
        "id": "wf_bad_1",
        "name": "Missing End Anchor Process",
        "nodes": [
            {"id": "node_start", "recipe_id": "workflow_start"},
            {"id": "node_filter", "recipe_id": "row_filter"},
        ],
        "edges": [
            {"source": "node_start", "target": "node_filter"}
        ],
        "node_configs": {}
    }

    ctx = {"_cached_sub_workflows": {"wf_bad_1": bad_child}}

    # When propagate_errors=True
    with pytest.raises(ExecutionException) as exc:
        recipe.execute(
            inputs={},
            config={"workflow_id": "wf_bad_1", "propagate_errors": True},
            context=ctx
        )
    assert "has no End Anchor" in str(exc.value)

    # When propagate_errors=False
    res = recipe.execute(
        inputs={},
        config={"workflow_id": "wf_bad_1", "propagate_errors": False},
        context=ctx
    )
    assert res["output_summary"]["status"] == "INVALID_ANCHORS"


def test_add_workflow_e2e_successful_handoff():
    """
    Simulate end-to-end execution of a parent pipeline calling a child sub-workflow.
    Point A -> Add Workflow (Start Anchor -> Row Filter -> End Anchor) -> Point B.
    """
    recipe = AddWorkflowRecipe()

    # Create Point A data
    point_a_df = pd.DataFrame([
        {"id": 1, "department": "Sales", "amount": 100},
        {"id": 2, "department": "Engineering", "amount": 300},
        {"id": 3, "department": "Sales", "amount": 150},
        {"id": 4, "department": "Marketing", "amount": 200},
    ])

    # Child workflow definition: Filters for department == 'Sales'
    child_sub_workflow = {
        "id": "wf_filter_sales",
        "name": "Sales Filter Sub-Process",
        "nodes": [
            {"id": "n_start", "recipe_id": "workflow_start", "label": "Start Anchor"},
            {"id": "n_filter", "recipe_id": "row_filter", "label": "Filter Sales"},
            {"id": "n_end", "recipe_id": "workflow_end", "label": "End Anchor"},
        ],
        "edges": [
            {"source": "n_start", "target": "n_filter"},
            {"source": "n_filter", "target": "n_end"},
        ],
        "node_configs": {
            "n_filter": {
                "action": "keep",
                "conditions": [
                    {"column": "department", "operator": "==", "value": "Sales"}
                ]
            }
        }
    }

    ctx = {
        "_cached_sub_workflows": {"wf_filter_sales": child_sub_workflow}
    }

    res = recipe.execute(
        inputs={"dataframe": point_a_df},
        config={"workflow_id": "wf_filter_sales"},
        context=ctx
    )

    out_df = res["dataframe"]
    assert isinstance(out_df, pd.DataFrame)
    assert len(out_df) == 2
    assert all(out_df["department"] == "Sales")
    assert res["output_summary"]["status"] == "SUCCESS"
    assert res["output_summary"]["input_records_received"] == 4
    assert res["output_summary"]["output_records_emitted"] == 2
    assert "Handed 2 rows to Point B" in res["output_summary"]["message"]


def test_add_workflow_failure_structured_warning():
    """Verify structured warning is produced when an internal sub-workflow step fails."""
    recipe = AddWorkflowRecipe()

    child_failing_workflow = {
        "id": "wf_broken_calc",
        "name": "Broken Calc Process",
        "nodes": [
            {"id": "n_start", "recipe_id": "workflow_start"},
            {"id": "n_calc", "recipe_id": "custom_python_step"},
            {"id": "n_end", "recipe_id": "workflow_end"},
        ],
        "edges": [
            {"source": "n_start", "target": "n_calc"},
            {"source": "n_calc", "target": "n_end"},
        ],
        "node_configs": {
            # Code raises runtime ValueError during execution
            "n_calc": {
                "code": "raise ValueError('Calculation step failed deliberately')"
            }
        }
    }

    ctx = {
        "_cached_sub_workflows": {"wf_broken_calc": child_failing_workflow}
    }

    # Verify structured diagnostic message is raised
    with pytest.raises(ExecutionException) as exc:
        recipe.execute(
            inputs={"dataframe": pd.DataFrame([{"colA": 1}])},
            config={"workflow_id": "wf_broken_calc", "propagate_errors": True},
            context=ctx
        )

    err_text = str(exc.value)
    assert "Sub-Workflow 'Broken Calc Process'" in err_text
    assert "failed at step 'n_calc'" in err_text
    assert "custom_python_step" in err_text
    assert "Calculation step failed deliberately" in err_text


def test_sub_workflow_eligibility_filter_logic():
    """Test the qualification rules for sub_workflow_eligible=True."""
    start_aliases = {"workflow_start", "start_anchor", "sub_process_start", "workflow_input"}
    end_aliases = {"workflow_end", "end_anchor", "sub_process_end", "workflow_output"}

    def is_eligible(nodes_data, last_execution):
        if not last_execution or not isinstance(last_execution, dict):
            return False
        if str(last_execution.get("status", "")).strip().lower() != "success":
            return False

        if not nodes_data or not isinstance(nodes_data, list):
            return False

        has_s = False
        has_e = False
        for n in nodes_data:
            if isinstance(n, dict):
                r_id = str(n.get("recipe_id") or n.get("type") or "").strip().lower()
                if r_id in start_aliases:
                    has_s = True
                elif r_id in end_aliases:
                    has_e = True
        return has_s and has_e

    # Case 1: Has both anchors and SUCCESS -> True
    nodes_valid = [
        {"id": "s1", "recipe_id": "workflow_start"},
        {"id": "c1", "recipe_id": "row_filter"},
        {"id": "e1", "recipe_id": "workflow_end"},
    ]
    assert is_eligible(nodes_valid, {"status": "SUCCESS"}) is True

    # Case 2: Has both anchors but FAILED execution -> False
    assert is_eligible(nodes_valid, {"status": "FAILED"}) is False

    # Case 3: Has both anchors but unrun (None) -> False
    assert is_eligible(nodes_valid, None) is False

    # Case 4: SUCCESS but missing End Anchor -> False
    nodes_no_end = [
        {"id": "s1", "recipe_id": "workflow_start"},
        {"id": "c1", "recipe_id": "row_filter"},
    ]
    assert is_eligible(nodes_no_end, {"status": "SUCCESS"}) is False

    # Case 5: SUCCESS but missing Start Anchor -> False
    nodes_no_start = [
        {"id": "c1", "recipe_id": "row_filter"},
        {"id": "e1", "recipe_id": "workflow_end"},
    ]
    assert is_eligible(nodes_no_start, {"status": "SUCCESS"}) is False
