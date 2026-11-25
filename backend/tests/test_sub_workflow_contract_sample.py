"""
A sub-process run on its own gets sample rows shaped like its Start Anchor's Expected Input
Columns (and schema inference reports those columns), so the steps after the anchor can be
configured against the columns the sub-process will really receive.
"""
import asyncio

import pandas as pd

from backend.app.recipes.flow_control.workflow_start import WorkflowStartRecipe
from backend.app.workflows.schema_inference import WorkflowSchemaInferencer

CONTRACT = [
    {"name": "SalePrice", "type": "integer", "required": True},
    {"name": "GrLivArea", "type": "integer"},
    {"name": "LotFrontage", "type": "number", "required": False},
    {"name": "Neighborhood", "type": "string"},
    {"name": "CentralAir", "type": "boolean"},
    {"name": "SoldOn", "type": "datetime"},
    {"name": "Notes", "type": "any"},
]


def test_standalone_start_anchor_uses_the_contract_columns():
    res = WorkflowStartRecipe().execute({}, {"input_columns": CONTRACT, "mock_record_count": 4}, context={})
    df = res["dataframe"]
    assert list(df.columns) == [c["name"] for c in CONTRACT]
    assert len(df) == 4
    assert df["SalePrice"].dtype.kind in "iu" and df["LotFrontage"].dtype.kind == "f"
    assert df["CentralAir"].dtype.kind == "b" and df["SoldOn"].dtype.kind == "M"
    assert res["output_summary"]["status"] == "STANDALONE_CONTRACT_SAMPLE"


def test_contract_from_json_string_and_without_contract():
    import json
    res = WorkflowStartRecipe().execute({}, {"input_columns": json.dumps(CONTRACT[:2])}, context={})
    assert list(res["dataframe"].columns) == ["SalePrice", "GrLivArea"]
    # no contract declared: the old demo rows, unchanged
    res = WorkflowStartRecipe().execute({}, {"input_columns": []}, context={})
    assert res["output_summary"]["status"] == "STANDALONE_MOCK"
    assert "customer_id" in res["dataframe"].columns


def test_parent_table_always_wins_over_the_sample():
    parent = pd.DataFrame({"SalePrice": [1, 2], "Extra": ["a", "b"]})
    res = WorkflowStartRecipe().execute({}, {"input_columns": CONTRACT}, context={"sub_workflow_input": parent})
    assert list(res["dataframe"].columns) == ["SalePrice", "Extra"]
    assert res["output_summary"]["status"] == "INJECTED"
    # even an empty parent table stays empty
    res = WorkflowStartRecipe().execute({}, {"input_columns": CONTRACT},
                                        context={"sub_workflow_input": pd.DataFrame()})
    assert res["dataframe"].empty


def test_schema_inference_offers_the_contract_columns_downstream():
    nodes = [{"id": "start", "recipe_id": "workflow_start"}, {"id": "flt", "recipe_id": "row_filter"}]
    edges = [{"source": "start", "target": "flt"}]
    configs = {"start": {"input_columns": CONTRACT}, "flt": {}}
    result = asyncio.run(WorkflowSchemaInferencer.infer_schema(nodes=nodes, edges=edges, node_configs=configs))
    by_id = result.node_schemas
    names = [c["name"] for c in CONTRACT]
    assert by_id["start"].columns == names
    assert by_id["flt"].available_left_columns == names
    types = {d.name: d.type for d in by_id["start"].column_details}
    assert types["SalePrice"] == "numeric" and types["Neighborhood"] == "string"
