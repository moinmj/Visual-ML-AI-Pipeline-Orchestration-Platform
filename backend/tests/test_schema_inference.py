import pytest
from backend.app.workflows.schema_inference import WorkflowSchemaInferencer

@pytest.mark.asyncio
async def test_schema_inference_for_api_and_anomaly_pipeline():
    nodes = [
        {"id": "cron_1", "recipe_id": "cron_trigger", "config": {}},
        {"id": "stock_1", "recipe_id": "stock_market_api", "config": {"mode": "historical", "symbols": "AAPL"}},
        {"id": "iso_1", "recipe_id": "isolation_forest", "config": {"contamination": 0.05}},
        {"id": "filter_1", "recipe_id": "row_filter", "config": {}}
    ]

    edges = [
        {"source": "cron_1", "target": "stock_1", "sourceHandle": "output", "targetHandle": "input"},
        {"source": "stock_1", "target": "iso_1", "sourceHandle": "output", "targetHandle": "input"},
        {"source": "iso_1", "target": "filter_1", "sourceHandle": "output 1", "targetHandle": "input"}
    ]

    res = await WorkflowSchemaInferencer.infer_schema(
        nodes=nodes,
        edges=edges,
        node_configs={}
    )

    assert res.success is True
    iso_schema = res.node_schemas["iso_1"]
    assert "is_anomaly" in iso_schema.columns
    assert "anomaly_score" in iso_schema.columns
    assert "close" in iso_schema.columns

    filter_schema = res.node_schemas["filter_1"]
    assert "is_anomaly" in filter_schema.available_left_columns
    assert "anomaly_score" in filter_schema.available_left_columns
    assert "close" in filter_schema.available_left_columns
    assert "date" in filter_schema.available_left_columns
