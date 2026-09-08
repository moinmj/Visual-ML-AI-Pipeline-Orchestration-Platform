import pytest
import pandas as pd
import numpy as np
from httpx import AsyncClient, ASGITransport

from backend.app.main import app
from backend.app.engine.dag.graph import WorkflowGraph, WorkflowNode, WorkflowEdge
from backend.app.engine.execution.executor import DAGExecutor
from backend.app.engine.execution.job_manager import job_manager
from backend.app.engine.inference.pipeline_inferencer import PipelineInferencer
from backend.app.engine.inference.schemas import PredictionRequest


@pytest.fixture
def sample_iris_df():
    np.random.seed(42)
    rows = []
    classes = ["setosa", "versicolor", "virginica"]
    for i in range(120):
        cls = classes[i % 3]
        base = 1.0 if cls == "setosa" else (4.0 if cls == "versicolor" else 5.5)
        rows.append({
            "sepal_length": round(float(np.random.normal(5.0 + base * 0.3, 0.4)), 2),
            "sepal_width": round(float(np.random.normal(3.0, 0.3)), 2),
            "petal_length": round(float(np.random.normal(base, 0.3)), 2),
            "petal_width": round(float(np.random.normal(base * 0.3, 0.1)), 2),
            "species": cls
        })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_ts_df():
    dates = pd.date_range(start="2026-01-01", periods=60, freq="D")
    y = np.linspace(100, 200, 60) + np.random.normal(0, 5, 60)
    return pd.DataFrame({
        "ds": dates.strftime("%Y-%m-%d"),
        "y": y
    })


@pytest.mark.asyncio
async def test_classification_inference_flow(sample_iris_df):
    """
    Test full end-to-end classification pipeline execution,
    schema extraction, class decoding, and prediction API.
    """
    nodes = [
        WorkflowNode(
            id="node_split",
            recipe_id="train_test_split",
            config={"target_column": "species", "test_size": 0.2, "random_state": 42}
        ),
        WorkflowNode(
            id="node_train",
            recipe_id="random_forest_trainer",
            config={"task_type": "classification", "n_estimators": 20, "random_state": 42}
        ),
        WorkflowNode(
            id="node_eval",
            recipe_id="model_evaluator",
            config={}
        )
    ]
    edges = [
        WorkflowEdge(source="node_split", target="node_train"),
        WorkflowEdge(source="node_train", target="node_eval")
    ]
    graph = WorkflowGraph(nodes=nodes, edges=edges)

    exec_res = DAGExecutor.execute_workflow(
        execution_id="test_exec_iris_01",
        workflow=graph,
        initial_df=sample_iris_df
    )

    assert exec_res.status == "SUCCESS"
    assert exec_res.inference_schema is not None
    schema = exec_res.inference_schema
    assert schema["task_type"] == "classification"
    assert "species" == schema["target_column"]
    assert len(schema["target_classes"]) == 3
    assert "setosa" in schema["target_classes"]

    # 1. Direct Engine Prediction
    bundle = job_manager.get_inference_bundle("test_exec_iris_01")
    assert bundle is not None

    req = PredictionRequest(inputs={
        "sepal_length": 5.0,
        "sepal_width": 3.4,
        "petal_length": 1.4,
        "petal_width": 0.2
    })
    pred_res = PipelineInferencer.predict(bundle=bundle, request=req)

    assert pred_res.status == "SUCCESS"
    assert pred_res.task_type == "classification"
    assert pred_res.prediction in ["setosa", "versicolor", "virginica"]
    assert pred_res.confidence is not None and pred_res.confidence > 0
    assert "setosa" in pred_res.probabilities

    # 2. Batch Scoring Prediction
    batch_req = PredictionRequest(inputs=[
        {"sepal_length": 5.0, "sepal_width": 3.4, "petal_length": 1.4, "petal_width": 0.2},
        {"sepal_length": 6.5, "sepal_width": 3.0, "petal_length": 5.2, "petal_width": 2.0}
    ])
    batch_res = PipelineInferencer.predict(bundle=bundle, request=batch_req)
    assert batch_res.status == "SUCCESS"
    assert len(batch_res.batch_predictions) == 2
    assert "Predicted_Class" in batch_res.batch_predictions[0]

    # 3. HTTP REST API Endpoint
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Schema endpoint
        resp_schema = await ac.get("/api/v1/workflows/test_exec_iris_01/schema")
        assert resp_schema.status_code == 200
        data_schema = resp_schema.json()
        assert data_schema["execution_id"] == "test_exec_iris_01"
        assert len(data_schema["features"]) >= 4

        # Predict endpoint
        resp_pred = await ac.post(
            "/api/v1/workflows/test_exec_iris_01/predict",
            json={"inputs": {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}}
        )
        assert resp_pred.status_code == 200
        data_pred = resp_pred.json()
        assert data_pred["status"] == "SUCCESS"
        assert data_pred["prediction"] in ["setosa", "versicolor", "virginica"]


@pytest.mark.asyncio
async def test_forecasting_inference_flow(sample_ts_df):
    """
    Test time-series forecasting pipeline execution and subsequent range prediction.
    """
    nodes = [
        WorkflowNode(
            id="node_prophet",
            recipe_id="prophet_forecaster",
            config={"date_column": "ds", "target_column": "y", "horizon": 15}
        )
    ]
    graph = WorkflowGraph(nodes=nodes, edges=[])

    exec_res = DAGExecutor.execute_workflow(
        execution_id="test_exec_ts_01",
        workflow=graph,
        initial_df=sample_ts_df
    )
    assert exec_res.status == "SUCCESS"

    bundle = job_manager.get_inference_bundle("test_exec_ts_01")
    assert bundle is not None

    req = PredictionRequest(forecast_horizon=10)
    pred_res = PipelineInferencer.predict(bundle=bundle, request=req)

    assert pred_res.status == "SUCCESS"
    assert pred_res.task_type == "time_series_forecasting"
    assert pred_res.forecast_horizon == 10
    assert len(pred_res.forecast_records) == 10
    assert "yhat" in pred_res.forecast_records[0]


@pytest.mark.asyncio
async def test_anomaly_detection_inference(sample_iris_df):
    """
    Test anomaly detection pipeline execution and single-record outlier risk scoring.
    """
    nodes = [
        WorkflowNode(
            id="node_iso",
            recipe_id="isolation_forest",
            config={"contamination": 0.05, "n_estimators": 30}
        )
    ]
    graph = WorkflowGraph(nodes=nodes, edges=[])

    numeric_df = sample_iris_df.drop(columns=["species"])
    exec_res = DAGExecutor.execute_workflow(
        execution_id="test_exec_anom_01",
        workflow=graph,
        initial_df=numeric_df
    )
    assert exec_res.status == "SUCCESS"

    bundle = job_manager.get_inference_bundle("test_exec_anom_01")
    assert bundle is not None

    req = PredictionRequest(inputs={
        "sepal_length": 50.0,  # Extreme outlier!
        "sepal_width": 30.0,
        "petal_length": 50.0,
        "petal_width": 20.0
    })
    pred_res = PipelineInferencer.predict(bundle=bundle, request=req)

    assert pred_res.status == "SUCCESS"
    assert pred_res.task_type == "anomaly_detection"
    assert pred_res.is_anomaly in [0, 1]
    assert pred_res.verdict in ["ANOMALOUS_OUTLIER", "NORMAL_RECORD"]
    assert pred_res.anomaly_score is not None
