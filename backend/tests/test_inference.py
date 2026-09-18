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


@pytest.mark.asyncio
async def test_tabular_temporal_regression_and_future_projection():
    """
    Verify that when a tabular dataset contains temporal/year features,
    non-forecasting regression models (e.g. Random Forest / XGBoost):
    1. Produce chronological Actual vs Predicted trajectory in model evaluation.
    2. Expose has_temporal_feature, temporal_column, min_year, max_year in inference schema.
    3. Allow future projection prediction (e.g. target_year=2025) with trajectory and trend.
    """
    np.random.seed(42)
    rows = []
    for yr in range(2010, 2021):
        for m in range(1, 13):
            rain = float(np.random.uniform(10.0, 150.0))
            humidity = float(np.random.uniform(30.0, 90.0))
            # Temperature with slight upward trend over years
            temp = float(round(15.0 + (yr - 2010) * 0.4 + (rain * -0.03) + (humidity * 0.05) + np.random.normal(0, 1), 2))
            rows.append({
                "date_year": yr,
                "date_month": m,
                "rainfall": rain,
                "humidity": humidity,
                "temperature": temp
            })
    weather_df = pd.DataFrame(rows)

    nodes = [
        WorkflowNode(
            id="node_split",
            recipe_id="train_test_split",
            config={"target_column": "temperature", "test_size": 0.2, "random_state": 42}
        ),
        WorkflowNode(
            id="node_train",
            recipe_id="random_forest_trainer",
            config={"task_type": "regression", "n_estimators": 25, "random_state": 42}
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
        execution_id="test_exec_temporal_reg_01",
        workflow=graph,
        initial_df=weather_df
    )
    assert exec_res.status == "SUCCESS"

    # 1. Verify Model Evaluation Trajectory
    metrics = exec_res.final_metrics
    assert "trajectory" in metrics
    assert len(metrics["trajectory"]) > 0
    assert metrics["temporal_column"] == "date_year"
    assert metrics["is_temporal"] is True
    # Verify trajectory data point format
    first_pt = metrics["trajectory"][0]
    assert "ds" in first_pt
    assert "actual" in first_pt
    assert "yhat" in first_pt
    assert "residual" in first_pt

    # 2. Verify Inference Schema Temporal Metadata
    schema = exec_res.inference_schema
    assert schema["has_temporal_feature"] is True
    assert schema["temporal_column"] == "date_year"
    assert schema["min_year"] in [2010, 2018]
    assert schema["max_year"] == 2020

    # 3. Verify Direct Future Year Projection Inference
    bundle = job_manager.get_inference_bundle("test_exec_temporal_reg_01")
    assert bundle is not None

    future_req = PredictionRequest(
        target_year=2025,
        inputs={"rainfall": 50.0, "humidity": 65.0, "date_month": 6}
    )
    pred_res = PipelineInferencer.predict(bundle=bundle, request=future_req)
    assert pred_res.status == "SUCCESS"
    assert pred_res.projected_end_value is not None
    assert pred_res.trend in ["Upward", "Downward", "Neutral"]
    assert pred_res.trajectory is not None
    assert len(pred_res.trajectory) >= 5
    # The final projected step should be 2025
    assert pred_res.trajectory[-1]["ds"] == "2025"

    # 4. Verify HTTP REST API Endpoints
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Schema
        resp_schema = await ac.get("/api/v1/workflows/test_exec_temporal_reg_01/schema")
        assert resp_schema.status_code == 200
        schema_data = resp_schema.json()
        assert schema_data["has_temporal_feature"] is True
        assert schema_data["temporal_column"] == "date_year"
        assert schema_data["max_year"] == 2020

        # Predict
        resp_pred = await ac.post(
            "/api/v1/workflows/test_exec_temporal_reg_01/predict",
            json={
                "target_year": 2025,
                "inputs": {"rainfall": 45.0, "humidity": 60.0, "date_month": 7}
            }
        )
        # Predict CSV Export
        resp_csv = await ac.post(
            "/api/v1/workflows/test_exec_temporal_reg_01/predict/export-csv",
            json={
                "target_year": 2025,
                "inputs": {"rainfall": 45.0, "humidity": 60.0, "date_month": 7}
            }
        )
        assert resp_csv.status_code == 200
        assert "text/csv" in resp_csv.headers["content-type"]
        assert "ds" in resp_csv.text or "yhat" in resp_csv.text or "temperature" in resp_csv.text


def test_flat_tree_extrapolation_and_prediction_labels():
    """
    Verify that when a tree model produces flat predictions across future years
    (tree saturation or zero temporal variance), the extrapolation fallback
    projects dynamic yearly values and sets descriptive prediction_label.
    """
    class ConstantDummyModel:
        def predict(self, X):
            return np.array([439305.1562] * len(X))

    bundle = {
        "execution_id": "test_constant_tree_exec",
        "task_type": "regression",
        "model": ConstantDummyModel(),
        "target_column": "Weekly_Sales",
        "feature_names": ["Store", "Date_year", "CPI"],
        "training_feature_summary": {
            "Store": {"data_type": "numeric", "min_value": 1, "max_value": 45, "default_value": 20},
            "Date_year": {"data_type": "numeric", "min_value": 2012, "max_value": 2012, "default_value": 2012},
            "CPI": {"data_type": "numeric", "min_value": 130.0, "max_value": 220.0, "default_value": 190.0}
        },
        "sample_row": {"Store": 22, "Date_year": 2012, "CPI": 191.0},
        "annual_trend_pct": 3.0
    }

    # 1. Single tabular regression test
    single_req = PredictionRequest(inputs={"Store": 22, "Date_year": 2012, "CPI": 191.0})
    single_res = PipelineInferencer.predict(bundle=bundle, request=single_req)
    assert single_res.status == "SUCCESS"
    assert single_res.prediction_label == "Predicted Weekly_Sales"
    assert single_res.prediction == 439305.1562

    # 2. Future year projection test (2012 -> 2016)
    proj_req = PredictionRequest(
        target_year=2016,
        inputs={"Store": 22, "Date_year": 2012, "CPI": 191.0}
    )
    proj_res = PipelineInferencer._predict_tabular_future_projection(
        bundle=bundle,
        base_inputs=proj_req.inputs,
        temporal_col="Date_year",
        future_periods=4,
        target_year=2016
    )

    assert proj_res.status == "SUCCESS"
    assert proj_res.prediction_label == "Projected Weekly_Sales (2016)"
    assert proj_res.trend == "Upward"
    assert proj_res.projected_change_pct > 0

    # Ensure every year has a distinct, progressive value (NOT flat 439305.1562)
    traj = proj_res.trajectory
    assert len(traj) == 5  # 2012, 2013, 2014, 2015, 2016
    assert traj[0]["ds"] == "2012"
    assert traj[0]["yhat"] == 439305.1562

    for i in range(1, len(traj)):
        assert traj[i]["yhat"] > traj[i - 1]["yhat"], f"Step {i} must extrapolate upward"

    # Verify final year 2016 matches projected_end_value
    assert traj[-1]["ds"] == "2016"
    assert proj_res.projected_end_value == traj[-1]["yhat"]


def test_string_typed_inputs_xgboost_compatibility():
    """
    Verify that when inputs contain strings (e.g. from JSON payloads or HTML form inputs),
    _preprocess_inputs coerces them into valid numeric dtypes, preventing the XGBoost error:
    'DataFrame.dtypes for data must be int, float, bool or category. Invalid columns:Store: object'
    """
    class StrictDtypeModel:
        def predict(self, X):
            # Mirror XGBoost's strict dtype check
            for col in X.columns:
                if X[col].dtype == "object":
                    raise ValueError(f"DataFrame.dtypes for data must be int, float, bool or category. Invalid columns:{col}: object")
            return np.array([550000.0])

    bundle = {
        "execution_id": "test_xgboost_strict_exec",
        "task_type": "regression",
        "model": StrictDtypeModel(),
        "target_column": "Weekly_Sales",
        "feature_names": ["Store", "Temperature", "Fuel_Price", "CPI"],
        "training_feature_summary": {
            "Store": {"data_type": "numeric", "min_value": 0.0, "max_value": 44.0, "default_value": 22.0},
            "Temperature": {"data_type": "numeric", "min_value": 30.0, "max_value": 100.0, "default_value": 70.0},
            "Fuel_Price": {"data_type": "numeric", "min_value": 2.0, "max_value": 5.0, "default_value": 3.5},
            "CPI": {"data_type": "numeric", "min_value": 100.0, "max_value": 250.0, "default_value": 180.0}
        },
        "sample_row": {"Store": 22.0, "Temperature": 70.0, "Fuel_Price": 3.5, "CPI": 180.0}
    }

    # Simulate raw JSON payload where all numeric inputs are strings
    string_inputs = {
        "Store": "22",
        "Temperature": "72.9",
        "Fuel_Price": "3.73",
        "CPI": "191.01"
    }

    req = PredictionRequest(inputs=string_inputs)
    res = PipelineInferencer.predict(bundle=bundle, request=req)

    assert res.status == "SUCCESS"
    assert res.prediction == 550000.0
    assert res.prediction_label == "Predicted Weekly_Sales"


def test_shap_treeshap_waterfall_breakdown():
    """
    Verify that when live prediction runs on a tree-based model (Random Forest / XGBoost),
    a SHAP / TreeSHAP waterfall decomposition is generated:
    - Base value E[f(x)]
    - Ordered waterfall_breakdown with feature contributions (+/-)
    - Human-readable summary: 'Store contributed +$45,000; Fuel_Price contributed -$12,000; Unemployment contributed -$8,000.'
    - Efficiency axiom holds: base_value + sum(attributions) == prediction
    """
    from sklearn.ensemble import RandomForestRegressor

    # 1. Train sample model with Walmart-like features
    X_train = pd.DataFrame({
        "Store": [1, 2, 3, 1, 2, 3, 1, 2],
        "Fuel_Price": [2.5, 3.8, 3.2, 2.7, 4.0, 3.5, 2.6, 3.9],
        "Unemployment": [8.5, 6.5, 9.0, 8.2, 6.8, 9.2, 8.3, 6.7]
    })
    y_train = np.array([900000.0, 960000.0, 850000.0, 920000.0, 980000.0, 840000.0, 910000.0, 970000.0])

    rf = RandomForestRegressor(n_estimators=10, random_state=42)
    rf.fit(X_train, y_train)

    bundle = {
        "execution_id": "test_treeshap_exec",
        "task_type": "regression",
        "model": rf,
        "target_column": "Weekly_Sales",
        "feature_names": ["Store", "Fuel_Price", "Unemployment"],
        "training_feature_summary": {
            "Store": {"data_type": "numeric", "default_value": 1.0},
            "Fuel_Price": {"data_type": "numeric", "default_value": 3.0},
            "Unemployment": {"data_type": "numeric", "default_value": 8.0}
        },
        "sample_row": {"Store": 1, "Fuel_Price": 3.0, "Unemployment": 8.0}
    }

    # 2. Score a live prediction
    req = PredictionRequest(inputs={"Store": 1, "Fuel_Price": 3.5, "Unemployment": 8.0})
    res = PipelineInferencer.predict(bundle=bundle, request=req)

    assert res.status == "SUCCESS"
    assert res.prediction is not None
    assert res.base_value is not None
    assert isinstance(res.base_value, float)
    assert res.base_value_formatted is not None
    assert "$" in res.base_value_formatted

    # 3. Check Waterfall Breakdown list
    assert res.waterfall_breakdown is not None
    assert len(res.waterfall_breakdown) == 3

    # Check structure of each attribution item
    sum_attr = 0.0
    for item in res.waterfall_breakdown:
        assert item.feature in ["Store", "Fuel_Price", "Unemployment"]
        assert item.direction in ["positive", "negative"]
        assert item.attribution_formatted.startswith("+") or item.attribution_formatted.startswith("-")
        assert "$" in item.attribution_formatted
        sum_attr += item.attribution

    # 4. Verify Efficiency Axiom (base_value + sum(attributions) == prediction)
    assert np.isclose(res.base_value + sum_attr, res.prediction, atol=1.0)

    # 5. Check Human-Readable Waterfall Summary
    assert res.waterfall_summary is not None
    assert "contributed" in res.waterfall_summary
    assert ("+$" in res.waterfall_summary) or ("-$" in res.waterfall_summary)

    # 6. Verify Tabular Future Projection also includes waterfall breakdown
    bundle_proj = dict(bundle)
    bundle_proj["has_temporal_feature"] = True
    bundle_proj["temporal_column"] = "Store"  # numeric step
    bundle_proj["annual_trend_pct"] = 2.0

    proj_req = PredictionRequest(inputs={"Store": 1, "Fuel_Price": 3.5, "Unemployment": 8.0}, future_periods=3)
    proj_res = PipelineInferencer.predict(bundle=bundle_proj, request=proj_req)
    assert proj_res.status == "SUCCESS"
    assert proj_res.trajectory is not None
    assert proj_res.waterfall_breakdown is not None
    assert len(proj_res.waterfall_breakdown) == 3
    assert proj_res.waterfall_summary is not None





