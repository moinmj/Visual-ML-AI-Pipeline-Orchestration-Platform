import pytest
from backend.app.recommendation.router import autowire_nodes, AutoWireRequest
from backend.app.recommendation.recommender import AIRecommender
from backend.app.engine.dag.graph import WorkflowGraph, WorkflowNode, WorkflowEdge
from backend.app.engine.execution.executor import DAGExecutor
import pandas as pd


@pytest.mark.asyncio
async def test_autowire_splitter_and_evaluator_handles():
    nodes = [
        {"id": "node_csv", "recipe_id": "csv_loader", "position": {"x": 40, "y": 100}},
        {"id": "node_select", "recipe_id": "column_selector", "position": {"x": 280, "y": 100}},
        {"id": "node_convert", "recipe_id": "data_type_converter", "position": {"x": 520, "y": 100}},
        {"id": "node_split", "recipe_id": "stratified_split", "position": {"x": 760, "y": 100}},
        {"id": "node_model", "recipe_id": "xgboost_trainer", "position": {"x": 1000, "y": 50}},
        {"id": "node_eval", "recipe_id": "model_evaluator", "position": {"x": 1240, "y": 100}},
        {"id": "node_governance", "recipe_id": "model_governance_card", "position": {"x": 1480, "y": 100}},
        {"id": "node_mlflow", "recipe_id": "mlflow_tracker", "position": {"x": 1720, "y": 100}}
    ]

    res = await autowire_nodes(AutoWireRequest(nodes=nodes))
    assert res["status"] == "AUTOWIRED"
    edges = res["edges"]

    # Verify edge from split to model trainer uses output_1 (Tr)
    split_model = next((e for e in edges if e["source"] == "node_split" and e["target"] == "node_model"), None)
    assert split_model is not None, "Missing split -> model edge"
    assert split_model.get("source_handle") == "output_1", f"Expected output_1, got {split_model.get('source_handle')}"

    # Verify edge from model trainer to evaluator uses input_1 (M)
    model_eval = next((e for e in edges if e["source"] == "node_model" and e["target"] == "node_eval"), None)
    assert model_eval is not None, "Missing model -> eval edge"
    assert model_eval.get("target_handle") == "input_1", f"Expected input_1, got {model_eval.get('target_handle')}"

    # Verify edge from split to evaluator uses output_2 (Te) -> input_2 (Te)
    split_eval = next((e for e in edges if e["source"] == "node_split" and e["target"] == "node_eval"), None)
    assert split_eval is not None, "Missing split -> eval edge"
    assert split_eval.get("source_handle") == "output_2", f"Expected output_2, got {split_eval.get('source_handle')}"
    assert split_eval.get("target_handle") == "input_2", f"Expected input_2, got {split_eval.get('target_handle')}"


def test_recommender_pipeline_generates_correct_handles():
    df = pd.DataFrame({
        "age": [25, 30, 35, 40, 45, 50, 55, 60],
        "salary": [50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000],
        "churn": [0, 1, 0, 1, 0, 1, 0, 1]
    })
    rec = AIRecommender._heuristic_recommend_pipeline(df, target_column="churn", task_type="classification")
    dag = rec.get("recommended_dag", {})
    edges = dag.get("edges", [])

    # Check split -> model
    split_model = next((e for e in edges if "split" in e.get("source", "") and "model" in e.get("target", "")), None)
    assert split_model is not None
    assert split_model.get("source_handle") == "output_1"

    # Check model -> eval
    model_eval = next((e for e in edges if "model" in e.get("source", "") and "eval" in e.get("target", "")), None)
    assert model_eval is not None
    assert model_eval.get("target_handle") == "input_1"

    # Check split -> eval
    split_eval = next((e for e in edges if "split" in e.get("source", "") and "eval" in e.get("target", "")), None)
    assert split_eval is not None
    assert split_eval.get("source_handle") == "output_2"
    assert split_eval.get("target_handle") == "input_2"
