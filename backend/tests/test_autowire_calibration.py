import pytest
import asyncio
from backend.app.recommendation.router import autowire_nodes, AutoWireRequest
from backend.app.engine.dag.graph import WorkflowGraph, WorkflowNode, WorkflowEdge


@pytest.mark.asyncio
async def test_autowire_trainer_before_calibration():
    # Intentionally pass model_calibration before catboost_trainer and with smaller x
    nodes = [
        {"id": "processor_train_test_split_6", "recipe_id": "train_test_split", "position": {"x": 100, "y": 100}},
        {"id": "processor_model_calibration_2", "recipe_id": "model_calibration", "position": {"x": 200, "y": 100}},
        {"id": "processor_catboost_trainer_3", "recipe_id": "catboost_trainer", "position": {"x": 300, "y": 100}},
        {"id": "processor_model_evaluator_1", "recipe_id": "model_evaluator", "position": {"x": 400, "y": 100}},
    ]

    res = await autowire_nodes(AutoWireRequest(nodes=nodes))
    assert res["status"] == "AUTOWIRED"
    edges = res["edges"]

    # 1. Train split -> CatBoost Trainer (output_1 -> input)
    split_to_trainer = next((e for e in edges if e["source"] == "processor_train_test_split_6" and e["target"] == "processor_catboost_trainer_3"), None)
    assert split_to_trainer is not None, f"Expected split -> catboost edge, but got: {edges}"
    assert split_to_trainer.get("source_handle") == "output_1"

    # 2. CatBoost Trainer -> Model Calibration (output -> input)
    trainer_to_calib = next((e for e in edges if e["source"] == "processor_catboost_trainer_3" and e["target"] == "processor_model_calibration_2"), None)
    assert trainer_to_calib is not None, f"Expected catboost -> calibration edge, but got: {edges}"
    assert trainer_to_calib.get("target_handle") == "input"

    # 3. Model Calibration -> Model Evaluator (output -> input_1)
    calib_to_eval = next((e for e in edges if e["source"] == "processor_model_calibration_2" and e["target"] == "processor_model_evaluator_1"), None)
    assert calib_to_eval is not None, f"Expected calibration -> evaluator edge, but got: {edges}"
    assert calib_to_eval.get("target_handle") == "input_1"

    # 4. Splitter -> Model Evaluator (output_2 -> input_2)
    split_to_eval = next((e for e in edges if e["source"] == "processor_train_test_split_6" and e["target"] == "processor_model_evaluator_1"), None)
    assert split_to_eval is not None, f"Expected split -> evaluator edge, but got: {edges}"
    assert split_to_eval.get("source_handle") == "output_2"
    assert split_to_eval.get("target_handle") == "input_2"


def test_dag_validation_calibration_after_trainer():
    wf_nodes = [
        WorkflowNode(id="processor_train_test_split_6", recipe_id="train_test_split", config={"target_column": "y"}),
        WorkflowNode(id="processor_catboost_trainer_3", recipe_id="catboost_trainer"),
        WorkflowNode(id="processor_model_calibration_2", recipe_id="model_calibration"),
        WorkflowNode(id="processor_model_evaluator_1", recipe_id="model_evaluator"),
    ]
    wf_edges = [
        WorkflowEdge(id="e1", source="processor_train_test_split_6", target="processor_catboost_trainer_3", source_handle="output_1", target_handle="input"),
        WorkflowEdge(id="e2", source="processor_catboost_trainer_3", target="processor_model_calibration_2", source_handle="output", target_handle="input"),
        WorkflowEdge(id="e3", source="processor_model_calibration_2", target="processor_model_evaluator_1", source_handle="output", target_handle="input_1"),
        WorkflowEdge(id="e4", source="processor_train_test_split_6", target="processor_model_evaluator_1", source_handle="output_2", target_handle="input_2"),
    ]
    graph = WorkflowGraph(nodes=wf_nodes, edges=wf_edges)
    diag = graph.get_diagnostics()
    # Ensure no incompatible connection error
    incomp_errors = [e for e in diag["errors"] if "Incompatible Connection" in str(e)]
    assert len(incomp_errors) == 0, f"Unexpected incompatible connection errors: {incomp_errors}"

