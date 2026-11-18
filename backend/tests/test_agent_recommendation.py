import json
import threading
import time

import pandas as pd
import pytest
from types import SimpleNamespace

import backend.app.recipes
from backend.app.engine.dag.graph import WorkflowGraph
from backend.app.engine.execution.executor import DAGExecutor
from backend.app.recipes.base.registry import recipe_registry
from backend.app.recommendation import agent_graph
from backend.app.recommendation.agent_jobs import (
    AgentJobManager,
    AgentQueueFullError,
)
from backend.app.recommendation.agent_service import (
    AgentPlanningError,
    agent_provider_available,
    build_agent_tools,
    build_agent_workflow,
    build_dataset_context,
    create_agent_plan,
)


def test_agent_dataset_context_contains_schema_but_no_row_values():
    df = pd.DataFrame(
        {
            "customer_name": ["Private Name A", "Private Name B"],
            "target": [0, 1],
        }
    )

    context = build_dataset_context(df)
    serialized = str(context)

    assert context["row_count"] == 2
    assert "customer_name" in context["columns"]
    assert "Private Name A" not in serialized
    assert "Private Name B" not in serialized
    assert "top_values" not in serialized


def test_agent_tools_discover_registered_trainers_for_requested_task():
    tools, recipe_tools = build_agent_tools(["feature", "target"], "regression")

    assert "select_recipe_missing_value_imputer" in recipe_tools
    expected_regression_trainers = {
        metadata.recipe_id
        for metadata in recipe_registry.list_all(category="training")
        if "regression"
        in recipe_registry.get(metadata.recipe_id)
        .get_schema()
        .get("properties", {})
        .get("task_type", {})
        .get("enum", [])
    }
    offered_trainers = {
        recipe_id
        for recipe_id in recipe_tools.values()
        if recipe_registry.get(recipe_id).category == "training"
    }
    assert offered_trainers == expected_regression_trainers
    assert "select_recipe_random_forest_trainer" in recipe_tools
    assert "select_recipe_model_evaluator" not in recipe_tools
    assert any(tool["function"]["name"] == "ask_user" for tool in tools)
    trainer_tool = next(
        tool["function"]
        for tool in tools
        if tool["function"]["name"] == "select_recipe_random_forest_trainer"
    )
    assert trainer_tool["parameters"]["properties"]["task_type"]["enum"] == [
        "regression"
    ]


def test_agent_builds_valid_classification_workflow():
    workflow = build_agent_workflow(
        dataset_id="dataset-123",
        target_column="target",
        recipe_selections=[
            ("missing_value_imputer", {"strategy": "median"}),
            ("random_forest_trainer", {"task_type": "classification"}),
        ],
    )

    assert isinstance(workflow, WorkflowGraph)
    assert workflow.get_diagnostics()["is_valid"] is True
    assert [node.recipe_id for node in workflow.nodes] == [
        "csv_loader",
        "missing_value_imputer",
        "train_test_split",
        "random_forest_trainer",
        "model_evaluator",
    ]
    assert workflow.nodes[2].config["target_column"] == "target"


def test_agent_builds_regression_workflow_with_regression_evaluation():
    workflow = build_agent_workflow(
        dataset_id="dataset-123",
        target_column="target",
        task_type="regression",
        recipe_selections=[
            ("linear_trainer", {"task_type": "regression"}),
        ],
    )

    assert workflow.get_diagnostics()["is_valid"] is True
    assert workflow.nodes[1].config["stratify"] is False
    assert workflow.nodes[-1].config["report_type"] == (
        "Regression Diagnostics (MAE, RMSE, R2, MAPE)"
    )


def test_agent_regression_workflow_executes_and_reports_regression_metrics():
    df = pd.DataFrame(
        {
            "feature": list(range(40)),
            "target": [value * 3.5 + 2 for value in range(40)],
        }
    )
    workflow = build_agent_workflow(
        dataset_id="dataset-123",
        target_column="target",
        task_type="regression",
        recipe_selections=[
            ("linear_trainer", {"task_type": "regression"}),
        ],
    )

    result = DAGExecutor.execute_workflow(
        execution_id="agent-regression-test",
        workflow=workflow,
        initial_df=df,
        context={"target_column": "target"},
    )

    assert result.status == "SUCCESS"
    assert result.final_metrics["task_type"] == "regression"
    assert "r2_score" in result.final_metrics
    assert "rmse" in result.final_metrics


@pytest.mark.asyncio
async def test_agent_plan_selects_regression_from_dynamic_recipe_catalog(monkeypatch):
    monkeypatch.setattr(agent_graph.settings, "OPENAI_API_KEY", None)
    monkeypatch.setattr(agent_graph.settings, "GROQ_API_KEY", "test-key")
    completion = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(
                    content="Use a regression trainer.",
                    tool_calls=[
                        SimpleNamespace(
                            function=SimpleNamespace(
                                name="choose_task_type",
                                arguments=json.dumps(
                                    {
                                        "task_type": "regression",
                                        "reason": "The user asked to predict a numeric price.",
                                    }
                                ),
                            )
                        ),
                        SimpleNamespace(
                            function=SimpleNamespace(
                                name="choose_target_column",
                                arguments=json.dumps(
                                    {"column": "price", "reason": "Requested outcome."}
                                ),
                            )
                        ),
                        SimpleNamespace(
                            function=SimpleNamespace(
                                name="select_recipe_random_forest_trainer",
                                arguments=json.dumps({"task_type": "regression"}),
                            )
                        ),
                    ],
                )
            )
        ]
    )
    captured_messages = {}

    async def fake_groq_request(**kwargs):
        captured_messages.update(kwargs)
        return completion

    monkeypatch.setattr(
        "backend.app.recommendation.agent_service._request_agent_tools",
        fake_groq_request,
    )
    plan = await create_agent_plan(
        dataset_id="dataset-123",
        dataset_context={
            "row_count": 20,
            "column_count": 2,
            "columns": {
                "feature": {"inferred_type": "numeric", "unique_count": 20},
                "price": {"inferred_type": "numeric", "unique_count": 20},
            },
        },
        user_request="Predict the price using regression.",
        requested_target=None,
    )

    assert plan["status"] == "ready_for_review"
    assert plan["task_type"] == "regression"
    assert plan["workflow"]["nodes"][1]["config"]["stratify"] is False
    catalog_message = captured_messages["messages"][1]["content"]
    assert "prophet_forecaster" in catalog_message
    assert "isolation_forest" in catalog_message


def test_agent_prefers_openai_and_falls_back_to_groq(monkeypatch):
    from backend.app.recommendation.agent_service import _get_agent_provider

    monkeypatch.setattr(agent_graph.settings, "OPENAI_API_KEY", " openai-key ")
    monkeypatch.setattr(agent_graph.settings, "OPENAI_MODEL", "test-openai-model")
    monkeypatch.setattr(agent_graph.settings, "GROQ_API_KEY", "groq-key")
    assert agent_provider_available()

    assert _get_agent_provider() == (
        "OpenAI",
        "openai-key",
        "test-openai-model",
        None,
    )

    monkeypatch.setattr(agent_graph.settings, "OPENAI_API_KEY", " ")
    assert agent_provider_available()
    assert _get_agent_provider() == (
        "Groq",
        "groq-key",
        agent_graph.settings.GROQ_MODEL,
        "https://api.groq.com/openai/v1",
    )

    monkeypatch.setattr(agent_graph.settings, "GROQ_API_KEY", " ")
    assert not agent_provider_available()


def test_agent_rejects_invalid_recipe_config():
    with pytest.raises(AgentPlanningError, match="Invalid configuration"):
        build_agent_workflow(
            dataset_id="dataset-123",
            target_column="target",
            recipe_selections=[
                ("random_forest_trainer", {"task_type": "regression"}),
            ],
        )


def test_agent_does_not_offer_target_leaking_encoders():
    tools, _ = build_agent_tools(["feature", "target"])
    encoder_tool = next(
        tool["function"]
        for tool in tools
        if tool["function"]["name"] == "select_recipe_categorical_encoder"
    )

    assert encoder_tool["parameters"]["properties"]["method"]["enum"] == [
        "one_hot",
        "label",
        "ordinal",
        "frequency",
        "binary",
    ]


@pytest.mark.asyncio
async def test_agent_revises_a_pipeline_until_it_accepts_a_candidate(monkeypatch):
    initial_workflow = build_agent_workflow(
        dataset_id="dataset-123",
        target_column="target",
        recipe_selections=[
            ("random_forest_trainer", {"task_type": "classification"}),
        ],
    )
    executions = iter(
        [
            SimpleNamespace(
                status="SUCCESS",
                final_metrics={"balanced_accuracy": 0.62, "f1_score": 0.6},
                node_results=[],
            ),
            SimpleNamespace(
                status="SUCCESS",
                final_metrics={"balanced_accuracy": 0.84, "f1_score": 0.83},
                node_results=[],
            ),
        ]
    )

    def fake_execute_workflow(**kwargs):
        return next(executions)

    reviews = iter(
        [
            {
                "action": "revise",
                "recipe_selections": [
                    ("linear_trainer", {"task_type": "classification"}),
                ],
            },
            {"action": "finish", "accepted": True, "reason": "Validation metrics meet the goal."},
        ]
    )

    async def fake_review_attempt(**kwargs):
        return next(reviews)

    async def fake_save_model(execution_id, metrics, task_type):
        return {"saved": True, "registered_model_name": "agent-test-model"}

    async def fake_create_plan(**kwargs):
        return {
            "status": "ready_for_review",
            "target_column": "target",
            "workflow": initial_workflow.model_dump(),
        }

    monkeypatch.setattr(
        agent_graph.DAGExecutor,
        "execute_workflow",
        staticmethod(fake_execute_workflow),
    )
    monkeypatch.setattr(agent_graph, "create_agent_plan", fake_create_plan)
    monkeypatch.setattr(agent_graph, "_review_attempt", fake_review_attempt)
    monkeypatch.setattr(agent_graph, "_save_accepted_model", fake_save_model)

    result = await agent_graph.run_pipeline_agent(
        dataset_id="dataset-123",
        dataframe=pd.DataFrame({"feature": [0, 1, 2, 3], "target": [0, 0, 1, 1]}),
        dataset_context={
            "row_count": 4,
            "columns": {
                "feature": {"unique_count": 4},
                "target": {"unique_count": 2},
            },
        },
        user_request="Create a classification model.",
        max_attempts=3,
    )

    assert result["status"] == "completed"
    assert len(result["attempts"]) == 2
    assert result["best_candidate"]["metrics"]["balanced_accuracy"] == 0.84
    assert result["model_saved"] is True


def test_regression_candidate_scoring_prefers_higher_r2_and_falls_back_to_rmse():
    first = agent_graph._candidate_score(
        {"r2_score": 0.72, "rmse": 12.0},
        "regression",
    )
    second = agent_graph._candidate_score(
        {"r2_score": 0.81, "rmse": 10.0},
        "regression",
    )
    fallback = agent_graph._candidate_score({"rmse": 10.0}, "regression")

    assert second[0] > first[0]
    assert fallback == (-10.0, "rmse", 10.0)


def test_pipeline_agent_graph_has_explicit_agent_and_execution_nodes():
    graph = agent_graph.build_pipeline_agent_graph()
    nodes = set(graph.get_graph().nodes)

    assert {
        "planner_agent",
        "execute_pipeline",
        "evaluator_reviser_agent",
        "revise_pipeline",
        "finalize",
    }.issubset(nodes)


@pytest.mark.asyncio
async def test_pipeline_agent_graph_returns_planner_questions_without_execution(monkeypatch):
    async def fake_create_plan(**kwargs):
        return {
            "status": "needs_clarification",
            "questions": [{"question": "Which column is the target?", "reason": "Required"}],
        }

    def unexpected_execution(**kwargs):
        pytest.fail("The graph must not execute a pipeline while clarification is pending.")

    monkeypatch.setattr(agent_graph, "create_agent_plan", fake_create_plan)
    monkeypatch.setattr(
        agent_graph.DAGExecutor,
        "execute_workflow",
        staticmethod(unexpected_execution),
    )

    result = await agent_graph.run_pipeline_agent(
        dataset_id="dataset-123",
        dataframe=pd.DataFrame({"feature": [0, 1], "target": [0, 1]}),
        dataset_context={
            "columns": {
                "feature": {"unique_count": 2},
                "target": {"unique_count": 2},
            }
        },
        user_request="Create a classification model.",
        max_attempts=2,
    )

    assert result["status"] == "needs_clarification"
    assert result["questions"][0]["question"] == "Which column is the target?"


def test_agent_job_manager_returns_job_id_and_rejects_over_capacity(monkeypatch):
    manager = AgentJobManager(max_workers=1, max_queued=0)
    worker_started = threading.Event()
    release_worker = threading.Event()

    async def blocked_agent(**kwargs):
        worker_started.set()
        release_worker.wait(timeout=5)
        return {"status": "completed"}

    monkeypatch.setattr(manager, "_run_agent", blocked_agent)
    try:
        job_id = manager.submit(
            dataset_id="dataset-123",
            dataset_path="datasets/dataset.csv",
            user_request="Train a regression model.",
            max_attempts=1,
        )
        assert job_id.startswith("agent_")
        assert worker_started.wait(timeout=2)

        with pytest.raises(AgentQueueFullError):
            manager.submit(
                dataset_id="dataset-456",
                dataset_path="datasets/other.csv",
                user_request="Train a classifier.",
                max_attempts=1,
            )

        release_worker.set()
        for _ in range(50):
            job = manager.get_job(job_id)
            if job and job["status"] in {"SUCCESS", "FAILED"}:
                break
            time.sleep(0.02)

        assert manager.get_job(job_id)["status"] == "SUCCESS"
    finally:
        release_worker.set()
        manager.shutdown()
