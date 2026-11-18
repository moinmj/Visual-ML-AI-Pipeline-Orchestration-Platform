import json
import math
import uuid
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import pandas as pd
from fastapi.concurrency import run_in_threadpool
from langgraph.graph import END, START, StateGraph
from jsonschema import ValidationError as JsonSchemaValidationError, validate

from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.engine.dag.graph import WorkflowGraph
from backend.app.engine.execution.executor import DAGExecutor
from backend.app.engine.execution.job_manager import job_manager
from backend.app.recipes.base.registry import recipe_registry
from backend.app.recommendation.agent_service import (
    AgentPlanningError,
    _recipe_parameters,
    build_agent_tools,
    build_agent_workflow,
    create_agent_plan,
    _request_agent_tools,
)


def _candidate_fingerprint(workflow: WorkflowGraph) -> str:
    selections = [
        (node.recipe_id, node.config)
        for node in workflow.nodes
        if recipe_registry.get(node.recipe_id).category in {"preprocessing", "nlp", "training"}
    ]
    return json.dumps(selections, sort_keys=True, separators=(",", ":"))


def _numeric_metrics(metrics: Optional[Dict[str, Any]]) -> Dict[str, float]:
    if not metrics:
        return {}
    return {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    }


def _review_tools(
    column_names: List[str],
    task_type: str,
    include_recipe_tools: bool,
    allow_rejection: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    tools, recipe_tools = build_agent_tools(column_names, task_type)
    tools = [
        tool
        for tool in tools
        if tool["function"]["name"] not in {"choose_target_column", "choose_task_type"}
        and (include_recipe_tools or tool["function"]["name"] not in recipe_tools)
    ]
    available_recipe_tools = {
        name: recipe_id
        for name, recipe_id in recipe_tools.items()
        if any(tool["function"]["name"] == name for tool in tools)
    }
    tools.append(
        {
            "type": "function",
            "function": {
                "name": "finish_search",
                "description": "Stop searching and report whether the best candidate meets the user's goal.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "accepted": (
                            {"type": "boolean"}
                            if allow_rejection
                            else {"type": "boolean", "enum": [True]}
                        ),
                        "reason": {"type": "string"},
                    },
                    "required": ["accepted", "reason"],
                    "additionalProperties": False,
                },
            },
        }
    )
    return tools, available_recipe_tools


async def _review_attempt(
    *,
    dataset_context: Dict[str, Any],
    user_request: str,
    target_column: str,
    task_type: str,
    history: List[Dict[str, Any]],
    attempt_number: int,
    max_attempts: int,
) -> Dict[str, Any]:
    tools, recipe_tools = _review_tools(
        list(dataset_context["columns"]),
        task_type=task_type,
        include_recipe_tools=attempt_number < max_attempts,
        allow_rejection=attempt_number >= max_attempts,
    )
    tool_schemas = {tool["function"]["name"]: tool["function"]["parameters"] for tool in tools}
    metric_guidance = (
        "Judge regression candidates primarily by R2 (higher is better), with RMSE and MAE (lower is better) as error metrics."
        if task_type == "regression"
        else "Judge classification candidates using balanced_accuracy and F1, accounting for class imbalance; do not rely on accuracy alone."
    )
    system_message = (
        f"You are the evaluation and revision controller for a {task_type} pipeline. "
        "Judge the best candidate using the user's stated objective, the reported validation metrics, "
        "dataset size, and the full attempt history. "
        + metric_guidance
        + " If the result is not good and another "
        "available configuration could help, call one or more recipe-selection tools to define a "
        "complete replacement pipeline (all preprocessing choices plus exactly one compatible trainer). "
        "Tool calls are configuration choices; the server validates and executes them. "
        "Call ask_user if the objective cannot be judged safely. Call finish_search with accepted=true "
        "only when the best result meets the user's goal. "
        + (
            "This is the final allowed attempt; call finish_search or ask_user and do not propose more recipes."
            if attempt_number >= max_attempts
            else "This is not the final attempt: revise the pipeline when results are not good. You may only accept early, not reject early."
        )
    )
    messages = [
        {"role": "system", "content": system_message},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "request": user_request,
                    "target_column": target_column,
                    "dataset_profile": dataset_context,
                    "attempt": attempt_number,
                    "max_attempts": max_attempts,
                    "attempt_history": history,
                },
                ensure_ascii=True,
            ),
        },
    ]
    completion = await _request_agent_tools(
        messages=messages,
        tools=tools,
        operation="review the pipeline result",
    )

    if not completion.choices or not completion.choices[0].message.tool_calls:
        raise AgentPlanningError("The model returned no decision for the pipeline result.")

    recipe_selections = []
    questions = []
    finish_decisions = []
    used_recipe_ids = set()
    message = completion.choices[0].message

    for tool_call in message.tool_calls:
        function_name = tool_call.function.name
        try:
            arguments = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as exc:
            raise AgentPlanningError("The model returned malformed pipeline review arguments.") from exc
        if not isinstance(arguments, dict):
            raise AgentPlanningError("OpenAI returned invalid pipeline review arguments.")
        try:
            validate(arguments, tool_schemas[function_name])
        except (KeyError, JsonSchemaValidationError) as exc:
            raise AgentPlanningError("OpenAI returned invalid pipeline review arguments.") from exc

        if function_name == "ask_user":
            questions.append({"question": arguments["question"], "reason": arguments["reason"]})
        elif function_name == "finish_search":
            finish_decisions.append(arguments)
        elif function_name in recipe_tools:
            recipe_id = recipe_tools[function_name]
            if recipe_id in used_recipe_ids:
                raise AgentPlanningError(f"OpenAI selected recipe '{recipe_id}' more than once.")
            try:
                validate(arguments, _recipe_parameters(recipe_id))
            except JsonSchemaValidationError as exc:
                raise AgentPlanningError(
                    f"OpenAI returned an invalid configuration for recipe '{recipe_id}'."
                ) from exc
            used_recipe_ids.add(recipe_id)
            recipe_selections.append((recipe_id, arguments))
        else:
            raise AgentPlanningError("OpenAI returned a tool that is not available to the reviewer.")

    if questions:
        return {"action": "needs_clarification", "questions": questions}
    if finish_decisions and recipe_selections:
        raise AgentPlanningError("OpenAI returned conflicting stop and recipe-change decisions.")
    if len(finish_decisions) > 1:
        raise AgentPlanningError("OpenAI returned multiple pipeline stop decisions.")
    if finish_decisions:
        return {
            "action": "finish",
            "accepted": finish_decisions[0]["accepted"],
            "reason": finish_decisions[0]["reason"],
        }
    if not recipe_selections:
        raise AgentPlanningError("OpenAI returned no replacement recipe selections.")
    return {"action": "revise", "recipe_selections": recipe_selections}


async def _save_accepted_model(
    execution_id: str,
    metrics: Dict[str, float],
    task_type: str = "classification",
) -> Dict[str, Any]:
    bundle = job_manager.get_inference_bundle(execution_id)
    model = bundle.get("model") if bundle else None
    if model is None:
        raise AgentPlanningError("The accepted candidate has no fitted model to save.")

    from backend.app.recipes.governance.mlflow_tracker import MLFLOW_AVAILABLE

    if not MLFLOW_AVAILABLE:
        return {"saved": False, "error": "MLflow is not installed."}
    from mlflow.exceptions import MlflowException

    tracker = recipe_registry.get("mlflow_tracker")
    task_label = task_type.title()
    model_name = f"Agent{task_label}_{uuid.uuid4().hex[:10]}"
    config = {
        "experiment_name": f"Agent_{task_label}_Pipelines",
        "registered_model_name": model_name,
        "stage": "Staging",
        "tracking_uri": settings.MLFLOW_TRACKING_URI,
    }
    try:
        saved = await run_in_threadpool(
            tracker.execute,
            inputs={"model": model, "metrics": metrics},
            config=config,
            context=None,
        )
    except (ImportError, MlflowException, OSError, ValueError) as exc:
        logger.error(f"Could not save accepted agent model: {exc}")
        return {"saved": False, "error": str(exc)}
    return {"saved": True, **saved.get("governance_record", {})}


class PipelineAgentState(TypedDict, total=False):
    dataset_id: str
    dataframe: Any
    dataset_context: Dict[str, Any]
    user_request: str
    requested_target: Optional[str]
    requested_task_type: Optional[str]
    task_type: str
    max_attempts: int
    plan: Dict[str, Any]
    target_column: str
    review_context: Dict[str, Any]
    workflow: Dict[str, Any]
    attempt_number: int
    attempts: List[Dict[str, Any]]
    history: List[Dict[str, Any]]
    tried_candidates: List[str]
    best_candidate: Optional[Dict[str, Any]]
    review: Dict[str, Any]
    decision: Dict[str, Any]
    final_result: Dict[str, Any]


async def _plan_pipeline(state: PipelineAgentState) -> Dict[str, Any]:
    plan = await create_agent_plan(
        dataset_id=state["dataset_id"],
        dataset_context=state["dataset_context"],
        user_request=state["user_request"],
        requested_target=state.get("requested_target"),
        requested_task_type=state.get("requested_task_type"),
    )
    if plan.get("status") != "ready_for_review":
        return {"plan": plan, "final_result": plan}

    workflow = WorkflowGraph(**plan["workflow"])
    target_column = plan["target_column"]
    task_type = plan.get("task_type", "classification")
    review_context = dict(state["dataset_context"])
    if task_type == "classification":
        target_counts = state["dataframe"][target_column].value_counts(dropna=True)
        review_context["classification_summary"] = {
            "class_count": int(target_counts.size),
            "majority_class_baseline_accuracy": (
                float(target_counts.max() / target_counts.sum()) if target_counts.sum() else 0.0
            ),
        }
    else:
        numeric_target = pd.to_numeric(
            state["dataframe"][target_column], errors="coerce"
        ).dropna()
        review_context["regression_summary"] = {
            "valid_target_count": int(numeric_target.size),
            "target_min": float(numeric_target.min()) if not numeric_target.empty else None,
            "target_median": float(numeric_target.median()) if not numeric_target.empty else None,
            "target_max": float(numeric_target.max()) if not numeric_target.empty else None,
        }

    return {
        "plan": plan,
        "target_column": target_column,
        "task_type": task_type,
        "review_context": review_context,
        "workflow": workflow.model_dump(),
        "attempt_number": 0,
        "attempts": [],
        "history": [],
        "tried_candidates": [_candidate_fingerprint(workflow)],
        "best_candidate": None,
    }


def _route_after_planning(state: PipelineAgentState) -> str:
    return END if state.get("final_result") else "execute_pipeline"


def _candidate_score(
    metrics: Dict[str, float],
    task_type: str,
) -> Tuple[Optional[float], Optional[str], Optional[float]]:
    if task_type == "regression":
        if "r2_score" in metrics:
            return metrics["r2_score"], "r2_score", metrics["r2_score"]
        if "rmse" in metrics:
            return -metrics["rmse"], "rmse", metrics["rmse"]
        return None, None, None

    for metric in ("balanced_accuracy", "f1_score"):
        if metric in metrics:
            return metrics[metric], metric, metrics[metric]
    return None, None, None


async def _execute_pipeline(state: PipelineAgentState) -> Dict[str, Any]:
    workflow = WorkflowGraph(**state["workflow"])
    execution_id = str(uuid.uuid4())
    result = await run_in_threadpool(
        DAGExecutor.execute_workflow,
        execution_id=execution_id,
        workflow=workflow,
        initial_df=state["dataframe"],
        context={"target_column": state["target_column"]},
        include_node_outputs=False,
    )
    metrics = _numeric_metrics(result.final_metrics)
    recipes = [node.recipe_id for node in workflow.nodes]
    failed_recipes = [
        node.recipe_id for node in result.node_results if node.status == "FAILED"
    ]
    attempt_number = state["attempt_number"] + 1
    attempt = {
        "attempt": attempt_number,
        "execution_id": execution_id,
        "status": result.status,
        "metrics": metrics,
        "recipes": recipes,
        "failed_recipes": failed_recipes,
    }
    history_entry = {
        "attempt": attempt_number,
        "status": result.status,
        "metrics": metrics,
        "recipes": recipes,
        "failed_recipes": failed_recipes,
    }

    best = state.get("best_candidate")
    score, score_metric, score_value = _candidate_score(metrics, state["task_type"])
    if result.status == "SUCCESS" and score is not None:
        if best is None or score > best["score"]:
            best = {
                "score": score,
                "score_metric": score_metric,
                "score_value": score_value,
                "execution_id": execution_id,
                "metrics": metrics,
                "workflow": workflow.model_dump(),
            }

    return {
        "attempt_number": attempt_number,
        "attempts": [*state["attempts"], attempt],
        "history": [*state["history"], history_entry],
        "best_candidate": best,
    }


async def _evaluate_candidate(state: PipelineAgentState) -> Dict[str, Any]:
    review = await _review_attempt(
        dataset_context=state["review_context"],
        user_request=state["user_request"],
        target_column=state["target_column"],
        task_type=state["task_type"],
        history=state["history"],
        attempt_number=state["attempt_number"],
        max_attempts=state["max_attempts"],
    )
    return {"review": review}


def _route_after_evaluation(state: PipelineAgentState) -> str:
    if state["review"]["action"] == "revise":
        return "revise_pipeline"
    return "finalize"


async def _revise_pipeline(state: PipelineAgentState) -> Dict[str, Any]:
    workflow = build_agent_workflow(
        dataset_id=state["dataset_id"],
        target_column=state["target_column"],
        recipe_selections=state["review"]["recipe_selections"],
        task_type=state["task_type"],
    )
    fingerprint = _candidate_fingerprint(workflow)
    if fingerprint in state["tried_candidates"]:
        return {
            "decision": {
                "accepted": False,
                "reason": "The agent repeated a pipeline already tried; stopped to avoid redundant runs.",
            }
        }

    return {
        "workflow": workflow.model_dump(),
        "tried_candidates": [*state["tried_candidates"], fingerprint],
    }


def _route_after_revision(state: PipelineAgentState) -> str:
    return "finalize" if state.get("decision") else "execute_pipeline"


async def _finalize_run(state: PipelineAgentState) -> Dict[str, Any]:
    review = state["review"]
    if review["action"] == "needs_clarification":
        return {
            "final_result": {
                "status": "needs_clarification",
                "questions": review["questions"],
                "attempts": state["attempts"],
                "best_candidate": state.get("best_candidate"),
            }
        }

    decision = state.get("decision") or {
        "accepted": review["accepted"],
        "reason": review["reason"],
    }
    best = state.get("best_candidate")
    if best is None:
        return {
            "final_result": {
                "status": "no_successful_candidate",
                "accepted": False,
                "reason": "No candidate completed successfully with usable evaluation metrics.",
                "attempts": state["attempts"],
                "model_saved": False,
            }
        }

    model_save = {"saved": False}
    if decision["accepted"]:
        model_save = await _save_accepted_model(
            best["execution_id"],
            best["metrics"],
            state["task_type"],
        )

    if decision["accepted"] and model_save["saved"]:
        status_text = "completed"
    elif decision["accepted"]:
        status_text = "accepted_but_model_save_failed"
    else:
        status_text = "best_candidate_not_accepted"

    return {
        "final_result": {
            "status": status_text,
            "task_type": state["task_type"],
            "accepted": decision["accepted"],
            "reason": decision["reason"],
            "best_candidate": {
                "execution_id": best["execution_id"],
                "metrics": best["metrics"],
                "score_metric": best["score_metric"],
                "score_value": best["score_value"],
                "workflow": best["workflow"],
            },
            "attempts": state["attempts"],
            "model_saved": model_save["saved"],
            "model_registry": {
                key: value for key, value in model_save.items() if key != "saved"
            },
            "evaluation_role": "validation",
            "notes": [
                "The same validation split is used to compare attempts. Its score is for model selection, not an unbiased final test score.",
                "Some preprocessing recipes learn values before the split in the current engine and may leak validation information.",
            ],
        }
    }


def build_pipeline_agent_graph():
    graph = StateGraph(PipelineAgentState)
    graph.add_node("planner_agent", _plan_pipeline)
    graph.add_node("execute_pipeline", _execute_pipeline)
    graph.add_node("evaluator_reviser_agent", _evaluate_candidate)
    graph.add_node("revise_pipeline", _revise_pipeline)
    graph.add_node("finalize", _finalize_run)

    graph.add_edge(START, "planner_agent")
    graph.add_conditional_edges(
        "planner_agent",
        _route_after_planning,
        {END: END, "execute_pipeline": "execute_pipeline"},
    )
    graph.add_edge("execute_pipeline", "evaluator_reviser_agent")
    graph.add_conditional_edges(
        "evaluator_reviser_agent",
        _route_after_evaluation,
        {"revise_pipeline": "revise_pipeline", "finalize": "finalize"},
    )
    graph.add_conditional_edges(
        "revise_pipeline",
        _route_after_revision,
        {"execute_pipeline": "execute_pipeline", "finalize": "finalize"},
    )
    graph.add_edge("finalize", END)
    return graph.compile()


async def run_pipeline_agent(
    *,
    dataset_id: str,
    dataframe: Any,
    dataset_context: Dict[str, Any],
    user_request: str,
    max_attempts: int,
    requested_target: Optional[str] = None,
    requested_task_type: Optional[str] = None,
) -> Dict[str, Any]:
    graph = build_pipeline_agent_graph()
    result = await graph.ainvoke(
        {
            "dataset_id": dataset_id,
            "dataframe": dataframe,
            "dataset_context": dataset_context,
            "user_request": user_request,
            "requested_target": requested_target,
            "requested_task_type": requested_task_type,
            "max_attempts": max_attempts,
        },
        config={"recursion_limit": max_attempts * 4 + 8},
    )
    final_result = result.get("final_result")
    if final_result is None:
        raise AgentPlanningError("The LangGraph agent run ended without a final result.")
    return final_result
