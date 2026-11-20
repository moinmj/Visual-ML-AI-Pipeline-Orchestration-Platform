import json
import threading
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from jsonschema import Draft202012Validator, ValidationError as JsonSchemaValidationError, validate
from jsonschema.exceptions import SchemaError
from openai import AsyncOpenAI, OpenAIError

import backend.app.recipes
from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.engine.dag.graph import WorkflowEdge, WorkflowGraph, WorkflowNode
from backend.app.profiling.profiler import DataProfiler
from backend.app.recipes.base.registry import recipe_registry


TRANSFORM_CATEGORIES = {"preprocessing", "nlp"}
SUPPORTED_TASK_TYPES = ("classification", "regression")

# Providers limit tokens per minute (e.g. Groq's free tier allows 8,000 for one request).
# The recipe tools are the bulk of each request, so they are compacted and trimmed to
# this estimated budget. Trainers are always kept; preprocessing tools are dropped
# largest-first when the budget is exceeded.
AGENT_TOOL_TOKEN_BUDGET = 4500
_STRIP_SCHEMA_KEYS = {"description", "title", "examples", "$comment"}


def _compact_schema(node: Any) -> Any:
    """Drop prose-only keywords from a JSON schema to save tokens (validation is unchanged)."""
    if isinstance(node, dict):
        return {
            key: _compact_schema(value)
            for key, value in node.items()
            if not (key in _STRIP_SCHEMA_KEYS and not isinstance(value, dict))
        }
    if isinstance(node, list):
        return [_compact_schema(value) for value in node]
    return node


def _estimate_tokens(obj: Any) -> int:
    """Rough, slightly pessimistic token estimate for a JSON-serializable object."""
    return len(json.dumps(obj, ensure_ascii=True)) // 3


class AgentPlanningError(Exception):
    """Raised when the model returns a plan that cannot safely become a workflow."""


class AgentConfigurationError(AgentPlanningError):
    """Raised when no supported model provider is configured."""


class AgentCapacityError(AgentPlanningError):
    """Raised when the process has reached its agent request concurrency limit."""


_AGENT_CAPACITY = threading.BoundedSemaphore(settings.AGENT_MAX_GROQ_CONCURRENT)


def agent_provider_available() -> bool:
    return any(
        api_key and api_key.strip()
        for api_key in (settings.OPENAI_API_KEY, settings.GROQ_API_KEY)
    )


def _get_agent_provider() -> Tuple[str, str, str, Optional[str]]:
    if settings.OPENAI_API_KEY and settings.OPENAI_API_KEY.strip():
        return "OpenAI", settings.OPENAI_API_KEY.strip(), settings.OPENAI_MODEL, None
    if settings.GROQ_API_KEY and settings.GROQ_API_KEY.strip():
        return (
            "Groq",
            settings.GROQ_API_KEY.strip(),
            settings.GROQ_MODEL,
            "https://api.groq.com/openai/v1",
        )
    raise AgentConfigurationError(
        "No recommendation-agent provider is configured. Set OPENAI_API_KEY or GROQ_API_KEY."
    )


async def _request_agent_tools(
    *,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    operation: str,
):
    provider, api_key, model, base_url = _get_agent_provider()
    if not _AGENT_CAPACITY.acquire(blocking=False):
        raise AgentCapacityError("The agent request capacity is full; retry shortly.")

    try:
        client_options = {"api_key": api_key, "timeout": 30.0, "max_retries": 2}
        if base_url:
            client_options["base_url"] = base_url
        async with AsyncOpenAI(**client_options) as client:
            return await client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="required",
                parallel_tool_calls=True,
            )
    except OpenAIError as exc:
        logger.error("%s agent %s request failed: %s", provider, operation, exc)
        raise AgentPlanningError(f"{provider} could not {operation}.") from exc
    finally:
        _AGENT_CAPACITY.release()

def build_dataset_context(df: pd.DataFrame) -> Dict[str, Any]:
    """Return schema and aggregate profile only; never send data rows or category values."""
    return build_dataset_context_from_profile(DataProfiler.profile_dataframe(df))


def build_dataset_context_from_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only the safe schema fields needed for planning from a stored data profile."""
    columns = {}
    profile_columns = profile.get("columns")
    if not isinstance(profile_columns, dict):
        profile_columns = {}
    for column_name, metadata in profile_columns.items():
        if not isinstance(metadata, dict):
            continue
        columns[column_name] = {
            "inferred_type": metadata.get("inferred_type"),
            "dtype": metadata.get("raw_dtype"),
            "null_count": metadata.get("null_count"),
            "null_percentage": metadata.get("null_percentage"),
            "unique_count": metadata.get("unique_count"),
        }

    return {
        "row_count": profile.get("row_count", 0),
        "column_count": profile.get("column_count") or len(columns),
        "columns": columns,
    }


def _recipe_tool_name(recipe_id: str) -> str:
    return f"select_recipe_{recipe_id}"


def _recipe_task_types(recipe_id: str) -> List[str]:
    schema = recipe_registry.get(recipe_id).get_schema()
    task_schema = schema.get("properties", {}).get("task_type", {})
    supported = task_schema.get("enum")
    if isinstance(supported, list):
        return [task for task in SUPPORTED_TASK_TYPES if task in supported]
    return []


def _recipe_parameters(
    recipe_id: str,
    task_type: Optional[str] = None,
) -> Dict[str, Any]:
    recipe = recipe_registry.get(recipe_id)
    schema = deepcopy(recipe.get_schema())
    schema.setdefault("additionalProperties", False)

    if recipe.category == "training":
        properties = schema.setdefault("properties", {})
        task_schema = properties.setdefault("task_type", {"type": "string"})
        compatible_tasks = _recipe_task_types(recipe_id)
        task_schema["enum"] = (
            [task_type] if task_type in compatible_tasks else compatible_tasks
        )
        if not task_schema["enum"]:
            raise AgentPlanningError(
                f"Training recipe '{recipe_id}' does not support a supervised task."
            )
        task_schema["default"] = task_type or (
            "classification" if "classification" in compatible_tasks else compatible_tasks[0]
        )
        required = schema.setdefault("required", [])
        if "task_type" not in required:
            required.append("task_type")
    elif recipe_id == "categorical_encoder":
        methods = schema.get("properties", {}).get("method", {}).get("enum", [])
        schema["properties"]["method"]["enum"] = [
            method for method in methods if method not in {"target", "woe"}
        ]

    return schema


def _available_task_types() -> List[str]:
    return sorted(
        {
            task
            for metadata in recipe_registry.list_all(category="training")
            if "train_data" in metadata.input_types and "model" in metadata.output_types
            for task in _recipe_task_types(metadata.recipe_id)
        }
    )


def build_agent_tools(
    column_names: List[str],
    task_type: Optional[str] = None,
    target_fixed: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    available_tasks = _available_task_types()
    if task_type is not None and task_type not in available_tasks:
        raise AgentPlanningError(f"Unsupported supervised task type '{task_type}'.")

    choose_task_type_tool = {
        "type": "function",
        "function": {
            "name": "choose_task_type",
            "description": (
                "Choose the supervised learning task requested by the user. "
                "Use regression for a numeric continuous outcome and classification "
                "for a discrete class label. Ask the user if the intent is ambiguous."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_type": {
                        "type": "string",
                        "enum": available_tasks,
                    },
                    "reason": {"type": "string"},
                },
                "required": ["task_type", "reason"],
                "additionalProperties": False,
            },
        },
    }
    choose_target_column_tool = {
        "type": "function",
        "function": {
            "name": "choose_target_column",
            "description": "Choose the dataset column the user wants the model to predict.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string", "enum": column_names},
                    "reason": {"type": "string"},
                },
                "required": ["column", "reason"],
                "additionalProperties": False,
            },
        },
    }
    ask_user_tool = {
        "type": "function",
        "function": {
            "name": "ask_user",
            "description": "Ask a concise clarification question when a safe plan cannot be made.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["question", "reason"],
                "additionalProperties": False,
            },
        },
    }

    # Only expose choose_* tools when the answer is not already fixed by the user.
    # Otherwise the model can satisfy tool_choice="required" with a redundant call
    # and never select a trainer.
    tools: List[Dict[str, Any]] = [ask_user_tool]
    if task_type is None:
        tools.append(choose_task_type_tool)
    if not target_fixed:
        tools.append(choose_target_column_tool)

    tool_recipes: Dict[str, str] = {}

    for metadata in recipe_registry.list_all():
        is_transform = (
            metadata.category in TRANSFORM_CATEGORIES
            and "dataframe" in metadata.input_types
            and "dataframe" in metadata.output_types
        )
        is_trainer = (
            metadata.category == "training"
            and "train_data" in metadata.input_types
            and "model" not in metadata.input_types  # stackers/tuners need an existing model
            and "model" in metadata.output_types
            and bool(_recipe_task_types(metadata.recipe_id))
            and (
                task_type is None
                or task_type in _recipe_task_types(metadata.recipe_id)
            )
        )
        if not (is_transform or is_trainer):
            continue

        # Groq validates every tool schema against JSON Schema draft 2020-12 and rejects
        # the whole request if one is invalid. Skip such recipes instead of failing.
        try:
            Draft202012Validator.check_schema(
                _recipe_parameters(metadata.recipe_id, task_type)
            )
        except SchemaError as exc:
            logger.warning(
                "Skipping recipe '%s' for the agent: invalid tool schema (%s)",
                metadata.recipe_id,
                exc.message,
            )
            continue

        recipe = recipe_registry.get(metadata.recipe_id)
        name = _recipe_tool_name(metadata.recipe_id)
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": (
                        f"Add the registered {recipe.category} recipe "
                        f"'{recipe.name}'. {(recipe.description or '')[:140]}"
                    ),
                    "parameters": _compact_schema(
                        _recipe_parameters(metadata.recipe_id, task_type)
                    ),
                },
            }
        )
        tool_recipes[name] = metadata.recipe_id

    # Keep the request under the provider's tokens-per-minute limit.
    total_tokens = sum(_estimate_tokens(tool) for tool in tools)
    if total_tokens > AGENT_TOOL_TOKEN_BUDGET:
        droppable = sorted(
            (
                tool
                for tool in tools
                if tool["function"]["name"] in tool_recipes
                and recipe_registry.get(tool_recipes[tool["function"]["name"]]).category
                != "training"
            ),
            key=_estimate_tokens,
            reverse=True,
        )
        dropped = []
        for tool in droppable:
            if total_tokens <= AGENT_TOOL_TOKEN_BUDGET:
                break
            tools.remove(tool)
            total_tokens -= _estimate_tokens(tool)
            tool_recipes.pop(tool["function"]["name"], None)
            dropped.append(tool["function"]["name"])
        logger.warning(
            "Agent tool list trimmed to fit the token budget; dropped %d preprocessing tools: %s",
            len(dropped),
            dropped,
        )
    logger.info(
        "Agent tools: %d tools, ~%d estimated tokens (budget %d)",
        len(tools),
        total_tokens,
        AGENT_TOOL_TOKEN_BUDGET,
    )

    return tools, tool_recipes


def build_agent_workflow(
    dataset_id: str,
    target_column: str,
    recipe_selections: List[Tuple[str, Dict[str, Any]]],
    task_type: str = "classification",
) -> WorkflowGraph:
    """Build and validate a supervised workflow from compatible registered recipes."""
    if task_type not in SUPPORTED_TASK_TYPES:
        raise AgentPlanningError(f"Unsupported supervised task type '{task_type}'.")
    if recipe_registry.get("model_evaluator") is None:
        raise AgentPlanningError("The model evaluator recipe is not registered.")

    trainer_selections = []
    preprocessing_selections = []
    for recipe_id, config in recipe_selections:
        recipe = recipe_registry.get(recipe_id)
        try:
            validate(config, _recipe_parameters(recipe_id, task_type))
        except JsonSchemaValidationError as exc:
            raise AgentPlanningError(
                f"Invalid configuration for recipe '{recipe_id}': {exc.message}"
            ) from exc

        if recipe.category == "training":
            if config.get("task_type") != task_type:
                raise AgentPlanningError(
                    f"Trainer '{recipe_id}' must use task type '{task_type}'."
                )
            trainer_selections.append((recipe_id, config))
        elif recipe.category in TRANSFORM_CATEGORIES:
            preprocessing_selections.append((recipe_id, config))
        else:
            raise AgentPlanningError(f"Recipe '{recipe_id}' is not supported by this planner.")

    if len(trainer_selections) != 1:
        raise AgentPlanningError("The plan must select exactly one compatible trainer.")

    nodes = [
        WorkflowNode(
            id="agent_data",
            recipe_id="csv_loader",
            label="Uploaded dataset",
            config={"dataset_id": dataset_id},
        )
    ]
    edges = []
    previous_node = "agent_data"

    for index, (recipe_id, config) in enumerate(preprocessing_selections, start=1):
        recipe = recipe_registry.get(recipe_id)
        step_config = dict(config)
        if "target_column" in recipe.get_schema().get("properties", {}):
            step_config.setdefault("target_column", target_column)
        node_id = f"agent_preprocess_{index}"
        nodes.append(
            WorkflowNode(
                id=node_id,
                recipe_id=recipe_id,
                label=recipe.name,
                config=step_config,
            )
        )
        edges.append(WorkflowEdge(source=previous_node, target=node_id))
        previous_node = node_id

    nodes.extend(
        [
            WorkflowNode(
                id="agent_split",
                recipe_id="train_test_split",
                label="Train / test split",
                config={
                    "target_column": target_column,
                    "test_size": 0.2,
                    "stratify": task_type == "classification",
                    "random_state": 42,
                },
            ),
            WorkflowNode(
                id="agent_trainer",
                recipe_id=trainer_selections[0][0],
                label=recipe_registry.get(trainer_selections[0][0]).name,
                config=trainer_selections[0][1],
            ),
            WorkflowNode(
                id="agent_evaluator",
                recipe_id="model_evaluator",
                label=f"Evaluate {task_type} model",
                config={
                    "report_type": (
                        "Regression Diagnostics (MAE, RMSE, R2, MAPE)"
                        if task_type == "regression"
                        else "Comprehensive (All Metrics + Confusion Matrix)"
                    )
                },
            ),
        ]
    )
    edges.extend(
        [
            WorkflowEdge(source=previous_node, target="agent_split"),
            WorkflowEdge(source="agent_split", target="agent_trainer"),
            WorkflowEdge(source="agent_split", target="agent_evaluator"),
            WorkflowEdge(source="agent_trainer", target="agent_evaluator"),
        ]
    )

    workflow = WorkflowGraph(nodes=nodes, edges=edges)
    diagnostics = workflow.get_diagnostics()
    if not diagnostics["is_valid"]:
        raise AgentPlanningError(
            "The generated recipe workflow is invalid: "
            + " | ".join(diagnostics["errors"])
        )
    return workflow


async def create_agent_plan(
    dataset_id: str,
    dataset_context: Dict[str, Any],
    user_request: str,
    requested_target: Optional[str],
    requested_task_type: Optional[str] = None,
) -> Dict[str, Any]:
    column_names = list(dataset_context["columns"])
    tools, tool_recipes = build_agent_tools(
        column_names,
        requested_task_type,
        target_fixed=bool(requested_target),
    )
    logger.info("========== AGENT PLANNER DEBUG ==========")
    logger.info("Requested task type: %s", requested_task_type)

    logger.info(
        "Registered training recipes: %s",
        [
            {
                "recipe_id": metadata.recipe_id,
                "name": metadata.name,
                "category": metadata.category,
                "input_types": metadata.input_types,
                "output_types": metadata.output_types,
                "supported_tasks": _recipe_task_types(metadata.recipe_id),
            }
            for metadata in recipe_registry.list_all(category="training")
        ],
    )

    logger.info(
        "Trainer tools exposed to recommendation agent: %s",
        [
            {"tool": name, "recipe_id": recipe_id}
            for name, recipe_id in tool_recipes.items()
            if recipe_registry.get(recipe_id).category == "training"
        ],
    )
    tool_schemas = {
        tool["function"]["name"]: tool["function"]["parameters"]
        for tool in tools
    }
    trainer_tool_names = [
        name
        for name, recipe_id in tool_recipes.items()
        if recipe_registry.get(recipe_id).category == "training"
    ]

    logger.info("Mandatory trainer choices available: %s", trainer_tool_names)
    # choose_task_type may not be exposed when the task is fixed, so don't read it from tools[0].
    supported_tasks = (
        [requested_task_type]
        if requested_task_type
        else _available_task_types()
    )
    task_instruction = (
        f"The user explicitly selected task type '{requested_task_type}'. "
        "Use it and do not select a different task."
        if requested_task_type
        else f"Select the task type from the available options: {supported_tasks}. "
        "If the request describes a different task family, explain the limitation and ask the user how to proceed."
    )
    target_instruction = (
        f"The user explicitly selected target column '{requested_target}'. "
        "Use it and do not select a different target."
        if requested_target
        else "Select a target column using choose_target_column. If the user's intent is ambiguous, call ask_user instead."
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a constrained supervised-pipeline planner. Use the supplied recipe tools "
                "to select zero or more preprocessing steps followed by exactly one compatible "
                "trainer. The server will add dataset loading, a train/test split used for candidate "
                "validation, and model evaluation. Call ask_user if required information is unclear. "
                "Never claim that a model has been trained or saved during planning. Tool calls only "
                "draft a plan; they do not execute recipes. Only choose recipe tools compatible with "
                "the selected task; the recipe catalog is informational and does not authorize direct execution. "

                # Dataset-aware trainer selection
                "IMPORTANT: You MUST select exactly one trainer based on the supplied dataset profile. "
                "Analyze the available dataset information, including row count, column count, feature "
                "types, target type, categorical feature cardinality, missing values, and other supplied "
                "statistics. Consider the characteristics of the dataset and the suitability of each "
                "available trainer. Compare the available trainer descriptions and select the most "
                "appropriate one. Do not always choose the same trainer, do not select the first trainer "
                "in the recipe catalog, and do not choose randomly. "
                "Your trainer selection must be supported by the dataset characteristics. "
                "If the user's request names a specific trainer, select that trainer. "
                "If the available dataset information is insufficient to make a reasonable choice, "
                "call ask_user rather than inventing dataset characteristics. "

                "Avoid target encoding because it can leak labels. For classification, do not choose "
                "a numeric target with many unique values without asking for confirmation. For "
                "regression, require a numeric target. Do not request or infer raw data values. "

                + task_instruction
                + " "
                + target_instruction
            ),

        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "request": user_request,
                    "dataset_profile": dataset_context,
                },
                ensure_ascii=True,
            ),
        },
    ]

    completion = await _request_agent_tools(
        messages=messages,
        tools=tools,
        operation="create a pipeline plan",
    )

    # Log the raw tool calls before parsing so failures can be diagnosed.
    logger.info(
        "Agent finish reason: %s",
        completion.choices[0].finish_reason if completion.choices else None,
    )
    if completion.choices:
        logger.info(
            "Agent tool calls: %s",
            [
                {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                }
                for call in (completion.choices[0].message.tool_calls or [])
            ],
        )

    if not completion.choices or not completion.choices[0].message.tool_calls:
        raise AgentPlanningError("The model returned no recipe selections for the pipeline plan.")

    message = completion.choices[0].message
    tool_calls = list(message.tool_calls)

    # parallel_tool_calls is not reliably honored: if the model skipped the trainer,
    # retry once with only recipe tools (plus ask_user).
    picked_trainer = any(tc.function.name in trainer_tool_names for tc in tool_calls)
    asked_user = any(tc.function.name == "ask_user" for tc in tool_calls)
    if not picked_trainer and not asked_user:
        logger.info("No trainer selected on first pass; retrying with recipe tools only.")
        retry_tools = [
            t
            for t in tools
            if t["function"]["name"] in tool_recipes
            or t["function"]["name"] == "ask_user"
        ]
        retry_messages = messages + [
            {
                "role": "system",
                "content": (
                    "Task type and target are already settled. Now call exactly one "
                    "trainer recipe tool, plus any needed preprocessing tools."
                ),
            }
        ]
        retry = await _request_agent_tools(
            messages=retry_messages,
            tools=retry_tools,
            operation="select a trainer",
        )
        if retry.choices and retry.choices[0].message.tool_calls:
            logger.info(
                "Agent retry tool calls: %s",
                [
                    {"name": call.function.name, "arguments": call.function.arguments}
                    for call in retry.choices[0].message.tool_calls
                ],
            )
            tool_calls.extend(retry.choices[0].message.tool_calls)

    questions = []
    selected_target = requested_target
    selected_task_type = requested_task_type
    recipe_selections = []
    used_recipe_ids = set()

    for tool_call in tool_calls:
        function_name = tool_call.function.name
        try:
            arguments = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as exc:
            raise AgentPlanningError("OpenAI returned malformed recipe configuration.") from exc
        if not isinstance(arguments, dict):
            raise AgentPlanningError("OpenAI returned invalid tool arguments.")
        try:
            validate(arguments, tool_schemas[function_name])
        except (KeyError, JsonSchemaValidationError) as exc:
            raise AgentPlanningError("OpenAI returned invalid tool arguments.") from exc

        if function_name == "ask_user":
            questions.append(
                {
                    "question": arguments["question"],
                    "reason": arguments["reason"],
                }
            )
        elif function_name == "choose_task_type":
            task = arguments.get("task_type")
            if task not in supported_tasks:
                raise AgentPlanningError("OpenAI selected a task type that is not supported.")
            if requested_task_type and task != requested_task_type:
                raise AgentPlanningError("OpenAI did not use the explicitly requested task type.")
            if selected_task_type and task != selected_task_type:
                raise AgentPlanningError("OpenAI returned conflicting task types.")
            selected_task_type = task
        elif function_name == "choose_target_column":
            column = arguments.get("column")
            if column not in column_names:
                raise AgentPlanningError("OpenAI selected a target column not present in the dataset.")
            if requested_target and column != requested_target:
                raise AgentPlanningError("OpenAI did not use the explicitly requested target column.")
            if selected_target and column != selected_target:
                raise AgentPlanningError("OpenAI returned conflicting target columns.")
            selected_target = column
        elif function_name in tool_recipes:
            recipe_id = tool_recipes[function_name]
            if recipe_id in used_recipe_ids:
                raise AgentPlanningError(f"OpenAI selected recipe '{recipe_id}' more than once.")
            used_recipe_ids.add(recipe_id)
            recipe_selections.append((recipe_id, arguments))
        else:
            raise AgentPlanningError("OpenAI returned a tool that is not available to the planner.")

    if questions:
        return {"status": "needs_clarification", "questions": questions}
    if not selected_task_type:
        return {
            "status": "needs_clarification",
            "questions": [
                {
                    "question": "Should the model classify categories or predict a numeric value?",
                    "reason": "The request and dataset profile do not make the supervised task type clear.",
                }
            ],
        }
    if not selected_target:
        return {
            "status": "needs_clarification",
            "questions": [
                {
                    "question": "Which column should the model predict?",
                    "reason": "A supervised workflow requires a target column.",
                }
            ],
        }
    target_profile = dataset_context["columns"][selected_target]
    unique_count = target_profile.get("unique_count")
    if not isinstance(unique_count, (int, float)) or unique_count < 2:
        return {
            "status": "needs_clarification",
            "questions": [
                {
                    "question": f"'{selected_target}' has fewer than two distinct values. Which target column should be used?",
                    "reason": "A supervised target must contain at least two distinct values.",
                }
            ],
        }
    if (
        selected_task_type == "regression"
        and target_profile.get("inferred_type") != "numeric"
    ):
        return {
            "status": "needs_clarification",
            "questions": [
                {
                    "question": f"'{selected_target}' does not appear numeric. Which numeric column should be predicted for regression?",
                    "reason": "The selected regression task requires a numeric target column.",
                }
            ],
        }

    # Log parsed selections BEFORE building the workflow. The workflow builder
    # can raise when zero or multiple compatible trainers were selected.
    logger.info("========== AGENT SELECTION DEBUG ==========")
    logger.info("Selected task type: %s", selected_task_type)
    logger.info("Selected target column: %s", selected_target)
    logger.info("All recipes selected by agent: %s", recipe_selections)
    logger.info(
        "Selected trainers: %s",
        [
            recipe_id
            for recipe_id, _ in recipe_selections
            if recipe_registry.get(recipe_id).category == "training"
        ],
    )
    logger.info("========== END GROQ SELECTION DEBUG ==========")

    workflow = build_agent_workflow(
        dataset_id=dataset_id,
        target_column=selected_target,
        recipe_selections=recipe_selections,
        task_type=selected_task_type,
    )
    return {
        "status": "ready_for_review",
        "task_type": selected_task_type,
        "target_column": selected_target,
        "summary": message.content or "A classification workflow was generated for review.",
        "workflow": workflow.model_dump(),
        "validation": workflow.get_diagnostics(),
        "notes": (
            [
                "Some preprocessing recipes learn values before the train/test split in the current engine; "
                "evaluation scores may be optimistic when those recipes are selected."
            ]
            if any(
                recipe_registry.get(recipe_id).category in TRANSFORM_CATEGORIES
                for recipe_id, _ in recipe_selections
            )
            else []
        ),
        "execution_endpoint": "/api/v1/workflows/execute",
        "approval_required": True,
    }