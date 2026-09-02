import time
import traceback
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field

from backend.app.engine.dag.graph import WorkflowGraph, WorkflowNode
from backend.app.recipes.base.registry import recipe_registry
from backend.app.core.exceptions import ExecutionException
from backend.app.core.logging import logger


class NodeExecutionResult(BaseModel):
    node_id: str
    recipe_id: str
    status: str  # "SUCCESS", "FAILED", "SKIPPED"
    duration_ms: float
    error_message: Optional[str] = None
    error_title: Optional[str] = None
    error_suggestion: Optional[str] = None
    output_summary: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}


def diagnose_execution_error(
    recipe: Any,
    error: Exception,
    inputs: Dict[str, Any],
    config: Dict[str, Any],
    context: Dict[str, Any]
) -> Dict[str, str]:
    """
    Translates cryptic low-level Python/C++/ML library errors into plain-English root causes
    and concrete, actionable step-by-step UI suggestions for the visual pipeline canvas.
    """
    err_str = str(error)
    rec_name = getattr(recipe, "name", "Processor")

    # 1. Unencoded Categorical/String Columns in ML Trainers
    if any(k in err_str for k in ["enable_categorical", "DataFrame.dtypes for data must be int, float", "could not convert string to float", "cannot convert string"]):
        return {
            "title": "Unencoded Categorical Features",
            "message": f"Trainer '{rec_name}' received non-numeric text columns. Machine learning algorithms require categorical features to be encoded into numbers.",
            "suggestion": "Insert a 'Categorical Feature Encoder' processor before 'Train / Test Splitter' to choose an encoding strategy (One-Hot, Target, Label, or Binary)."
        }

    # 2. Missing Train/Test Split or Missing Partition Keys
    if any(k in err_str for k in ["expects 'X_train' and 'y_train'", "expects 'X_test' and 'y_test'"]):
        return {
            "title": "Missing Dataset Split",
            "message": f"'{rec_name}' requires train/test dataset partitions but did not find them in upstream inputs.",
            "suggestion": "Connect a 'Train / Test Splitter' processor before this component and ensure a target column is selected."
        }

    # 3. Missing Model in Model Evaluator
    if "expects a trained 'model'" in err_str:
        return {
            "title": "Missing Trained Model",
            "message": f"'{rec_name}' requires a trained machine learning model to evaluate.",
            "suggestion": "Connect a model trainer (e.g. XGBoost, Random Forest, LightGBM, CatBoost) to this Evaluator."
        }

    # 4. Time Series Forecaster Missing Date Column
    if any(k in err_str for k in ["'ds'", "time-series observations", "date_column"]):
        return {
            "title": "Time-Series Date Column Issue",
            "message": f"Forecaster '{rec_name}' could not identify a valid sequential date column.",
            "suggestion": "Open processor settings and configure 'date_column' to your date/timestamp column (e.g., 'OrderDate', 'Date')."
        }

    # 5. Target Column Missing or Not Found
    if any(k in err_str for k in ["Target column", "target_column", "not found in dataframe"]):
        return {
            "title": "Target Column Not Found",
            "message": f"The target variable specified does not exist in the incoming dataset.",
            "suggestion": "Open processor configuration and select a valid target column from your dataset."
        }

    # 6. Feature Dimension Mismatch
    if "not aligned" in err_str or "feature_names" in err_str:
        return {
            "title": "Feature Dimension Mismatch",
            "message": f"The feature columns in test data do not match what the model was trained on.",
            "suggestion": "Ensure the exact same preprocessing steps (Imputer, Scaler, Encoder) are applied before both training and testing."
        }

    # Default fallback
    return {
        "title": f"Execution Error in {rec_name}",
        "message": err_str,
        "suggestion": "Check incoming connection handles and verify the processor configuration parameters."
    }


class WorkflowExecutionResult(BaseModel):
    execution_id: str
    status: str  # "SUCCESS", "FAILED"
    total_duration_ms: float
    node_results: List[Any] = Field(default_factory=list)
    final_metrics: Optional[Dict[str, Any]] = None
    anomaly_summary: Optional[Dict[str, Any]] = None
    forecasting_summary: Optional[Dict[str, Any]] = None
    governance_summary: Optional[Dict[str, Any]] = None
    logs: List[str] = Field(default_factory=list)
    node_outputs: Dict[str, Any] = Field(default_factory=dict)
    step_snapshots: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}


def make_json_safe(obj: Any) -> Any:
    """
    Recursively converts arbitrary Python/ML objects (DataFrames, ndarrays, numpy scalars,
    models) into standard, JSON-serializable primitives for FastAPI Pydantic responses.
    """
    if obj is None:
        return None
    if isinstance(obj, pd.DataFrame):
        clean_df = obj.replace({float("nan"): None, float("inf"): None, float("-inf"): None})
        return {
            "type": "DataFrame",
            "shape": list(obj.shape),
            "columns": list(obj.columns),
            "records": clean_df.to_dict(orient="records")
        }
    elif isinstance(obj, pd.Series):
        clean_s = obj.replace({float("nan"): None, float("inf"): None, float("-inf"): None})
        return clean_s.to_dict()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [make_json_safe(v) for v in obj]
    elif hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        try:
            return make_json_safe(obj.to_dict())
        except Exception:
            return str(obj)
    elif type(obj).__module__ != "builtins":
        return f"<{type(obj).__name__} Object>"
    else:
        return obj


class DAGExecutor:
    """
    Unified In-Memory DAG Execution Engine.
    Executes nodes in topological order, manages context passing,
    and captures metrics, summaries, diagnostic artifacts, and step snapshots.
    """

    @classmethod
    def execute_workflow(
        cls,
        execution_id: str,
        workflow: WorkflowGraph,
        initial_df: Optional[pd.DataFrame] = None,
        context: Optional[Dict[str, Any]] = None,
        include_node_outputs: bool = False
    ) -> WorkflowExecutionResult:
        start_time = time.time()
        logs: List[str] = []
        node_results: List[NodeExecutionResult] = []
        node_outputs: Dict[str, Dict[str, Any]] = {}
        step_snapshots: Dict[str, Dict[str, Any]] = {}
        
        final_metrics = None
        anomaly_summary = None
        forecasting_summary = None
        governance_summary = None

        pipeline_context = dict(context or {})
        if initial_df is not None:
            pipeline_context["dataframe"] = initial_df.copy()

        logs.append(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] Starting execution '{execution_id}'")

        # 1. Topological Sorting
        try:
            ordered_nodes = workflow.get_topological_order()
        except Exception as e:
            err_msg = f"DAG Resolution Error: {str(e)}"
            logs.append(f"❌ {err_msg}")
            return WorkflowExecutionResult(
                execution_id=execution_id,
                status="FAILED",
                total_duration_ms=0.0,
                node_results=[],
                logs=logs
            )

        # 2. Build In-Edge map to find parent nodes
        parent_map: Dict[str, List[str]] = {n.id: [] for n in workflow.nodes}
        for edge in workflow.edges:
            parent_map[edge.target].append(edge.source)

        overall_status = "SUCCESS"

        # 3. Step-by-Step Node Execution
        for node in ordered_nodes:
            node_start = time.time()
            parents = parent_map[node.id]

            try:
                recipe = recipe_registry.get(node.recipe_id)
            except Exception as e:
                err_msg = f"Recipe '{node.recipe_id}' not found in registry: {str(e)}"
                logs.append(f"❌ Node '{node.id}' failed: {err_msg}")
                overall_status = "FAILED"
                node_results.append(NodeExecutionResult(
                    node_id=node.id,
                    recipe_id=node.recipe_id,
                    status="FAILED",
                    duration_ms=0.0,
                    error_message=err_msg
                ))
                break

            # Collect inputs from parents
            node_inputs: Dict[str, Any] = {}
            if not parents:
                # If root node, supply initial dataframe only if recipe accepts dataframe
                if "dataframe" in recipe.input_types or "any" in recipe.input_types or not recipe.input_types:
                    if initial_df is not None:
                        node_inputs = {"dataframe": initial_df.copy()}
                else:
                    # Model Trainer or Evaluator dropped without parents!
                    err_msg = (
                        f"Node '{node.id}' [{recipe.name}] requires inputs {recipe.input_types}, "
                        f"but has 0 incoming connections. It cannot run as an unparented root node."
                    )
                    duration_ms = round((time.time() - node_start) * 1000.0, 2)
                    logs.append(f"❌ Node '{node.id}' failed: {err_msg}")
                    node_results.append(NodeExecutionResult(
                        node_id=node.id,
                        recipe_id=node.recipe_id,
                        status="FAILED",
                        duration_ms=duration_ms,
                        error_message=err_msg
                    ))
                    overall_status = "FAILED"
                    break
            else:
                for parent_id in parents:
                    parent_out = node_outputs.get(parent_id, {})
                    node_inputs.update(parent_out)

            # Execution
            try:
                outputs = recipe.execute(inputs=node_inputs, config=node.config, context=pipeline_context)
                node_outputs[node.id] = outputs

                # Propagate standard artifacts to shared context
                for key in ["X_test", "y_test", "X_train", "y_train", "dataframe", "forecast_df", "model", "scaler", "encoder", "task_type", "feature_names", "feature_importances"]:
                    if key in outputs:
                        pipeline_context[key] = outputs[key]

                # Capture summaries & KPIs
                if "metrics" in outputs:
                    final_metrics = outputs["metrics"]
                if "anomaly_summary" in outputs:
                    anomaly_summary = outputs["anomaly_summary"]
                if "forecasting_summary" in outputs:
                    forecasting_summary = outputs["forecasting_summary"]
                if "governance_record" in outputs:
                    governance_summary = outputs["governance_record"]

                duration_ms = round((time.time() - node_start) * 1000.0, 2)
                logs.append(f"✅ Node `{node.id}` [{recipe.name}] ➔ Finished in {duration_ms}ms (SUCCESS)")

                # Create serializable summary & Step Snapshot (n8n/Boomi step inspection)
                summary: Dict[str, Any] = {}
                snapshot_info: Dict[str, Any] = {
                    "node_id": node.id,
                    "recipe_id": node.recipe_id,
                    "recipe_name": recipe.name,
                    "duration_ms": duration_ms,
                    "input_keys": list(node_inputs.keys()),
                    "output_keys": list(outputs.keys()),
                    "row_count": None,
                    "columns": [],
                    "preview_rows": []
                }

                for k, v in outputs.items():
                    if isinstance(v, pd.DataFrame):
                        summary[k] = {"shape": list(v.shape), "type": "DataFrame"}
                        snapshot_info["row_count"] = int(v.shape[0])
                        snapshot_info["columns"] = list(v.columns)
                        # Save top 5 rows for step inspection
                        try:
                            clean_v = v.head(5).replace({float("nan"): None, float("inf"): None, float("-inf"): None})
                            snapshot_info["preview_rows"] = clean_v.to_dict(orient="records")
                        except Exception:
                            pass
                    elif hasattr(v, "shape"):
                        summary[k] = {"shape": list(v.shape), "type": "Array"}
                    elif k in ["metrics", "anomaly_summary", "forecasting_summary", "feature_importances"]:
                        summary[k] = make_json_safe(v)
                    else:
                        summary[k] = {"type": type(v).__name__}

                step_snapshots[node.id] = snapshot_info

                node_results.append(NodeExecutionResult(
                    node_id=node.id,
                    recipe_id=node.recipe_id,
                    status="SUCCESS",
                    duration_ms=duration_ms,
                    output_summary=summary
                ))

            except Exception as e:
                duration_ms = round((time.time() - node_start) * 1000.0, 2)
                diag = diagnose_execution_error(recipe, e, node_inputs, node.config, pipeline_context)
                err_title = diag["title"]
                err_msg = diag["message"]
                err_sugg = diag["suggestion"]

                logs.append(f"❌ Node `{node.id}` [{recipe.name}] failed in {duration_ms}ms: {err_title} ➔ {err_msg}")
                logs.append(f"💡 Suggestion: {err_sugg}")
                logger.error(f"Execution failed on node {node.id} ({err_title}): {traceback.format_exc()}")

                node_results.append(NodeExecutionResult(
                    node_id=node.id,
                    recipe_id=node.recipe_id,
                    status="FAILED",
                    duration_ms=duration_ms,
                    error_message=err_msg,
                    error_title=err_title,
                    error_suggestion=err_sugg
                ))
                overall_status = "FAILED"
                break

        total_duration = round((time.time() - start_time) * 1000.0, 2)
        logs.append(f"🏁 Execution finished with status '{overall_status}' in {total_duration}ms")

        # Sanitize all outputs to be 100% JSON serializable for FastAPI responses
        safe_node_outputs = make_json_safe(node_outputs) if include_node_outputs else {}
        safe_final_metrics = make_json_safe(final_metrics)
        safe_anomaly_summary = make_json_safe(anomaly_summary)
        safe_forecasting_summary = make_json_safe(forecasting_summary)
        safe_governance_summary = make_json_safe(governance_summary)
        safe_step_snapshots = make_json_safe(step_snapshots)

        return WorkflowExecutionResult(
            execution_id=execution_id,
            status=overall_status,
            total_duration_ms=total_duration,
            node_results=node_results,
            final_metrics=safe_final_metrics,
            anomaly_summary=safe_anomaly_summary,
            forecasting_summary=safe_forecasting_summary,
            governance_summary=safe_governance_summary,
            logs=logs,
            node_outputs=safe_node_outputs,
            step_snapshots=safe_step_snapshots
        )
