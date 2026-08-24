import time
import traceback
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
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
    output_summary: Dict[str, Any] = Field(default_factory=dict)


class WorkflowExecutionResult(BaseModel):
    execution_id: str
    status: str  # "SUCCESS", "FAILED"
    total_duration_ms: float
    node_results: List[NodeExecutionResult]
    final_metrics: Optional[Dict[str, Any]] = None
    logs: List[str] = Field(default_factory=list)


class DAGExecutor:
    """
    In-memory DAG execution engine. Passes artifacts between nodes,
    handles errors, and tracks node-level states.
    """

    @classmethod
    def execute_workflow(
        cls,
        execution_id: str,
        workflow: WorkflowGraph,
        context: Optional[Any] = None
    ) -> WorkflowExecutionResult:
        start_time = time.time()
        logs: List[str] = []
        node_results: List[NodeExecutionResult] = []
        node_outputs: Dict[str, Dict[str, Any]] = {}
        final_metrics = None

        logs.append(f"[{datetime.now(timezone.utc).isoformat()}] Starting execution '{execution_id}'")

        # 1. Topological Order
        try:
            ordered_nodes = workflow.get_topological_order()
        except Exception as e:
            logs.append(f"Validation Error: {str(e)}")
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

        for node in ordered_nodes:
            node_start = time.time()
            logs.append(f"Executing node '{node.id}' [Recipe: {node.recipe_id}]")

            # Collect inputs from all parent nodes
            node_inputs: Dict[str, Any] = {}
            for parent_id in parent_map[node.id]:
                parent_out = node_outputs.get(parent_id, {})
                node_inputs.update(parent_out)

            try:
                recipe = recipe_registry.get(node.recipe_id)

                # Validate recipe config
                config_errors = recipe.validate_config(node.config)
                if config_errors:
                    raise ValueError(f"Configuration errors: {', '.join(config_errors)}")

                # Execute recipe
                outputs = recipe.execute(inputs=node_inputs, config=node.config, context=context)
                node_outputs[node.id] = outputs

                duration_ms = round((time.time() - node_start) * 1000.0, 2)
                logs.append(f"Node '{node.id}' finished in {duration_ms}ms (SUCCESS)")

                # Create output summary (serializable)
                summary: Dict[str, Any] = {}
                for k, v in outputs.items():
                    if hasattr(v, "shape"):
                        summary[k] = {"shape": list(v.shape), "type": "DataFrame"}
                    elif k == "metrics":
                        summary[k] = v
                        final_metrics = v
                    elif k == "feature_importances":
                        summary[k] = v
                    else:
                        summary[k] = {"type": type(v).__name__}

                node_results.append(NodeExecutionResult(
                    node_id=node.id,
                    recipe_id=node.recipe_id,
                    status="SUCCESS",
                    duration_ms=duration_ms,
                    output_summary=summary
                ))

            except Exception as e:
                duration_ms = round((time.time() - node_start) * 1000.0, 2)
                err_msg = f"{str(e)}"
                logs.append(f"Node '{node.id}' failed in {duration_ms}ms: {err_msg}")
                logger.error(f"Execution failed on node {node.id}: {traceback.format_exc()}")

                node_results.append(NodeExecutionResult(
                    node_id=node.id,
                    recipe_id=node.recipe_id,
                    status="FAILED",
                    duration_ms=duration_ms,
                    error_message=err_msg
                ))
                overall_status = "FAILED"
                break

        total_duration = round((time.time() - start_time) * 1000.0, 2)
        logs.append(f"[{datetime.now(timezone.utc).isoformat()}] Execution finished with status '{overall_status}' in {total_duration}ms")

        return WorkflowExecutionResult(
            execution_id=execution_id,
            status=overall_status,
            total_duration_ms=total_duration,
            node_results=node_results,
            final_metrics=final_metrics,
            logs=logs
        )
