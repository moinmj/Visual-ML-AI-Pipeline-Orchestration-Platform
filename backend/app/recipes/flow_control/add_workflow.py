"""
Add Workflow Recipe (Ingestion & Orchestration -> Control Flow & Reliability).

Executes another modular workflow (which has both Start and End anchor nodes) as an
encapsulated single step on the visual canvas between Point A and Point B.
Passes upstream Point A DataFrame into the child workflow's Start Anchor,
runs the internal steps, and emits the child End Anchor's dataset to Point B.
"""

from typing import Dict, Any, List, Optional
import time
import json
from datetime import datetime, timezone
import pandas as pd

from backend.app.recipes.base.recipe import BaseRecipe, RecipePort
from backend.app.core.exceptions import ExecutionException, ValidationException
from backend.app.core.logging import logger


class AddWorkflowRecipe(BaseRecipe):
    recipe_id = "add_workflow"
    name = "Add Workflow"
    version = "1.0.0"
    category = "flow_control"
    group = "Ingestion & Orchestration"
    subgroup = "Control Flow & Reliability"
    description = (
        "Integrates and executes another workflow (containing Start and End anchors) "
        "as a single step between upstream Point A and downstream Point B."
    )
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    inputs = [
        RecipePort(
            id="input",
            label="Input Table",
            type="dataframe",
            required=False,
            max_connections=1,
            description="Dataset from upstream Point A to feed into child workflow Start Anchor",
        )
    ]

    outputs = [
        RecipePort(
            id="output",
            label="Output Table",
            type="dataframe",
            description="Transformed dataset emitted by child workflow End Anchor to Point B",
        ),
        RecipePort(
            id="metrics",
            label="Sub-Workflow Telemetry",
            type="metrics",
            description="Duration, node execution counts, and sub-workflow diagnostics",
        ),
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "workflow_id": {
                    "type": "string",
                    "title": "Select Sub-Workflow",
                    "input_type": "select",
                    "endpoint": "/api/v1/workflows?sub_workflow_eligible=true",
                    "description": "Select an eligible workflow (must contain Start & End anchors and have succeeded).",
                },
                "workflow_name": {
                    "type": "string",
                    "title": "Workflow Title / Display Label",
                    "input_type": "text",
                    "description": "Optional human-readable label for the selected sub-workflow.",
                },
                "propagate_errors": {
                    "type": "boolean",
                    "title": "Halt Pipeline on Sub-Workflow Error",
                    "default": True,
                    "input_type": "checkbox",
                    "description": "Raise an immediate diagnostic error if any step inside the sub-workflow fails.",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "title": "Execution Timeout (Seconds)",
                    "default": 300,
                    "minimum": 1,
                    "maximum": 3600,
                    "input_type": "number",
                    "description": "Maximum allowed runtime for the sub-workflow before timeout.",
                },
            },
            "required": ["workflow_id"],
        }

    _sync_engine = None

    @classmethod
    def _get_sync_engine(cls):
        """One shared sync engine for the fallback loader (was: a new engine per call, never disposed)."""
        if cls._sync_engine is None:
            from sqlalchemy import create_engine
            from backend.app.core.config import settings
            sync_url = str(settings.DATABASE_URL).replace("+aiosqlite", "").replace("+asyncpg", "")
            cls._sync_engine = create_engine(sync_url, pool_pre_ping=True)
        return cls._sync_engine

    @classmethod
    def _load_child_workflow_sync(cls, workflow_id: str) -> Optional[Any]:
        """
        Fallback for callers that didn't prepare the child in the API request (normally
        preload_sub_workflows does). Uses an ORM Session, so the tenant guard filters the row
        to the caller's tenant, and portable boolean/NULL checks that work on Postgres and SQLite.
        Saved credentials and datasets of the child are NOT preloaded on this path.
        """
        try:
            from sqlalchemy import select
            from sqlalchemy.orm import Session
            from backend.app.workflows.models import Workflow
            from backend.app.workflows.router import db_workflow_to_graph
            with Session(cls._get_sync_engine()) as session:
                wf = session.execute(
                    select(Workflow).where(
                        Workflow.id == workflow_id,
                        Workflow.is_active.is_(True),
                        Workflow.deleted_at.is_(None),
                    )
                ).scalar_one_or_none()
                if wf is None:
                    return None
                graph = db_workflow_to_graph(wf)
                graph.id = graph.workflow_id = str(wf.id)
                graph.name = wf.name
                logger.warning(
                    f"Sub-workflow '{workflow_id}' was loaded without preparation; "
                    "its saved credentials and datasets were not preloaded."
                )
                return graph
        except Exception as e:
            logger.warning(f"Failed to load child workflow '{workflow_id}' from DB: {e}")
        return None

    @staticmethod
    def _graph_from_dict(child: Dict[str, Any]) -> Any:
        """Builds a runnable graph from a {id, name, nodes, edges, node_configs} dict."""
        from types import SimpleNamespace
        from backend.app.workflows.router import db_workflow_to_graph

        def _parse(value, default):
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except Exception:
                    return default
            return value or default

        graph = db_workflow_to_graph(SimpleNamespace(
            nodes=_parse(child.get("nodes"), []),
            edges=_parse(child.get("edges"), []),
            node_configs=_parse(child.get("node_configs"), {}),
        ))
        graph.id = graph.workflow_id = str(child.get("id") or "")
        graph.name = child.get("name")
        return graph

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        t0 = time.time()
        workflow_id = str(config.get("workflow_id", "")).strip()
        workflow_name = str(config.get("workflow_name", "")).strip() or workflow_id or "Sub-Workflow"
        propagate_errors = bool(config.get("propagate_errors", True))
        timeout_seconds = max(1, min(int(config.get("timeout_seconds", 300)), 3600))

        # 1. Grab Point A input DataFrame
        in_df = None
        if inputs:
            for k in ("dataframe", "input"):
                if isinstance(inputs.get(k), pd.DataFrame):
                    in_df = inputs[k]
                    break
        if in_df is None:
            in_df = pd.DataFrame()

        # If node is unconfigured (no workflow selected yet), pass through with warning
        if not workflow_id:
            return {
                "dataframe": in_df,
                "output_summary": {
                    "title": "Add Workflow (Unconfigured)",
                    "status": "PENDING_CONFIGURATION",
                    "records": len(in_df),
                    "message": "No sub-workflow selected yet. Please configure 'workflow_id' in node settings.",
                },
                "metrics": {
                    "status": "PENDING_CONFIGURATION",
                    "records": len(in_df),
                    "duration_ms": 0.0,
                }
            }

        # 2. Cycle Detection (Prevent recursion A -> B -> A)
        call_stack = []
        if context and isinstance(context, dict):
            call_stack = list(context.get("_sub_workflow_stack", []))
        if workflow_id in call_stack:
            chain_str = " -> ".join(call_stack + [workflow_id])
            err_msg = f"Circular sub-workflow recursion detected: {chain_str}. Halting execution to prevent infinite loop."
            logger.error(err_msg)
            raise ExecutionException(err_msg)

        # 3. Load Child Workflow
        # Normally prepared in the API request by preload_sub_workflows (tenant-checked,
        # credentials and datasets preloaded); see backend/app/workflows/sub_workflows.py.
        from backend.app.workflows.sub_workflows import SUB_WORKFLOW_DEFINITION_KEY, SUB_WORKFLOW_ERROR_KEY
        child_graph = None
        prepare_error = config.get(SUB_WORKFLOW_ERROR_KEY)
        if not prepare_error:
            child_graph = config.get(SUB_WORKFLOW_DEFINITION_KEY)
            if child_graph is None and context and isinstance(context, dict):
                cached = (context.get("_cached_sub_workflows") or {}).get(workflow_id)
                if isinstance(cached, dict):
                    child_graph = self._graph_from_dict(cached)
                elif cached is not None:
                    child_graph = cached
            if child_graph is None:
                child_graph = self._load_child_workflow_sync(workflow_id)

        if child_graph is None:
            err_msg = prepare_error or (
                f"Sub-workflow '{workflow_name}' (ID: {workflow_id}) could not be found. "
                "It may have been deleted, or it belongs to another workspace."
            )
            logger.error(err_msg)
            if propagate_errors:
                raise ExecutionException(err_msg)
            return {
                "dataframe": in_df,
                "output_summary": {
                    "title": f"Sub-Workflow: {workflow_name}",
                    "status": "NOT_FOUND",
                    "error": err_msg,
                },
                "metrics": {"status": "NOT_FOUND", "duration_ms": round((time.time() - t0) * 1000, 2)}
            }

        actual_name = getattr(child_graph, "name", None) or workflow_name
        child_nodes = list(getattr(child_graph, "nodes", None) or [])

        # 4. Verify Start and End Anchors exist
        start_aliases = {"workflow_start", "start_anchor", "sub_process_start", "workflow_input"}
        end_aliases = {"workflow_end", "end_anchor", "sub_process_end", "workflow_output"}
        start_nodes = [n for n in child_nodes if str(n.recipe_id).strip().lower() in start_aliases]
        end_nodes = [n for n in child_nodes if str(n.recipe_id).strip().lower() in end_aliases]

        if not start_nodes or not end_nodes:
            missing_parts = []
            if not start_nodes:
                missing_parts.append("Start Anchor ('workflow_start')")
            if not end_nodes:
                missing_parts.append("End Anchor ('workflow_end')")
            err_msg = (
                f"Sub-Workflow '{actual_name}' (ID: {workflow_id}) is missing required anchors: "
                f"{', '.join(missing_parts)}. Workflows integrated via 'Add Workflow' must have both Start and End anchors."
            )
            logger.error(err_msg)
            if propagate_errors:
                raise ExecutionException(err_msg)
            return {
                "dataframe": in_df,
                "output_summary": {
                    "title": f"Sub-Workflow: {actual_name}",
                    "status": "INVALID_ANCHORS",
                    "error": err_msg,
                },
                "metrics": {"status": "INVALID_ANCHORS", "duration_ms": round((time.time() - t0) * 1000, 2)}
            }

        # 5. Build Child Context
        from backend.app.engine.execution.executor import DAGExecutor

        child_context = dict(context or {})
        child_context["sub_workflow_input"] = in_df.copy()
        child_context["_sub_workflow_stack"] = call_stack + [workflow_id]

        # 6. Execute Sub-Workflow DAG
        sub_exec_id = f"sub_{workflow_id[:8]}_{int(time.time())}"
        child_result = DAGExecutor.execute_workflow(
            execution_id=sub_exec_id,
            workflow=child_graph,
            initial_df=in_df.copy(),
            context=child_context,
            include_node_outputs=True
        )

        # 7. Check for Sub-Workflow Failure & Format Structured Warning
        if child_result.status != "SUCCESS":
            # Identify which node failed
            failed_node_id = "unknown"
            failed_recipe = "unknown"
            failed_reason = "Internal step execution error"

            for nr in (child_result.node_results or []):
                nr_status = getattr(nr, "status", None) if hasattr(nr, "status") else (nr.get("status") if isinstance(nr, dict) else None)
                if nr_status in ("FAILED", "ERROR"):
                    failed_node_id = getattr(nr, "node_id", None) if hasattr(nr, "node_id") else (nr.get("node_id") if isinstance(nr, dict) else "unknown")
                    failed_recipe = getattr(nr, "recipe_id", None) if hasattr(nr, "recipe_id") else (nr.get("recipe_id") if isinstance(nr, dict) else "unknown")
                    failed_reason = getattr(nr, "error_message", None) if hasattr(nr, "error_message") else (nr.get("error_message") if isinstance(nr, dict) else "Unknown step failure")
                    break

            if failed_node_id == "unknown" and child_result.logs:
                import re
                for log_line in child_result.logs:
                    log_str = str(log_line)
                    if any(kw in log_str for kw in ("❌", "DAG Resolution Error", "Configuration Error", "failed")):
                        failed_reason = log_str
                        m_node = re.search(r"['`]([a-zA-Z0-9_\-]+)['`]\s*\[(.*?)\]", log_str)
                        if m_node:
                            failed_node_id = m_node.group(1)
                            failed_recipe = m_node.group(2)
                        elif "node_id':" in log_str:
                            m_id = re.search(r"'node_id':\s*'([^']+)'", log_str)
                            m_rec = re.search(r"'recipe_id':\s*'([^']+)'", log_str)
                            if m_id:
                                failed_node_id = m_id.group(1)
                            if m_rec:
                                failed_recipe = m_rec.group(1)
                        break

            structured_warning = (
                f"❌ Sub-Workflow '{actual_name}' (ID: {workflow_id}) failed at step "
                f"'{failed_node_id}' [Recipe: {failed_recipe}]: {failed_reason}"
            )
            logger.error(structured_warning)

            if propagate_errors:
                raise ExecutionException(structured_warning)

            return {
                "dataframe": in_df,
                "output_summary": {
                    "title": f"Sub-Workflow: {actual_name}",
                    "status": "FAILED",
                    "failed_step": failed_node_id,
                    "failed_recipe": failed_recipe,
                    "error_message": failed_reason,
                    "warning": structured_warning,
                },
                "metrics": {
                    "status": "FAILED",
                    "failed_step": failed_node_id,
                    "sub_workflow_id": workflow_id,
                    "duration_ms": child_result.total_duration_ms,
                }
            }

        # 8. Extract Final DataFrame from End Anchor
        out_df = None
        end_node_id = end_nodes[0].id

        # Priority 1: Check NodeStepExecutor disk cache for uncompressed parquet output
        try:
            from backend.app.engine.execution.step_executor import NodeStepExecutor
            cached_out = NodeStepExecutor.load_node_output(sub_exec_id, end_node_id)
            if cached_out and isinstance(cached_out.get("dataframe"), pd.DataFrame):
                out_df = cached_out["dataframe"]
        except Exception as e:
            logger.debug(f"Could not load cached parquet for sub-workflow end node {end_node_id}: {e}")

        # Priority 2: Extract from child_result.node_outputs (handles serialized JSON DataFrame dict)
        if out_df is None and child_result.node_outputs and end_node_id in child_result.node_outputs:
            end_out = child_result.node_outputs[end_node_id]
            if isinstance(end_out, dict):
                df_val = end_out.get("dataframe")
                if isinstance(df_val, pd.DataFrame):
                    out_df = df_val
                elif isinstance(df_val, dict) and "records" in df_val:
                    out_df = pd.DataFrame(df_val["records"])
            elif isinstance(end_out, pd.DataFrame):
                out_df = end_out

        # Priority 3: Registered in child_context by workflow_end recipe
        if out_df is None and "sub_workflow_final_output" in child_context:
            val = child_context["sub_workflow_final_output"]
            if isinstance(val, pd.DataFrame):
                out_df = val

        # Priority 4: Fallback to empty or input DataFrame if sub-process emitted nothing
        if out_df is None:
            out_df = in_df.copy()

        elapsed_ms = round((time.time() - t0) * 1000, 2)
        total_sub_nodes = len(child_nodes)

        output_summary = {
            "title": f"Sub-Workflow: {actual_name}",
            "status": "SUCCESS",
            "sub_workflow_id": workflow_id,
            "sub_workflow_name": actual_name,
            "input_records_received": len(in_df),
            "output_records_emitted": len(out_df),
            "total_sub_steps_executed": total_sub_nodes,
            "duration_ms": elapsed_ms,
            "message": f"Successfully executed '{actual_name}' ({total_sub_nodes} steps). Handed {len(out_df)} rows to Point B.",
        }

        telemetry = {
            "status": "SUCCESS",
            "sub_workflow_id": workflow_id,
            "sub_workflow_name": actual_name,
            "duration_ms": elapsed_ms,
            "steps_count": total_sub_nodes,
            "input_rows": len(in_df),
            "output_rows": len(out_df),
        }

        return {
            "dataframe": out_df,
            "output_summary": output_summary,
            "metrics": telemetry,
        }

