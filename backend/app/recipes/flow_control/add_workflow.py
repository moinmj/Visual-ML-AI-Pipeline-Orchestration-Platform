"""
Add Workflow Recipe (Ingestion & Orchestration -> Control Flow & Reliability).

Executes another modular workflow (which has both Start and End anchor nodes) as an
encapsulated single step on the visual canvas between Point A and Point B.
Passes upstream Point A DataFrame into the child workflow's Start Anchor,
runs the internal steps, and emits the child End Anchor's dataset to Point B.
"""

from typing import Dict, Any, List, Optional
import concurrent.futures
import contextvars
import json
import re
import threading
import time
import uuid
from collections import deque
import pandas as pd

from backend.app.recipes.base.recipe import BaseRecipe, RecipePort
from backend.app.core.exceptions import ExecutionException, ValidationException
from backend.app.core.logging import logger


class _LinkedCancel:
    """Stop signal for one sub-process run; also counts as set when the caller's own run was stopped,
    so a timeout of an outer sub-process stops the ones nested inside it."""

    def __init__(self, parent: Optional[Any] = None):
        self._own = threading.Event()
        self._parent = parent

    def set(self) -> None:
        self._own.set()

    def is_set(self) -> bool:
        return self._own.is_set() or bool(self._parent is not None and self._parent.is_set())


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
                    "description": (
                        "On: if the sub-workflow fails, the whole pipeline stops with that error. "
                        "Off: this node is marked as a warning and the steps after it are skipped; "
                        "other branches of the pipeline keep running."
                    ),
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

    @staticmethod
    def _soft_failure(status: str, message: str, title: str, in_df: pd.DataFrame,
                      metrics: Optional[Dict[str, Any]] = None,
                      summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Result when the sub-process didn't produce output and 'Halt Pipeline on Sub-Workflow Error'
        is off. The executor shows this node as WARNING and skips every step after it (other
        branches of the parent keep running), so the parent's input is never passed on as if
        the sub-process had worked.
        """
        return {
            "dataframe": in_df,
            "_node_status": "WARNING",
            "_warning": message,
            "output_summary": {"title": title, "status": status, "error": message, **(summary or {})},
            "metrics": {"status": status, **(metrics or {})},
        }

    @staticmethod
    def _anchor_structure_error(child_graph: Any, start_nodes: List[Any], end_nodes: List[Any]) -> Optional[str]:
        """Why the child can't be used as a sub-process, or None when its anchors are valid."""
        if not start_nodes and not end_nodes:
            return "has no Start Anchor and no End Anchor."
        if not start_nodes:
            return "has no Start Anchor."
        if not end_nodes:
            return "has no End Anchor."
        if len(start_nodes) > 1:
            return f"has {len(start_nodes)} Start Anchors ({', '.join(n.id for n in start_nodes)}); keep one."
        if len(end_nodes) > 1:
            return f"has {len(end_nodes)} End Anchors ({', '.join(n.id for n in end_nodes)}); keep one."

        downstream: Dict[str, List[str]] = {}
        for e in getattr(child_graph, "edges", None) or []:
            downstream.setdefault(e.source, []).append(e.target)
        start_id, end_id = start_nodes[0].id, end_nodes[0].id
        seen, queue = {start_id}, deque([start_id])
        while queue:
            for nxt in downstream.get(queue.popleft(), []):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        if end_id not in seen:
            return f"has its End Anchor ('{end_id}') not connected to its Start Anchor ('{start_id}')."
        return None

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

        # No sub-workflow selected yet: warn and skip what follows, never pass the input on as if it ran
        if not workflow_id:
            msg = "No sub-workflow selected yet. Choose one in 'Select Sub-Workflow' in this node's settings."
            return self._soft_failure(
                "PENDING_CONFIGURATION", msg, "Add Workflow (Unconfigured)", in_df,
                metrics={"records": len(in_df), "duration_ms": 0.0},
                summary={"records": len(in_df), "message": msg},
            )

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
            return self._soft_failure(
                "NOT_FOUND", err_msg, f"Sub-Workflow: {workflow_name}", in_df,
                metrics={"duration_ms": round((time.time() - t0) * 1000, 2)},
            )

        actual_name = getattr(child_graph, "name", None) or workflow_name
        child_nodes = list(getattr(child_graph, "nodes", None) or [])

        # 4. Verify Start and End Anchors exist
        start_aliases = {"workflow_start", "start_anchor", "sub_process_start", "workflow_input"}
        end_aliases = {"workflow_end", "end_anchor", "sub_process_end", "workflow_output"}
        start_nodes = [n for n in child_nodes if str(n.recipe_id).strip().lower() in start_aliases]
        end_nodes = [n for n in child_nodes if str(n.recipe_id).strip().lower() in end_aliases]

        anchor_error = self._anchor_structure_error(child_graph, start_nodes, end_nodes)
        if anchor_error:
            err_msg = (
                f"Sub-Workflow '{actual_name}' (ID: {workflow_id}) {anchor_error} "
                "Workflows used through 'Add Workflow' need exactly one Start Anchor and one End Anchor, "
                "with the End Anchor connected (directly or through other steps) to the Start Anchor."
            )
            logger.error(err_msg)
            if propagate_errors:
                raise ExecutionException(err_msg)
            return self._soft_failure(
                "INVALID_ANCHORS", err_msg, f"Sub-Workflow: {actual_name}", in_df,
                metrics={"duration_ms": round((time.time() - t0) * 1000, 2)},
            )

        # 5. Build Child Context
        parent_exec_id = re.sub(r"[^A-Za-z0-9_-]", "", str((context or {}).get("execution_id") or "run"))[:40] or "run"
        sub_exec_id = f"{parent_exec_id}_sub_{uuid.uuid4().hex[:12]}"   # unique per call, even within one second
        result_holder: Dict[str, Any] = {}                              # filled by this run's End Anchor only
        cancel = _LinkedCancel((context or {}).get("_cancel_event"))

        child_context = dict(context or {})
        child_context["sub_workflow_input"] = in_df.copy()             # only the Start Anchor reads this
        child_context["_sub_workflow_stack"] = call_stack + [workflow_id]
        child_context["_sub_workflow_result"] = result_holder
        child_context["_cancel_event"] = cancel
        child_context["_cancel_reason"] = f"sub-process '{actual_name}' timed out after {timeout_seconds}s"
        child_context["execution_id"] = sub_exec_id
        child_context["workflow_id"] = workflow_id

        # 6. Execute Sub-Workflow DAG in a worker thread so timeout_seconds can be enforced.
        # The tenant/user context variables are copied into the thread.
        from backend.app.engine.execution.executor import DAGExecutor
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="sub-workflow")
        try:
            future = pool.submit(
                contextvars.copy_context().run,
                DAGExecutor.execute_workflow,
                execution_id=sub_exec_id,
                workflow=child_graph,
                initial_df=None,            # the parent's table goes to the Start Anchor only
                context=child_context,
                include_node_outputs=False,
            )
            try:
                child_result = future.result(timeout=timeout_seconds)
            except concurrent.futures.TimeoutError:
                cancel.set()   # the child stops before its next step
                err_msg = (
                    f"Sub-Workflow '{actual_name}' (ID: {workflow_id}) timed out after {timeout_seconds} seconds "
                    "and was stopped. Increase 'Execution Timeout' or make the sub-process faster."
                )
                logger.error(err_msg)
                if propagate_errors:
                    raise ExecutionException(err_msg)
                return self._soft_failure(
                    "TIMEOUT", err_msg, f"Sub-Workflow: {actual_name}", in_df,
                    metrics={"sub_workflow_id": workflow_id, "duration_ms": round((time.time() - t0) * 1000, 2)},
                )
        finally:
            pool.shutdown(wait=False)

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

            return self._soft_failure(
                "FAILED", structured_warning, f"Sub-Workflow: {actual_name}", in_df,
                metrics={"failed_step": failed_node_id, "sub_workflow_id": workflow_id,
                         "duration_ms": child_result.total_duration_ms},
                summary={"failed_step": failed_node_id, "failed_recipe": failed_recipe,
                         "error_message": failed_reason, "warning": structured_warning},
            )

        # 8. Final table, handed over by this run's End Anchor (same DataFrame, dtypes intact)
        out_df = result_holder.get("dataframe")
        if not isinstance(out_df, pd.DataFrame):
            err_msg = (
                f"Sub-Workflow '{actual_name}' (ID: {workflow_id}) finished but its End Anchor produced no output. "
                "Check that the End Anchor is connected to the steps before it."
            )
            logger.error(err_msg)
            if propagate_errors:
                raise ExecutionException(err_msg)
            return self._soft_failure(
                "NO_OUTPUT", err_msg, f"Sub-Workflow: {actual_name}", in_df,
                metrics={"sub_workflow_id": workflow_id, "duration_ms": round((time.time() - t0) * 1000, 2)},
            )

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

