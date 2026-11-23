"""
Sub-process ("Add Workflow") preparation, done in the API request before a run starts.

For every Add Workflow node in a graph this:
  1. loads the selected child workflow with the async ORM session. Every query goes through
     the tenant guard (infrastructure/database/tenant_guard.py), so a workflow of another
     tenant behaves as if it did not exist;
  2. converts it with db_workflow_to_graph (handles flat and nested saved node configs);
  3. preloads the child's datasets and saved credentials, exactly like the parent's;
  4. repeats this for sub-processes nested inside the child (cycle- and depth-checked);
  5. attaches the prepared child graph to the node's in-memory config under
     SUB_WORKFLOW_DEFINITION_KEY.

The Add Workflow recipe then runs that prepared graph and never opens its own database
connection. The key is runtime-only: _storable_config() in the workflows router drops it, so
it is never written to the workflow, its last execution or its history.
"""

from typing import Any, Dict, Optional, Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.logging import logger

SUB_WORKFLOW_DEFINITION_KEY = "__sub_workflow_definition"
SUB_WORKFLOW_ERROR_KEY = "__sub_workflow_error"
MAX_SUB_WORKFLOW_DEPTH = 5

# recipe_id of the Add Workflow node and the aliases the registry maps to it
ADD_WORKFLOW_RECIPE_IDS = {"add_workflow", "sub_workflow", "sub_process", "insert_workflow", "call_sub_workflow"}


def is_runtime_config_key(key: Any) -> bool:
    """Config keys that only exist in memory during a run and must never be saved."""
    return isinstance(key, str) and key.startswith("__")


def is_add_workflow_node(node: Any) -> bool:
    return str(getattr(node, "recipe_id", "") or "").strip().lower() in ADD_WORKFLOW_RECIPE_IDS


async def load_sub_workflow_graph(
    workflow_id: str,
    db: AsyncSession,
    tenant_id: Optional[int],
    call_stack: Sequence[str] = (),
) -> Any:
    """
    Loads one child workflow, ready to run. Raises ValueError with a message for the user
    when it can't be used (not found in this tenant, calls itself, nested too deep).
    """
    from backend.app.workflows.models import Workflow
    from backend.app.workflows.router import (
        db_workflow_to_graph,
        preload_workflow_credentials,
        preload_workflow_datasets,
    )

    workflow_id = str(workflow_id).strip()
    if workflow_id in call_stack:
        chain = " -> ".join(list(call_stack) + [workflow_id])
        raise ValueError(f"Circular sub-workflow recursion detected: {chain}.")
    if len(call_stack) >= MAX_SUB_WORKFLOW_DEPTH:
        raise ValueError(
            f"Sub-workflows are nested more than {MAX_SUB_WORKFLOW_DEPTH} levels deep "
            f"({' -> '.join(list(call_stack) + [workflow_id])})."
        )

    # Tenant filter is added by the tenant guard; no skip_tenant_filter here on purpose.
    res = await db.execute(
        select(Workflow).where(
            Workflow.id == workflow_id,
            Workflow.is_active.is_(True),
            Workflow.deleted_at.is_(None),
        )
    )
    wf = res.scalar_one_or_none()
    if wf is None:
        raise ValueError(
            f"Sub-workflow '{workflow_id}' could not be found. It may have been deleted, "
            "or it belongs to another workspace."
        )

    graph = db_workflow_to_graph(wf)
    graph.id = str(wf.id)
    graph.workflow_id = str(wf.id)
    graph.name = wf.name
    await preload_workflow_datasets(workflow=graph, db=db, tenant_id=tenant_id)
    await preload_workflow_credentials(workflow=graph, db=db, tenant_id=tenant_id)
    await preload_sub_workflows(graph, db, tenant_id, call_stack=list(call_stack) + [workflow_id])
    return graph


async def preload_sub_workflows(
    workflow: Any,
    db: AsyncSession,
    tenant_id: Optional[int],
    call_stack: Sequence[str] = (),
) -> None:
    """
    Prepares every Add Workflow node in `workflow` (see module docstring). A child that
    can't be used doesn't stop the request: the reason is stored on the node and the
    Add Workflow recipe reports it when the node runs, so the parent's other nodes and
    the propagate_errors setting behave as usual.
    """
    own_id = str(getattr(workflow, "id", None) or getattr(workflow, "workflow_id", None) or "").strip()
    stack = list(call_stack) or ([own_id] if own_id else [])

    for node in getattr(workflow, "nodes", None) or []:
        if not is_add_workflow_node(node):
            continue
        cfg: Dict[str, Any] = dict(node.config or {})
        cfg.pop(SUB_WORKFLOW_DEFINITION_KEY, None)
        cfg.pop(SUB_WORKFLOW_ERROR_KEY, None)
        child_id = str(cfg.get("workflow_id") or "").strip()
        if child_id:
            try:
                cfg[SUB_WORKFLOW_DEFINITION_KEY] = await load_sub_workflow_graph(child_id, db, tenant_id, stack)
            except ValueError as e:
                cfg[SUB_WORKFLOW_ERROR_KEY] = str(e)
            except Exception as e:
                logger.warning(f"Could not prepare sub-workflow '{child_id}' for node '{node.id}': {e}")
                cfg[SUB_WORKFLOW_ERROR_KEY] = f"Sub-workflow '{child_id}' could not be loaded: {e}"
        node.config = cfg
