"""
Sub-process ("Add Workflow") support shared by the API, the scheduler and the recipes.

Preparation, done in the API request before a run starts. For every Add Workflow node:
  1. load the selected child workflow (or the pinned version of it) with the async ORM session.
     Every query goes through the tenant guard (infrastructure/database/tenant_guard.py), so a
     workflow of another tenant behaves as if it did not exist;
  2. convert it with db_workflow_to_graph (handles flat and nested saved node configs);
  3. preload the child's datasets and saved credentials, exactly like the parent's;
  4. repeat for sub-processes nested inside the child (cycle- and depth-checked);
  5. attach the prepared graph and version info to the node's in-memory config under the
     reserved "__" keys below. Those keys are runtime-only: _storable_config() in the workflows
     router drops them, so they are never written to the workflow, its last execution or history.

Versions: whenever a workflow with Start and End Anchors is saved with a changed graph, a row is
added to workflow_versions. An Add Workflow node runs "latest" (the workflow as saved now) unless
its workflow_version setting pins a version number.

Contract and parameters, declared on the child's Start Anchor and checked by Add Workflow:
  input_columns: [{"name", "type", "required"}]   type: any|string|integer|number|boolean|datetime
  parameters:    [{"name", "type", "required", "default", "description"}]   type: string|integer|number|boolean
Child steps use a parameter as {{params.<name>}} in any text setting.
"""

import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.logging import logger

SUB_WORKFLOW_DEFINITION_KEY = "__sub_workflow_definition"
SUB_WORKFLOW_ERROR_KEY = "__sub_workflow_error"
SUB_WORKFLOW_VERSION_INFO_KEY = "__sub_workflow_version_info"
MAX_SUB_WORKFLOW_DEPTH = 5

# recipe_id of the Add Workflow node and the aliases the registry maps to it
ADD_WORKFLOW_RECIPE_IDS = {"add_workflow", "sub_workflow", "sub_process", "insert_workflow", "call_sub_workflow"}
START_ANCHOR_IDS = {"workflow_start", "start_anchor", "sub_process_start", "workflow_input"}
END_ANCHOR_IDS = {"workflow_end", "end_anchor", "sub_process_end", "workflow_output"}


def is_runtime_config_key(key: Any) -> bool:
    """Config keys that only exist in memory during a run and must never be saved."""
    return isinstance(key, str) and key.startswith("__")


def is_add_workflow_node(node: Any) -> bool:
    return str(getattr(node, "recipe_id", "") or "").strip().lower() in ADD_WORKFLOW_RECIPE_IDS


def _recipe_of(node: Any) -> str:
    if isinstance(node, dict):
        return str(node.get("recipe_id") or node.get("type") or "").strip().lower()
    return str(getattr(node, "recipe_id", "") or "").strip().lower()


def has_both_anchors(nodes: Any) -> bool:
    """True when a saved node list contains a Start Anchor and an End Anchor."""
    if not isinstance(nodes, list):
        return False
    recipes = {_recipe_of(n) for n in nodes}
    return bool(recipes & START_ANCHOR_IDS) and bool(recipes & END_ANCHOR_IDS)


def definition_hash(nodes: Any, edges: Any, node_configs: Any) -> str:
    payload = json.dumps({"nodes": nodes or [], "edges": edges or [], "node_configs": node_configs or {}},
                         sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ── Versions ───────────────────────────────────────────────────────────────

async def latest_version_number(db: AsyncSession, workflow_id: str) -> Optional[int]:
    from backend.app.workflows.models import WorkflowVersion
    res = await db.execute(
        select(func.max(WorkflowVersion.version_number)).where(WorkflowVersion.workflow_id == str(workflow_id))
    )
    return res.scalar()


async def record_sub_workflow_version(db: AsyncSession, wf: Any) -> Optional[int]:
    """
    After a workflow was saved: updates its is_sub_workflow flag and, when it has both anchors
    and its graph differs from the latest saved version, stores a new version. Commits.
    Returns the latest version number (None when the workflow isn't a sub-process).
    """
    from backend.app.workflows.models import WorkflowVersion

    is_sub = has_both_anchors(wf.nodes)
    wf.is_sub_workflow = is_sub
    latest: Optional[int] = None
    if is_sub:
        digest = definition_hash(wf.nodes, wf.edges, wf.node_configs)
        res = await db.execute(
            select(WorkflowVersion)
            .where(WorkflowVersion.workflow_id == str(wf.id))
            .order_by(WorkflowVersion.version_number.desc())
            .limit(1)
        )
        last = res.scalar_one_or_none()
        latest = last.version_number if last else None
        if last is None or last.definition_hash != digest:
            latest = (latest or 0) + 1
            db.add(WorkflowVersion(
                workflow_id=str(wf.id), version_number=latest, name=wf.name, definition_hash=digest,
                nodes=wf.nodes or [], edges=wf.edges or [], node_configs=wf.node_configs or {},
            ))
    await db.commit()
    return latest


async def list_sub_workflow_versions(db: AsyncSession, wf: Any) -> List[Dict[str, Any]]:
    """Versions of a sub-process, newest first. Creates version 1 for a workflow saved before versions existed."""
    from backend.app.workflows.models import WorkflowVersion

    res = await db.execute(
        select(WorkflowVersion)
        .where(WorkflowVersion.workflow_id == str(wf.id))
        .order_by(WorkflowVersion.version_number.desc())
    )
    rows = list(res.scalars().all())
    if not rows and has_both_anchors(wf.nodes):
        await record_sub_workflow_version(db, wf)
        return await list_sub_workflow_versions(db, wf)
    current = definition_hash(wf.nodes, wf.edges, wf.node_configs)
    return [
        {
            "version": r.version_number,
            "name": r.name,
            "created_at": r.created_at,
            "is_latest": i == 0,
            "matches_current_definition": r.definition_hash == current,
        }
        for i, r in enumerate(rows)
    ]


# ── Loading ────────────────────────────────────────────────────────────────

def parse_version(value: Any) -> Optional[int]:
    """'latest' / empty -> None (latest), otherwise a positive version number. Raises ValueError."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in ("", "latest", "none", "null"):
        return None
    if not text.isdigit() or int(text) < 1:
        raise ValueError(f"Sub-workflow version '{value}' is not valid. Use 'latest' or a version number.")
    return int(text)


async def load_sub_workflow_graph(
    workflow_id: str,
    db: AsyncSession,
    tenant_id: Optional[int],
    call_stack: Sequence[str] = (),
    version: Any = None,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Loads one child workflow, ready to run: (graph, version_info). Raises ValueError with a
    message for the user when it can't be used (not found in this tenant, unknown version,
    calls itself, nested too deep).
    """
    from backend.app.workflows.models import Workflow, WorkflowVersion
    from backend.app.workflows.router import (
        db_workflow_to_graph,
        preload_workflow_credentials,
        preload_workflow_datasets,
    )

    workflow_id = str(workflow_id).strip()
    pinned = parse_version(version)
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

    latest = await latest_version_number(db, workflow_id)
    source: Any = wf
    if pinned is not None:
        v_res = await db.execute(
            select(WorkflowVersion).where(
                WorkflowVersion.workflow_id == workflow_id,
                WorkflowVersion.version_number == pinned,
            )
        )
        source = v_res.scalar_one_or_none()
        if source is None:
            raise ValueError(
                f"Version {pinned} of sub-workflow '{wf.name}' does not exist"
                + (f" (latest is {latest})." if latest else ".")
            )

    graph = db_workflow_to_graph(source)
    graph.id = str(wf.id)
    graph.workflow_id = str(wf.id)
    graph.name = wf.name
    await preload_workflow_datasets(workflow=graph, db=db, tenant_id=tenant_id)
    await preload_workflow_credentials(workflow=graph, db=db, tenant_id=tenant_id)
    await preload_sub_workflows(graph, db, tenant_id, call_stack=list(call_stack) + [workflow_id])
    info = {
        "version": pinned if pinned is not None else "latest",
        "latest_version": latest,
        "update_available": bool(pinned is not None and latest is not None and pinned < latest),
    }
    return graph, info


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
        for key in (SUB_WORKFLOW_DEFINITION_KEY, SUB_WORKFLOW_ERROR_KEY, SUB_WORKFLOW_VERSION_INFO_KEY):
            cfg.pop(key, None)
        child_id = str(cfg.get("workflow_id") or "").strip()
        if child_id:
            try:
                graph, info = await load_sub_workflow_graph(
                    child_id, db, tenant_id, stack, version=cfg.get("workflow_version"))
                cfg[SUB_WORKFLOW_DEFINITION_KEY] = graph
                cfg[SUB_WORKFLOW_VERSION_INFO_KEY] = info
            except ValueError as e:
                cfg[SUB_WORKFLOW_ERROR_KEY] = str(e)
            except Exception as e:
                logger.warning(f"Could not prepare sub-workflow '{child_id}' for node '{node.id}': {e}")
                cfg[SUB_WORKFLOW_ERROR_KEY] = f"Sub-workflow '{child_id}' could not be loaded: {e}"
        node.config = cfg


# ── Input contract ─────────────────────────────────────────────────────────

_TYPE_KINDS = {
    "integer": "iu",
    "number": "iuf",
    "boolean": "b",
    "datetime": "M",
}


def _as_list(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return []
    return [v for v in (value or []) if isinstance(v, dict)] if isinstance(value, list) else []


def check_input_contract(df: Any, columns: Any) -> List[str]:
    """Problems with the parent's table against the Start Anchor's input_columns ([] = fine)."""
    import pandas as pd

    problems: List[str] = []
    for col in _as_list(columns):
        name = str(col.get("name") or "").strip()
        if not name:
            continue
        col_type = str(col.get("type") or "any").strip().lower()
        required = col.get("required", True) not in (False, "false", "False", 0)
        if name not in df.columns:
            if required:
                problems.append(f"missing column '{name}'")
            continue
        if col_type in ("any", "") or len(df) == 0:
            continue
        series = df[name]
        if col_type == "string":
            ok = pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)
        elif col_type in _TYPE_KINDS:
            ok = series.dtype.kind in _TYPE_KINDS[col_type]
        else:
            continue
        if not ok:
            problems.append(f"column '{name}' should be {col_type} but is {series.dtype}")
    return problems


# ── Parameters ─────────────────────────────────────────────────────────────

_PARAM_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PARAM_REF = re.compile(r"\{\{\s*params\.([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


def _coerce(value: Any, ptype: str) -> Any:
    if ptype == "integer":
        if isinstance(value, bool):
            raise ValueError
        if isinstance(value, float) and not value.is_integer():
            raise ValueError
        return int(str(value).strip()) if not isinstance(value, (int, float)) else int(value)
    if ptype == "number":
        if isinstance(value, bool):
            raise ValueError
        return float(value)
    if ptype == "boolean":
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in ("true", "1", "yes", "on"):
            return True
        if text in ("false", "0", "no", "off"):
            return False
        raise ValueError
    return "" if value is None else str(value)


def resolve_parameters(declared: Any, provided: Any) -> Tuple[Dict[str, Any], List[str]]:
    """
    Values for the child's parameters: the caller's values (Add Workflow 'parameters'),
    falling back to each declaration's default, converted to the declared type.
    Returns (values, problems).
    """
    if isinstance(provided, str):
        try:
            provided = json.loads(provided) if provided.strip() else {}
        except Exception:
            return {}, ["'parameters' must be a JSON object of name/value pairs"]
    provided = provided if isinstance(provided, dict) else {}
    values: Dict[str, Any] = {}
    problems: List[str] = []
    names = set()
    for decl in _as_list(declared):
        name = str(decl.get("name") or "").strip()
        if not name:
            continue
        if not _PARAM_NAME.match(name):
            problems.append(f"parameter name '{name}' is not valid (use letters, digits and _)")
            continue
        names.add(name)
        ptype = str(decl.get("type") or "string").strip().lower()
        required = decl.get("required", False) in (True, "true", "True", 1)
        if name in provided and provided[name] not in (None, ""):
            raw = provided[name]
        elif decl.get("default") not in (None, ""):
            raw = decl.get("default")
        elif required:
            problems.append(f"parameter '{name}' is required")
            continue
        else:
            values[name] = None
            continue
        try:
            values[name] = _coerce(raw, ptype)
        except (ValueError, TypeError):
            problems.append(f"parameter '{name}' should be {ptype}, got {raw!r}")
    for extra in sorted(set(provided) - names):
        problems.append(
            f"unknown parameter '{extra}'"
            + (f" (declared: {', '.join(sorted(names))})" if names else " (the sub-workflow declares none)")
        )
    return values, problems


def substitute_params(value: Any, params: Dict[str, Any]) -> Any:
    """Replaces {{params.name}} in text settings. A setting that is only a reference keeps the value's type."""
    if isinstance(value, str):
        whole = _PARAM_REF.fullmatch(value.strip())
        if whole and whole.group(1) in params:
            return params[whole.group(1)]
        def _replace(m: "re.Match") -> str:
            name = m.group(1)
            if name not in params:
                return m.group(0)          # not a declared parameter: leave the text as written
            return "" if params[name] is None else str(params[name])
        return _PARAM_REF.sub(_replace, value)
    if isinstance(value, dict):
        return {k: substitute_params(v, params) for k, v in value.items()}
    if isinstance(value, list):
        return [substitute_params(v, params) for v in value]
    return value
