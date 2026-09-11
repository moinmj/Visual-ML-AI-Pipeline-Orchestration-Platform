from fastapi import APIRouter, Depends, HTTPException, status, Body, Query
from fastapi.encoders import jsonable_encoder
from typing import Dict, Any, List, Optional, Union
import uuid
import pandas as pd
from datetime import datetime, timezone, timedelta
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from backend.app.infrastructure.database.session import get_db
from backend.app.engine.dag.graph import WorkflowGraph
from backend.app.engine.execution.executor import DAGExecutor, WorkflowExecutionResult
from backend.app.engine.execution.job_manager import job_manager
from backend.app.workflows.models import Workflow, WorkflowExecution
from backend.app.workflows.schemas import (
    WorkflowCreate,
    WorkflowUpdate,
    WorkflowResponse,
    WorkflowExecutionSummaryResponse,
    WorkflowExecutionDetailResponse,
    WorkflowCompareResponse
)
from backend.app.engine.inference import (
    PipelineInferencer,
    PredictionRequest,
    PredictionResponse,
    InferenceSchemaResponse,
    FeatureSchemaItem
)
from backend.app.core.security import get_current_user, require_role, require_permission

from backend.app.datasets.models import Dataset

router = APIRouter(prefix="/workflows", tags=["Workflows & DAG Execution"])


async def resolve_workflow_dataset(
    db: AsyncSession,
    dataset_id: Optional[str] = None,
    dataset_name: Optional[str] = None,
    nodes: Optional[List[Dict[str, Any]]] = None,
    node_configs: Optional[Dict[str, Any]] = None
) -> tuple[Optional[str], Optional[str]]:
    """
    Resolves dataset_id and dataset_name from payload or by inspecting
    the workflow's ingestion nodes (e.g. csv_loader).
    """
    resolved_id = dataset_id
    resolved_name = dataset_name

    # If dataset_id was not explicitly passed, inspect node configs for csv_loader
    if not resolved_id:
        if node_configs and isinstance(node_configs, dict):
            for n_id, n_data in node_configs.items():
                cfg = n_data.get("config", {}) if isinstance(n_data, dict) else {}
                if "dataset_id" in cfg and cfg["dataset_id"]:
                    resolved_id = str(cfg["dataset_id"])
                    break
        if not resolved_id and nodes and isinstance(nodes, list):
            for n in nodes:
                cfg = n.get("config", {}) if isinstance(n, dict) else {}
                if "dataset_id" in cfg and cfg["dataset_id"]:
                    resolved_id = str(cfg["dataset_id"])
                    break

    # Look up human-readable dataset_name if we have an ID
    if resolved_id and not resolved_name:
        try:
            ds_res = await db.execute(select(Dataset).where(Dataset.id == resolved_id))
            ds = ds_res.scalar_one_or_none()
            if ds:
                resolved_name = ds.name
        except Exception:
            pass

    return resolved_id, resolved_name


async def resolve_or_normalize_last_execution(
    db: AsyncSession,
    last_execution: Optional[Dict[str, Any]] = None,
    execution_id: Optional[str] = None,
    existing_wf: Optional[Workflow] = None,
    dataset_id: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Resolves, normalizes, and protects last_execution diagnostics and reports:
    1. If a valid, non-empty last_execution dictionary is supplied, normalizes logs and returns it JSON-safe.
    2. If execution_id is provided, searches DB workflows and job_manager for the matching execution report.
    3. If updating an existing workflow and no new execution is provided, preserves the existing last_execution.
    4. Auto-adoption fallback: If user executed an unsaved canvas workflow and then clicked Save Workflow,
       locates the recent auto-saved execution run and adopts its reports.
    """
    # 1. Normalize provided dictionary if it contains execution content
    if last_execution and isinstance(last_execution, dict):
        has_content = any(k in last_execution for k in [
            "execution_id", "status", "final_metrics", "node_results", "reports", "metrics", "logs", "execution_logs"
        ])
        if has_content:
            norm_exec = dict(last_execution)
            if "logs" in norm_exec and "execution_logs" not in norm_exec:
                norm_exec["execution_logs"] = norm_exec["logs"]
            if "execution_logs" in norm_exec and "logs" not in norm_exec:
                norm_exec["logs"] = norm_exec["execution_logs"]
            return jsonable_encoder(norm_exec)

    target_exec_id = execution_id
    if not target_exec_id and isinstance(last_execution, dict):
        target_exec_id = last_execution.get("execution_id")

    # 2. Look up by execution_id if provided
    if target_exec_id:
        # Check running/recent jobs
        job = job_manager.get_job(target_exec_id)
        if job and job.get("results"):
            res = job["results"]
            norm_job = {
                "execution_id": target_exec_id,
                "status": job.get("status", "SUCCESS"),
                "total_duration_ms": job.get("duration_ms", 0.0),
                "final_metrics": res.get("final_metrics") if isinstance(res, dict) else getattr(res, "final_metrics", None),
                "node_results": res.get("node_results") if isinstance(res, dict) else getattr(res, "node_results", []),
                "execution_logs": job.get("logs", []),
                "logs": job.get("logs", []),
                "step_snapshots": res.get("step_snapshots", {}) if isinstance(res, dict) else getattr(res, "step_snapshots", {}),
            }
            return jsonable_encoder(norm_job)

        # Check existing workflows in DB
        wf_q = select(Workflow).where(Workflow.last_execution.is_not(None)).order_by(Workflow.updated_at.desc()).limit(20)
        wf_res = await db.execute(wf_q)
        for cand in wf_res.scalars().all():
            cand_exec = cand.last_execution
            if isinstance(cand_exec, dict) and cand_exec.get("execution_id") == target_exec_id:
                return jsonable_encoder(cand_exec)

    # 3. If updating an existing workflow, preserve existing last_execution
    if existing_wf and existing_wf.last_execution:
        return existing_wf.last_execution

    # 4. Auto-adoption: Find recent auto-saved execution run (within last 30 minutes)
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=30)
        auto_q = select(Workflow).where(
            Workflow.last_execution.is_not(None),
            Workflow.name.ilike("Auto-Saved Pipeline%"),
            Workflow.updated_at >= cutoff
        ).order_by(Workflow.updated_at.desc()).limit(5)
        auto_res = await db.execute(auto_q)
        candidates = auto_res.scalars().all()
        for cand in candidates:
            # Match by dataset_id if present, or take the latest auto-saved run
            if dataset_id and cand.dataset_id and cand.dataset_id == dataset_id:
                adopted = cand.last_execution
                cand.is_active = False
                cand.deleted_at = datetime.now(timezone.utc)
                return jsonable_encoder(adopted)
            elif not dataset_id:
                adopted = cand.last_execution
                cand.is_active = False
                cand.deleted_at = datetime.now(timezone.utc)
                return jsonable_encoder(adopted)
    except Exception:
        pass

    return None


async def record_workflow_execution_history(
    db: AsyncSession,
    workflow_id: str,
    execution_id: str,
    status_str: str,
    total_duration_ms: float,
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    node_configs: Dict[str, Any],
    metrics: Optional[Dict[str, Any]] = None,
    reports: Optional[Dict[str, Any]] = None,
    step_snapshots: Optional[Dict[str, Any]] = None,
    logs: Optional[Any] = None,
    run_label: Optional[str] = None
) -> WorkflowExecution:
    """
    Persists an immutable historical execution snapshot linking the exact graph configuration
    (nodes, edges, node_configs) with its execution diagnostics (metrics, reports, step snapshots, logs).
    """
    # Check if this execution_id already exists to prevent duplicate entries
    existing = await db.execute(select(WorkflowExecution).where(WorkflowExecution.id == execution_id))
    ex_row = existing.scalar_one_or_none()
    if ex_row:
        return ex_row

    # Determine next auto-incrementing version number for this workflow
    max_ver_stmt = select(func.max(WorkflowExecution.version_number)).where(WorkflowExecution.workflow_id == workflow_id)
    max_ver_res = await db.execute(max_ver_stmt)
    current_max = max_ver_res.scalar() or 0
    next_ver = current_max + 1

    exec_record = WorkflowExecution(
        id=execution_id,
        workflow_id=workflow_id,
        version_number=next_ver,
        run_label=run_label or f"Run #{next_ver}",
        status=status_str or "SUCCESS",
        total_duration_ms=total_duration_ms or 0.0,
        snapshot_nodes=jsonable_encoder(nodes or []),
        snapshot_edges=jsonable_encoder(edges or []),
        snapshot_node_configs=jsonable_encoder(node_configs or {}),
        metrics=jsonable_encoder(metrics or {}),
        reports=jsonable_encoder(reports or {}),
        step_snapshots=jsonable_encoder(step_snapshots or {}),
        logs=jsonable_encoder(logs or [])
    )
    db.add(exec_record)
    await db.commit()
    await db.refresh(exec_record)
    return exec_record


# -------------------------------------------------------------
# WORKFLOW PERSISTENCE & WORKBOOK RETRIEVAL ENDPOINTS
# -------------------------------------------------------------

@router.post(
    "/",
    response_model=WorkflowResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_role("Tenant Admin", "Data Scientist", "ML Engineer"))]
)
async def save_workflow(
    payload: WorkflowCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Save / create / upsert a pipeline workbook with exact node configs, parameters, layout, and edges.
    Seamlessly captures and persists execution reports (last_execution) across new and updated workbooks.
    """
    target_id = payload.id or str(uuid.uuid4())
    result = await db.execute(select(Workflow).where(Workflow.id == target_id))
    wf = result.scalar_one_or_none()

    ds_id, ds_name = await resolve_workflow_dataset(
        db=db,
        dataset_id=payload.dataset_id,
        dataset_name=payload.dataset_name,
        nodes=payload.nodes,
        node_configs=payload.node_configs
    )

    resolved_last_exec = await resolve_or_normalize_last_execution(
        db=db,
        last_execution=payload.last_execution,
        execution_id=payload.execution_id,
        existing_wf=wf,
        dataset_id=ds_id
    )

    if wf:
        # Update existing
        if payload.name:
            wf.name = payload.name
        if payload.description is not None:
            wf.description = payload.description
        if ds_id is not None:
            wf.dataset_id = ds_id
        if ds_name is not None:
            wf.dataset_name = ds_name
        if payload.nodes is not None:
            wf.nodes = payload.nodes
        if payload.edges is not None:
            wf.edges = payload.edges
        if payload.node_configs is not None:
            wf.node_configs = payload.node_configs
        if resolved_last_exec is not None:
            wf.last_execution = resolved_last_exec
        wf.is_active = True
        wf.deleted_at = None
        wf.updated_at = datetime.now(timezone.utc)
    else:
        # Create new
        wf = Workflow(
            id=target_id,
            name=payload.name,
            description=payload.description,
            dataset_id=ds_id,
            dataset_name=ds_name,
            nodes=payload.nodes,
            edges=payload.edges,
            node_configs=payload.node_configs,
            last_execution=resolved_last_exec,
            is_active=True
        )
        db.add(wf)

    await db.commit()
    await db.refresh(wf)
    return wf


@router.get("/", response_model=List[WorkflowResponse])
async def list_workflows(
    include_deleted: bool = False,
    db: AsyncSession = Depends(get_db)
):
    """
    List all saved pipeline workbooks. By default filters out soft-deleted pipelines.
    Pass include_deleted=True to retrieve archived/trash items.
    """
    query = select(Workflow)
    if not include_deleted:
        query = query.where(Workflow.is_active == True)
    
    query = query.order_by(Workflow.updated_at.desc())
    result = await db.execute(query)
    workflows = result.scalars().all()
    return workflows


@router.get("/jobs")
async def list_jobs(limit: int = 50):
    """
    List recent workflow execution jobs and their runtime statuses.
    """
    return job_manager.list_jobs(limit=limit)


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """
    Poll the status, logs, step snapshots, and results for an async workflow job.
    """
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job


@router.get("/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow(
    workflow_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Retrieve a specific saved pipeline workbook by ID with full exact configuration, parameters, and saved execution report.
    """
    result = await db.execute(select(Workflow).where(Workflow.id == workflow_id))
    wf = result.scalar_one_or_none()
    if not wf:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow workbook '{workflow_id}' not found."
        )

    # Auto-resolve dataset_id if not explicitly set on older records
    if not wf.dataset_id:
        ds_id, ds_name = await resolve_workflow_dataset(
            db=db,
            dataset_id=None,
            dataset_name=None,
            nodes=wf.nodes or [],
            node_configs=wf.node_configs or {}
        )
        if ds_id:
            wf.dataset_id = ds_id
            wf.dataset_name = ds_name
            await db.commit()
            await db.refresh(wf)

    return wf


@router.put(
    "/{workflow_id}",
    response_model=WorkflowResponse,
    dependencies=[Depends(require_role("Tenant Admin", "Data Scientist", "ML Engineer"))]
)
async def upsert_workflow(
    workflow_id: str,
    payload: WorkflowCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Upsert Endpoint (Create or Update).
    If workflow_id exists in DB -> Updates existing pipeline.
    If workflow_id does NOT exist (or is 'new'/'create') -> Creates new pipeline record.
    """
    target_id = workflow_id
    if workflow_id.lower() in ["new", "create", "0", "undefined", "null"]:
        target_id = payload.id or str(uuid.uuid4())

    result = await db.execute(select(Workflow).where(Workflow.id == target_id))
    wf = result.scalar_one_or_none()

    ds_id, ds_name = await resolve_workflow_dataset(
        db=db,
        dataset_id=payload.dataset_id,
        dataset_name=payload.dataset_name,
        nodes=payload.nodes,
        node_configs=payload.node_configs
    )

    resolved_last_exec = await resolve_or_normalize_last_execution(
        db=db,
        last_execution=payload.last_execution,
        execution_id=payload.execution_id,
        existing_wf=wf,
        dataset_id=ds_id
    )

    if wf:
        # Update existing
        if payload.name:
            wf.name = payload.name
        if payload.description is not None:
            wf.description = payload.description
        if ds_id is not None:
            wf.dataset_id = ds_id
        if ds_name is not None:
            wf.dataset_name = ds_name
        if payload.nodes is not None:
            wf.nodes = payload.nodes
        if payload.edges is not None:
            wf.edges = payload.edges
        if payload.node_configs is not None:
            wf.node_configs = payload.node_configs
        if resolved_last_exec is not None:
            wf.last_execution = resolved_last_exec
        wf.is_active = True
        wf.deleted_at = None
        wf.updated_at = datetime.now(timezone.utc)
    else:
        # Create new
        wf = Workflow(
            id=target_id,
            name=payload.name or "Untitled Pipeline",
            description=payload.description,
            dataset_id=ds_id,
            dataset_name=ds_name,
            nodes=payload.nodes or [],
            edges=payload.edges or [],
            node_configs=payload.node_configs or {},
            last_execution=resolved_last_exec,
            is_active=True
        )
        db.add(wf)

    await db.commit()
    await db.refresh(wf)
    return wf


@router.post(
    "/{workflow_id}/save-execution",
    response_model=WorkflowResponse,
    dependencies=[Depends(require_role("Tenant Admin", "Data Scientist", "ML Engineer"))]
)
async def save_workflow_execution_report(
    workflow_id: str,
    execution_payload: Dict[str, Any] = Body(..., description="Execution report, metrics, and diagnostics to attach to the workflow"),
    db: AsyncSession = Depends(get_db)
):
    """
    Directly attach, update, or save execution reports (final_metrics, node_results, confusion matrix, logs)
    to a specific workflow workbook in the database.
    """
    result = await db.execute(select(Workflow).where(Workflow.id == workflow_id))
    wf = result.scalar_one_or_none()
    if not wf:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow workbook '{workflow_id}' not found."
        )

    resolved_exec = await resolve_or_normalize_last_execution(
        db=db,
        last_execution=execution_payload,
        execution_id=execution_payload.get("execution_id"),
        existing_wf=wf,
        dataset_id=wf.dataset_id
    )
    exec_dict = resolved_exec or dict(execution_payload)
    wf.last_execution = jsonable_encoder(exec_dict)
    wf.updated_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(wf)

    # Record history snapshot
    try:
        e_id = exec_dict.get("execution_id") or str(uuid.uuid4())
        await record_workflow_execution_history(
            db=db,
            workflow_id=workflow_id,
            execution_id=e_id,
            status_str=exec_dict.get("status", "SUCCESS"),
            total_duration_ms=exec_dict.get("total_duration_ms", 0.0),
            nodes=wf.nodes or [],
            edges=wf.edges or [],
            node_configs=wf.node_configs or {},
            metrics=exec_dict.get("final_metrics") or exec_dict.get("metrics"),
            reports={
                "anomaly_summary": exec_dict.get("anomaly_summary"),
                "forecasting_summary": exec_dict.get("forecasting_summary"),
                "governance_summary": exec_dict.get("governance_summary"),
                "node_results": exec_dict.get("node_results"),
                "inference_schema": exec_dict.get("inference_schema"),
            },
            step_snapshots=exec_dict.get("step_snapshots"),
            logs=exec_dict.get("execution_logs") or exec_dict.get("logs"),
            run_label=wf.name
        )
    except Exception:
        pass

    return wf


@router.delete(
    "/{workflow_id}",
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(require_role("Tenant Admin", "Data Scientist"))]
)
async def delete_workflow(
    workflow_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Soft-delete a saved pipeline workbook by ID (sets is_active=False, deleted_at=now).
    """
    result = await db.execute(select(Workflow).where(Workflow.id == workflow_id))
    wf = result.scalar_one_or_none()
    if not wf:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow workbook '{workflow_id}' not found."
        )

    wf.is_active = False
    wf.deleted_at = datetime.now(timezone.utc)
    await db.commit()
    return {
        "status": "SOFT_DELETED",
        "workflow_id": workflow_id,
        "message": f"Workflow workbook '{workflow_id}' archived/soft-deleted successfully."
    }


@router.post(
    "/{workflow_id}/restore",
    response_model=WorkflowResponse,
    dependencies=[Depends(require_role("Tenant Admin", "Data Scientist"))]
)
async def restore_workflow(
    workflow_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Restore a soft-deleted pipeline workbook back to active state.
    """
    result = await db.execute(select(Workflow).where(Workflow.id == workflow_id))
    wf = result.scalar_one_or_none()
    if not wf:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow workbook '{workflow_id}' not found."
        )

    wf.is_active = True
    wf.deleted_at = None
    wf.updated_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(wf)
    return wf


# -------------------------------------------------------------
# DAG VALIDATION & EXECUTION ENDPOINTS
# -------------------------------------------------------------

def workflow_graph_to_db_payload(workflow: WorkflowGraph) -> tuple:
    """Helper to convert WorkflowGraph instance to DB nodes, edges, and node_configs."""
    nodes_payload = []
    node_configs = {}
    for n in workflow.nodes:
        nodes_payload.append({
            "id": n.id,
            "recipe_id": n.recipe_id,
            "label": n.label or n.id,
            "content": n.label or n.id,
            "config": n.config
        })
        node_configs[n.id] = {
            "recipe_id": n.recipe_id,
            "label": n.label or n.id,
            "config": n.config
        }
    edges_payload = [
        {"id": f"e_{e.source}_{e.target}", "source": e.source, "target": e.target}
        for e in workflow.edges
    ]
    return nodes_payload, edges_payload, node_configs


def db_workflow_to_graph(wf: Workflow) -> WorkflowGraph:
    """Helper to convert a stored DB Workflow model into an executable WorkflowGraph."""
    from backend.app.engine.dag.graph import WorkflowNode, WorkflowEdge
    nodes = []
    saved_configs = wf.node_configs or {}
    for nd in wf.nodes or []:
        nid = nd["id"]
        n_cfg = saved_configs.get(nid, {})
        recipe_id = nd.get("recipe_id") or n_cfg.get("recipe_id", "csv_loader")
        config = nd.get("config") or n_cfg.get("config", {})
        label = nd.get("label") or nd.get("content") or n_cfg.get("label", nid)
        nodes.append(WorkflowNode(id=nid, recipe_id=recipe_id, config=config, label=label))
    edges = [
        WorkflowEdge(source=ed["source"], target=ed["target"])
        for ed in wf.edges or []
    ]
    return WorkflowGraph(nodes=nodes, edges=edges)


@router.post(
    "/validate",
    response_model=Dict[str, Any],
    dependencies=[Depends(get_current_user)]
)
async def validate_workflow(workflow: WorkflowGraph):
    """
    Validate a workflow graph for cycle detection, valid node connectivity, schema requirements, and best-practice recommendations.
    """
    diag = workflow.get_diagnostics()
    return {
        "valid": diag["is_valid"],
        "errors": diag["errors"],
        "warnings": diag["warnings"],
        "recommendations": diag["recommendations"]
    }


@router.post(
    "/{workflow_id}/validate",
    response_model=Dict[str, Any],
    dependencies=[Depends(get_current_user)]
)
async def validate_workflow_by_id(
    workflow_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Validate an existing saved pipeline workbook by ID directly from the database.
    """
    result = await db.execute(select(Workflow).where(Workflow.id == workflow_id))
    wf = result.scalar_one_or_none()
    if not wf:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow workbook '{workflow_id}' not found."
        )

    graph = db_workflow_to_graph(wf)
    diag = graph.get_diagnostics()
    return {
        "workflow_id": workflow_id,
        "name": wf.name,
        "valid": diag["is_valid"],
        "errors": diag["errors"],
        "warnings": diag["warnings"],
        "recommendations": diag["recommendations"]
    }


@router.post(
    "/execute",
    response_model=WorkflowExecutionResult,
    dependencies=[Depends(require_role("Tenant Admin", "Data Scientist", "ML Engineer"))]
)
async def execute_workflow(
    workflow: WorkflowGraph,
    include_node_outputs: bool = Query(False, description="Opt-in to include full raw data of every node (default: false for lean response)"),
    auto_save: bool = Query(True, description="Automatically upsert/save current workflow to DB before executing"),
    workflow_id: Optional[str] = Query(None, description="Optional workflow ID to link/update in database"),
    workflow_name: Optional[str] = Query(None, description="Optional workflow title if auto-saving to DB"),
    db: AsyncSession = Depends(get_db)
):
    """
    Execute a full workflow DAG end-to-end synchronously.
    Supports auto-saving: saves/upserts the workflow, links its active dataset, and persists execution metrics/reports to DB.
    """
    target_id = (
        workflow_id
        or getattr(workflow, "workflow_id", None)
        or getattr(workflow, "id", None)
        or getattr(workflow, "pipeline_id", None)
    )
    if auto_save or target_id:
        target_id = target_id or str(uuid.uuid4())
        nodes_payload, edges_payload, node_configs = workflow_graph_to_db_payload(workflow)
        
        ds_id, ds_name = await resolve_workflow_dataset(
            db=db,
            dataset_id=workflow.dataset_id,
            dataset_name=workflow.dataset_name,
            nodes=nodes_payload,
            node_configs=node_configs
        )

        result = await db.execute(select(Workflow).where(Workflow.id == target_id))
        wf = result.scalar_one_or_none()
        if wf:
            if workflow_name or workflow.name:
                wf.name = workflow_name or workflow.name
            if ds_id is not None:
                wf.dataset_id = ds_id
            if ds_name is not None:
                wf.dataset_name = ds_name
            wf.nodes = nodes_payload
            wf.edges = edges_payload
            wf.node_configs = node_configs
            wf.is_active = True
            wf.deleted_at = None
            wf.updated_at = datetime.now(timezone.utc)
        else:
            wf = Workflow(
                id=target_id,
                name=workflow.name or workflow_name or "Auto-Saved Pipeline",
                description="Auto-saved during execution run",
                dataset_id=ds_id,
                dataset_name=ds_name,
                nodes=nodes_payload,
                edges=edges_payload,
                node_configs=node_configs,
                is_active=True
            )
            db.add(wf)
        await db.commit()

    execution_id = str(uuid.uuid4())
    result = DAGExecutor.execute_workflow(
        execution_id=execution_id,
        workflow=workflow,
        include_node_outputs=include_node_outputs
    )
    if target_id:
        result.workflow_id = target_id
        res_wf = await db.execute(select(Workflow).where(Workflow.id == target_id))
        wf_rec = res_wf.scalar_one_or_none()
        if wf_rec:
            wf_rec.last_execution = jsonable_encoder({
                "execution_id": result.execution_id,
                "status": result.status,
                "total_duration_ms": result.total_duration_ms,
                "final_metrics": result.final_metrics,
                "anomaly_summary": result.anomaly_summary,
                "forecasting_summary": result.forecasting_summary,
                "governance_summary": result.governance_summary,
                "node_results": result.node_results,
                "execution_logs": result.logs,
                "step_snapshots": result.step_snapshots,
                "inference_schema": getattr(result, "inference_schema", None),
            })
            await db.commit()

            # Record immutable history snapshot
            try:
                await record_workflow_execution_history(
                    db=db,
                    workflow_id=target_id,
                    execution_id=result.execution_id,
                    status_str=result.status,
                    total_duration_ms=result.total_duration_ms or 0.0,
                    nodes=wf_rec.nodes or [],
                    edges=wf_rec.edges or [],
                    node_configs=wf_rec.node_configs or {},
                    metrics=result.final_metrics,
                    reports={
                        "anomaly_summary": result.anomaly_summary,
                        "forecasting_summary": result.forecasting_summary,
                        "governance_summary": result.governance_summary,
                        "node_results": result.node_results,
                        "inference_schema": getattr(result, "inference_schema", None),
                    },
                    step_snapshots=result.step_snapshots,
                    logs=result.logs,
                    run_label=workflow_name or getattr(workflow, "name", None) or wf_rec.name
                )
            except Exception:
                pass
    return result


@router.post(
    "/{workflow_id}/execute",
    response_model=WorkflowExecutionResult,
    dependencies=[Depends(require_role("Tenant Admin", "Data Scientist", "ML Engineer"))]
)
async def execute_workflow_by_id(
    workflow_id: str,
    workflow: Optional[WorkflowGraph] = None,
    include_node_outputs: bool = Query(False, description="Opt-in to include full raw data of every node"),
    db: AsyncSession = Depends(get_db)
):
    """
    Execute an existing workflow directly by ID.
    If an updated workflow graph body is provided, it auto-updates the saved record in the database first.
    If no body is passed, it executes the saved configuration from the database.
    """
    result = await db.execute(select(Workflow).where(Workflow.id == workflow_id))
    wf = result.scalar_one_or_none()
    if not wf:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow workbook '{workflow_id}' not found."
        )

    if workflow is not None:
        nodes_payload, edges_payload, node_configs = workflow_graph_to_db_payload(workflow)
        ds_id, ds_name = await resolve_workflow_dataset(
            db=db,
            dataset_id=workflow.dataset_id,
            dataset_name=workflow.dataset_name,
            nodes=nodes_payload,
            node_configs=node_configs
        )
        if ds_id is not None:
            wf.dataset_id = ds_id
        if ds_name is not None:
            wf.dataset_name = ds_name
        wf.nodes = nodes_payload
        wf.edges = edges_payload
        wf.node_configs = node_configs
        wf.is_active = True
        wf.deleted_at = None
        wf.updated_at = datetime.now(timezone.utc)
        await db.commit()
        graph_to_execute = workflow
    else:
        graph_to_execute = db_workflow_to_graph(wf)

    execution_id = str(uuid.uuid4())
    exec_result = DAGExecutor.execute_workflow(
        execution_id=execution_id,
        workflow=graph_to_execute,
        include_node_outputs=include_node_outputs
    )
    exec_result.workflow_id = workflow_id
    wf.last_execution = jsonable_encoder({
        "execution_id": exec_result.execution_id,
        "status": exec_result.status,
        "total_duration_ms": exec_result.total_duration_ms,
        "final_metrics": exec_result.final_metrics,
        "anomaly_summary": exec_result.anomaly_summary,
        "forecasting_summary": exec_result.forecasting_summary,
        "governance_summary": exec_result.governance_summary,
        "node_results": exec_result.node_results,
        "execution_logs": exec_result.logs,
        "step_snapshots": exec_result.step_snapshots,
        "inference_schema": getattr(exec_result, "inference_schema", None),
    })
    await db.commit()

    # Record immutable history snapshot
    try:
        await record_workflow_execution_history(
            db=db,
            workflow_id=workflow_id,
            execution_id=exec_result.execution_id,
            status_str=exec_result.status,
            total_duration_ms=exec_result.total_duration_ms or 0.0,
            nodes=wf.nodes or [],
            edges=wf.edges or [],
            node_configs=wf.node_configs or {},
            metrics=exec_result.final_metrics,
            reports={
                "anomaly_summary": exec_result.anomaly_summary,
                "forecasting_summary": exec_result.forecasting_summary,
                "governance_summary": exec_result.governance_summary,
                "node_results": exec_result.node_results,
                "inference_schema": getattr(exec_result, "inference_schema", None),
            },
            step_snapshots=exec_result.step_snapshots,
            logs=exec_result.logs,
            run_label=wf.name
        )
    except Exception:
        pass

    return exec_result


@router.post(
    "/async-execute",
    dependencies=[Depends(require_role("Tenant Admin", "Data Scientist", "ML Engineer"))]
)
async def submit_async_workflow(
    workflow: WorkflowGraph,
    include_node_outputs: bool = Query(False, description="Opt-in to include full raw data of every node")
):
    """
    Submit a workflow for asynchronous background execution (n8n/Boomi worker pool pattern).
    Returns immediately with a job_id for status polling.
    """
    job_id = job_manager.submit_job(
        workflow=workflow,
        trigger_type="api_async",
        include_node_outputs=include_node_outputs
    )
    return {
        "job_id": job_id,
        "status": "PENDING",
        "message": "Workflow execution dispatched to background worker pool."
    }




@router.post("/trigger/{webhook_path}")
async def trigger_webhook(
    webhook_path: str,
    payload: Union[List[Dict[str, Any]], Dict[str, Any]] = Body(...)
):
    """
    Inbound Webhook Trigger Endpoint (n8n / Boomi pattern).
    Receives external HTTP JSON payloads, normalizes into a DataFrame, and returns trigger receipt.
    """
    if isinstance(payload, list):
        df = pd.json_normalize(payload)
    else:
        df = pd.json_normalize([payload])

    return {
        "status": "TRIGGERED",
        "webhook_path": webhook_path,
        "records_ingested": len(df),
        "columns_received": list(df.columns),
        "sample_preview": df.head(3).to_dict(orient="records")
    }


@router.post("/autowire")
async def autowire_workflow_nodes(payload: Dict[str, Any] = Body(...)):
    """
    Auto-Wire DAG Endpoint. Automatically links whiteboard nodes into an intelligent pipeline DAG.
    """
    from backend.app.recommendation.router import autowire_nodes, AutoWireRequest
    nodes = payload.get("nodes", [])
    return await autowire_nodes(AutoWireRequest(nodes=nodes))


# -------------------------------------------------------------
# MODEL INFERENCE & LIVE PREDICTION ENDPOINTS
# -------------------------------------------------------------

@router.get("/{execution_id}/schema", response_model=InferenceSchemaResponse)
async def get_execution_inference_schema(execution_id: str):
    """
    Retrieve the dynamic input schema, feature boundaries, default values,
    and sample request payload for a trained pipeline execution.
    """
    bundle = job_manager.get_inference_bundle(execution_id)
    if not bundle:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No live model inference bundle found for execution ID '{execution_id}'. Please execute the pipeline first."
        )

    feat_items: List[FeatureSchemaItem] = []
    features_summary = bundle.get("training_feature_summary", {})
    feature_names = bundle.get("feature_names", [])

    for f_name in feature_names:
        f_info = features_summary.get(f_name, {})
        feat_items.append(FeatureSchemaItem(
            name=f_name,
            data_type=f_info.get("data_type", "numeric"),
            min_value=f_info.get("min_value"),
            max_value=f_info.get("max_value"),
            median_value=f_info.get("median_value"),
            mean_value=f_info.get("mean_value"),
            default_value=f_info.get("default_value", 0.0),
            allowed_categories=f_info.get("allowed_categories")
        ))

    return InferenceSchemaResponse(
        execution_id=execution_id,
        task_type=bundle.get("task_type", "classification"),
        target_column=bundle.get("target_column"),
        target_classes=bundle.get("target_classes", []),
        features=feat_items,
        sample_payload=bundle.get("sample_row", {}),
        time_series_meta=bundle.get("forecasting_summary")
    )


# -------------------------------------------------------------
# WORKFLOW VERSION HISTORY, AUDIT TRAIL & ROLLBACK ENDPOINTS (Option B)
# -------------------------------------------------------------

@router.get(
    "/{workflow_id}/history/compare",
    response_model=WorkflowCompareResponse,
    dependencies=[Depends(get_current_user)]
)
async def compare_workflow_executions(
    workflow_id: str,
    run_a: str = Query(..., description="First execution ID to compare"),
    run_b: str = Query(..., description="Second execution ID to compare"),
    db: AsyncSession = Depends(get_db)
):
    """
    Endpoint 4: Side-by-Side Execution Diffing.
    Compares metrics, hyperparameter changes, and graph modifications between two runs.
    """
    res_a = await db.execute(select(WorkflowExecution).where(WorkflowExecution.workflow_id == workflow_id, WorkflowExecution.id == run_a))
    exec_a = res_a.scalar_one_or_none()
    if not exec_a:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Execution '{run_a}' not found for workflow '{workflow_id}'.")

    res_b = await db.execute(select(WorkflowExecution).where(WorkflowExecution.workflow_id == workflow_id, WorkflowExecution.id == run_b))
    exec_b = res_b.scalar_one_or_none()
    if not exec_b:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Execution '{run_b}' not found for workflow '{workflow_id}'.")

    # 1. Compare Metrics
    metrics_a = exec_a.metrics or {}
    metrics_b = exec_b.metrics or {}
    all_metric_keys = sorted(list(set(list(metrics_a.keys()) + list(metrics_b.keys()))))
    
    metrics_diff: Dict[str, Dict[str, Any]] = {}
    lower_is_better_keys = {"loss", "log_loss", "mae", "mse", "rmse", "mean_squared_error", "mean_absolute_error"}

    for k in all_metric_keys:
        val_a = metrics_a.get(k)
        val_b = metrics_b.get(k)
        if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
            delta = round(float(val_b) - float(val_a), 4)
            pct_change = round((delta / abs(val_a)) * 100, 2) if val_a != 0 else None
            higher_is_better = k.lower() not in lower_is_better_keys
            improved = (delta > 0) if higher_is_better else (delta < 0)
            metrics_diff[k] = {
                "val_a": val_a,
                "val_b": val_b,
                "delta": delta,
                "pct_change": pct_change,
                "improved": improved
            }

    # 2. Compare Graph Nodes & Configs
    nodes_a = {n.get("id"): n for n in (exec_a.snapshot_nodes or []) if isinstance(n, dict)}
    nodes_b = {n.get("id"): n for n in (exec_b.snapshot_nodes or []) if isinstance(n, dict)}

    nodes_added = [
        {"id": nid, "recipe_id": n.get("recipe_id") or n.get("data", {}).get("recipe_id"), "label": n.get("content") or n.get("label")}
        for nid, n in nodes_b.items() if nid not in nodes_a
    ]
    nodes_removed = [
        {"id": nid, "recipe_id": n.get("recipe_id") or n.get("data", {}).get("recipe_id"), "label": n.get("content") or n.get("label")}
        for nid, n in nodes_a.items() if nid not in nodes_b
    ]

    configs_a = exec_a.snapshot_node_configs or {}
    configs_b = exec_b.snapshot_node_configs or {}
    common_node_ids = set(nodes_a.keys()).intersection(set(nodes_b.keys()))

    parameter_changes = []
    for nid in common_node_ids:
        cfg_entry_a = configs_a.get(nid, {})
        cfg_entry_b = configs_b.get(nid, {})
        params_a = cfg_entry_a.get("config", {}) if isinstance(cfg_entry_a, dict) else {}
        params_b = cfg_entry_b.get("config", {}) if isinstance(cfg_entry_b, dict) else {}
        r_id = (cfg_entry_b.get("recipe_id") or cfg_entry_a.get("recipe_id") or 
                nodes_b[nid].get("recipe_id") or nodes_b[nid].get("data", {}).get("recipe_id"))

        all_param_keys = set(list(params_a.keys()) + list(params_b.keys()))
        for p in all_param_keys:
            v_a = params_a.get(p)
            v_b = params_b.get(p)
            if v_a != v_b:
                parameter_changes.append({
                    "node_id": nid,
                    "recipe_id": r_id,
                    "parameter": p,
                    "val_run_a": v_a,
                    "val_run_b": v_b
                })

    config_diff = {
        "nodes_added": nodes_added,
        "nodes_removed": nodes_removed,
        "parameter_changes": parameter_changes,
        "total_nodes_run_a": len(nodes_a),
        "total_nodes_run_b": len(nodes_b),
    }

    return WorkflowCompareResponse(
        workflow_id=workflow_id,
        run_a={
            "id": exec_a.id,
            "version_number": exec_a.version_number,
            "run_label": exec_a.run_label,
            "status": exec_a.status,
            "created_at": exec_a.created_at.isoformat() if exec_a.created_at else "",
            "metrics": exec_a.metrics
        },
        run_b={
            "id": exec_b.id,
            "version_number": exec_b.version_number,
            "run_label": exec_b.run_label,
            "status": exec_b.status,
            "created_at": exec_b.created_at.isoformat() if exec_b.created_at else "",
            "metrics": exec_b.metrics
        },
        metrics_diff=metrics_diff,
        config_diff=config_diff
    )


@router.get(
    "/{workflow_id}/history",
    response_model=List[WorkflowExecutionSummaryResponse],
    dependencies=[Depends(get_current_user)]
)
async def list_workflow_history(
    workflow_id: str,
    limit: int = Query(50, ge=1, le=200, description="Max history runs to retrieve"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    db: AsyncSession = Depends(get_db)
):
    """
    Endpoint 1: Lightweight History Timeline List.
    Retrieves execution history sorted in descending version order.
    Includes backward-compatibility auto-synthesis if the workflow has an existing last_execution.
    """
    # Verify workflow exists
    wf_res = await db.execute(select(Workflow).where(Workflow.id == workflow_id))
    wf = wf_res.scalar_one_or_none()
    if not wf:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow workbook '{workflow_id}' not found."
        )

    stmt = (
        select(WorkflowExecution)
        .where(WorkflowExecution.workflow_id == workflow_id)
        .order_by(WorkflowExecution.version_number.desc())
        .offset(offset)
        .limit(limit)
    )
    res = await db.execute(stmt)
    executions = res.scalars().all()

    # Backward-compatibility fallback: synthesize Run #1 if none exist but last_execution exists
    if not executions and wf.last_execution and isinstance(wf.last_execution, dict):
        last_ex = wf.last_execution
        synth_rec = WorkflowExecution(
            id=last_ex.get("execution_id") or str(uuid.uuid4()),
            workflow_id=workflow_id,
            version_number=1,
            run_label="Run #1 (Initial Execution)",
            status=last_ex.get("status", "SUCCESS"),
            total_duration_ms=last_ex.get("total_duration_ms", 0.0),
            snapshot_nodes=jsonable_encoder(wf.nodes or []),
            snapshot_edges=jsonable_encoder(wf.edges or []),
            snapshot_node_configs=jsonable_encoder(wf.node_configs or {}),
            metrics=jsonable_encoder(last_ex.get("final_metrics") or last_ex.get("metrics") or {}),
            reports=jsonable_encoder({
                "anomaly_summary": last_ex.get("anomaly_summary"),
                "forecasting_summary": last_ex.get("forecasting_summary"),
                "governance_summary": last_ex.get("governance_summary"),
                "node_results": last_ex.get("node_results"),
                "inference_schema": last_ex.get("inference_schema"),
            }),
            step_snapshots=jsonable_encoder(last_ex.get("step_snapshots") or {}),
            logs=jsonable_encoder(last_ex.get("execution_logs") or last_ex.get("logs") or []),
        )
        db.add(synth_rec)
        await db.commit()
        await db.refresh(synth_rec)
        executions = [synth_rec]

    return [
        WorkflowExecutionSummaryResponse(
            id=ex.id,
            workflow_id=ex.workflow_id,
            version_number=ex.version_number,
            run_label=ex.run_label,
            status=ex.status,
            total_duration_ms=ex.total_duration_ms,
            metrics=ex.metrics,
            nodes_count=len(ex.snapshot_nodes or []),
            edges_count=len(ex.snapshot_edges or []),
            created_at=ex.created_at
        )
        for ex in executions
    ]


@router.get(
    "/{workflow_id}/history/{execution_id}",
    response_model=WorkflowExecutionDetailResponse,
    dependencies=[Depends(get_current_user)]
)
async def get_workflow_execution_detail(
    workflow_id: str,
    execution_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Endpoint 2: Run Deep-Dive (Full Frozen Snapshot & Diagnostics).
    Returns complete frozen graph configurations, reports, confusion matrices, and step logs.
    """
    stmt = select(WorkflowExecution).where(
        WorkflowExecution.workflow_id == workflow_id,
        WorkflowExecution.id == execution_id
    )
    res = await db.execute(stmt)
    ex = res.scalar_one_or_none()
    if not ex:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Execution '{execution_id}' not found for workflow '{workflow_id}'."
        )

    return WorkflowExecutionDetailResponse(
        id=ex.id,
        workflow_id=ex.workflow_id,
        version_number=ex.version_number,
        run_label=ex.run_label,
        status=ex.status,
        total_duration_ms=ex.total_duration_ms,
        snapshot_nodes=ex.snapshot_nodes or [],
        snapshot_edges=ex.snapshot_edges or [],
        snapshot_node_configs=ex.snapshot_node_configs or {},
        metrics=ex.metrics,
        reports=ex.reports,
        step_snapshots=ex.step_snapshots,
        logs=ex.logs,
        created_at=ex.created_at
    )


@router.post(
    "/{workflow_id}/history/{execution_id}/rollback",
    response_model=WorkflowResponse,
    dependencies=[Depends(require_role("Tenant Admin", "Data Scientist", "ML Engineer"))]
)
async def rollback_workflow_to_execution(
    workflow_id: str,
    execution_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Endpoint 3: One-Click Rollback / Restore.
    Restores the workflow's active nodes, edges, node_configs, and last_execution to match this historical run.
    """
    res_exec = await db.execute(
        select(WorkflowExecution).where(
            WorkflowExecution.workflow_id == workflow_id,
            WorkflowExecution.id == execution_id
        )
    )
    ex = res_exec.scalar_one_or_none()
    if not ex:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Execution '{execution_id}' not found for workflow '{workflow_id}'."
        )

    res_wf = await db.execute(select(Workflow).where(Workflow.id == workflow_id))
    wf = res_wf.scalar_one_or_none()
    if not wf:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow '{workflow_id}' not found."
        )

    # Overwrite canvas with historical snapshot
    wf.nodes = ex.snapshot_nodes
    wf.edges = ex.snapshot_edges
    wf.node_configs = ex.snapshot_node_configs

    # Reconstruct last_execution payload from snapshot
    restored_last_exec = {
        "execution_id": ex.id,
        "status": ex.status,
        "total_duration_ms": ex.total_duration_ms,
        "final_metrics": ex.metrics,
        **(ex.reports or {}),
        "step_snapshots": ex.step_snapshots,
        "execution_logs": ex.logs,
        "rolled_back_from_version": ex.version_number
    }
    wf.last_execution = jsonable_encoder(restored_last_exec)
    wf.updated_at = datetime.now(timezone.utc)

    await db.commit()
    await db.refresh(wf)
    return wf


@router.post("/predict", response_model=PredictionResponse)
async def predict_latest(
    execution_id: str = Query(..., description="The execution ID of the trained pipeline"),
    payload: PredictionRequest = Body(...)
):
    """
    Convenience endpoint for live predictions passing execution_id as a query parameter.
    """
    return await predict_with_execution(execution_id=execution_id, payload=payload)


@router.post("/{execution_id}/predict", response_model=PredictionResponse)
async def predict_with_execution(
    execution_id: str,
    payload: PredictionRequest = Body(...)
):
    """
    Execute live model inference using the trained pipeline artifacts from a specific execution run.
    Supports single feature dictionaries, batch CSV records, or time-series forecast horizon / date ranges.
    """
    bundle = job_manager.get_inference_bundle(execution_id)
    if not bundle:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No live model inference bundle found for execution ID '{execution_id}'. Please execute the pipeline first."
        )

    response = PipelineInferencer.predict(bundle=bundle, request=payload)
    if response.status == "FAILED":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Prediction failed: {response.error_message}"
        )
    return response


