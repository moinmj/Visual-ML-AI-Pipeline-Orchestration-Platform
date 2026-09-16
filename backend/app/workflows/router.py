from fastapi import APIRouter, Depends, HTTPException, status, Body, Query, Response
from fastapi.encoders import jsonable_encoder
from typing import Dict, Any, List, Optional, Union
import math
import uuid
import pandas as pd
from datetime import datetime, timezone, timedelta
from sqlalchemy import func, or_
from sqlalchemy.orm import load_only
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
    WorkflowListItemResponse,
    WorkflowListPaginatedResponse,
    WorkflowStatusFilter,
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
    dataset_id: Optional[str] = None,
    workflow_id: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Resolves, normalizes, and protects last_execution diagnostics and reports:
    1. If a valid, non-empty last_execution dictionary is supplied, normalizes logs and returns it JSON-safe.
    2. If execution_id is provided, searches DB workflows and job_manager for the matching execution report.
    3. If no execution payload was supplied, checks if a newer in-memory execution exists in job_manager for this workflow.
    4. If updating an existing workflow and no newer execution is found, preserves the existing last_execution.
    5. Auto-adoption fallback: If user executed an unsaved canvas workflow and then clicked Save Workflow,
       locates the recent auto-saved execution run and adopts its reports.
    """
    wf_target_id = workflow_id or (existing_wf.id if existing_wf else None)

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
        if job and (job.get("results") or job.get("result")):
            res = job.get("results") or job.get("result")
            norm_job = {
                "execution_id": target_exec_id,
                "status": job.get("status", "SUCCESS"),
                "total_duration_ms": job.get("duration_ms", 0.0),
                "final_metrics": res.get("final_metrics") if isinstance(res, dict) else getattr(res, "final_metrics", None),
                "node_results": res.get("node_results") if isinstance(res, dict) else getattr(res, "node_results", []),
                "execution_logs": job.get("logs", []),
                "logs": job.get("logs", []),
                "step_snapshots": res.get("step_snapshots", {}) if isinstance(res, dict) else getattr(res, "step_snapshots", {}),
                "anomaly_summary": res.get("anomaly_summary") if isinstance(res, dict) else getattr(res, "anomaly_summary", None),
                "forecasting_summary": res.get("forecasting_summary") if isinstance(res, dict) else getattr(res, "forecasting_summary", None),
                "governance_summary": res.get("governance_summary") if isinstance(res, dict) else getattr(res, "governance_summary", None),
                "inference_schema": res.get("inference_schema") if isinstance(res, dict) else getattr(res, "inference_schema", None),
            }
            return jsonable_encoder(norm_job)

        # Check existing workflows in DB
        wf_q = select(Workflow).where(Workflow.last_execution.is_not(None)).order_by(Workflow.updated_at.desc()).limit(20)
        wf_res = await db.execute(wf_q)
        for cand in wf_res.scalars().all():
            cand_exec = cand.last_execution
            if isinstance(cand_exec, dict) and cand_exec.get("execution_id") == target_exec_id:
                return jsonable_encoder(cand_exec)

    # 3. Check if a newer in-memory execution exists for this workflow in job_manager
    if wf_target_id:
        recent_job = job_manager.get_latest_execution_for_workflow(wf_target_id)
        if recent_job and (recent_job.get("results") or recent_job.get("result")):
            existing_exec_id = None
            if existing_wf and isinstance(existing_wf.last_execution, dict):
                existing_exec_id = existing_wf.last_execution.get("execution_id")

            if not existing_exec_id or recent_job.get("job_id") != existing_exec_id:
                res = recent_job.get("results") or recent_job.get("result")
                norm_job = {
                    "execution_id": recent_job.get("job_id"),
                    "status": recent_job.get("status", "SUCCESS"),
                    "total_duration_ms": recent_job.get("duration_ms", 0.0),
                    "final_metrics": res.get("final_metrics") if isinstance(res, dict) else getattr(res, "final_metrics", None),
                    "node_results": res.get("node_results") if isinstance(res, dict) else getattr(res, "node_results", []),
                    "execution_logs": recent_job.get("logs", []),
                    "logs": recent_job.get("logs", []),
                    "step_snapshots": res.get("step_snapshots", {}) if isinstance(res, dict) else getattr(res, "step_snapshots", {}),
                    "anomaly_summary": res.get("anomaly_summary") if isinstance(res, dict) else getattr(res, "anomaly_summary", None),
                    "forecasting_summary": res.get("forecasting_summary") if isinstance(res, dict) else getattr(res, "forecasting_summary", None),
                    "governance_summary": res.get("governance_summary") if isinstance(res, dict) else getattr(res, "governance_summary", None),
                    "inference_schema": res.get("inference_schema") if isinstance(res, dict) else getattr(res, "inference_schema", None),
                }
                return jsonable_encoder(norm_job)

    # 4. If updating an existing workflow, preserve existing last_execution
    if existing_wf and existing_wf.last_execution:
        return existing_wf.last_execution

    # 5. Auto-adoption: Find recent auto-saved execution run (within last 30 minutes)
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


BULKY_METRIC_KEYS = {
    "trajectory",
    "actual_vs_predicted_time_series",
    "confusion_matrix",
    "classification_report",
    "predictions",
    "predictions_sample",
    "probabilities",
    "residuals",
    "raw_residuals",
    "data_drift_report",
    "step_snapshots",
    "node_results",
    "reports",
    "logs",
    "execution_logs",
    "sample_payload",
    "forecast_df",
    "dataframe",
}


def normalize_history_status(status_str: Optional[str]) -> str:
    """Normalizes status strings to standard 'SUCCESS', 'FAILED', or 'UNRUN'."""
    if not status_str:
        return "UNRUN"
    s = str(status_str).strip().upper()
    if "FAIL" in s or "ERR" in s or "UNSUCCESS" in s:
        return "FAILED"
    if "SUCC" in s or "PASS" in s or "DONE" in s:
        return "SUCCESS"
    if "UNRUN" in s or "DRAFT" in s or "NEVER" in s:
        return "UNRUN"
    return s


def sanitize_summary_metrics(metrics: Any) -> Dict[str, Any]:
    """
    Sanitizes metrics dictionary for the lightweight history timeline summary.
    Excludes heavy reports/time-series/matrices and preserves compact scalar and list attributes.
    Keeps API responses shorter and prevents Swagger UI hanging.
    """
    if not isinstance(metrics, dict):
        return {}

    sanitized: Dict[str, Any] = {}
    for k, v in metrics.items():
        if k in BULKY_METRIC_KEYS:
            continue
        # Primitive scalars (int, float, bool, str)
        if isinstance(v, (int, float, bool, str)) or v is None:
            if isinstance(v, float):
                sanitized[k] = round(v, 4)
            else:
                sanitized[k] = v
        # Short primitive lists (e.g. encoded_columns: ["Date"], target_classes, feature names)
        elif isinstance(v, list):
            if len(v) <= 15 and all(isinstance(x, (int, float, bool, str)) for x in v):
                sanitized[k] = v
            elif len(v) > 15 and all(isinstance(x, (int, float, bool, str)) for x in v):
                # Keep first 15 items to prevent huge payload
                sanitized[k] = v[:15]
        # Small flat sub-dictionaries (e.g. { class_0: 0.9, class_1: 0.8 })
        elif isinstance(v, dict):
            if len(v) <= 6 and all(isinstance(sub_v, (int, float, bool, str)) for sub_v in v.values()):
                sanitized[k] = {
                    sub_k: (round(sub_v, 4) if isinstance(sub_v, float) else sub_v)
                    for sub_k, sub_v in v.items()
                }

    return sanitized


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

    # If no executions recorded yet, but workflow already had an earlier last_execution,
    # preserve that prior run as Version 1 so it isn't overwritten by the new run
    if current_max == 0:
        wf_chk = await db.execute(select(Workflow).where(Workflow.id == workflow_id))
        existing_wf_rec = wf_chk.scalar_one_or_none()
        if (
            existing_wf_rec
            and isinstance(existing_wf_rec.last_execution, dict)
            and existing_wf_rec.last_execution.get("execution_id")
            and existing_wf_rec.last_execution.get("execution_id") != execution_id
        ):
            prior_ex = existing_wf_rec.last_execution
            prior_rec = WorkflowExecution(
                id=prior_ex.get("execution_id"),
                workflow_id=workflow_id,
                version_number=1,
                run_label="Run #1",
                status=normalize_history_status(prior_ex.get("status") or "SUCCESS"),
                total_duration_ms=round(float(prior_ex.get("total_duration_ms") or 0.0), 2),
                snapshot_nodes=jsonable_encoder(existing_wf_rec.nodes or []),
                snapshot_edges=jsonable_encoder(existing_wf_rec.edges or []),
                snapshot_node_configs=jsonable_encoder(existing_wf_rec.node_configs or {}),
                metrics=jsonable_encoder(prior_ex.get("final_metrics") or prior_ex.get("metrics") or {}),
                reports=jsonable_encoder({
                    "anomaly_summary": prior_ex.get("anomaly_summary"),
                    "forecasting_summary": prior_ex.get("forecasting_summary"),
                    "governance_summary": prior_ex.get("governance_summary"),
                    "node_results": prior_ex.get("node_results"),
                    "inference_schema": prior_ex.get("inference_schema"),
                }),
                step_snapshots=jsonable_encoder(prior_ex.get("step_snapshots") or {}),
                logs=jsonable_encoder(prior_ex.get("execution_logs") or prior_ex.get("logs") or [])
            )
            db.add(prior_rec)
            await db.commit()
            current_max = 1

    next_ver = current_max + 1
    default_label = f"Run #{next_ver}"

    exec_record = WorkflowExecution(
        id=execution_id,
        workflow_id=workflow_id,
        version_number=next_ver,
        run_label=run_label or default_label,
        status=normalize_history_status(status_str or "SUCCESS"),
        total_duration_ms=round(float(total_duration_ms or 0.0), 2),
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

    # Auto-prune older versions if exceeding MAX_SAVED_VERSIONS (keep last 5 runs to prevent DB bloat)
    try:
        all_vers_stmt = (
            select(WorkflowExecution)
            .where(WorkflowExecution.workflow_id == workflow_id)
            .order_by(WorkflowExecution.version_number.desc())
        )
        all_vers_res = await db.execute(all_vers_stmt)
        all_vers = all_vers_res.scalars().all()
        MAX_SAVED_VERSIONS = 5
        if len(all_vers) > MAX_SAVED_VERSIONS:
            for old_ver in all_vers[MAX_SAVED_VERSIONS:]:
                await db.delete(old_ver)
            await db.commit()
    except Exception:
        pass

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
        dataset_id=ds_id,
        workflow_id=target_id
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

    # Record immutable history version snapshot when workbook is explicitly saved with execution results
    if resolved_last_exec and isinstance(resolved_last_exec, dict):
        try:
            e_id = resolved_last_exec.get("execution_id") or str(uuid.uuid4())
            await record_workflow_execution_history(
                db=db,
                workflow_id=wf.id,
                execution_id=e_id,
                status_str=resolved_last_exec.get("status", "SUCCESS"),
                total_duration_ms=resolved_last_exec.get("total_duration_ms", 0.0),
                nodes=wf.nodes or [],
                edges=wf.edges or [],
                node_configs=wf.node_configs or {},
                metrics=resolved_last_exec.get("final_metrics") or resolved_last_exec.get("metrics"),
                reports={
                    "anomaly_summary": resolved_last_exec.get("anomaly_summary"),
                    "forecasting_summary": resolved_last_exec.get("forecasting_summary"),
                    "governance_summary": resolved_last_exec.get("governance_summary"),
                    "node_results": resolved_last_exec.get("node_results"),
                    "inference_schema": resolved_last_exec.get("inference_schema"),
                },
                step_snapshots=resolved_last_exec.get("step_snapshots"),
                logs=resolved_last_exec.get("execution_logs") or resolved_last_exec.get("logs"),
                run_label=None
            )
        except Exception:
            pass

    await db.refresh(wf)
    return wf


@router.get("/", response_model=WorkflowListPaginatedResponse)
async def list_workflows(
    include_deleted: bool = False,
    limit: int = Query(10, ge=1, le=200, description="Max workbooks to retrieve per page"),
    skip: int = Query(0, ge=0, description="Number of records to skip (offset)"),
    offset: Optional[int] = Query(None, ge=0, description="Alias for skip"),
    page: Optional[int] = Query(None, ge=1, description="Optional 1-indexed page number (e.g. page=2 with limit=10 sets skip=10)"),
    filters: Optional[WorkflowStatusFilter] = Query(None, description="Filter workbooks by execution status: 'success', 'failed', 'unrun'"),
    search: Optional[str] = Query(None, description="Search keyword to filter workflows by title or description"),
    status: Optional[str] = Query(None, include_in_schema=False, description="Alias for filters"),
    filter: Optional[str] = Query(None, include_in_schema=False, description="Alias for filters"),
    db: AsyncSession = Depends(get_db)
):
    """
    List all saved pipeline workbooks with pagination, execution status filtering ('success', 'failed', 'unrun'),
    and keyword search.
    Returns: total_records, skip, limit, current_page, total_pages, and data array.
    """
    actual_skip = skip
    if page is not None and skip == 0:
        actual_skip = (page - 1) * limit
    elif offset is not None and skip == 0:
        actual_skip = offset

    # Build SQL filter conditions
    conditions = []
    if not include_deleted:
        conditions.append(Workflow.is_active == True)

    # 1. Search filter by title / description keyword
    if search and search.strip():
        term = f"%{search.strip()}%"
        conditions.append(or_(
            Workflow.name.ilike(term),
            Workflow.description.ilike(term)
        ))

    # 2. Execution status filter ('success', 'failed', 'unrun')
    status_raw = filters.value if isinstance(filters, WorkflowStatusFilter) else (filters or status or filter)
    if status_raw and str(status_raw).strip():
        s = str(status_raw).strip().lower()
        if s == "success":
            conditions.append(
                func.lower(func.json_extract(Workflow.last_execution, '$.status')) == 'success'
            )
        elif s == "failed":
            conditions.append(
                or_(
                    func.lower(func.json_extract(Workflow.last_execution, '$.status')) == 'failed',
                    func.lower(func.json_extract(Workflow.last_execution, '$.status')) == 'error'
                )
            )
        elif s in ("unrun", "never_run", "draft", "not_run"):
            conditions.append(
                or_(
                    Workflow.last_execution == None,
                    func.json_extract(Workflow.last_execution, '$.status') == None,
                    func.json_extract(Workflow.last_execution, '$.status') == ""
                )
            )

    # Count total records matching filter conditions
    count_stmt = select(func.count(Workflow.id))
    for cond in conditions:
        count_stmt = count_stmt.where(cond)
    total_res = await db.execute(count_stmt)
    total_records = total_res.scalar() or 0

    current_page = (actual_skip // limit) + 1 if limit > 0 else 1
    total_pages = math.ceil(total_records / limit) if (limit > 0 and total_records > 0) else (1 if total_records == 0 else 1)

    # Retrieve matching records
    query = select(Workflow)
    for cond in conditions:
        query = query.where(cond)
    query = query.order_by(Workflow.updated_at.desc()).offset(actual_skip).limit(limit)
    result = await db.execute(query)
    workflows = result.scalars().all()

    items = []
    for wf in workflows:
        last_exec = wf.last_execution if isinstance(wf.last_execution, dict) else {}
        items.append(
            WorkflowListItemResponse(
                id=wf.id,
                name=wf.name,
                description=wf.description,
                dataset_id=wf.dataset_id,
                dataset_name=wf.dataset_name,
                is_active=wf.is_active,
                deleted_at=wf.deleted_at,
                created_at=wf.created_at,
                updated_at=wf.updated_at,
                last_execution_status=last_exec.get("status"),
                last_execution_id=last_exec.get("execution_id"),
                nodes_count=len(wf.nodes or []),
                edges_count=len(wf.edges or []),
            )
        )

    return WorkflowListPaginatedResponse(
        total_records=total_records,
        skip=actual_skip,
        limit=limit,
        current_page=current_page,
        total_pages=total_pages,
        data=items
    )


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
        dataset_id=ds_id,
        workflow_id=target_id
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

    # Record immutable history version snapshot when workbook is saved with execution results
    if resolved_last_exec and isinstance(resolved_last_exec, dict):
        try:
            e_id = resolved_last_exec.get("execution_id") or str(uuid.uuid4())
            await record_workflow_execution_history(
                db=db,
                workflow_id=wf.id,
                execution_id=e_id,
                status_str=resolved_last_exec.get("status", "SUCCESS"),
                total_duration_ms=resolved_last_exec.get("total_duration_ms", 0.0),
                nodes=wf.nodes or [],
                edges=wf.edges or [],
                node_configs=wf.node_configs or {},
                metrics=resolved_last_exec.get("final_metrics") or resolved_last_exec.get("metrics"),
                reports={
                    "anomaly_summary": resolved_last_exec.get("anomaly_summary"),
                    "forecasting_summary": resolved_last_exec.get("forecasting_summary"),
                    "governance_summary": resolved_last_exec.get("governance_summary"),
                    "node_results": resolved_last_exec.get("node_results"),
                    "inference_schema": resolved_last_exec.get("inference_schema"),
                },
                step_snapshots=resolved_last_exec.get("step_snapshots"),
                logs=resolved_last_exec.get("execution_logs") or resolved_last_exec.get("logs"),
                run_label=None
            )
        except Exception:
            pass

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
    auto_save: bool = Query(False, description="Automatically upsert/save current workflow to DB before executing"),
    workflow_id: Optional[str] = Query(None, description="Optional workflow ID to link in execution result"),
    workflow_name: Optional[str] = Query(None, description="Optional workflow title if auto-saving to DB"),
    db: AsyncSession = Depends(get_db)
):
    """
    Execute a full workflow DAG end-to-end synchronously.
    Execution does NOT auto-save to DB by default; workflows and execution results are saved via the Save Workbook endpoint.
    """
    target_id = (
        workflow_id
        or getattr(workflow, "workflow_id", None)
        or getattr(workflow, "id", None)
        or getattr(workflow, "pipeline_id", None)
    )
    if auto_save:
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

    # Cache execution result in memory for lookup if user subsequently saves the workbook
    try:
        job_manager.register_job_result(result.execution_id, result, workflow_id=target_id)
    except Exception:
        pass

    if auto_save and target_id:
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
    auto_save: bool = Query(False, description="Automatically save updated graph and execution to DB"),
    db: AsyncSession = Depends(get_db)
):
    """
    Execute an existing workflow directly by ID.
    If an updated workflow graph body is provided and auto_save=True, it updates the saved record in the database first.
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
        if auto_save:
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

    try:
        job_manager.register_job_result(exec_result.execution_id, exec_result, workflow_id=workflow_id)
    except Exception:
        pass

    if auto_save:
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
        time_series_meta=bundle.get("forecasting_summary"),
        has_temporal_feature=bool(bundle.get("has_temporal_feature", False)),
        temporal_column=bundle.get("temporal_column"),
        min_year=bundle.get("min_year"),
        max_year=bundle.get("max_year")
    )


# -------------------------------------------------------------
# WORKFLOW VERSION HISTORY & AUDIT TRAIL ENDPOINTS (Option B)
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
    limit: int = Query(20, ge=1, le=100, description="Max history runs to retrieve"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    db: AsyncSession = Depends(get_db)
):
    """
    Endpoint 1: Lightweight History Timeline List.
    Retrieves execution history sorted in descending version order.
    Defers heavy columns (step_snapshots, reports, logs) to keep Swagger and client responses blazing fast.
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
        .options(
            load_only(
                WorkflowExecution.id,
                WorkflowExecution.workflow_id,
                WorkflowExecution.version_number,
                WorkflowExecution.run_label,
                WorkflowExecution.status,
                WorkflowExecution.total_duration_ms,
                WorkflowExecution.metrics,
                WorkflowExecution.snapshot_nodes,
                WorkflowExecution.snapshot_edges,
                WorkflowExecution.created_at,
            )
        )
        .where(WorkflowExecution.workflow_id == workflow_id)
        .order_by(WorkflowExecution.version_number.desc())
        .offset(offset)
        .limit(limit)
    )
    res = await db.execute(stmt)
    executions = res.scalars().all()

    # Backward-compatibility fallback: synthesize Run #1 if none exist but last_execution exists
    if not executions:
        if wf.last_execution and isinstance(wf.last_execution, dict) and wf.last_execution.get("status") not in ("UNRUN", "unrun", None):
            last_ex = wf.last_execution
            norm_status = normalize_history_status(last_ex.get("status"))
            synth_rec = WorkflowExecution(
                id=last_ex.get("execution_id") or str(uuid.uuid4()),
                workflow_id=workflow_id,
                version_number=1,
                run_label="Run #1",
                status=norm_status,
                total_duration_ms=round(float(last_ex.get("total_duration_ms", 0.0)), 2),
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
        else:
            # UNRUN workbook fallback: Always return standard lightweight UNRUN history summary
            return [
                WorkflowExecutionSummaryResponse(
                    id=f"unrun_{wf.id}",
                    workflow_id=wf.id,
                    version_number=1,
                    run_label="Run #1",
                    status="UNRUN",
                    total_duration_ms=0.0,
                    metrics={},
                    nodes_count=len(wf.nodes or []),
                    edges_count=len(wf.edges or []),
                    created_at=wf.created_at or datetime.now(timezone.utc)
                )
            ]

    response_items: List[WorkflowExecutionSummaryResponse] = []
    for ex in executions:
        label = ex.run_label or f"Run #{ex.version_number}"
        if " (Initial Execution)" in label:
            label = label.replace(" (Initial Execution)", "")

        response_items.append(
            WorkflowExecutionSummaryResponse(
                id=ex.id,
                workflow_id=ex.workflow_id,
                version_number=ex.version_number,
                run_label=label,
                status=normalize_history_status(ex.status),
                total_duration_ms=round(float(ex.total_duration_ms or 0.0), 2),
                metrics=sanitize_summary_metrics(ex.metrics),
                nodes_count=len(ex.snapshot_nodes or []),
                edges_count=len(ex.snapshot_edges or []),
                created_at=ex.created_at
            )
        )
    return response_items


@router.delete(
    "/{workflow_id}/history",
    dependencies=[Depends(require_role("Tenant Admin", "Data Scientist"))]
)
async def prune_workflow_history(
    workflow_id: str,
    keep_latest: bool = Query(True, description="If true, keeps the latest execution as version 1; if false, deletes all versions"),
    db: AsyncSession = Depends(get_db)
):
    """
    Prunes execution history for a workflow to keep Swagger and history API lightweight.
    When keep_latest=True, retains only the latest active execution as Version 1 and removes older bloated runs.
    """
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
        .order_by(WorkflowExecution.version_number.desc(), WorkflowExecution.created_at.desc())
    )
    res = await db.execute(stmt)
    executions = res.scalars().all()

    if not executions:
        return {"workflow_id": workflow_id, "deleted_count": 0, "message": "No history versions found."}

    if keep_latest:
        latest = executions[0]
        latest.version_number = 1
        latest.run_label = "Run #1"
        deleted_count = 0
        for older in executions[1:]:
            await db.delete(older)
            deleted_count += 1
        await db.commit()
        await db.refresh(latest)
        return {
            "workflow_id": workflow_id,
            "deleted_count": deleted_count,
            "active_version": 1,
            "execution_id": latest.id,
            "message": f"Successfully deleted {deleted_count} older versions. Active version set to Version 1."
        }
    else:
        deleted_count = len(executions)
        for ex in executions:
            await db.delete(ex)
        await db.commit()
        return {
            "workflow_id": workflow_id,
            "deleted_count": deleted_count,
            "message": f"Successfully deleted all {deleted_count} versions."
        }


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
        if execution_id.startswith("unrun_") or execution_id == "unrun":
            wf_res = await db.execute(select(Workflow).where(Workflow.id == workflow_id))
            wf = wf_res.scalar_one_or_none()
            if wf:
                return WorkflowExecutionDetailResponse(
                    id=execution_id,
                    workflow_id=workflow_id,
                    version_number=1,
                    run_label="Run #1",
                    status="UNRUN",
                    total_duration_ms=0.0,
                    snapshot_nodes=wf.nodes or [],
                    snapshot_edges=wf.edges or [],
                    snapshot_node_configs=wf.node_configs or {},
                    metrics={},
                    reports={},
                    step_snapshots={},
                    logs=[],
                    created_at=wf.created_at or datetime.now(timezone.utc)
                )
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


def _convert_prediction_to_csv(response: PredictionResponse) -> str:
    trajectory = response.trajectory or response.forecast_records or []
    if trajectory:
        df_csv = pd.DataFrame(trajectory)
        return df_csv.to_csv(index=False)
    elif response.probabilities:
        df_csv = pd.DataFrame([
            {"class_label": k, "probability_pct": round(v * 100, 2), "is_predicted": (k == str(response.prediction))}
            for k, v in response.probabilities.items()
        ])
        return df_csv.to_csv(index=False)
    else:
        record = response.inferred_inputs or {}
        record[response.target_column or "prediction"] = response.prediction
        if response.confidence is not None:
            record["confidence_pct"] = round(response.confidence * (100 if response.confidence <= 1 else 1), 1)
        if response.risk_level:
            record["risk_level"] = response.risk_level
        df_csv = pd.DataFrame([record])
        return df_csv.to_csv(index=False)


@router.post("/{execution_id}/predict/export-csv")
async def export_prediction_csv(
    execution_id: str,
    payload: PredictionRequest = Body(...)
):
    """
    Executes model inference and returns the predicted output directly as a downloadable CSV file.
    """
    response = await predict_with_execution(execution_id=execution_id, payload=payload)
    csv_str = _convert_prediction_to_csv(response)
    return Response(
        content=csv_str,
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=prediction_export_{execution_id}.csv"
        }
    )



