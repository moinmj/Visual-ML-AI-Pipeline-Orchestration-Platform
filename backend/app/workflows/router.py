from fastapi import APIRouter, Depends, HTTPException, status, Body, Query
from typing import Dict, Any, List, Optional, Union
import uuid
import pandas as pd
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from backend.app.infrastructure.database.session import get_db
from backend.app.engine.dag.graph import WorkflowGraph
from backend.app.engine.execution.executor import DAGExecutor, WorkflowExecutionResult
from backend.app.engine.execution.job_manager import job_manager
from backend.app.workflows.models import Workflow
from backend.app.workflows.schemas import WorkflowCreate, WorkflowUpdate, WorkflowResponse
from backend.app.engine.inference import (
    PipelineInferencer,
    PredictionRequest,
    PredictionResponse,
    InferenceSchemaResponse,
    FeatureSchemaItem
)

router = APIRouter(prefix="/workflows", tags=["Workflows & DAG Execution"])


# -------------------------------------------------------------
# WORKFLOW PERSISTENCE & WORKBOOK RETRIEVAL ENDPOINTS
# -------------------------------------------------------------

@router.post("/", response_model=WorkflowResponse, status_code=status.HTTP_201_CREATED)
async def save_workflow(
    payload: WorkflowCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Save / create / upsert a pipeline workbook with exact node configs, parameters, layout, and edges.
    """
    target_id = payload.id or str(uuid.uuid4())
    result = await db.execute(select(Workflow).where(Workflow.id == target_id))
    wf = result.scalar_one_or_none()

    if wf:
        # Update existing
        if payload.name:
            wf.name = payload.name
        if payload.description is not None:
            wf.description = payload.description
        if payload.nodes is not None:
            wf.nodes = payload.nodes
        if payload.edges is not None:
            wf.edges = payload.edges
        if payload.node_configs is not None:
            wf.node_configs = payload.node_configs
        wf.updated_at = datetime.now(timezone.utc)
    else:
        # Create new
        wf = Workflow(
            id=target_id,
            name=payload.name,
            description=payload.description,
            nodes=payload.nodes,
            edges=payload.edges,
            node_configs=payload.node_configs
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


@router.get("/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow(
    workflow_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Retrieve a specific saved pipeline workbook by ID with full exact configuration and params.
    """
    result = await db.execute(select(Workflow).where(Workflow.id == workflow_id))
    wf = result.scalar_one_or_none()
    if not wf:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow workbook '{workflow_id}' not found."
        )
    return wf


@router.put("/{workflow_id}", response_model=WorkflowResponse)
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

    if wf:
        # Update existing
        if payload.name:
            wf.name = payload.name
        if payload.description is not None:
            wf.description = payload.description
        if payload.nodes is not None:
            wf.nodes = payload.nodes
        if payload.edges is not None:
            wf.edges = payload.edges
        if payload.node_configs is not None:
            wf.node_configs = payload.node_configs
        wf.is_active = True
        wf.deleted_at = None
        wf.updated_at = datetime.now(timezone.utc)
    else:
        # Create new
        wf = Workflow(
            id=target_id,
            name=payload.name or "Untitled Pipeline",
            description=payload.description,
            nodes=payload.nodes or [],
            edges=payload.edges or [],
            node_configs=payload.node_configs or {},
            is_active=True
        )
        db.add(wf)

    await db.commit()
    await db.refresh(wf)
    return wf


@router.delete("/{workflow_id}", status_code=status.HTTP_200_OK)
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


@router.post("/{workflow_id}/restore", response_model=WorkflowResponse)
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
    return {"status": "DELETED", "message": f"Workflow workbook '{workflow_id}' deleted successfully."}


# -------------------------------------------------------------
# DAG VALIDATION & EXECUTION ENDPOINTS
# -------------------------------------------------------------

@router.post("/validate", response_model=Dict[str, Any])
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


@router.post("/execute", response_model=WorkflowExecutionResult)
async def execute_workflow(
    workflow: WorkflowGraph,
    include_node_outputs: bool = Query(False, description="Opt-in to include full raw data of every node (default: false for lean response)")
):
    """
    Execute a full workflow DAG end-to-end synchronously.
    Returns lean response by default (node_results, step_snapshots 5-row previews, final_metrics, logs).
    """
    execution_id = str(uuid.uuid4())
    result = DAGExecutor.execute_workflow(
        execution_id=execution_id,
        workflow=workflow,
        include_node_outputs=include_node_outputs
    )
    return result


@router.post("/async-execute")
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


@router.post("/predict", response_model=PredictionResponse)
async def predict_latest(
    execution_id: str = Query(..., description="The execution ID of the trained pipeline"),
    payload: PredictionRequest = Body(...)
):
    """
    Convenience endpoint for live predictions passing execution_id as a query parameter.
    """
    return await predict_with_execution(execution_id=execution_id, payload=payload)


