from fastapi import APIRouter, Depends, HTTPException, status, Body
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
    Save / create a pipeline workbook with exact node configs, parameters, layout, and edges.
    """
    wf = Workflow(
        id=str(uuid.uuid4()),
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
    db: AsyncSession = Depends(get_db)
):
    """
    List all saved pipeline workbooks.
    """
    result = await db.execute(select(Workflow).order_by(Workflow.updated_at.desc()))
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
async def update_workflow(
    workflow_id: str,
    payload: WorkflowUpdate,
    db: AsyncSession = Depends(get_db)
):
    """
    Update an existing saved pipeline workbook.
    """
    result = await db.execute(select(Workflow).where(Workflow.id == workflow_id))
    wf = result.scalar_one_or_none()
    if not wf:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow workbook '{workflow_id}' not found."
        )

    if payload.name is not None:
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

    await db.commit()
    await db.refresh(wf)
    return wf


@router.delete("/{workflow_id}", status_code=status.HTTP_200_OK)
async def delete_workflow(
    workflow_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a saved pipeline workbook by ID.
    """
    result = await db.execute(select(Workflow).where(Workflow.id == workflow_id))
    wf = result.scalar_one_or_none()
    if not wf:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow workbook '{workflow_id}' not found."
        )

    await db.delete(wf)
    await db.commit()
    return {"status": "DELETED", "message": f"Workflow workbook '{workflow_id}' deleted successfully."}


# -------------------------------------------------------------
# DAG VALIDATION & EXECUTION ENDPOINTS
# -------------------------------------------------------------

@router.post("/validate", response_model=Dict[str, Any])
async def validate_workflow(workflow: WorkflowGraph):
    """
    Validate a workflow graph for cycle detection, valid node connectivity, and schema requirements.
    """
    errors = workflow.validate_graph()
    return {
        "valid": len(errors) == 0,
        "errors": errors
    }


@router.post("/execute", response_model=WorkflowExecutionResult)
async def execute_workflow(workflow: WorkflowGraph):
    """
    Execute a full workflow DAG end-to-end synchronously.
    """
    execution_id = str(uuid.uuid4())
    result = DAGExecutor.execute_workflow(execution_id=execution_id, workflow=workflow)
    return result


@router.post("/async-execute")
async def submit_async_workflow(workflow: WorkflowGraph):
    """
    Submit a workflow for asynchronous background execution (n8n/Boomi worker pool pattern).
    Returns immediately with a job_id for status polling.
    """
    job_id = job_manager.submit_job(workflow=workflow, trigger_type="api_async")
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
