from fastapi import APIRouter, Depends, HTTPException, status, Body
from typing import Dict, Any, List, Optional, Union
import uuid
import pandas as pd

from backend.app.engine.dag.graph import WorkflowGraph
from backend.app.engine.execution.executor import DAGExecutor, WorkflowExecutionResult
from backend.app.engine.execution.job_manager import job_manager

router = APIRouter(prefix="/workflows", tags=["Workflows & DAG Execution"])


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

