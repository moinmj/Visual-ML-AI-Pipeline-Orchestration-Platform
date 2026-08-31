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
from backend.app.workflows.schemas import (
    WorkflowCreate, WorkflowUpdate, WorkflowResponse,
    AutoWireRequest, AutoWireResponse
)

router = APIRouter(prefix="/workflows", tags=["Workflows & DAG Execution"])


@router.post("/autowire", response_model=AutoWireResponse)
async def autowire_workflow(payload: AutoWireRequest):
    """
    Smart Recipe-Aware Auto-Wire API Endpoint.
    Receives an array of visual canvas nodes and computes optimal DAG edge connections.
    """
    nodes = payload.nodes
    if not nodes or len(nodes) < 2:
        return {"edges": [], "count": 0}

    configs = payload.node_configs or {}

    def get_recipe_type(n: Dict[str, Any]) -> str:
        nid = n.get("id", "")
        if "recipe_id" in n and n["recipe_id"]:
            return n["recipe_id"]
        cfg = configs.get(nid, {})
        return cfg.get("recipe_id", "")

    def get_node_pos(n: Dict[str, Any]) -> float:
        pos = n.get("position") or n.get("pos") or [0, 0]
        if isinstance(pos, (list, tuple)) and len(pos) >= 1:
            return float(pos[0])
        elif isinstance(pos, dict):
            return float(pos.get("x", 0))
        return 0.0

    def get_category_order(n: Dict[str, Any]) -> int:
        r_id = get_recipe_type(n)
        if r_id in ["cron_trigger", "webhook_trigger"]:
            return 0
        if r_id in ["csv_loader"]:
            return 1
        if r_id in ["missing_value_imputer", "feature_scaler", "categorical_encoder", "statistical_guardrail", "lag_feature_engineering"]:
            return 2
        if r_id in ["train_test_split"]:
            return 3
        if r_id in ["xgboost_trainer", "lightgbm_trainer", "catboost_trainer", "random_forest_trainer", "linear_trainer", "isolation_forest", "prophet_forecaster", "arima_forecaster"]:
            return 4
        if r_id in ["model_evaluator", "mlflow_tracker"]:
            return 5
        return 2

    # Categorize nodes
    triggers = [n for n in nodes if get_category_order(n) == 0]
    ingestions = [n for n in nodes if get_category_order(n) == 1]
    preprocessings = [n for n in nodes if get_category_order(n) == 2]
    splitters = [n for n in nodes if get_category_order(n) == 3]
    ml_models = [n for n in nodes if get_category_order(n) == 4]
    evaluators = [n for n in nodes if get_category_order(n) == 5]

    # Sort within categories by X coordinate
    triggers.sort(key=get_node_pos)
    ingestions.sort(key=get_node_pos)
    preprocessings.sort(key=get_node_pos)
    splitters.sort(key=get_node_pos)
    ml_models.sort(key=get_node_pos)
    evaluators.sort(key=get_node_pos)

    new_edges = []
    added_pairs = set()

    def add_edge(src_id: str, tgt_id: str):
        if src_id and tgt_id and src_id != tgt_id and (src_id, tgt_id) not in added_pairs:
            added_pairs.add((src_id, tgt_id))
            new_edges.append({
                "id": f"auto_{src_id}_{tgt_id}",
                "source": src_id,
                "target": tgt_id,
                "animated": True
            })

    # 1. Triggers -> First Ingestion/Data Node
    data_candidates = ingestions + preprocessings + splitters + ml_models
    first_data_node = data_candidates[0].get("id") if data_candidates else None
    for t_node in triggers:
        if first_data_node:
            add_edge(t_node.get("id"), first_data_node)

    # 2. Ingestion + Preprocessing linear chain
    data_chain = ingestions + preprocessings
    for i in range(len(data_chain) - 1):
        add_edge(data_chain[i].get("id"), data_chain[i+1].get("id"))

    last_prep_node = data_chain[-1].get("id") if data_chain else (triggers[-1].get("id") if triggers else None)

    # 3. Splitting & Models & Evaluation
    if splitters:
        main_splitter = splitters[0].get("id")
        if last_prep_node:
            add_edge(last_prep_node, main_splitter)
        
        for m in ml_models:
            add_edge(main_splitter, m.get("id"))
        for ev in evaluators:
            add_edge(main_splitter, ev.get("id"))
        for m in ml_models:
            for ev in evaluators:
                add_edge(m.get("id"), ev.get("id"))
    else:
        if ml_models:
            for m in ml_models:
                if last_prep_node:
                    add_edge(last_prep_node, m.get("id"))
                for ev in evaluators:
                    add_edge(m.get("id"), ev.get("id"))
        elif evaluators and last_prep_node:
            for ev in evaluators:
                add_edge(last_prep_node, ev.get("id"))

    # Fallback if no edges built
    all_sorted = sorted(nodes, key=lambda n: (get_category_order(n), get_node_pos(n)))
    if not new_edges and len(all_sorted) >= 2:
        for i in range(len(all_sorted) - 1):
            add_edge(all_sorted[i].get("id"), all_sorted[i+1].get("id"))

    return {"edges": new_edges, "count": len(new_edges)}


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
