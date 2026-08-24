from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any, List
import uuid
from backend.app.engine.dag.graph import WorkflowGraph
from backend.app.engine.execution.executor import DAGExecutor, WorkflowExecutionResult

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
    Execute a full workflow DAG end-to-end and return real-time node outputs, logs, and evaluation metrics.
    """
    execution_id = str(uuid.uuid4())
    result = DAGExecutor.execute_workflow(execution_id=execution_id, workflow=workflow)
    return result
