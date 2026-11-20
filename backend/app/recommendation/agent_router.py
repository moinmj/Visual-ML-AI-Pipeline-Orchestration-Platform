from typing import Annotated, Any, Dict, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.security import TokenData, get_current_user
from backend.app.infrastructure.database.session import get_db
from backend.app.datasets.service import dataset_service
from backend.app.recommendation.agent_jobs import (
    AgentQueueFullError,
    agent_job_manager,
)
from backend.app.recommendation.agent_service import (
    AgentCapacityError,
    AgentConfigurationError,
    AgentPlanningError,
    agent_provider_available,
    build_dataset_context,
    build_dataset_context_from_profile,
    create_agent_plan,
)


router = APIRouter(
    prefix="/agent/pipelines",
    tags=["Recommendation Agent"],
    dependencies=[Depends(get_current_user)],
)
CurrentUser = Annotated[TokenData, Depends(get_current_user)]


class AgentPlanRequest(BaseModel):
    dataset_id: str = Field(..., min_length=1)
    request: str = Field(..., min_length=3, max_length=4000)
    target_column: Optional[str] = None
    task_type: Optional[Literal["classification", "regression"]] = None


class AgentRunRequest(AgentPlanRequest):
    max_attempts: int = Field(5, ge=1, le=5)


async def _load_agent_dataset(dataset_id: str, db: AsyncSession, tenant_id: int):
    dataset = await dataset_service.get_dataset(db, dataset_id, tenant_id)
    try:
        from backend.app.infrastructure.storage.storage_manager import storage_manager

        dataframe = await run_in_threadpool(
            storage_manager.read_dataframe,
            dataset.storage_path,
            tenant_id,
        )
    except (OSError, ValueError, ImportError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not read the uploaded dataset: {exc}",
        ) from exc

    if not all(isinstance(column, str) for column in dataframe.columns):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The agent planner requires dataset column names to be strings.",
        )
    if dataframe.empty or len(dataframe.columns) < 2:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Supervised planning requires a non-empty dataset with a target and at least one feature column.",
        )
    return dataframe


@router.post("/plan", response_model=Dict[str, Any])
async def plan_pipeline_with_agent(
    payload: AgentPlanRequest,
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
):
    """Create a reviewable supervised workflow using registered recipes."""
    dataset = await dataset_service.get_dataset(db, payload.dataset_id, user.tenant_id)
    dataset_context = await run_in_threadpool(
        build_dataset_context_from_profile,
        dataset.profile or {},
    )
    if not dataset_context["columns"]:
        dataframe = await _load_agent_dataset(payload.dataset_id, db, user.tenant_id)
        dataset_context = await run_in_threadpool(build_dataset_context, dataframe)
    if not all(isinstance(column, str) for column in dataset_context["columns"]):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The agent planner requires dataset column names to be strings.",
        )
    if payload.target_column and payload.target_column not in dataset_context["columns"]:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Target column '{payload.target_column}' is not present in the dataset profile.",
        )

    try:
        return await create_agent_plan(
            dataset_id=payload.dataset_id,
            dataset_context=dataset_context,
            user_request=payload.request,
            requested_target=payload.target_column,
            requested_task_type=payload.task_type,
        )
    except AgentPlanningError as exc:
        if isinstance(exc, AgentCapacityError):
            code = status.HTTP_429_TOO_MANY_REQUESTS
        elif isinstance(exc, AgentConfigurationError):
            code = status.HTTP_503_SERVICE_UNAVAILABLE
        else:
            code = status.HTTP_502_BAD_GATEWAY
        raise HTTPException(status_code=code, detail=str(exc)) from exc


@router.post("/run", response_model=Dict[str, Any], status_code=status.HTTP_202_ACCEPTED)
async def run_pipeline_with_agent(
    payload: AgentRunRequest,
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
):
    """Queue an autonomous supervised-model search and return without waiting for training."""
    if not agent_provider_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No recommendation-agent provider is configured. Set OPENAI_API_KEY or GROQ_API_KEY.",
        )
    dataset = await dataset_service.get_dataset(db, payload.dataset_id, user.tenant_id)
    try:
        job_id = agent_job_manager.submit(
            dataset_id=payload.dataset_id,
            dataset_path=dataset.storage_path,
            user_request=payload.request,
            max_attempts=payload.max_attempts,
            requested_target=payload.target_column,
            requested_task_type=payload.task_type,
            tenant_id=user.tenant_id,
            user_id=user.sub,
        )
    except AgentQueueFullError as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(exc),
            headers={"Retry-After": "5"},
        ) from exc

    return {
        "job_id": job_id,
        "status": "PENDING",
        "status_endpoint": f"/api/v1/agent/pipelines/jobs/{job_id}",
    }


@router.get("/jobs/{job_id}", response_model=Dict[str, Any])
async def get_agent_job(job_id: str, user: CurrentUser):
    """Get status and the result of an asynchronously running agent job."""
    job = agent_job_manager.get_job(job_id, tenant_id=user.tenant_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent job '{job_id}' was not found or has expired.",
        )
    return job