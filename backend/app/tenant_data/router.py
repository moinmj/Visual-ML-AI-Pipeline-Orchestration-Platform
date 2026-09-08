from typing import Annotated, List, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.security import TokenData, get_current_user
from backend.app.infrastructure.database.session import get_db
from backend.app.infrastructure.database.tenant_session import get_tenant_db
from backend.app.datasets.schemas import DatasetResponse
from backend.app.tenant_data.schemas import (
    EnvironmentResponse,
    IngestModelTableRequest,
    ModelResponse,
)
from backend.app.tenant_data.service import tenant_data_service

router = APIRouter(prefix="/tenant-data", tags=["Tenant Data Ingestion"])


@router.get("/environments", response_model=List[EnvironmentResponse])
async def list_environments(
    user: Annotated[TokenData, Depends(get_current_user)],
    environment_id: Optional[int] = Query(None, description="Filter to a single environment"),
    tenant_db: AsyncSession = Depends(get_tenant_db),
):
    """
    Lists the caller's tenant's environments, each with its models
    (and each model's measures/dimensions) nested inline.

    tenant_id always comes from the caller's token, never from the
    request, so one tenant can never list another tenant's environments.
    """
    return await tenant_data_service.list_environments(
        tenant_db, tenant_id=user.tenant_id, environment_id=environment_id
    )


@router.get("/environments/{environment_id}/models", response_model=List[ModelResponse])
async def list_models(
    environment_id: int,
    user: Annotated[TokenData, Depends(get_current_user)],
    tenant_db: AsyncSession = Depends(get_tenant_db),
):
    """Lists models (with measures/dimensions) for one environment owned by the caller's tenant."""
    return await tenant_data_service.list_models(tenant_db, tenant_id=user.tenant_id, environment_id=environment_id)


@router.post(
    "/environments/{environment_id}/models/{model_id}/ingest",
    response_model=DatasetResponse,
    status_code=201,
)
async def ingest_model_table(
    environment_id: int,
    model_id: int,
    payload: IngestModelTableRequest,
    user: Annotated[TokenData, Depends(get_current_user)],
    app_db: AsyncSession = Depends(get_db),
    tenant_db: AsyncSession = Depends(get_tenant_db),
):
    """
    Extracts the physical table backing a model (its
    `druid_datasource_name`) using the columns declared by that model's
    measures/dimensions, and stores the result as a platform Dataset -
    the same Dataset type CSV uploads produce, so the recipes/pipeline
    builder can use it exactly like any other ingested dataset.

    To restrict this to specific roles/permissions (e.g. only "Tenant
    Admin" or callers with a "can_ingest_data" permission), add:
        Depends(require_role("Tenant Admin"))
    or
        Depends(require_permission("can_ingest_data"))
    as an additional dependency here.
    """
    return await tenant_data_service.extract_model_table(
        app_db,
        tenant_db,
        tenant_id=user.tenant_id,
        environment_id=environment_id,
        model_id=model_id,
        datasource_name=payload.datasource_name,
        name=payload.name,
        description=payload.description,
        row_limit=payload.row_limit,
    )