from typing import Annotated, List, Optional

from fastapi import APIRouter, Depends, UploadFile, File, Form, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.security import TokenData, get_current_user
from backend.app.infrastructure.database.session import get_db
from backend.app.datasets.schemas import DatasetResponse, DatasetPreviewResponse, DatasetProfileResponse
from backend.app.datasets.service import dataset_service

# Every route here requires a valid Bearer token. The tenant is ALWAYS taken from
# the token (user.tenant_id) - never from the path, query or body - so a caller can
# only ever list, preview, register or delete their own tenant's data.
router = APIRouter(prefix="/datasets", tags=["Datasets"])

CurrentUser = Annotated[TokenData, Depends(get_current_user)]


@router.post("/upload", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    user: CurrentUser,
    file: UploadFile = File(...),
    name: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload a dataset (CSV, XLSX, JSON, Parquet).
    Automatically computes schema, column metrics, and statistical profile.
    Stored in the caller's tenant folder in Ozone (ml-workflow/t<tenant_id>/datasets/...).
    """
    return await dataset_service.upload_dataset(
        db, tenant_id=user.tenant_id, file=file, name=name, description=description
    )


@router.get("/ozone/browse")
async def browse_ozone_datasets(
    user: CurrentUser,
    prefix: str = Query("", description="Optional key prefix to filter by, relative to your tenant folder"),
):
    """
    Lists files sitting in the caller's tenant folder in Ozone, so one can be
    picked and registered as a dataset without re-uploading it.
    """
    return dataset_service.browse_ozone(tenant_id=user.tenant_id, prefix=prefix)


@router.post("/ozone/register", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
async def register_ozone_dataset(
    user: CurrentUser,
    ozone_key: str = Form(..., description="Object key returned by /ozone/browse"),
    name: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db)
):
    """
    Registers a file that's already in the caller's Ozone tenant folder as a dataset,
    without re-uploading it.
    """
    return await dataset_service.register_from_ozone(
        db, tenant_id=user.tenant_id, ozone_key=ozone_key, name=name, description=description
    )


@router.get("/", response_model=List[DatasetResponse])
async def list_datasets(user: CurrentUser, db: AsyncSession = Depends(get_db)):
    """
    List the caller's tenant's datasets (only those stored in its Ozone folder).
    """
    return await dataset_service.list_datasets(db, tenant_id=user.tenant_id)


@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(dataset_id: str, user: CurrentUser, db: AsyncSession = Depends(get_db)):
    """
    Get dataset metadata by ID (404 if it belongs to another tenant).
    """
    return await dataset_service.get_dataset(db, dataset_id, user.tenant_id)


@router.get("/{dataset_id}/preview", response_model=DatasetPreviewResponse)
async def get_dataset_preview(
    dataset_id: str,
    user: CurrentUser,
    limit: int = Query(10, ge=1, le=1000, description="Number of rows to return per page"),
    offset: int = Query(0, ge=0, description="Row offset index for pagination"),
    page: Optional[int] = Query(None, ge=1, description="Optional 1-indexed page number (e.g. page=2 with limit=100 sets offset=100)"),
    db: AsyncSession = Depends(get_db)
):
    """
    Preview rows of the dataset with full pagination support (limit, offset, and page).
    """
    actual_offset = offset
    if page is not None and offset == 0:
        actual_offset = (page - 1) * limit

    return await dataset_service.get_preview(
        db, dataset_id, tenant_id=user.tenant_id, limit=limit, offset=actual_offset
    )


@router.get("/{dataset_id}/profile", response_model=DatasetProfileResponse)
async def get_dataset_profile(dataset_id: str, user: CurrentUser, db: AsyncSession = Depends(get_db)):
    """
    Retrieve the full statistical and quality profile of a dataset.
    """
    dataset = await dataset_service.get_dataset(db, dataset_id, user.tenant_id)
    return {
        "id": dataset.id,
        "name": dataset.name,
        "profile": dataset.profile or {}
    }


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(dataset_id: str, user: CurrentUser, db: AsyncSession = Depends(get_db)):
    """
    Delete a dataset and its physical file from storage.
    """
    await dataset_service.delete_dataset(db, dataset_id, user.tenant_id)
    return None