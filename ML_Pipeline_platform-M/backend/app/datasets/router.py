from fastapi import APIRouter, Depends, UploadFile, File, Form, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional

from backend.app.infrastructure.database.session import get_db
from backend.app.datasets.schemas import DatasetResponse, DatasetPreviewResponse, DatasetProfileResponse
from backend.app.datasets.service import dataset_service

router = APIRouter(prefix="/datasets", tags=["Datasets"])


@router.post("/upload", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload a dataset (CSV, XLSX, JSON, Parquet).
    Automatically computes schema, column metrics, and statistical profile.
    """
    return await dataset_service.upload_dataset(db, file=file, name=name, description=description)


@router.get("/", response_model=List[DatasetResponse])
async def list_datasets(db: AsyncSession = Depends(get_db)):
    """
    List all uploaded datasets.
    """
    return await dataset_service.list_datasets(db)


@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(dataset_id: str, db: AsyncSession = Depends(get_db)):
    """
    Get dataset metadata by ID.
    """
    return await dataset_service.get_dataset(db, dataset_id)


@router.get("/{dataset_id}/preview", response_model=DatasetPreviewResponse)
async def get_dataset_preview(
    dataset_id: str,
    limit: int = Query(10, ge=1, le=100),
    db: AsyncSession = Depends(get_db)
):
    """
    Preview the top N rows of the dataset.
    """
    return await dataset_service.get_preview(db, dataset_id, limit=limit)


@router.get("/{dataset_id}/profile", response_model=DatasetProfileResponse)
async def get_dataset_profile(dataset_id: str, db: AsyncSession = Depends(get_db)):
    """
    Retrieve the full statistical and quality profile of a dataset.
    """
    dataset = await dataset_service.get_dataset(db, dataset_id)
    return {
        "id": dataset.id,
        "name": dataset.name,
        "profile": dataset.profile or {}
    }


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(dataset_id: str, db: AsyncSession = Depends(get_db)):
    """
    Delete a dataset and its physical file from storage.
    """
    await dataset_service.delete_dataset(db, dataset_id)
    return None
