from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


class DatasetBase(BaseModel):
    name: str = Field(..., description="User-friendly name of the dataset")
    description: Optional[str] = Field(None, description="Optional dataset description")


class DatasetCreate(DatasetBase):
    pass


class DatasetResponse(DatasetBase):
    id: str
    file_name: str
    file_format: str
    file_size_bytes: int
    storage_path: str
    row_count: int
    column_count: int
    quality_score: float
    created_at: datetime
    updated_at: datetime

    model_config = {
        "from_attributes": True
    }


class DatasetPreviewResponse(BaseModel):
    id: str
    name: str
    columns: List[str]
    total_rows: int
    preview_rows: List[Dict[str, Any]]


class DatasetProfileResponse(BaseModel):
    id: str
    name: str
    profile: Dict[str, Any]
