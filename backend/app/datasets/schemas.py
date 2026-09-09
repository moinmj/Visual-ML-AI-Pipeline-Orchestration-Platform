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


class ColumnSchemaItem(BaseModel):
    name: str = Field(..., description="Column name")
    data_type: str = Field(..., description="Inferred semantic data type: numeric, categorical, text, datetime, boolean")
    raw_type: str = Field(..., description="Underlying Pandas / NumPy dtype (e.g. int64, float64, object)")


class DatasetPreviewResponse(BaseModel):
    id: str
    name: str
    columns: List[str]
    column_types: Dict[str, str] = Field(default_factory=dict, description="Dictionary mapping each column name to its inferred data type")
    columns_schema: List[ColumnSchemaItem] = Field(default_factory=list, description="List of columns with rich type descriptors")
    total_rows: int
    preview_rows: List[Dict[str, Any]]


class DatasetProfileResponse(BaseModel):
    id: str
    name: str
    profile: Dict[str, Any]
