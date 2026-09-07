from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class MeasureResponse(BaseModel):
    name: str
    display_name: Optional[str] = None
    source_column: Optional[str] = None
    aggregation_type: Optional[str] = "SUM"


class DimensionResponse(BaseModel):
    dimension_name: Optional[str] = None
    key_column: Optional[str] = None
    text_column: Optional[str] = None
    columns: List[str] = Field(default_factory=list)
    join_key_fact_column: Optional[str] = None
    join_key_dim_column: Optional[str] = None


class ModelResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    druid_datasource_name: Optional[str] = None
    dimensions: List[DimensionResponse] = Field(default_factory=list)
    measures: List[MeasureResponse] = Field(default_factory=list)


class EnvironmentResponse(BaseModel):
    environment_id: int
    name: str
    description: Optional[str] = None
    models: List[ModelResponse] = Field(default_factory=list)


class IngestModelTableRequest(BaseModel):
    name: Optional[str] = Field(None, description="Optional display name for the created dataset")
    description: Optional[str] = Field(None, description="Optional dataset description")
    row_limit: Optional[int] = Field(
        None, ge=1, le=1_000_000, description="Optional cap on rows pulled from the source table"
    )


class IngestModelTableResponse(BaseModel):
    dataset_id: str
    dataset_name: str
    row_count: int
    column_count: int
    source: Dict[str, Any]
