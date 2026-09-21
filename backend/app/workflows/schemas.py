from enum import Enum
from pydantic import BaseModel, Field, BeforeValidator, PlainSerializer
from typing import List, Dict, Any, Optional, Union, Annotated
from datetime import datetime, timezone


def _ensure_utc_datetime(v: Any) -> Optional[datetime]:
    if v is None:
        return None
    if isinstance(v, str):
        v = datetime.fromisoformat(v.replace("Z", "+00:00"))
    if isinstance(v, datetime):
        if v.tzinfo is None:
            v = v.replace(tzinfo=timezone.utc)
        else:
            v = v.astimezone(timezone.utc)
    return v


UTCDateTime = Annotated[
    datetime,
    BeforeValidator(_ensure_utc_datetime),
    PlainSerializer(lambda dt: dt.isoformat().replace("+00:00", "Z"), return_type=str, when_used="json")
]

OptUTCDateTime = Annotated[
    Optional[datetime],
    BeforeValidator(_ensure_utc_datetime),
    PlainSerializer(lambda dt: dt.isoformat().replace("+00:00", "Z") if dt else None, return_type=Optional[str], when_used="json")
]


class WorkflowStatusFilter(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    UNRUN = "unrun"


class WorkflowCreate(BaseModel):
    id: Optional[str] = Field(None, description="Optional pipeline ID (generated if not provided)")
    workflow_id: Optional[str] = Field(None, description="Alias for id")
    pipeline_id: Optional[str] = Field(None, description="Alias for id")
    name: str = Field("Untitled Pipeline", description="Name of the pipeline workbook")
    workflow_name: Optional[str] = Field(None, description="Alias for name")
    description: Optional[str] = Field(None, description="Optional pipeline description")
    dataset_id: Optional[str] = Field(None, description="ID of the active dataset associated with this pipeline")
    dataset_name: Optional[str] = Field(None, description="Name of the active dataset associated with this pipeline")
    nodes: List[Dict[str, Any]] = Field(default_factory=list, description="Visual canvas nodes")
    edges: List[Dict[str, Any]] = Field(default_factory=list, description="DAG edges")
    node_configs: Dict[str, Any] = Field(default_factory=dict, description="Full recipe node configurations & parameters")
    execution_id: Optional[str] = Field(None, description="Optional execution ID from a recent run to link/attach reports")
    executionId: Optional[str] = Field(None, description="CamelCase alias for execution_id")
    run_id: Optional[str] = Field(None, description="Alias for execution_id")
    job_id: Optional[str] = Field(None, description="Alias for execution_id")
    last_execution: Optional[Dict[str, Any]] = Field(None, description="Saved execution diagnostics, metrics, and logs")

    model_config = {"extra": "allow"}


class WorkflowUpdate(BaseModel):
    id: Optional[str] = None
    workflow_id: Optional[str] = None
    pipeline_id: Optional[str] = None
    name: Optional[str] = None
    workflow_name: Optional[str] = None
    description: Optional[str] = None
    dataset_id: Optional[str] = None
    dataset_name: Optional[str] = None
    nodes: Optional[List[Dict[str, Any]]] = None
    edges: Optional[List[Dict[str, Any]]] = None
    node_configs: Optional[Dict[str, Any]] = None
    execution_id: Optional[str] = None
    executionId: Optional[str] = None
    run_id: Optional[str] = None
    job_id: Optional[str] = None
    last_execution: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None

    model_config = {"extra": "allow"}


class WorkflowResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    dataset_id: Optional[str] = None
    dataset_name: Optional[str] = None
    nodes: List[Dict[str, Any]] = Field(default_factory=list)
    edges: List[Dict[str, Any]] = Field(default_factory=list)
    node_configs: Dict[str, Any] = Field(default_factory=dict)
    last_execution: Optional[Dict[str, Any]] = None
    is_active: bool = True
    deleted_at: OptUTCDateTime = None
    created_at: UTCDateTime
    updated_at: UTCDateTime
    model_config = {"from_attributes": True}


class WorkflowListItemResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    dataset_id: Optional[str] = None
    dataset_name: Optional[str] = None
    is_active: bool = True
    deleted_at: OptUTCDateTime = None
    created_at: UTCDateTime
    updated_at: UTCDateTime
    last_execution_status: Optional[str] = Field(None, description="Latest run status: SUCCESS, FAILED, or None if never executed")
    last_execution_id: Optional[str] = Field(None, description="Execution ID of the latest run")
    nodes_count: int = Field(0, description="Total number of nodes in this pipeline")
    edges_count: int = Field(0, description="Total number of edge connections in this pipeline")

    model_config = {"from_attributes": True}


WorkflowSummaryResponse = WorkflowListItemResponse


class WorkflowListPaginatedResponse(BaseModel):
    total_records: int = Field(..., description="Total number of workflows matching filter")
    skip: int = Field(0, description="Number of records skipped")
    limit: int = Field(10, description="Page limit")
    current_page: int = Field(1, description="1-indexed current page number")
    total_pages: int = Field(1, description="Total number of pages")
    data: List[WorkflowListItemResponse] = Field(default_factory=list, description="List of lightweight workflow summaries")

    model_config = {"from_attributes": True}


class WorkflowExecutionSummaryResponse(BaseModel):
    id: str
    workflow_id: str
    version_number: int
    run_label: Optional[str] = None
    status: str
    total_duration_ms: Optional[float] = 0.0
    metrics: Optional[Dict[str, Any]] = None
    nodes_count: int = 0
    edges_count: int = 0
    created_at: UTCDateTime

    model_config = {"from_attributes": True}


class WorkflowExecutionDetailResponse(BaseModel):
    id: str
    workflow_id: str
    version_number: int
    run_label: Optional[str] = None
    status: str
    total_duration_ms: Optional[float] = 0.0
    snapshot_nodes: List[Dict[str, Any]] = Field(default_factory=list)
    snapshot_edges: List[Dict[str, Any]] = Field(default_factory=list)
    snapshot_node_configs: Dict[str, Any] = Field(default_factory=dict)
    metrics: Optional[Dict[str, Any]] = None
    reports: Optional[Dict[str, Any]] = None
    step_snapshots: Optional[Dict[str, Any]] = None
    logs: Optional[Union[List[str], List[Dict[str, Any]]]] = None
    created_at: UTCDateTime

    model_config = {"from_attributes": True}


class WorkflowCompareResponse(BaseModel):
    workflow_id: str
    run_a: Dict[str, Any]
    run_b: Dict[str, Any]
    metrics_diff: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    config_diff: Dict[str, Any] = Field(default_factory=dict)

