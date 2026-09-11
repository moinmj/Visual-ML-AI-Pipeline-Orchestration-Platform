from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime


class WorkflowCreate(BaseModel):
    id: Optional[str] = Field(None, description="Optional pipeline ID (generated if not provided)")
    name: str = Field("Untitled Pipeline", description="Name of the pipeline workbook")
    description: Optional[str] = Field(None, description="Optional pipeline description")
    dataset_id: Optional[str] = Field(None, description="ID of the active dataset associated with this pipeline")
    dataset_name: Optional[str] = Field(None, description="Name of the active dataset associated with this pipeline")
    nodes: List[Dict[str, Any]] = Field(default_factory=list, description="Visual canvas nodes")
    edges: List[Dict[str, Any]] = Field(default_factory=list, description="DAG edges")
    node_configs: Dict[str, Any] = Field(default_factory=dict, description="Full recipe node configurations & parameters")
    execution_id: Optional[str] = Field(None, description="Optional execution ID from a recent run to link/attach reports")
    last_execution: Optional[Dict[str, Any]] = Field(None, description="Saved execution diagnostics, metrics, and logs")


class WorkflowUpdate(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    dataset_id: Optional[str] = None
    dataset_name: Optional[str] = None
    nodes: Optional[List[Dict[str, Any]]] = None
    edges: Optional[List[Dict[str, Any]]] = None
    node_configs: Optional[Dict[str, Any]] = None
    execution_id: Optional[str] = None
    last_execution: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None


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
    deleted_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

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
    created_at: datetime

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
    created_at: datetime

    model_config = {"from_attributes": True}


class WorkflowCompareResponse(BaseModel):
    workflow_id: str
    run_a: Dict[str, Any]
    run_b: Dict[str, Any]
    metrics_diff: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    config_diff: Dict[str, Any] = Field(default_factory=dict)

