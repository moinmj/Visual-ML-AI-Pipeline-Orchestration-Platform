from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime


class WorkflowCreate(BaseModel):
    id: Optional[str] = Field(None, description="Optional pipeline ID (generated if not provided)")
    name: str = Field("Untitled Pipeline", description="Name of the pipeline workbook")
    description: Optional[str] = Field(None, description="Optional pipeline description")
    nodes: List[Dict[str, Any]] = Field(default_factory=list, description="Visual canvas nodes")
    edges: List[Dict[str, Any]] = Field(default_factory=list, description="DAG edges")
    node_configs: Dict[str, Any] = Field(default_factory=dict, description="Full recipe node configurations & parameters")


class WorkflowUpdate(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    nodes: Optional[List[Dict[str, Any]]] = None
    edges: Optional[List[Dict[str, Any]]] = None
    node_configs: Optional[Dict[str, Any]] = None


class WorkflowResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    nodes: List[Dict[str, Any]] = Field(default_factory=list)
    edges: List[Dict[str, Any]] = Field(default_factory=list)
    node_configs: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
