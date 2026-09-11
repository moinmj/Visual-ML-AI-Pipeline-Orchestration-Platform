import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, DateTime, JSON, Text, Boolean, Integer, Float, ForeignKey
from backend.app.infrastructure.database.session import Base


class Workflow(Base):
    __tablename__ = "workflows"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)

    # Associated Dataset
    dataset_id = Column(String(36), nullable=True)
    dataset_name = Column(String(255), nullable=True)

    # Full Graph Payload
    nodes = Column(JSON, nullable=False, default=list)
    edges = Column(JSON, nullable=False, default=list)
    node_configs = Column(JSON, nullable=False, default=dict)
    last_execution = Column(JSON, nullable=True)

    # Soft Delete & Governance
    is_active = Column(Boolean, nullable=False, default=True)
    deleted_at = Column(DateTime, nullable=True)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class WorkflowExecution(Base):
    __tablename__ = "workflow_executions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    workflow_id = Column(String(36), ForeignKey("workflows.id", ondelete="CASCADE"), nullable=False, index=True)
    version_number = Column(Integer, nullable=False, default=1)
    run_label = Column(String(255), nullable=True)

    # Status & Timing
    status = Column(String(50), nullable=False, default="SUCCESS")
    total_duration_ms = Column(Float, nullable=True, default=0.0)

    # Frozen Graph Snapshot at Run Time
    snapshot_nodes = Column(JSON, nullable=False, default=list)
    snapshot_edges = Column(JSON, nullable=False, default=list)
    snapshot_node_configs = Column(JSON, nullable=False, default=dict)

    # Execution Outputs & Diagnostics
    metrics = Column(JSON, nullable=True)
    reports = Column(JSON, nullable=True)
    step_snapshots = Column(JSON, nullable=True)
    logs = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

