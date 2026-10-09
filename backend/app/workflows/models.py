import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, DateTime, JSON, Text, Boolean, Integer, Float, ForeignKey
from backend.app.infrastructure.database.session import Base


class Workflow(Base):
    __tablename__ = "workflows"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    # Owning tenant (from the caller's token). Rows are only ever visible to this tenant.
    tenant_id = Column(Integer, nullable=True, index=True)
    # User who created it (the token's "sub"); filled in automatically on create, never changed.
    user_id = Column(String(255), nullable=True, index=True)
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
    deleted_at = Column(DateTime(timezone=True), nullable=True)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class WorkflowExecution(Base):
    __tablename__ = "workflow_executions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    workflow_id = Column(String(36), ForeignKey("workflows.id", ondelete="CASCADE"), nullable=False, index=True)
    # Owning tenant (same as the parent workflow's).
    tenant_id = Column(Integer, nullable=True, index=True)
    # User who started this run (the token's "sub"); filled in automatically.
    user_id = Column(String(255), nullable=True, index=True)
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

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))