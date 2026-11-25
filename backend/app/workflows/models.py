import uuid
from datetime import datetime, timezone
from typing import Any
from sqlalchemy import event, inspect as sa_inspect
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

    # True when the graph has both a Start and an End Anchor, i.e. it can be used through
    # "Add Workflow". Set on every save (see workflows/sub_workflows.py); NULL = not yet computed.
    is_sub_workflow = Column(Boolean, nullable=True, index=True)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class WorkflowVersion(Base):
    """
    Saved definition of a sub-process workflow. A new version is written whenever a workflow
    with Start and End Anchors is saved with a changed graph, so an "Add Workflow" node can pin
    a version and keep running it after the sub-process is edited.
    """
    __tablename__ = "workflow_versions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    workflow_id = Column(String(36), ForeignKey("workflows.id", ondelete="CASCADE"), nullable=False, index=True)
    tenant_id = Column(Integer, nullable=True, index=True)
    user_id = Column(String(255), nullable=True, index=True)
    version_number = Column(Integer, nullable=False)
    name = Column(String(255), nullable=False)
    definition_hash = Column(String(64), nullable=False)

    nodes = Column(JSON, nullable=False, default=list)
    edges = Column(JSON, nullable=False, default=list)
    node_configs = Column(JSON, nullable=False, default=dict)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


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

# ── Credentials are never stored in plaintext ─────────────────────────────
# Every write of a workflow, a sub-process version or an execution snapshot goes through
# protect_graph_secrets (core/crypto.py): inline passwords / tokens / keys are encrypted at any
# depth and nodes using a vaulted credential lose their copy of the secret. This catches every
# path that writes these rows, including ones that build the payload from a graph whose
# credentials were decrypted in memory for a run (scheduler, /execute, imports, saved reports).
# Explicit save paths still run sanitize_vaulted_node_configs first (it restores masked values
# from the stored row); this is the backstop. Core / raw SQL writes are not covered.

def _graph_changed(target: Any, *attrs: str) -> bool:
    state = sa_inspect(target)
    return any(state.attrs[a].history.has_changes() for a in attrs)


def _protect_graph_columns(target: Any, nodes_attr: str, configs_attr: str, is_insert: bool) -> None:
    from backend.app.core.crypto import protect_graph_secrets
    if not is_insert and not _graph_changed(target, nodes_attr, configs_attr):
        return
    nodes, configs = getattr(target, nodes_attr), getattr(target, configs_attr)
    new_nodes, new_configs = protect_graph_secrets(nodes, configs)
    if new_nodes != nodes:
        setattr(target, nodes_attr, new_nodes)
    if new_configs != configs:
        setattr(target, configs_attr, new_configs)


def _protect_workflow(target: "Workflow", is_insert: bool) -> None:
    from backend.app.core.crypto import protect_execution_secrets
    _protect_graph_columns(target, "nodes", "node_configs", is_insert)
    if is_insert or _graph_changed(target, "last_execution"):
        last = target.last_execution
        safe = protect_execution_secrets(last)
        if safe != last:
            target.last_execution = safe


@event.listens_for(Workflow, "before_insert")
def _workflow_before_insert(mapper, connection, target):
    _protect_workflow(target, True)


@event.listens_for(Workflow, "before_update")
def _workflow_before_update(mapper, connection, target):
    _protect_workflow(target, False)


@event.listens_for(WorkflowVersion, "before_insert")
def _version_before_insert(mapper, connection, target):
    _protect_graph_columns(target, "nodes", "node_configs", True)


@event.listens_for(WorkflowVersion, "before_update")
def _version_before_update(mapper, connection, target):
    _protect_graph_columns(target, "nodes", "node_configs", False)


@event.listens_for(WorkflowExecution, "before_insert")
def _execution_before_insert(mapper, connection, target):
    _protect_graph_columns(target, "snapshot_nodes", "snapshot_node_configs", True)


@event.listens_for(WorkflowExecution, "before_update")
def _execution_before_update(mapper, connection, target):
    _protect_graph_columns(target, "snapshot_nodes", "snapshot_node_configs", False)
