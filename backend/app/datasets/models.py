import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, Float, DateTime, JSON, Text, Boolean
from backend.app.infrastructure.database.session import Base


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))

    # Owning tenant (taken from the caller's token, never from the request).
    # Every read/list/delete filters on it. Rows created before tenant scoping
    # existed have NULL here, so they match no tenant and are invisible to all.
    tenant_id = Column(Integer, nullable=True, index=True)

    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    file_name = Column(String(255), nullable=False)
    file_format = Column(String(50), nullable=False)  # csv, xlsx, json, parquet
    file_size_bytes = Column(Integer, nullable=False)
    storage_path = Column(String(512), nullable=False)
    
    # Quick metadata
    row_count = Column(Integer, default=0)
    column_count = Column(Integer, default=0)
    quality_score = Column(Float, default=100.0)

    # Statistical profile JSON
    profile = Column(JSON, nullable=True)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))