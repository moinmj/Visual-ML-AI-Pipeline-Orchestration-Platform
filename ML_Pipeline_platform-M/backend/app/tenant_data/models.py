"""
Tenant -> Environment -> Model metadata.

There is no separate "Tenant" table: a tenant is just the `tenant_id`
claim carried in the caller's JWT (see backend/app/core/security.py),
and every EnvironmentV3 row is scoped to one. This mirrors the previous
part of the project (environment_loader.py / model_loader.py), just
translated onto this platform's async SQLAlchemy Base and its own table
names so it doesn't collide with anything from the earlier codebase.

Each ModelV3 describes a physical table (`druid_datasource_name`) that
lives in an external warehouse; `tenant_data/service.py` uses the
measures + dimensions declared here to know which columns to pull when
that table is ingested into the platform as a Dataset.
"""
import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from backend.app.infrastructure.database.tenant_session import TenantMetaBase as Base


from sqlalchemy import Enum as SAEnum

class AggregationType(str, enum.Enum):
    SUM = "SUM"
    COUNT = "COUNT"
    COUNT_DISTINCT = "COUNT_DISTINCT"
    AVG = "AVG"
    MIN = "MIN"
    MAX = "MAX"
    CALCULATED = "CALCULATED"


_agg_type = lambda: SAEnum(
    AggregationType,
    name="aggregation_type_v3",
    create_type=False,
    values_callable=lambda e: [m.value for m in e],
)


class EnvironmentV3(Base):
    __tablename__ = "environments_v3"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    tenant_id = Column(BigInteger, nullable=False, index=True)
    name = Column(Text, nullable=False)
    description = Column(Text, nullable=True)
    color_hex = Column(Text, nullable=True)
    is_active = Column(Boolean, nullable=False, default=True)
    created_by = Column(BigInteger, nullable=False, default=1)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))

    models = relationship("ModelV3", back_populates="environment", cascade="all, delete-orphan")
    dimensions = relationship("DimensionV3", back_populates="environment", cascade="all, delete-orphan")


class DimensionV3(Base):
    __tablename__ = "dimensions_v3"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    environment_id = Column(BigInteger, ForeignKey("environments_v3.id"), nullable=False, index=True)
    tenant_id = Column(BigInteger, nullable=False, default=1)
    name = Column(Text, nullable=True)
    description = Column(Text, nullable=True)
    key_column = Column(Text, nullable=True)
    text_column = Column(Text, nullable=True)
    is_active = Column(Boolean, nullable=False, default=True)
    created_by = Column(BigInteger, nullable=False, default=1)

    environment = relationship("EnvironmentV3", back_populates="dimensions")
    attributes = relationship(
        "DimensionAttributeV3", back_populates="dimension", cascade="all, delete-orphan"
    )


class DimensionAttributeV3(Base):
    __tablename__ = "dimension_attributes_v3"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    dimension_id = Column(BigInteger, ForeignKey("dimensions_v3.id"), nullable=False, index=True)
    column_name = Column(Text, nullable=True)
    display_name = Column(Text, nullable=True)
    data_type = Column(Text, nullable=False, default="string")
    order_index = Column(Integer, nullable=False, default=1)
    is_active = Column(Boolean, nullable=False, default=True)

    dimension = relationship("DimensionV3", back_populates="attributes")


class ModelV3(Base):
    __tablename__ = "models_v3"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    environment_id = Column(BigInteger, ForeignKey("environments_v3.id"), nullable=False, index=True)
    tenant_id = Column(BigInteger, nullable=False, default=1)
    name = Column(Text, nullable=False)
    description = Column(Text, nullable=True)
    fact_table_name = Column(Text, nullable=True)
    fact_table_path = Column(Text, nullable=True)
    druid_datasource_name = Column(Text, nullable=True)
    is_active = Column(Boolean, nullable=False, default=True)
    created_by = Column(BigInteger, nullable=False, default=1)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))

    environment = relationship("EnvironmentV3", back_populates="models")
    measures = relationship("ModelMeasureV3", back_populates="model", cascade="all, delete-orphan")
    model_dimensions = relationship(
        "ModelDimensionV3", back_populates="model", cascade="all, delete-orphan"
    )


class ModelMeasureV3(Base):
    __tablename__ = "model_measures_v3"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    model_id = Column(BigInteger, ForeignKey("models_v3.id"), nullable=False, index=True)
    name = Column(Text, nullable=False)
    display_name = Column(Text, nullable=True)
    source_column = Column(Text, nullable=True)
    aggregation_type = Column(_agg_type(), nullable=False, default=AggregationType.SUM)
    order_index = Column(Integer, nullable=False, default=1)
    is_active = Column(Boolean, nullable=False, default=True)

    model = relationship("ModelV3", back_populates="measures")


class ModelDimensionV3(Base):
    """Join between a Model and a Dimension (a model can use many dimensions)."""

    __tablename__ = "model_dimensions_v3"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    model_id = Column(BigInteger, ForeignKey("models_v3.id"), nullable=False, index=True)
    dimension_id = Column(BigInteger, ForeignKey("dimensions_v3.id"), nullable=False, index=True)
    # Column on the model's fact table used to join to the dimension.
    join_key_fact_column = Column(Text, nullable=True)
    order_index = Column(Integer, nullable=False, default=1)
    is_active = Column(Boolean, nullable=False, default=True)

    model = relationship("ModelV3", back_populates="model_dimensions")
    dimension = relationship("DimensionV3")