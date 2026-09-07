import asyncio
import uuid
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.app.core.config import settings
from backend.app.core.exceptions import NotFoundException, ValidationException
from backend.app.core.logging import logger
from backend.app.datasets.models import Dataset
from backend.app.profiling.profiler import DataProfiler
from backend.app.infrastructure.storage.storage_manager import storage_manager
from backend.app.tenant_data.models import (
    DimensionV3,
    EnvironmentV3,
    ModelDimensionV3,
    ModelV3,
)

_source_engine_cache: Dict[str, Any] = {}


def _to_sync_url(url: str) -> str:
    """Strip the async driver so plain SQLAlchemy `create_engine` can use it."""
    if not url:
        return url
    return (
        url.replace("+aiosqlite", "")
        .replace("postgresql+asyncpg://", "postgresql://")
        .replace("+asyncpg", "")
    )


def _get_source_engine():
    """
    Lazily builds (and caches) a plain/sync SQLAlchemy engine pointed at the
    warehouse that hosts models' physical tables. Defaults to the tenant metadata
    DB (P_DATABASE_URL) or platform DB (DATABASE_URL) so ingestion works out of
    the box in dev/demo setups.
    """
    raw_url = settings.SOURCE_DATABASE_URL
    # If SOURCE_DATABASE_URL is a Druid URL (handled via HTTP REST), use P_DATABASE_URL or DATABASE_URL
    if not raw_url or "druid" in raw_url.lower():
        raw_url = settings.P_DATABASE_URL or settings.DATABASE_URL
    url = _to_sync_url(raw_url)
    if url not in _source_engine_cache:
        connect_args = {"check_same_thread": False} if "sqlite" in url else {}
        _source_engine_cache[url] = create_engine(url, connect_args=connect_args)
    return _source_engine_cache[url]


def _extract_druid_table(table_or_ds: str, columns: List[str], row_limit: Optional[int]) -> Optional[pd.DataFrame]:
    """Attempts to query Druid's native SQL endpoint over HTTP REST."""
    import requests
    druid_url = settings.SOURCE_DATABASE_URL
    if not druid_url or "druid" not in druid_url.lower():
        return None

    http_url = druid_url.replace("druid://", "http://").rstrip("/")
    if not http_url.endswith("/druid/v2/sql"):
        http_url = f"{http_url}/druid/v2/sql"

    try:
        r = requests.post(http_url, json={"query": "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES"}, timeout=10)
        if r.status_code != 200:
            return None
        druid_tables = {row["TABLE_NAME"].lower(): row["TABLE_NAME"] for row in r.json() if "TABLE_NAME" in row}
        target_ds = druid_tables.get(table_or_ds.lower())
        if not target_ds:
            return None

        # Datasource exists in Druid - query it!
        quoted_cols = ", ".join(f'"{c}"' for c in columns) if columns else "*"
        sql = f'SELECT {quoted_cols} FROM "{target_ds}"'
        if row_limit:
            sql += f" LIMIT {int(row_limit)}"

        r_data = requests.post(http_url, json={"query": sql}, timeout=60)
        if r_data.status_code == 200:
            data = r_data.json()
            logger.info(f"Extracted {len(data)} rows from Druid datasource '{target_ds}'.")
            return pd.DataFrame(data)
        else:
            logger.warning(f"Druid query for datasource '{target_ds}' failed: {r_data.text}")
    except Exception as e:
        logger.warning(f"Druid extraction failed for '{table_or_ds}': {e}")

    return None


def _resolve_physical_table(engine, candidate_names: List[Optional[str]]) -> Optional[str]:
    """Resolves an encoded datasource/model name to the physical table in the warehouse."""
    from sqlalchemy import inspect
    try:
        insp = inspect(engine)
        existing_tables = set(insp.get_table_names())
    except Exception as e:
        logger.warning(f"Could not inspect table names from source engine: {e}")
        return None

    for candidate in candidate_names:
        if not candidate:
            continue
        candidate_str = str(candidate).strip()
        # 1. Exact match
        if candidate_str in existing_tables:
            return candidate_str
        if candidate_str.lower() in existing_tables:
            return candidate_str.lower()
        # 2. Extract after __ delimiter (e.g. t1_e172_...__local_db_t1_u2_dataingestion_sales_2025)
        parts = candidate_str.split("__")
        last_part = parts[-1]
        if last_part in existing_tables:
            return last_part
        if last_part.lower() in existing_tables:
            return last_part.lower()
        # 3. Subparts by underscore
        sub_parts = last_part.split("_")
        for i in range(len(sub_parts)):
            sub = "_".join(sub_parts[i:])
            if sub in existing_tables:
                return sub
            if sub.lower() in existing_tables:
                return sub.lower()
            if f"v3_{sub}" in existing_tables:
                return f"v3_{sub}"
            if f"v3_{sub.lower()}" in existing_tables:
                return f"v3_{sub.lower()}"
        # 4. Dot-separated path (e.g. local.db.t1_u2_postgresqlmodels_v3_fact_sales)
        if "." in candidate_str:
            last_dot = candidate_str.split(".")[-1]
            if last_dot in existing_tables:
                return last_dot
            sub_parts = last_dot.split("_")
            for i in range(len(sub_parts)):
                sub = "_".join(sub_parts[i:])
                if sub in existing_tables:
                    return sub
                if f"v3_{sub}" in existing_tables:
                    return f"v3_{sub}"

    return None


def _build_model_dict(model: ModelV3) -> Dict[str, Any]:
    """Shapes a ModelV3 ORM object into the API/response dict, mirroring
    the column-collection logic from the reference model_loader.py."""
    measures = [
        {
            "name": m.name,
            "display_name": m.display_name,
            "source_column": m.source_column,
            "aggregation_type": m.aggregation_type.value if hasattr(m.aggregation_type, "value") else str(m.aggregation_type or "SUM"),
        }
        for m in (model.measures or [])
        if getattr(m, "is_active", True)
    ]

    dimensions = []
    for md in (model.model_dimensions or []):
        if not getattr(md, "is_active", True):
            continue
        columns: List[str] = []

        # Fact table join column
        if md.join_key_fact_column:
            columns.append(md.join_key_fact_column)

        # Dimension key column
        if md.dimension and md.dimension.key_column:
            if md.dimension.key_column not in columns:
                columns.append(md.dimension.key_column)

        # Dimension attributes
        if md.dimension and hasattr(md.dimension, "attributes") and md.dimension.attributes:
            for attr in md.dimension.attributes:
                if getattr(attr, "is_active", True) and attr.column_name and attr.column_name not in columns:
                    columns.append(attr.column_name)

        dimensions.append(
            {
                "dimension_name": md.dimension.name if md.dimension else None,
                "key_column": md.dimension.key_column if md.dimension else None,
                "text_column": md.dimension.text_column if md.dimension else None,
                "columns": columns,
                "join_key_fact_column": md.join_key_fact_column,
                "join_key_dim_column": md.dimension.key_column if md.dimension else None,
            }
        )

    return {
        "id": model.id,
        "name": model.name,
        "description": model.description,
        "druid_datasource_name": model.druid_datasource_name,
        "dimensions": dimensions,
        "measures": measures,
    }


class TenantDataService:
    @staticmethod
    async def list_environments(
        tenant_db: AsyncSession, tenant_id: int, environment_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        filters = [EnvironmentV3.tenant_id == tenant_id, EnvironmentV3.is_active == True]  # noqa: E712
        # Filtering by BOTH tenant_id and id means an environment_id
        # belonging to a different tenant simply comes back empty instead
        # of leaking another tenant's environment.
        if environment_id is not None:
            filters.append(EnvironmentV3.id == environment_id)

        result = await tenant_db.execute(
            select(EnvironmentV3)
            .options(
                selectinload(EnvironmentV3.models).selectinload(ModelV3.measures),
                selectinload(EnvironmentV3.models)
                .selectinload(ModelV3.model_dimensions)
                .selectinload(ModelDimensionV3.dimension)
                .selectinload(DimensionV3.attributes),
            )
            .where(*filters)
        )
        environments = result.scalars().unique().all()

        return [
            {
                "environment_id": env.id,
                "name": env.name,
                "description": env.description,
                "models": [_build_model_dict(m) for m in env.models if m.is_active],
            }
            for env in environments
        ]

    @staticmethod
    async def _get_environment_for_tenant(
        tenant_db: AsyncSession, tenant_id: int, environment_id: int
    ) -> EnvironmentV3:
        result = await tenant_db.execute(
            select(EnvironmentV3).where(
                EnvironmentV3.id == environment_id,
                EnvironmentV3.tenant_id == tenant_id,
                EnvironmentV3.is_active == True,  # noqa: E712
            )
        )
        env = result.scalar_one_or_none()
        if not env:
            raise NotFoundException("Environment", environment_id)
        return env

    @staticmethod
    async def list_models(tenant_db: AsyncSession, tenant_id: int, environment_id: int) -> List[Dict[str, Any]]:
        # Ensures the environment actually belongs to this tenant before
        # returning anything about the models inside it.
        await TenantDataService._get_environment_for_tenant(tenant_db, tenant_id, environment_id)

        result = await tenant_db.execute(
            select(ModelV3)
            .options(
                selectinload(ModelV3.measures),
                selectinload(ModelV3.model_dimensions)
                .selectinload(ModelDimensionV3.dimension)
                .selectinload(DimensionV3.attributes),
            )
            .where(ModelV3.environment_id == environment_id, ModelV3.is_active == True)  # noqa: E712
        )
        models = result.scalars().unique().all()
        return [_build_model_dict(m) for m in models]

    @staticmethod
    async def _get_model_scoped(
        tenant_db: AsyncSession, tenant_id: int, environment_id: int, model_id: int
    ) -> ModelV3:
        # Two-step scoping: environment must belong to the tenant, then the
        # model must belong to that environment. This is what prevents a
        # caller from reaching another tenant's model just by guessing IDs.
        await TenantDataService._get_environment_for_tenant(tenant_db, tenant_id, environment_id)

        result = await tenant_db.execute(
            select(ModelV3)
            .options(
                selectinload(ModelV3.measures),
                selectinload(ModelV3.model_dimensions)
                .selectinload(ModelDimensionV3.dimension)
                .selectinload(DimensionV3.attributes),
            )
            .where(
                ModelV3.id == model_id,
                ModelV3.environment_id == environment_id,
                ModelV3.is_active == True,  # noqa: E712
            )
        )
        model = result.scalar_one_or_none()
        if not model:
            raise NotFoundException("Model", model_id)
        return model

    @staticmethod
    def _extract_table_sync(table_name: str, columns: List[str], row_limit: Optional[int]) -> pd.DataFrame:
        # 1. Check if datasource exists in Druid
        df_druid = _extract_druid_table(table_name, columns, row_limit)
        if df_druid is not None:
            return df_druid

        # 2. Extract from relational warehouse / Postgres
        from sqlalchemy import inspect, text
        engine = _get_source_engine()
        try:
            insp = inspect(engine)
            available_cols = {col["name"] for col in insp.get_columns(table_name)}
        except Exception:
            available_cols = set()

        valid_cols = [c for c in columns if c in available_cols] if available_cols and columns else []
        prep = engine.dialect.identifier_preparer
        quoted_cols = ", ".join(prep.quote(c) for c in valid_cols) if valid_cols else "*"
        query = f"SELECT {quoted_cols} FROM {prep.quote(table_name)}"
        if row_limit:
            query += f" LIMIT {int(row_limit)}"
        with engine.connect() as conn:
            return pd.read_sql(text(query), con=conn)

    @staticmethod
    async def extract_model_table(
        app_db: AsyncSession,
        tenant_db: AsyncSession,
        tenant_id: int,
        environment_id: int,
        model_id: int,
        name: Optional[str] = None,
        description: Optional[str] = None,
        row_limit: Optional[int] = None,
    ) -> Dataset:
        """
        app_db: session on the platform's own DB (DATABASE_URL) - where the
                resulting Dataset row gets written.
        tenant_db: session on the tenant metadata DB (P_DATABASE_URL) - where
                Environment/Model/Measure/Dimension are read from.
        The physical table's rows themselves are pulled separately via
        SOURCE_DATABASE_URL / P_DATABASE_URL (see _extract_table_sync / _get_source_engine).
        """
        model = await TenantDataService._get_model_scoped(tenant_db, tenant_id, environment_id, model_id)
        model_dict = _build_model_dict(model)

        # Whitelist of columns actually declared on this model - these come
        # from our own trusted metadata tables, not client input, so it's
        # safe to interpolate them (quoted) into the generated SQL.
        columns: List[str] = []
        for measure in model_dict["measures"]:
            if measure["source_column"] and measure["source_column"] not in columns:
                columns.append(measure["source_column"])
        for dim in model_dict["dimensions"]:
            for col in dim["columns"]:
                if col and col not in columns:
                    columns.append(col)

        engine = _get_source_engine()
        candidate_names = [
            model.druid_datasource_name,
            getattr(model, "fact_table_name", None),
            getattr(model, "fact_table_path", None),
            model.name,
        ]
        resolved_table = _resolve_physical_table(engine, candidate_names)
        if not resolved_table:
            if model.druid_datasource_name:
                resolved_table = model.druid_datasource_name
            else:
                raise ValidationException(
                    f"No physical database table could be resolved for model '{model.name}' (id={model_id}). "
                    f"Checked candidates: {[c for c in candidate_names if c]}."
                )

        table_name = resolved_table

        try:
            df = await asyncio.to_thread(
                TenantDataService._extract_table_sync, table_name, columns, row_limit
            )
        except Exception as e:
            logger.error(f"Failed to extract table '{table_name}' for model {model_id}: {e}")
            raise ValidationException(
                f"Failed to extract source table '{table_name}' for model '{model.name}': {e}"
            )

        if df.empty and len(df.columns) == 0:
            raise ValidationException(
                f"Source table '{table_name}' returned no columns. Check the model's "
                f"measures/dimensions and SOURCE_DATABASE_URL configuration."
            )

        dataset_id = str(uuid.uuid4())
        dataset_name = name or f"{model.name} ({model_dict['druid_datasource_name']})"
        storage_rel_path = f"datasets/{dataset_id}_{model.name}.parquet"

        storage_manager.save_dataframe(df, storage_rel_path, file_format="parquet")
        profile_data = DataProfiler.profile_dataframe(df)
        profile_data["source"] = {
            "type": "tenant_model_table",
            "tenant_id": tenant_id,
            "environment_id": environment_id,
            "model_id": model_id,
            "model_name": model.name,
            "table": table_name,
        }

        dataset = Dataset(
            id=dataset_id,
            name=dataset_name,
            description=description or f"Ingested from model '{model.name}' (table '{table_name}').",
            file_name=f"{model.name}.parquet",
            file_format="parquet",
            file_size_bytes=int(df.memory_usage(deep=True).sum()),
            storage_path=storage_rel_path,
            row_count=profile_data["row_count"],
            column_count=profile_data["column_count"],
            quality_score=profile_data["quality_score"],
            profile=profile_data,
        )

        app_db.add(dataset)
        await app_db.commit()
        await app_db.refresh(dataset)
        logger.info(
            f"Ingested model '{model.name}' (tenant={tenant_id}, env={environment_id}) "
            f"as dataset {dataset.id} with {dataset.row_count} rows."
        )
        return dataset


tenant_data_service = TenantDataService()