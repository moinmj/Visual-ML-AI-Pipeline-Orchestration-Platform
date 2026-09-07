import asyncio
import uuid
from typing import Any, Dict, List, Optional, Tuple

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


def _get_druid_datasources_for_model(
    tenant_id: int,
    environment_id: int,
    model_name: str,
    druid_datasource_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Discovers all actual Druid datasources matching this tenant, environment, and model."""
    import requests
    druid_url = settings.SOURCE_DATABASE_URL
    if not druid_url or "druid" not in druid_url.lower():
        return []
    http_url = druid_url.replace("druid://", "http://").rstrip("/")
    if not http_url.endswith("/druid/v2/sql"):
        http_url = f"{http_url}/druid/v2/sql"

    try:
        r = requests.post(
            http_url,
            json={"query": "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = 'druid'"},
            timeout=10,
        )
        if r.status_code != 200:
            return []
        all_tables = [row["TABLE_NAME"] for row in r.json() if "TABLE_NAME" in row]
    except Exception as e:
        logger.warning(f"Could not fetch Druid tables: {e}")
        return []

    clean_model = model_name.lower().replace(" ", "_").replace("-", "_")
    prefix = f"t{tenant_id}_e{environment_id}_{clean_model}__"
    general_prefix = f"t{tenant_id}_e{environment_id}_"

    matches = []
    for t in all_tables:
        tl = t.lower()
        if druid_datasource_name and tl == druid_datasource_name.lower():
            matches.append(t)
        elif tl.startswith(prefix):
            matches.append(t)
        elif clean_model in tl and tl.startswith(general_prefix):
            matches.append(t)

    unique_matches = []
    for m in matches:
        if m not in unique_matches:
            unique_matches.append(m)

    results = []
    for m in unique_matches:
        cnt = None
        try:
            r_cnt = requests.post(http_url, json={"query": f'SELECT COUNT(*) as cnt FROM "{m}"'}, timeout=5)
            if r_cnt.status_code == 200 and r_cnt.json():
                cnt = r_cnt.json()[0].get("cnt")
        except Exception:
            pass
        results.append({"datasource_name": m, "row_count": cnt})

    results.sort(key=lambda x: (x["row_count"] is not None, x["row_count"] or 0), reverse=True)
    return results


def _extract_druid_table(
    candidate_tables: List[str],
    columns: Optional[List[str]] = None,
    row_limit: Optional[int] = None,
) -> Optional[Tuple[pd.DataFrame, str]]:
    """
    Attempts to query Druid's native SQL endpoint over HTTP REST.
    Tries candidate datasource names in order, inspects actual Druid columns to avoid
    'Column not found' errors, and returns (DataFrame, resolved_datasource_name).
    """
    import requests
    druid_url = settings.SOURCE_DATABASE_URL
    if not druid_url or "druid" not in druid_url.lower():
        return None

    http_url = druid_url.replace("druid://", "http://").rstrip("/")
    if not http_url.endswith("/druid/v2/sql"):
        http_url = f"{http_url}/druid/v2/sql"

    try:
        r = requests.post(
            http_url,
            json={"query": "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = 'druid'"},
            timeout=10,
        )
        if r.status_code != 200:
            return None
        druid_tables = {row["TABLE_NAME"].lower(): row["TABLE_NAME"] for row in r.json() if "TABLE_NAME" in row}
    except Exception as e:
        logger.warning(f"Could not connect to Druid: {e}")
        return None

    target_ds = None
    for cand in candidate_tables:
        if not cand:
            continue
        cand_str = str(cand).strip()
        if cand_str.lower() in druid_tables:
            target_ds = druid_tables[cand_str.lower()]
            break

    if not target_ds:
        return None

    try:
        r_cols = requests.post(
            http_url,
            json={"query": f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = 'druid' AND TABLE_NAME = '{target_ds}'"},
            timeout=10,
        )
        actual_cols = []
        if r_cols.status_code == 200:
            actual_cols = [row["COLUMN_NAME"] for row in r_cols.json() if "COLUMN_NAME" in row]

        valid_cols = [c for c in (columns or []) if c in actual_cols and c != "__time"]

        if valid_cols:
            quoted_cols = ", ".join(f'"{c}"' for c in valid_cols)
        else:
            cols_to_pull = [c for c in actual_cols if c != "__time"]
            quoted_cols = ", ".join(f'"{c}"' for c in cols_to_pull) if cols_to_pull else "*"

        sql = f'SELECT {quoted_cols} FROM "{target_ds}"'
        if row_limit:
            sql += f" LIMIT {int(row_limit)}"

        r_data = requests.post(http_url, json={"query": sql}, timeout=60)
        if r_data.status_code != 200:
            logger.warning(f"Druid query failed ({r_data.text}), falling back to SELECT * FROM \"{target_ds}\"")
            sql_fallback = f'SELECT * FROM "{target_ds}"'
            if row_limit:
                sql_fallback += f" LIMIT {int(row_limit)}"
            r_data = requests.post(http_url, json={"query": sql_fallback}, timeout=60)

        if r_data.status_code == 200:
            data = r_data.json()
            logger.info(f"Successfully extracted {len(data)} rows from Druid datasource '{target_ds}'.")
            df = pd.DataFrame(data)
            if "__time" in df.columns and len(df) > 0:
                first_val = str(df["__time"].iloc[0])
                if "2000-01-01" in first_val and df["__time"].nunique() <= 1:
                    df = df.drop(columns=["__time"])
            return df, target_ds
        else:
            logger.warning(f"Druid query for datasource '{target_ds}' failed: {r_data.text}")
    except Exception as e:
        logger.warning(f"Druid extraction failed for '{target_ds}': {e}")

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
    def get_druid_datasources(
        tenant_id: int,
        environment_id: int,
        model_name: str,
        druid_datasource_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return _get_druid_datasources_for_model(
            tenant_id=tenant_id,
            environment_id=environment_id,
            model_name=model_name,
            druid_datasource_name=druid_datasource_name,
        )

    @staticmethod
    def _extract_table_sync(candidate_tables: List[str], columns: List[str], row_limit: Optional[int]) -> Tuple[pd.DataFrame, str]:
        # 1. Check if candidate datasource exists in Druid
        druid_res = _extract_druid_table(candidate_tables, columns, row_limit)
        if druid_res is not None:
            return druid_res

        # 2. Extract from relational warehouse / Postgres
        from sqlalchemy import inspect, text
        engine = _get_source_engine()
        resolved_table = _resolve_physical_table(engine, candidate_tables)
        if not resolved_table:
            for c in candidate_tables:
                if c:
                    resolved_table = str(c)
                    break
        if not resolved_table:
            raise ValidationException(f"Could not resolve physical table from candidates: {candidate_tables}")

        try:
            insp = inspect(engine)
            available_cols = {col["name"] for col in insp.get_columns(resolved_table)}
        except Exception:
            available_cols = set()

        valid_cols = [c for c in columns if c in available_cols] if available_cols and columns else []
        prep = engine.dialect.identifier_preparer
        quoted_cols = ", ".join(prep.quote(c) for c in valid_cols) if valid_cols else "*"
        query = f"SELECT {quoted_cols} FROM {prep.quote(resolved_table)}"
        if row_limit:
            query += f" LIMIT {int(row_limit)}"
        with engine.connect() as conn:
            return pd.read_sql(text(query), con=conn), resolved_table

    @staticmethod
    async def extract_model_table(
        app_db: AsyncSession,
        tenant_db: AsyncSession,
        tenant_id: int,
        environment_id: int,
        model_id: int,
        datasource_name: Optional[str] = None,
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

        # Whitelist of columns actually declared on this model
        columns: List[str] = []
        for measure in model_dict["measures"]:
            if measure["source_column"] and measure["source_column"] not in columns:
                columns.append(measure["source_column"])
        for dim in model_dict["dimensions"]:
            for col in dim["columns"]:
                if col and col not in columns:
                    columns.append(col)

        candidate_names: List[str] = []
        if datasource_name:
            candidate_names.append(datasource_name)

        # Discover matching Druid datasources for this model
        try:
            discovered = _get_druid_datasources_for_model(
                tenant_id=tenant_id,
                environment_id=environment_id,
                model_name=model.name,
                druid_datasource_name=model.druid_datasource_name,
            )
            for d in discovered:
                ds = d["datasource_name"]
                if ds not in candidate_names:
                    candidate_names.append(ds)
        except Exception as e:
            logger.warning(f"Could not discover Druid datasources: {e}")

        for cand in [
            model.druid_datasource_name,
            getattr(model, "fact_table_name", None),
            getattr(model, "fact_table_path", None),
            model.name,
        ]:
            if cand and cand not in candidate_names:
                candidate_names.append(cand)

        try:
            df, resolved_table = await asyncio.to_thread(
                TenantDataService._extract_table_sync, candidate_names, columns, row_limit
            )
        except Exception as e:
            logger.error(f"Failed to extract table for model {model_id}: {e}")
            raise ValidationException(
                f"Failed to extract source table for model '{model.name}': {e}"
            )

        if df.empty and len(df.columns) == 0:
            raise ValidationException(
                f"Source table '{resolved_table}' returned no columns. Check the model's "
                f"measures/dimensions and SOURCE_DATABASE_URL configuration."
            )

        dataset_id = str(uuid.uuid4())
        dataset_name = name or f"{model.name} ({resolved_table})"
        storage_rel_path = f"datasets/{dataset_id}_{model.name}.parquet"

        storage_manager.save_dataframe(df, storage_rel_path, file_format="parquet")
        profile_data = DataProfiler.profile_dataframe(df)
        profile_data["source"] = {
            "type": "tenant_model_table",
            "tenant_id": tenant_id,
            "environment_id": environment_id,
            "model_id": model_id,
            "model_name": model.name,
            "table": resolved_table,
            "datasource_name": resolved_table,
        }

        dataset = Dataset(
            id=dataset_id,
            name=dataset_name,
            description=description or f"Ingested from model '{model.name}' (datasource '{resolved_table}').",
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
            f"as dataset {dataset.id} with {dataset.row_count} rows from '{resolved_table}'."
        )
        return dataset


tenant_data_service = TenantDataService()