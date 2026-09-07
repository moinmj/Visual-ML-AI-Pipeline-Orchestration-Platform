"""
Synchronous helper functions for interacting with tenant metadata and ingesting
model tables, designed for use in Streamlit UI, notebooks, and background jobs.
"""
import asyncio
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from backend.app.infrastructure.database.session import get_db
from backend.app.infrastructure.database.tenant_session import get_tenant_db
from backend.app.infrastructure.storage.storage_manager import storage_manager
from backend.app.tenant_data.service import tenant_data_service


def fetch_tenant_environments(tenant_id: int = 1) -> List[Dict[str, Any]]:
    """Fetches all environments (and their nested models) for a given tenant."""
    async def _fetch():
        async for db in get_tenant_db():
            return await tenant_data_service.list_environments(db, tenant_id=tenant_id)
        return []

    return asyncio.run(_fetch())


def fetch_tenant_models(tenant_id: int, environment_id: int) -> List[Dict[str, Any]]:
    """Fetches models with measures and dimensions for a specific environment."""
    async def _fetch():
        async for db in get_tenant_db():
            return await tenant_data_service.list_models(
                db, tenant_id=tenant_id, environment_id=environment_id
            )
        return []

    return asyncio.run(_fetch())


def ingest_tenant_model(
    tenant_id: int,
    environment_id: int,
    model_id: int,
    row_limit: Optional[int] = None,
    custom_name: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Extracts the physical warehouse/database table backing a tenant's model,
    registers it as a platform Dataset (parquet), and returns the loaded DataFrame
    along with its metadata dictionary.
    """
    async def _ingest():
        async for app_db in get_db():
            async for tenant_db in get_tenant_db():
                dataset = await tenant_data_service.extract_model_table(
                    app_db=app_db,
                    tenant_db=tenant_db,
                    tenant_id=tenant_id,
                    environment_id=environment_id,
                    model_id=model_id,
                    name=custom_name,
                    row_limit=row_limit,
                )
                df = storage_manager.read_dataframe(dataset.storage_path)
                meta = {
                    "dataset_id": dataset.id,
                    "dataset_name": dataset.name,
                    "row_count": dataset.row_count,
                    "column_count": dataset.column_count,
                    "storage_path": dataset.storage_path,
                }
                return df, meta
            break
        raise RuntimeError("Failed to establish database sessions")

    return asyncio.run(_ingest())
