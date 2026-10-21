import uuid
import pandas as pd
from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete, func, or_
from fastapi import UploadFile

from backend.app.datasets.models import Dataset
try:
    from backend.app.infrastructure.storage.storage_manager import storage_manager
except ImportError:
    from backend.app.infrastructure.storage import storage_manager
from backend.app.profiling.profiler import DataProfiler
from backend.app.core.exceptions import NotFoundException, ValidationException
from backend.app.core.logging import logger


class DatasetService:
    @staticmethod
    async def upload_dataset(
        db: AsyncSession,
        tenant_id: int,
        file: UploadFile,
        name: Optional[str] = None,
        description: Optional[str] = None
    ) -> Dataset:
        # Validate format
        file_name = file.filename or "dataset.csv"
        ext = file_name.split(".")[-1].lower()
        if ext not in ["csv", "xlsx", "xls", "json", "parquet"]:
            raise ValidationException(f"Unsupported file format: '.{ext}'. Allowed: CSV, XLSX, JSON, Parquet.")

        content = await file.read()
        file_size = len(content)
        if file_size == 0:
            raise ValidationException("The uploaded file is empty.")

        dataset_id = str(uuid.uuid4())
        dataset_name = name or file_name.rsplit(".", 1)[0]
        storage_rel_path = f"datasets/{dataset_id}_{file_name}"

        # 1. Save file to storage (returns the local path, or an s3://bucket/key
        #    Ozone path when STORAGE_BACKEND="ozone" -- must be used from here on,
        #    not the raw key we asked it to save under).
        storage_path = await storage_manager.save_file(content, storage_rel_path, tenant_id)

        # 2. Read DataFrame and perform automatic statistical profiling
        try:
            df = storage_manager.read_dataframe(storage_path, tenant_id)
            profile_data = DataProfiler.profile_dataframe(df)
        except Exception as e:
            # Clean up on failure
            storage_manager.delete_file(storage_path, tenant_id)
            logger.error(f"Error parsing uploaded tabular file: {e}")
            raise ValidationException(f"Failed to parse and profile dataset: {str(e)}")

        # 3. Create DB Record
        dataset = Dataset(
            id=dataset_id,
            tenant_id=tenant_id,
            name=dataset_name,
            description=description,
            file_name=file_name,
            file_format=ext,
            file_size_bytes=file_size,
            storage_path=storage_path,
            row_count=profile_data["row_count"],
            column_count=profile_data["column_count"],
            quality_score=profile_data["quality_score"],
            profile=profile_data
        )

        db.add(dataset)
        await db.commit()
        await db.refresh(dataset)
        logger.info(f"Dataset successfully created: {dataset.id} ({dataset.name}) with {dataset.row_count} rows.")
        return dataset

    @staticmethod
    async def list_datasets(
        db: AsyncSession,
        tenant_id: int,
        search: Optional[str] = None,
        file_format: Optional[str] = None,
        sort: str = "newest",
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> Tuple[List[Dataset], int]:
        """
        The tenant's datasets (only those stored in its Ozone folder), optionally
        searched, filtered, sorted and paged. Returns (page_of_datasets, total_matches).

        search: case-insensitive "contains" match on dataset ID, name, file name and
        description; several words must ALL match (e.g. "sales 2024").
        """
        from backend.app.core.config import settings
        from backend.app.infrastructure.storage.ozone_client import tenant_prefix
        folder_path = f"s3://{settings.OZONE_BUCKET}/{tenant_prefix(tenant_id)}"
        local_prefix = f"t{int(tenant_id)}/"

        query = (
            select(Dataset)
            .where(Dataset.tenant_id == tenant_id)
            .where(or_(
                Dataset.storage_path.startswith(folder_path, autoescape=True),
                Dataset.storage_path.startswith(local_prefix, autoescape=True),
                ~Dataset.storage_path.startswith("s3://")
            ))
        )

        for term in (search or "").split():
            term = term.lower()
            query = query.where(or_(
                func.lower(Dataset.id).contains(term, autoescape=True),
                func.lower(Dataset.name).contains(term, autoescape=True),
                func.lower(Dataset.file_name).contains(term, autoescape=True),
                func.lower(func.coalesce(Dataset.description, "")).contains(term, autoescape=True),
            ))

        if file_format:
            query = query.where(func.lower(Dataset.file_format) == file_format.lower().lstrip("."))

        total = (await db.execute(select(func.count()).select_from(query.subquery()))).scalar() or 0

        order = {
            "newest": Dataset.created_at.desc(),
            "oldest": Dataset.created_at.asc(),
            "name": func.lower(Dataset.name).asc(),
            "name_desc": func.lower(Dataset.name).desc(),
            "largest": Dataset.file_size_bytes.desc(),
            "rows": Dataset.row_count.desc(),
        }.get(sort, Dataset.created_at.desc())
        query = query.order_by(order, Dataset.id)

        if offset:
            query = query.offset(offset)
        if limit:
            query = query.limit(limit)

        result = await db.execute(query)
        return list(result.scalars().all()), total

    @staticmethod
    def browse_ozone(
        tenant_id: int,
        scope: str = "ml_workflow",
        folder: str = "all",
        media_type: Optional[str] = None,
        prefix: str = "",
        include_folders: bool = False,
        limit: int = 1000,
        meta: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Lists objects in the caller's tenant space in Ozone, formatted identically
        to DatasetResponse so frontend dataset list components can render them.
        """
        from datetime import datetime, timezone
        from backend.app.core.config import settings

        try:
            raw_objects = storage_manager.list_ozone_objects(
                tenant_id=tenant_id,
                prefix=prefix,
                scope=scope,
                folder=folder,
                media_type=media_type,
                max_keys=limit,
                include_folders=include_folders,
                meta=meta,
            )
        except ValueError as e:
            raise ValidationException(str(e))

        now_utc = datetime.now(timezone.utc)
        results = []
        for obj in raw_objects:
            key = obj["key"]
            file_name = obj["file_name"]
            ext = obj["file_format"]
            m_type = obj.get("media_type")
            is_folder = m_type == "folder"
            stem = file_name if is_folder or "." not in file_name else file_name.rsplit(".", 1)[0]

            preview_url = None
            if m_type == "image":
                preview_url = f"/api/v1/datasets/media/preview?key={key}"

            last_mod = obj.get("last_modified")
            if last_mod:
                try:
                    dt = datetime.fromisoformat(last_mod.replace("Z", "+00:00"))
                except Exception:
                    dt = now_utc
            else:
                dt = now_utc

            folder_label = f"ml-workflow/{folder}" if scope == "ml_workflow" else "all_tenant_data"
            results.append({
                "id": key,
                "name": stem,
                "description": f"Ozone {'folder' if is_folder else 'object'} [{folder_label}]",
                "file_name": file_name,
                "file_format": ext,
                "file_size_bytes": obj.get("size_bytes", 0),
                "storage_path": f"s3://{settings.OZONE_BUCKET}/{key}",
                "row_count": 0,
                "column_count": 0,
                "quality_score": 100.0,
                "created_at": dt,
                "updated_at": dt,
                "media_type": m_type,
                "preview_url": preview_url,
                "key": key,
                "relative_path": obj.get("relative_path", key),
                "scope": scope,
            })
        return results

    @staticmethod
    async def register_from_ozone(
        db: AsyncSession,
        tenant_id: int,
        ozone_key: str,
        name: Optional[str] = None,
        description: Optional[str] = None
    ) -> Dataset:
        """
        Registers an existing Ozone object as a dataset, without re-uploading it.
        Reads it once to compute the same profile/schema an upload would get.
        """
        ext = ozone_key.rsplit(".", 1)[-1].lower() if "." in ozone_key else ""
        if ext not in ["csv", "xlsx", "xls", "json", "parquet"]:
            raise ValidationException(f"Unsupported file format: '.{ext}'. Allowed: CSV, XLSX, JSON, Parquet.")

        from backend.app.core.config import settings
        from backend.app.infrastructure.storage.ozone_client import is_in_tenant_prefix, tenant_prefix
        if not is_in_tenant_prefix(ozone_key, tenant_id):
            raise ValidationException(f"Only files under '{tenant_prefix(tenant_id)}' can be registered.")
        storage_path = f"s3://{settings.OZONE_BUCKET}/{ozone_key}"
        dataset_id = str(uuid.uuid4())
        file_name = ozone_key.rsplit("/", 1)[-1]
        dataset_name = name or file_name.rsplit(".", 1)[0]

        try:
            df = storage_manager.read_dataframe(storage_path, tenant_id)
            profile_data = DataProfiler.profile_dataframe(df)
        except Exception as e:
            logger.error(f"Error reading/profiling Ozone object '{ozone_key}': {e}")
            raise ValidationException(f"Failed to read and profile '{ozone_key}': {str(e)}")

        from backend.app.infrastructure.storage.ozone_client import get_s3_client
        file_size = get_s3_client().head_object(Bucket=settings.OZONE_BUCKET, Key=ozone_key)["ContentLength"]

        dataset = Dataset(
            id=dataset_id,
            tenant_id=tenant_id,
            name=dataset_name,
            description=description,
            file_name=file_name,
            file_format=ext,
            file_size_bytes=file_size,
            storage_path=storage_path,
            row_count=profile_data["row_count"],
            column_count=profile_data["column_count"],
            quality_score=profile_data["quality_score"],
            profile=profile_data
        )
        db.add(dataset)
        await db.commit()
        await db.refresh(dataset)
        logger.info(f"Dataset registered from Ozone: {dataset.id} ({dataset.name}) -> {storage_path}")
        return dataset

    @staticmethod
    async def get_dataset(db: AsyncSession, dataset_id: str, tenant_id: int) -> Dataset:
        # A dataset that belongs to another tenant is reported as "not found" (404),
        # so its existence is never revealed to the caller.
        result = await db.execute(
            select(Dataset).where(Dataset.id == dataset_id, Dataset.tenant_id == tenant_id)
        )
        dataset = result.scalar_one_or_none()
        if not dataset:
            raise NotFoundException("Dataset", dataset_id)
        return dataset

    @staticmethod
    async def get_preview(db: AsyncSession, dataset_id: str, tenant_id: int, limit: int = 10, offset: int = 0) -> Dict[str, Any]:
        dataset = await DatasetService.get_dataset(db, dataset_id, tenant_id)
        df = storage_manager.read_dataframe(dataset.storage_path, tenant_id)

        total_rows = int(len(df))
        start_idx = max(0, offset)
        end_idx = min(total_rows, start_idx + limit)
        preview_df = df.iloc[start_idx:end_idx].replace({float("nan"): None, float("inf"): None, float("-inf"): None})

        from backend.app.profiling.profiler import DataProfiler
        column_types: Dict[str, str] = {}
        columns_schema: List[Dict[str, str]] = []
        for col in df.columns:
            inferred = DataProfiler._infer_column_type(df[col])
            raw_dtype = str(df[col].dtype)
            column_types[str(col)] = inferred
            columns_schema.append({
                "name": str(col),
                "data_type": inferred,
                "raw_type": raw_dtype
            })

        inferred_target = (dataset.profile or {}).get("inferred_target_column")
        if not inferred_target:
            profile_res = DataProfiler.profile_dataframe(df)
            inferred_target = profile_res.get("inferred_target_column")

        return {
            "id": dataset.id,
            "name": dataset.name,
            "columns": list(df.columns),
            "column_types": column_types,
            "columns_schema": columns_schema,
            "inferred_target_column": inferred_target,
            "total_rows": total_rows,
            "limit": limit,
            "offset": start_idx,
            "preview_rows": preview_df.to_dict(orient="records")
        }

    @staticmethod
    async def delete_dataset(db: AsyncSession, dataset_id: str, tenant_id: int) -> bool:
        dataset = await DatasetService.get_dataset(db, dataset_id, tenant_id)
        storage_manager.delete_file(dataset.storage_path, tenant_id)
        await db.delete(dataset)
        await db.commit()
        logger.info(f"Deleted dataset {dataset_id}")
        return True


dataset_service = DatasetService()