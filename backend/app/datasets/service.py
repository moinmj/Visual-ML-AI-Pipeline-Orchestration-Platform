import uuid
import pandas as pd
from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
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
        storage_path = await storage_manager.save_file(content, storage_rel_path)

        # 2. Read DataFrame and perform automatic statistical profiling
        try:
            df = storage_manager.read_dataframe(storage_path)
            profile_data = DataProfiler.profile_dataframe(df)
        except Exception as e:
            # Clean up on failure
            storage_manager.delete_file(storage_path)
            logger.error(f"Error parsing uploaded tabular file: {e}")
            raise ValidationException(f"Failed to parse and profile dataset: {str(e)}")

        # 3. Create DB Record
        dataset = Dataset(
            id=dataset_id,
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
    async def list_datasets(db: AsyncSession) -> List[Dataset]:
        result = await db.execute(select(Dataset).order_by(Dataset.created_at.desc()))
        return list(result.scalars().all())

    @staticmethod
    def browse_ozone(prefix: str = "") -> List[Dict[str, Any]]:
        """
        Lists objects already in the Ozone datalake bucket, so the user can pick
        an existing file instead of uploading a new one.
        """
        return storage_manager.list_ozone_objects(prefix=prefix)

    @staticmethod
    async def register_from_ozone(
        db: AsyncSession,
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
        storage_path = f"s3://{settings.OZONE_BUCKET}/{ozone_key}"
        dataset_id = str(uuid.uuid4())
        file_name = ozone_key.rsplit("/", 1)[-1]
        dataset_name = name or file_name.rsplit(".", 1)[0]

        try:
            df = storage_manager.read_dataframe(storage_path)
            profile_data = DataProfiler.profile_dataframe(df)
        except Exception as e:
            logger.error(f"Error reading/profiling Ozone object '{ozone_key}': {e}")
            raise ValidationException(f"Failed to read and profile '{ozone_key}': {str(e)}")

        from backend.app.infrastructure.storage.ozone_client import get_s3_client
        file_size = get_s3_client().head_object(Bucket=settings.OZONE_BUCKET, Key=ozone_key)["ContentLength"]

        dataset = Dataset(
            id=dataset_id,
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
    async def get_dataset(db: AsyncSession, dataset_id: str) -> Dataset:
        result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
        dataset = result.scalar_one_or_none()
        if not dataset:
            raise NotFoundException("Dataset", dataset_id)
        return dataset

    @staticmethod
    async def get_preview(db: AsyncSession, dataset_id: str, limit: int = 10, offset: int = 0) -> Dict[str, Any]:
        dataset = await DatasetService.get_dataset(db, dataset_id)
        df = storage_manager.read_dataframe(dataset.storage_path)

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
    async def delete_dataset(db: AsyncSession, dataset_id: str) -> bool:
        dataset = await DatasetService.get_dataset(db, dataset_id)
        storage_manager.delete_file(dataset.storage_path)
        await db.delete(dataset)
        await db.commit()
        logger.info(f"Deleted dataset {dataset_id}")
        return True


dataset_service = DatasetService()