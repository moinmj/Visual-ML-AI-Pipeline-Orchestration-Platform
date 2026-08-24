import uuid
import pandas as pd
from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from fastapi import UploadFile

from backend.app.datasets.models import Dataset
from backend.app.infrastructure.storage.storage_manager import storage_manager
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

        # 1. Save file to storage
        await storage_manager.save_file(content, storage_rel_path)

        # 2. Read DataFrame and perform automatic statistical profiling
        try:
            df = storage_manager.read_dataframe(storage_rel_path)
            profile_data = DataProfiler.profile_dataframe(df)
        except Exception as e:
            # Clean up on failure
            storage_manager.delete_file(storage_rel_path)
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
            storage_path=storage_rel_path,
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
    async def get_dataset(db: AsyncSession, dataset_id: str) -> Dataset:
        result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
        dataset = result.scalar_one_or_none()
        if not dataset:
            raise NotFoundException("Dataset", dataset_id)
        return dataset

    @staticmethod
    async def get_preview(db: AsyncSession, dataset_id: str, limit: int = 10) -> Dict[str, Any]:
        dataset = await DatasetService.get_dataset(db, dataset_id)
        df = storage_manager.read_dataframe(dataset.storage_path)
        preview_df = df.head(limit).replace({float("nan"): None, float("inf"): None, float("-inf"): None})
        
        return {
            "id": dataset.id,
            "name": dataset.name,
            "columns": list(df.columns),
            "total_rows": int(len(df)),
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
