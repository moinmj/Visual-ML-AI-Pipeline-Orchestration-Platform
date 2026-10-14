import os
import shutil
import mimetypes
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Union, Optional
from backend.app.core.config import settings
from backend.app.core.logging import logger
import pandas as pd

try:
    import aiofiles
    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False


class StorageManager:
    """
    Unified Storage abstraction for saving/retrieving datasets,
    transformed dataframes, and ML artifacts.
    """

    def __init__(self):
        self.base_dir = Path(settings.LOCAL_STORAGE_DIR)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    async def save_file(self, file_content: bytes, destination_rel_path: str, tenant_id: int) -> str:
        """
        Saves raw bytes to the Ozone datalake, inside the tenant's folder
        (ml-workflow/t<tenant_id>/...), and returns the "s3://bucket/key" path.
        Dataset uploads never touch local disk. tenant_id is required on purpose:
        there is no default folder to fall back to.
        """
        from backend.app.infrastructure.storage.ozone_client import upload_file_to_ozone
        content_type = mimetypes.guess_type(destination_rel_path)[0] or "application/octet-stream"
        return upload_file_to_ozone(file_content, destination_rel_path, tenant_id, content_type)

    def get_absolute_path(self, relative_path: str) -> Path:
        """
        Resolves a relative storage path to an absolute path on the local filesystem.
        """
        return self.base_dir / relative_path

    @staticmethod
    def safe_to_parquet(df: pd.DataFrame, dest_path: Union[Path, str]) -> bool:
        """
        Safely writes a DataFrame to a local Parquet file.
        If PyArrow raises ArrowInvalid or ArrowTypeError (e.g. mixed struct/dict and scalar values),
        it automatically serializes complex/mixed object columns into clean JSON strings
        so the Parquet table can be reliably saved without errors.
        """
        p = Path(dest_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            df.to_parquet(p, index=False)
            return True
        except Exception:
            try:
                import json
                df_copy = df.copy()
                for col in df_copy.columns:
                    if df_copy[col].dtype == "object":
                        df_copy[col] = df_copy[col].apply(
                            lambda x: json.dumps(x, default=str) if isinstance(x, (dict, list)) else (str(x) if (x is not None and not pd.isna(x)) else None)
                        )
                df_copy.to_parquet(p, index=False)
                return True
            except Exception as e:
                logger.warning(f"safe_to_parquet failed writing to {p}: {e}")
                return False

    def read_dataframe(self, relative_path: str, tenant_id: Optional[int] = None) -> pd.DataFrame:
        """
        Reads tabular data from storage into a Pandas DataFrame based on file extension.
        Works for both a local relative path and an "s3://bucket/key" Ozone path,
        so callers (e.g. dataset preview, csv_loader) don't need to know which backend
        a given dataset was stored on.
        When tenant_id is given, an Ozone path must be inside that tenant's folder.
        """
        if relative_path.startswith("s3://"):
            from backend.app.infrastructure.storage.ozone_client import get_file_bytes_from_ozone
            key = relative_path.split("/", 3)[-1]  # strip "s3://bucket/"
            suffix = Path(key).suffix.lower()
            data = get_file_bytes_from_ozone(key, tenant_id)
            buf = BytesIO(data)
            if suffix == ".csv":
                return pd.read_csv(buf)
            elif suffix in [".xlsx", ".xls"]:
                return pd.read_excel(buf)
            elif suffix == ".parquet":
                return pd.read_parquet(buf)
            elif suffix == ".json":
                return pd.read_json(buf)
            else:
                raise ValueError(f"Unsupported tabular file format: {suffix}")

        abs_path = self.get_absolute_path(relative_path)
        if not abs_path.exists():
            raise FileNotFoundError(f"File not found in storage: {relative_path}")

        suffix = abs_path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(abs_path)
        elif suffix in [".xlsx", ".xls"]:
            return pd.read_excel(abs_path)
        elif suffix == ".parquet":
            return pd.read_parquet(abs_path)
        elif suffix == ".json":
            return pd.read_json(abs_path)
        else:
            raise ValueError(f"Unsupported tabular file format: {suffix}")

    def save_dataframe(self, df: pd.DataFrame, destination_rel_path: str, tenant_id: int, file_format: str = "parquet") -> str:
        """
        Saves a Pandas DataFrame to the Ozone datalake, inside the tenant's folder
        (ml-workflow/t<tenant_id>/...), as Parquet or CSV, and returns the
        "s3://bucket/key" path. Nothing is written to local disk.
        tenant_id is required on purpose: there is no default folder to fall back to.
        """
        if tenant_id is None:
            raise ValueError("tenant_id is required to save a dataframe to Ozone")

        from backend.app.infrastructure.storage.ozone_client import upload_file_to_ozone
        buf = BytesIO()
        if destination_rel_path.endswith(".csv") or (file_format == "csv" and not destination_rel_path.endswith(".parquet")):
            df.to_csv(buf, index=False)
            content_type = "text/csv"
        else:
            df.to_parquet(buf, index=False)
            content_type = "application/octet-stream"
        return upload_file_to_ozone(buf.getvalue(), destination_rel_path, tenant_id, content_type)

    def delete_file(self, relative_path: str, tenant_id: Optional[int] = None) -> bool:
        """
        Deletes a file from storage (local or Ozone, based on the path form).
        """
        if relative_path.startswith("s3://"):
            from backend.app.infrastructure.storage.ozone_client import delete_file_from_ozone
            key = relative_path.split("/", 3)[-1]
            return delete_file_from_ozone(key, tenant_id)

        full_path = self.base_dir / relative_path
        if full_path.exists():
            full_path.unlink()
            return True
        return False

    def list_ozone_objects(
        self,
        tenant_id: int,
        prefix: str = "",
        scope: str = "ml_workflow",
        folder: str = "all",
        media_type: Optional[str] = None,
        max_keys: int = 1000,
    ):
        """
        Lists objects in the caller's tenant space in Ozone.
        Supports scope ('all_tenant_data' or 'ml_workflow') and folder filtering.
        """
        from backend.app.infrastructure.storage.ozone_client import list_ozone_objects
        return list_ozone_objects(
            tenant_id=tenant_id,
            prefix=prefix,
            scope=scope,
            folder=folder,
            media_type=media_type,
            max_keys=max_keys,
        )


storage_manager = StorageManager()