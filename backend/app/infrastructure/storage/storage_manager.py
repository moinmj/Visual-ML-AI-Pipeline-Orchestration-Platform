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

    async def save_file(self, file_content: bytes, destination_rel_path: str) -> str:
        """
        Saves raw bytes to the Ozone datalake and returns the "s3://bucket/key" path.
        Dataset uploads never touch local disk.
        """
        from backend.app.infrastructure.storage.ozone_client import upload_file_to_ozone
        content_type = mimetypes.guess_type(destination_rel_path)[0] or "application/octet-stream"
        return upload_file_to_ozone(file_content, destination_rel_path, content_type)

        #logger.info(f"Saved file to storage: {destination_rel_path}")
        #return destination_rel_path

    def get_absolute_path(self, relative_path: str) -> Path:
        """
        Resolves a relative storage path to an absolute path on the local filesystem.
        """
        return self.base_dir / relative_path

    def read_dataframe(self, relative_path: str) -> pd.DataFrame:
        """
        Reads tabular data from storage into a Pandas DataFrame based on file extension.
        Works for both a local relative path and an "s3://bucket/key" Ozone path,
        so callers (e.g. dataset preview, csv_loader) don't need to know which backend
        a given dataset was stored on.
        """
        if relative_path.startswith("s3://"):
            from backend.app.infrastructure.storage.ozone_client import get_file_bytes_from_ozone
            key = relative_path.split("/", 3)[-1]  # strip "s3://bucket/"
            suffix = Path(key).suffix.lower()
            data = get_file_bytes_from_ozone(key)
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

    def save_dataframe(self, df: pd.DataFrame, destination_rel_path: str, file_format: str = "parquet") -> str:
        """
        Saves a Pandas DataFrame into storage as Parquet or CSV.
        """
        full_path = self.base_dir / destination_rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)

        if file_format == "parquet" or destination_rel_path.endswith(".parquet"):
            df.to_parquet(full_path, index=False)
        elif file_format == "csv" or destination_rel_path.endswith(".csv"):
            df.to_csv(full_path, index=False)
        else:
            df.to_parquet(full_path, index=False)

        return destination_rel_path

    def delete_file(self, relative_path: str) -> bool:
        """
        Deletes a file from storage (local or Ozone, based on the path form).
        """
        if relative_path.startswith("s3://"):
            from backend.app.infrastructure.storage.ozone_client import delete_file_from_ozone
            key = relative_path.split("/", 3)[-1]
            return delete_file_from_ozone(key)

        full_path = self.base_dir / relative_path
        if full_path.exists():
            full_path.unlink()
            return True
        return False

    def list_ozone_objects(self, prefix: str = ""):
        """
        Lists objects already sitting in the Ozone bucket (the datalake "browse
        existing data" option), so one can be registered as a dataset without
        re-uploading it.
        """
        from backend.app.infrastructure.storage.ozone_client import list_ozone_objects
        return list_ozone_objects(prefix=prefix)


storage_manager = StorageManager()