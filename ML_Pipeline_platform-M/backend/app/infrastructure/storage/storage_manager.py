import os
import shutil
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
        Saves raw bytes to storage and returns the relative storage path.
        """
        full_path = self.base_dir / destination_rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)

        if AIOFILES_AVAILABLE:
            async with aiofiles.open(full_path, "wb") as f:
                await f.write(file_content)
        else:
            with open(full_path, "wb") as f:
                f.write(file_content)

        logger.info(f"Saved file to storage: {destination_rel_path}")
        return destination_rel_path

    def get_absolute_path(self, relative_path: str) -> Path:
        """
        Resolves a relative storage path to an absolute path on the local filesystem.
        """
        return self.base_dir / relative_path

    def read_dataframe(self, relative_path: str) -> pd.DataFrame:
        """
        Reads tabular data from storage into a Pandas DataFrame based on file extension.
        """
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
        Deletes a file from storage.
        """
        full_path = self.base_dir / relative_path
        if full_path.exists():
            full_path.unlink()
            return True
        return False


storage_manager = StorageManager()
