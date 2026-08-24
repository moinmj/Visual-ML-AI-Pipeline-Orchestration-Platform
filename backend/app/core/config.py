from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import os
from pathlib import Path


class Settings(BaseSettings):
    PROJECT_NAME: str = "Visual AI/ML Pipeline Platform"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = "development-secret-key-change-in-production"

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./data/platform.db"

    # Storage Settings
    STORAGE_BACKEND: str = "local"  # "local" or "s3"
    LOCAL_STORAGE_DIR: str = "./data/storage"

    # S3 / MinIO Settings
    S3_ENDPOINT_URL: Optional[str] = "http://localhost:9000"
    S3_ACCESS_KEY: Optional[str] = "minioadmin"
    S3_SECRET_KEY: Optional[str] = "minioadmin"
    S3_BUCKET_NAME: str = "pipeline-artifacts"

    # MLflow Settings
    MLFLOW_TRACKING_URI: str = "sqlite:///./data/mlflow.db"

    model_config = {
        "env_file": ".env",
        "extra": "ignore"
    }


settings = Settings()

# Ensure local data directories exist
Path("./data").mkdir(parents=True, exist_ok=True)
Path(settings.LOCAL_STORAGE_DIR).mkdir(parents=True, exist_ok=True)
Path(f"{settings.LOCAL_STORAGE_DIR}/datasets").mkdir(parents=True, exist_ok=True)
Path(f"{settings.LOCAL_STORAGE_DIR}/artifacts").mkdir(parents=True, exist_ok=True)
