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

    # Auth / JWT Settings
    # Tokens are expected to be issued by an identity provider (or the
    # dev-only /api/v1/auth/dev-token endpoint) and carry: sub, tenant_id,
    # roles (list[str]) and permissions (list[str]) claims.
    JWT_SECRET_KEY: str = "CHANGE_ME"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # Tenant Data Source Settings
    # Connection string for the warehouse that physically hosts each
    # Model's table (e.g. the Druid/analytics datasource referenced by
    # ModelV3.druid_datasource_name). Defaults to the platform's own
    # DATABASE_URL so the ingestion path works out-of-the-box in dev/demo
    # environments; point this at your real warehouse in production.
    SOURCE_DATABASE_URL: Optional[str] = None

    # Tenant metadata DB (Environments/Models/Measures/Dimensions live here,
    # in a separate Postgres database from the platform's own DATABASE_URL).
    P_DATABASE_URL: Optional[str] = None
    TENANT_DB_AUTO_CREATE: bool = False

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