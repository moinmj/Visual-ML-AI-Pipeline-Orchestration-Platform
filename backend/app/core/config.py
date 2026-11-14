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
    BACKEND_PUBLIC_URL: str = "https://ml-crt.aiiotgeeks.com"
    FRONTEND_PUBLIC_URL: str = "https://crt.aiiotgeeks.com"
    SECRET_KEY: str = "development-secret-key-change-in-production"

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./data/platform.db"
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 30
    DB_POOL_TIMEOUT: int = 60
    DB_POOL_RECYCLE: int = 1800
    UM_DB_POOL_SIZE: int = 15
    UM_DB_MAX_OVERFLOW: int = 20
 
    # Storage Settings
    STORAGE_BACKEND: str = "ozone"  # datasets are always stored in Ozone (OZONE_PREFIX folder)
    LOCAL_STORAGE_DIR: str = "./data/storage"

    # S3 / MinIO Settings
    S3_ENDPOINT_URL: Optional[str] = "http://localhost:9000"
    S3_ACCESS_KEY: Optional[str] = "minioadmin"
    S3_SECRET_KEY: Optional[str] = "minioadmin"
    S3_BUCKET_NAME: str = "pipeline-artifacts"

    # Apache Ozone (S3-compatible datalake). Set STORAGE_BACKEND="ozone" to use it.
    OZONE_ENDPOINT: Optional[str] = None
    OZONE_ACCESS_KEY: Optional[str] = None
    OZONE_SECRET_KEY: Optional[str] = None
    OZONE_BUCKET: Optional[str] = None
    OZONE_REGION: str = "us-east-1"
    # Folder (key prefix) inside the bucket that this platform reads/writes. "" = whole bucket.
    OZONE_PREFIX: str = "ml-workflow"
    # OZONE_PREFIX: str = ""
    # Lets a "Tenant Admin" list the WHOLE bucket (landing/, files/, warehouse/, every tenant's folder)
    # with GET /datasets/ozone/browse?scope=bucket. OFF by default because it shows other tenants' file names.
    OZONE_ALLOW_BUCKET_BROWSE: bool = False
    # Folder (relative to the tenant folder, i.e. ml-workflow/t<tenant_id>/) where "Save to Ozone" puts the
    # result table of a workflow node: <OZONE_EXPORT_FOLDER>/<execution_id>/<node_id>.csv
    OZONE_EXPORT_FOLDER: str = "outputs/workflow_exports"

    # MLflow Settings
    MLFLOW_TRACKING_URI: str = "sqlite:///./data/mlflow.db"

    # LLM Provider Settings (Groq, Gemini, OpenAI)
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: str = "openai/gpt-oss-20b"
    GEMINI_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None

    # Auth / JWT Settings (LEGACY — kept so old config/.env files don't break;
    # the service no longer issues or validates JWTs).
    JWT_SECRET_KEY: str = "Z73MyQ5WFZNjAQYkIGqjxBR55xAMv7r/v/qv/dMhl121zVQYbC3Y3Hhx"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440

    # ── Session auth (User-Management cookie sessions) ────────────────────────
    # Auth is now session based. This service reads public.user_sessions in the
    # User-Management database directly and validates the `sid` cookie.
    #   UM_DATABASE_URL    : async DSN to user_management_db (URL-encode the pw).
    #   SESSION_SECRET_KEY : MUST equal UM's SECRET_KEY (used for the CSRF HMAC).
    UM_DATABASE_URL: Optional[str] = None
    SESSION_SECRET_KEY: str = "Z73MyQ5WFZNjAQYkIGqjxBR55xAMv7r/v/qv/dMhl121zVQYbC3Y3Hhx"
    SESSION_IDLE_MINUTES: int = 30
    SESSION_CSRF_ENABLED: bool = True

    # Export control (who may download / export data, and from where)
    # Comma-separated role names that may export or download data.
    EXPORT_ALLOWED_ROLES: str = "Tenant Admin,Data Scientist,ML Engineer"
    # Comma-separated IPs / CIDR ranges allowed to export, e.g. "10.0.0.0/8,203.0.113.5".
    # Empty = no IP restriction. (A browser cannot report a MAC address, so the client IP is used instead.)
    EXPORT_ALLOWED_IPS: str = ""
    # Reverse proxies whose X-Forwarded-For header is trusted when working out the client IP.
    TRUSTED_PROXY_IPS: str = "127.0.0.1,::1"
    # How long per-node downloadable artifacts (train/test sets, metrics, model file) are kept.
    NODE_ARTIFACT_TTL_HOURS: int = 168
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    FERNET_KEY: Optional[str] = "Rc4YoP_92ZxBPiwTRsprsTUA__aqFDBqZPCz7sLFR_g="
    RESET_TOKEN_EXPIRE_HOURS: int = 24

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

    # External Warehouse & Connectors
    ICEBERG_WAREHOUSE: Optional[str] = None
    LANDING_FOLDER: Optional[str] = None
    AIRFLOW_BASE_URL: Optional[str] = None
    DRUID_HOST: Optional[str] = None
    DRUID_ROUTER_PORT: Optional[int] = 8888
    DRUID_OVERLORD_PORT: Optional[int] = 8081
    DRUID_COORDINATOR_PORT: Optional[int] = 8081

    # OAuth & Integration Credentials
    DISCORD_CLIENT_ID: Optional[str] = None
    DISCORD_CLIENT_SECRET: Optional[str] = None
    DISCORD_REDIRECT_URI: Optional[str] = None
    TEAMS_CLIENT_ID: Optional[str] = None
    TEAMS_CLIENT_SECRET: Optional[str] = None
    TEAMS_TENANT_ID: Optional[str] = None
    TEAMS_REDIRECT_URI: Optional[str] = None
    AZURE_CLIENT_ID: Optional[str] = None
    AZURE_CLIENT_SECRET: Optional[str] = None
    AZURE_TENANT_ID: Optional[str] = None

    # SMS & Twilio Credentials
    SMS_PROVIDER: Optional[str] = "twilio"
    TWILIO_ACCOUNT_SID: Optional[str] = None
    TWILIO_AUTH_TOKEN: Optional[str] = None
    TWILIO_PHONE_NUMBER: Optional[str] = None

    # Maps & Location API Keys
    GOOGLE_MAPS_API_KEY: Optional[str] = None
    MAPBOX_ACCESS_TOKEN: Optional[str] = None

    # CRM & Business Integration API Keys
    SALESFORCE_ACCESS_TOKEN: Optional[str] = None
    SALESFORCE_INSTANCE_URL: Optional[str] = None
    HUBSPOT_ACCESS_TOKEN: Optional[str] = None
 
    model_config = {
        "env_file": ".env",
        "extra": "ignore"
    }


settings = Settings()

# Datasets are stored only in Ozone (s3://<OZONE_BUCKET>/<OZONE_PREFIX>/t<tenant_id>/...).
# The local dirs below are only for the SQLite/MLflow defaults and temporary
# execution / test-session caches - no datasets are written here anymore.
Path("./data").mkdir(parents=True, exist_ok=True)
Path(settings.LOCAL_STORAGE_DIR).mkdir(parents=True, exist_ok=True)

_missing_ozone = [k for k in ("OZONE_ENDPOINT", "OZONE_ACCESS_KEY", "OZONE_SECRET_KEY", "OZONE_BUCKET") if not getattr(settings, k)]
if _missing_ozone and str(getattr(settings, "ENVIRONMENT", "development")).lower() in ("production", "prod"):
    raise RuntimeError(f"Ozone storage is required but these settings are missing: {', '.join(_missing_ozone)}")