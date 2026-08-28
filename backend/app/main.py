import os
import sys

# Ensure repository root is on sys.path for PM2, Docker, and systemd process managers
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from backend.app.core.config import settings
from backend.app.core.logging import setup_logging, logger
from backend.app.core.exceptions import PlatformException
from backend.app.infrastructure.database.session import init_db

# Import routers
from backend.app.datasets.router import router as datasets_router
from backend.app.recipes.router import router as recipes_router
from backend.app.workflows.router import router as workflows_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    setup_logging()
    logger.info("Initializing platform database...")
    await init_db()
    logger.info("Platform database initialized successfully.")
    yield
    # Shutdown
    logger.info("Platform shutting down.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Enterprise Visual AI/ML Pipeline Orchestration Engine (Dataiku / n8n / Boomi inspired)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
)

# CORS configuration for future frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global Exception Handler
@app.exception_handler(PlatformException)
async def platform_exception_handler(request: Request, exc: PlatformException):
    status_code = status.HTTP_400_BAD_REQUEST
    if exc.code == "NOT_FOUND":
        status_code = status.HTTP_404_NOT_FOUND
    elif exc.code == "VALIDATION_ERROR":
        status_code = status.HTTP_422_UNPROCESSABLE_ENTITY

    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
                "details": exc.details
            }
        }
    )


# Health check endpoint
@app.get("/health", tags=["System"])
async def health_check():
    return {
        "status": "healthy",
        "environment": settings.ENVIRONMENT,
        "version": "1.0.0"
    }


# Include sub-routers
app.include_router(datasets_router, prefix=settings.API_V1_STR)
app.include_router(recipes_router, prefix=settings.API_V1_STR)
app.include_router(workflows_router, prefix=settings.API_V1_STR)
