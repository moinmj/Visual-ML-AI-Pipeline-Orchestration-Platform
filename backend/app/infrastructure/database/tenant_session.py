"""
Connection to the tenant metadata database (Environments/Models/Measures/
Dimensions), which is a SEPARATE Postgres database from this platform's
own DATABASE_URL (which only holds Datasets/Workflows).

Deliberately uses its own DeclarativeBase (`TenantMetaBase`) so that
`init_db()` in `infrastructure/database/session.py` - which only calls
`Base.metadata.create_all(...)` against the platform's own engine - never
attempts to create or alter these tables. They already exist and are
owned by another service; we only ever SELECT from them here.
"""
from typing import AsyncGenerator, Optional

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool

from backend.app.core.config import settings


class TenantMetaBase(DeclarativeBase):
    pass


def _to_asyncpg_url(url: str) -> str:
    """Normalizes a plain postgres URL to the asyncpg driver SQLAlchemy needs."""
    if url.startswith("postgresql+asyncpg://"):
        return url
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


_tenant_engine = None
_TenantSessionLocal: Optional[async_sessionmaker] = None


def _ensure_initialized():
    global _tenant_engine, _TenantSessionLocal
    if _tenant_engine is not None:
        return
    url = settings.P_DATABASE_URL or settings.DATABASE_URL
    if "postgresql" in url:
        _tenant_engine = create_async_engine(
            _to_asyncpg_url(url),
            echo=False,
            future=True,
            poolclass=NullPool,
        )
    else:
        connect_args = {"check_same_thread": False} if "sqlite" in url else {}
        _tenant_engine = create_async_engine(
            url,
            echo=False,
            future=True,
            connect_args=connect_args,
        )
    _TenantSessionLocal = async_sessionmaker(
        bind=_tenant_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )


async def get_tenant_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency: yields a session bound to the tenant metadata DB."""
    _ensure_initialized()
    async with _TenantSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_tenant_db() -> None:
    """Create tenant metadata tables when explicitly enabled for development."""
    _ensure_initialized()
    from backend.app.tenant_data import models  # noqa: F401

    async with _tenant_engine.begin() as connection:
        await connection.run_sync(TenantMetaBase.metadata.create_all)