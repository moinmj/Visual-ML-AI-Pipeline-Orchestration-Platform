from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase
from backend.app.core.config import settings
from typing import AsyncGenerator


class Base(DeclarativeBase):
    pass


def _to_async_url(url: str) -> str:
    """Accept a plain postgresql:// URL and switch it to the asyncpg driver."""
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+asyncpg://", 1)
    return url


_DB_URL = _to_async_url(settings.DATABASE_URL)
_IS_SQLITE = "sqlite" in _DB_URL

# Create async engine
if _IS_SQLITE:
    engine = create_async_engine(
        _DB_URL,
        echo=False,
        future=True,
        connect_args={"check_same_thread": False},
    )
else:
    # pool_pre_ping: transparently replaces connections dropped by an SSH tunnel / idle timeout.
    engine = create_async_engine(
        _DB_URL,
        echo=False,
        future=True,
        pool_pre_ping=True,
        pool_recycle=1800,
    )

# Async session factory
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


from sqlalchemy import text


async def init_db():
    import backend.app.workflows.models
    import backend.app.datasets.models
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # create_all never alters an existing table, so add datasets.tenant_id to databases
    # created before tenant scoping. Idempotent; existing rows keep NULL (visible to no tenant).
    try:
        async with engine.begin() as conn:
            if _IS_SQLITE:
                await conn.execute(text("ALTER TABLE datasets ADD COLUMN tenant_id INTEGER"))
            else:
                await conn.execute(text("ALTER TABLE datasets ADD COLUMN IF NOT EXISTS tenant_id INTEGER"))
    except Exception:
        pass  # column already exists (SQLite has no IF NOT EXISTS)
    try:
        async with engine.begin() as conn:
            await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_datasets_tenant_id ON datasets (tenant_id)"))
    except Exception:
        pass

    # Same for workflows / workflow_executions (tenant-scoped workbooks). Existing rows
    # keep NULL and are visible to no tenant until assigned, e.g.:
    #   UPDATE workflows SET tenant_id = 1 WHERE tenant_id IS NULL;
    #   UPDATE workflow_executions SET tenant_id = 1 WHERE tenant_id IS NULL;
    for table in ("workflows", "workflow_executions"):
        try:
            async with engine.begin() as conn:
                if _IS_SQLITE:
                    await conn.execute(text(f"ALTER TABLE {table} ADD COLUMN tenant_id INTEGER"))
                else:
                    await conn.execute(text(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS tenant_id INTEGER"))
        except Exception:
            pass  # column already exists
        try:
            async with engine.begin() as conn:
                await conn.execute(text(f"CREATE INDEX IF NOT EXISTS ix_{table}_tenant_id ON {table} (tenant_id)"))
        except Exception:
            pass

    # user_id (creator, from the token's "sub") on existing tables. Idempotent;
    # rows created before this keep NULL, e.g. back-fill with:
    #   UPDATE workflows SET user_id = '2' WHERE user_id IS NULL;
    for table in ("datasets", "workflows", "workflow_executions"):
        try:
            async with engine.begin() as conn:
                if _IS_SQLITE:
                    await conn.execute(text(f"ALTER TABLE {table} ADD COLUMN user_id VARCHAR(255)"))
                else:
                    await conn.execute(text(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS user_id VARCHAR(255)"))
        except Exception:
            pass  # column already exists
        try:
            async with engine.begin() as conn:
                await conn.execute(text(f"CREATE INDEX IF NOT EXISTS ix_{table}_user_id ON {table} (user_id)"))
        except Exception:
            pass

    # Legacy SQLite-only column back-fill (databases created before soft-delete support).
    # Not needed on PostgreSQL: create_all builds the full schema on a fresh database.
    if _IS_SQLITE:
        for ddl in (
            "ALTER TABLE workflows ADD COLUMN is_active BOOLEAN DEFAULT 1",
            "ALTER TABLE workflows ADD COLUMN deleted_at DATETIME",
            "ALTER TABLE workflows ADD COLUMN last_execution JSON",
            "ALTER TABLE workflows ADD COLUMN dataset_id VARCHAR(36)",
            "ALTER TABLE workflows ADD COLUMN dataset_name VARCHAR(255)",
        ):
            try:
                async with engine.begin() as conn:
                    await conn.execute(text(ddl))
            except Exception:
                pass
