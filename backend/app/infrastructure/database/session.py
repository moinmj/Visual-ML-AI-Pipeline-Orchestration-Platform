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

    # Auto-clean legacy history versions: keep the latest execution per workflow as version 1.
    # Runs in its own transaction so a failure can't poison the rest of startup (PostgreSQL
    # aborts the whole transaction on any error).
    try:
        async with engine.begin() as conn:
            # 1. Orphaned executions (workflows that no longer exist or are inactive)
            await conn.execute(text("""
                DELETE FROM workflow_executions
                WHERE workflow_id NOT IN (
                    SELECT id FROM workflows WHERE is_active = TRUE AND deleted_at IS NULL
                )
            """))
            # 2. Active workflows with several executions: keep latest, delete older
            multi_res = await conn.execute(text("""
                SELECT workflow_id, count(*) AS cnt
                FROM workflow_executions
                GROUP BY workflow_id
                HAVING count(*) > 1
            """))
            for (wf_id, _) in multi_res.fetchall():
                latest_res = await conn.execute(text("""
                    SELECT id FROM workflow_executions
                    WHERE workflow_id = :wf_id
                    ORDER BY version_number DESC, created_at DESC
                    LIMIT 1
                """), {"wf_id": wf_id})
                latest_id = latest_res.scalar()
                if latest_id:
                    await conn.execute(text("""
                        DELETE FROM workflow_executions
                        WHERE workflow_id = :wf_id AND id != :latest_id
                    """), {"wf_id": wf_id, "latest_id": latest_id})
                    await conn.execute(text("""
                        UPDATE workflow_executions
                        SET version_number = 1, run_label = 'Run #1'
                        WHERE id = :latest_id
                    """), {"latest_id": latest_id})
            # 3. Any remaining single execution gets version_number = 1
            await conn.execute(text("""
                UPDATE workflow_executions
                SET version_number = 1, run_label = 'Run #1'
                WHERE version_number != 1
            """))
    except Exception:
        pass