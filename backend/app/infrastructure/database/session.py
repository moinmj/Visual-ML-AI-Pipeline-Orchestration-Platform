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