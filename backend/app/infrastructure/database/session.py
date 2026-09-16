from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase
from backend.app.core.config import settings
from typing import AsyncGenerator


class Base(DeclarativeBase):
    pass


# Create async engine
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    future=True,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}
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
        # Auto-migrate workflows table columns for soft delete support
        try:
            await conn.execute(text("ALTER TABLE workflows ADD COLUMN is_active BOOLEAN DEFAULT 1"))
        except Exception:
            pass
        try:
            await conn.execute(text("ALTER TABLE workflows ADD COLUMN deleted_at DATETIME"))
        except Exception:
            pass
        try:
            await conn.execute(text("ALTER TABLE workflows ADD COLUMN last_execution JSON"))
        except Exception:
            pass
        try:
            await conn.execute(text("ALTER TABLE workflows ADD COLUMN dataset_id VARCHAR(36)"))
        except Exception:
            pass
        try:
            await conn.execute(text("ALTER TABLE workflows ADD COLUMN dataset_name VARCHAR(255)"))
        except Exception:
            pass

        # Auto-clean legacy history versions: keep active executions as version 1, prune older versions
        try:
            # 1. Clean up orphaned executions (workflows that no longer exist or are inactive)
            await conn.execute(text("""
                DELETE FROM workflow_executions 
                WHERE workflow_id NOT IN (
                    SELECT id FROM workflows WHERE is_active = 1 AND deleted_at IS NULL
                )
            """))
            # 2. For active workflows with multiple historical executions, keep latest and delete older
            multi_res = await conn.execute(text("""
                SELECT workflow_id, count(*) as cnt 
                FROM workflow_executions 
                GROUP BY workflow_id 
                HAVING cnt > 1
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
            # 3. Ensure any remaining single active execution has version_number = 1
            await conn.execute(text("""
                UPDATE workflow_executions 
                SET version_number = 1, run_label = 'Run #1' 
                WHERE version_number != 1
            """))
        except Exception:
            pass