import asyncio
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.core.tenant_context import set_current_tenant_id, set_current_user_id


class AgentQueueFullError(Exception):
    """Raised when the bounded agent worker queue has reached capacity."""


class AgentJobManager:
    """Runs long agent workflows off the API event loop with bounded admission."""

    def __init__(
        self,
        max_workers: int,
        max_queued: int,
        max_history: int = 500,
    ):
        if max_workers < 1 or max_queued < 0:
            raise ValueError("Agent worker and queue sizes must be non-negative.")
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="PipelineAgent",
        )
        self._capacity = threading.BoundedSemaphore(max_workers + max_queued)
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._job_tenants: Dict[str, int] = {}
        self._lock = threading.Lock()
        self._max_history = max_history

    def submit(
        self,
        *,
        dataset_id: str,
        dataset_path: str,
        user_request: str,
        max_attempts: int,
        tenant_id: int,
        user_id: Optional[str] = None,
        requested_target: Optional[str] = None,
        requested_task_type: Optional[str] = None,
    ) -> str:
        if not self._capacity.acquire(blocking=False):
            raise AgentQueueFullError("Agent worker capacity is full; retry the request shortly.")

        job_id = f"agent_{uuid.uuid4().hex}"
        with self._lock:
            self._prune_history()
            self._job_tenants[job_id] = int(tenant_id)
            self._jobs[job_id] = {
                "job_id": job_id,
                "status": "PENDING",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "started_at": None,
                "completed_at": None,
                "duration_ms": 0.0,
                "result": None,
                "error": None,
            }

        try:
            self._executor.submit(
                self._run_job,
                job_id,
                dataset_id,
                dataset_path,
                user_request,
                max_attempts,
                requested_target,
                requested_task_type,
                tenant_id,
                user_id,
            )
        except RuntimeError as exc:
            with self._lock:
                self._jobs.pop(job_id, None)
                self._job_tenants.pop(job_id, None)
            self._capacity.release()
            logger.error(f"Could not submit agent job {job_id}: {exc}")
            raise AgentQueueFullError("Agent workers are shutting down; retry the request.") from exc
        return job_id

    def get_job(self, job_id: str, tenant_id: int) -> Optional[Dict[str, Any]]:
        """Return the job only if it belongs to the caller's tenant."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or self._job_tenants.get(job_id) != int(tenant_id):
                return None
            return dict(job)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)

    def _run_job(
        self,
        job_id: str,
        dataset_id: str,
        dataset_path: str,
        user_request: str,
        max_attempts: int,
        requested_target: Optional[str],
        requested_task_type: Optional[str],
        tenant_id: int,
        user_id: Optional[str],
    ) -> None:
        started = time.monotonic()
        # Worker threads do not inherit the request's context, so set the caller's
        # tenant/user here. The DB tenant guard and executors read it from there.
        set_current_tenant_id(tenant_id)
        set_current_user_id(user_id)
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job["status"] = "RUNNING"
                job["started_at"] = datetime.now(timezone.utc).isoformat()

        try:
            result = asyncio.run(
                self._run_agent(
                    dataset_path=dataset_path,
                    dataset_id=dataset_id,
                    user_request=user_request,
                    max_attempts=max_attempts,
                    requested_target=requested_target,
                    requested_task_type=requested_task_type,
                    tenant_id=tenant_id,
                )
            )
            with self._lock:
                job = self._jobs.get(job_id)
                if job:
                    job["status"] = "SUCCESS"
                    job["result"] = result
        except Exception as exc:
            logger.exception(f"Agent job {job_id} failed: {exc}")
            with self._lock:
                job = self._jobs.get(job_id)
                if job:
                    job["status"] = "FAILED"
                    job["error"] = str(exc)
        finally:
            set_current_tenant_id(None)
            set_current_user_id(None)
            with self._lock:
                job = self._jobs.get(job_id)
                if job:
                    job["completed_at"] = datetime.now(timezone.utc).isoformat()
                    job["duration_ms"] = round((time.monotonic() - started) * 1000, 2)
            self._capacity.release()

    @staticmethod
    async def _run_agent(
        *,
        dataset_path: str,
        dataset_id: str,
        user_request: str,
        max_attempts: int,
        requested_target: Optional[str],
        requested_task_type: Optional[str],
        tenant_id: int,
    ) -> Dict[str, Any]:
        from backend.app.infrastructure.storage.storage_manager import storage_manager
        from backend.app.recommendation.agent_graph import run_pipeline_agent
        from backend.app.recommendation.agent_service import build_dataset_context

        dataframe = storage_manager.read_dataframe(dataset_path, tenant_id)
        if not all(isinstance(column, str) for column in dataframe.columns):
            raise ValueError("The agent planner requires dataset column names to be strings.")
        if dataframe.empty or len(dataframe.columns) < 2:
            raise ValueError(
                "Supervised planning requires a non-empty dataset with a target and at least one feature column."
            )
        if requested_target and requested_target not in dataframe.columns:
            raise ValueError(f"Target column '{requested_target}' is not present in the dataset.")

        return await run_pipeline_agent(
            dataset_id=dataset_id,
            dataframe=dataframe,
            dataset_context=build_dataset_context(dataframe),
            user_request=user_request,
            max_attempts=max_attempts,
            requested_target=requested_target,
            requested_task_type=requested_task_type,
        )

    def _prune_history(self) -> None:
        terminal_jobs = sorted(
            (
                (job["created_at"], job_id)
                for job_id, job in self._jobs.items()
                if job["status"] in {"SUCCESS", "FAILED"}
            ),
            reverse=True,
        )
        for _, job_id in terminal_jobs[self._max_history :]:
            self._jobs.pop(job_id, None)
            self._job_tenants.pop(job_id, None)


agent_job_manager = AgentJobManager(
    max_workers=settings.AGENT_MAX_WORKERS,
    max_queued=settings.AGENT_MAX_QUEUED,
)