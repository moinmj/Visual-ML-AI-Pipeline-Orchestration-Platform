import uuid
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import pandas as pd

from backend.app.engine.dag.graph import WorkflowGraph
from backend.app.engine.execution.executor import DAGExecutor, WorkflowExecutionResult
from backend.app.core.logging import logger


class PipelineJobManager:
    """
    Asynchronous Background Job Manager & Worker Pool.
    Manages non-blocking concurrent pipeline execution, status tracking,
    and execution history in enterprise orchestrator fashion (n8n/Boomi style).
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(PipelineJobManager, cls).__new__(cls)
                cls._instance._init_manager()
            return cls._instance

    def _init_manager(self, max_workers: int = 4):
        self.executor_pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="PipelineWorker")
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.job_futures: Dict[str, Any] = {}
        self.lock = threading.Lock()

    def submit_job(
        self,
        workflow: WorkflowGraph,
        initial_df: Optional[pd.DataFrame] = None,
        context: Optional[Dict[str, Any]] = None,
        trigger_type: str = "manual",
        trigger_id: Optional[str] = None,
        include_node_outputs: bool = False
    ) -> str:
        """Submits a DAG execution job to the background worker pool."""
        job_id = f"job_{uuid.uuid4().hex[:8]}"
        created_at = datetime.now(timezone.utc).isoformat()

        with self.lock:
            self.jobs[job_id] = {
                "job_id": job_id,
                "status": "PENDING",  # PENDING, RUNNING, SUCCESS, FAILED
                "created_at": created_at,
                "started_at": None,
                "completed_at": None,
                "duration_ms": 0.0,
                "trigger_type": trigger_type,
                "trigger_id": trigger_id,
                "node_count": len(workflow.nodes),
                "result": None,
                "error": None
            }

        # Dispatch execution to background thread
        future = self.executor_pool.submit(
            self._run_job_worker,
            job_id=job_id,
            workflow=workflow,
            initial_df=initial_df,
            context=context,
            include_node_outputs=include_node_outputs
        )
        self.job_futures[job_id] = future
        return job_id

    def _run_job_worker(
        self,
        job_id: str,
        workflow: WorkflowGraph,
        initial_df: Optional[pd.DataFrame],
        context: Optional[Dict[str, Any]],
        include_node_outputs: bool = False
    ):
        with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id]["status"] = "RUNNING"
                self.jobs[job_id]["started_at"] = datetime.now(timezone.utc).isoformat()

        start_t = time.time()
        try:
            result = DAGExecutor.execute_workflow(
                execution_id=job_id,
                workflow=workflow,
                initial_df=initial_df,
                context=context,
                include_node_outputs=include_node_outputs
            )
            duration_ms = round((time.time() - start_t) * 1000.0, 2)
            with self.lock:
                if job_id in self.jobs:
                    self.jobs[job_id]["status"] = result.status
                    self.jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
                    self.jobs[job_id]["duration_ms"] = duration_ms
                    self.jobs[job_id]["result"] = result
        except Exception as e:
            duration_ms = round((time.time() - start_t) * 1000.0, 2)
            logger.error(f"Job {job_id} failed: {str(e)}")
            with self.lock:
                if job_id in self.jobs:
                    self.jobs[job_id]["status"] = "FAILED"
                    self.jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
                    self.jobs[job_id]["duration_ms"] = duration_ms
                    self.jobs[job_id]["error"] = str(e)

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves status, logs, and outputs for a job."""
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return None
            return dict(job)

    def list_jobs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Lists recent execution jobs sorted by creation time descending."""
        with self.lock:
            sorted_jobs = sorted(
                self.jobs.values(),
                key=lambda j: j["created_at"],
                reverse=True
            )
            return sorted_jobs[:limit]

    def clear_history(self):
        """Clears completed job records from memory."""
        with self.lock:
            self.jobs.clear()
            self.job_futures.clear()


# Global singleton instance
job_manager = PipelineJobManager()
