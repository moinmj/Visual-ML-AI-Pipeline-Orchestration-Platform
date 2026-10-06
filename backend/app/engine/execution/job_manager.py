import uuid
import time
import threading
import contextvars
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import pandas as pd

from backend.app.engine.dag.graph import WorkflowGraph
from backend.app.engine.execution.executor import DAGExecutor, WorkflowExecutionResult
from backend.app.core.logging import logger
from backend.app.core.tenant_context import get_current_tenant_id


class PipelineJobManager:
    """
    Asynchronous Background Job Manager & Worker Pool.
    Manages non-blocking concurrent pipeline execution, status tracking,
    and execution history in enterprise orchestrator fashion (n8n/Boomi style).

    Tenant isolation: every job and inference bundle is tagged with the tenant of
    the request that created it, and all getters only return entries of the
    CURRENT request's tenant (another tenant's job looks like "not found").
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
        self.inference_bundles: Dict[str, Dict[str, Any]] = {}
        self.inference_bundle_tenants: Dict[str, Optional[int]] = {}
        # keyed by (tenant_id, workflow_id) so one tenant can't overwrite another's pointer
        self.workflow_latest_executions: Dict[tuple, str] = {}
        self.lock = threading.RLock()

    @staticmethod
    def _visible(owner_tenant_id: Optional[int]) -> bool:
        """An entry is visible only to the tenant that created it (fail closed)."""
        return owner_tenant_id == get_current_tenant_id()

    def submit_job(
        self,
        workflow: WorkflowGraph,
        initial_df: Optional[pd.DataFrame] = None,
        context: Optional[Dict[str, Any]] = None,
        trigger_type: str = "manual",
        trigger_id: Optional[str] = None,
        include_node_outputs: bool = False,
        resume_from: Optional[str] = None
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
                "tenant_id": get_current_tenant_id(),
                "trigger_type": trigger_type,
                "trigger_id": trigger_id,
                "node_count": len(workflow.nodes),
                "resumed_from": resume_from,
                # Live progress, filled while the job runs (GET /workflows/jobs/{id}/logs)
                "logs": [],
                "node_status": {n.id: "PENDING" for n in workflow.nodes},
                "current_node": None,
                "progress": {"completed": 0, "total": len(workflow.nodes)},
                "result": None,
                "error": None
            }

        # Dispatch execution to background thread. The request's context (incl. the
        # tenant) is copied into the worker so the run stays inside that tenant.
        ctx = contextvars.copy_context()
        future = self.executor_pool.submit(
            ctx.run,
            self._run_job_worker,
            job_id=job_id,
            workflow=workflow,
            initial_df=initial_df,
            context=context,
            include_node_outputs=include_node_outputs,
            resume_from=resume_from
        )
        self.job_futures[job_id] = future
        return job_id

    def _on_progress(self, job_id: str, event: Dict[str, Any]) -> None:
        """Applies a live executor event to the job, so pollers see it immediately."""
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return
            etype = event.get("type")
            if etype == "log":
                job["logs"].append(event.get("message"))
            elif etype == "started":
                job["progress"]["total"] = event.get("total_nodes", job["progress"]["total"])
                for nid in event.get("node_ids", []):
                    job["node_status"].setdefault(nid, "PENDING")
            elif etype == "node_started":
                job["current_node"] = event.get("node_id")
                job["node_status"][event.get("node_id")] = "RUNNING"
            elif etype == "node_finished":
                status = event.get("status")
                job["node_status"][event.get("node_id")] = "REUSED" if event.get("reused") else status
                if status == "SUCCESS":
                    job["progress"]["completed"] += 1
                if job.get("current_node") == event.get("node_id"):
                    job["current_node"] = None

    def get_job_progress(self, job_id: str, since: int = 0) -> Optional[Dict[str, Any]]:
        """
        Live view of a job for the current tenant: status, the log lines added since
        index `since`, per-node status and progress. None if not found / not yours.
        """
        with self.lock:
            job = self.jobs.get(job_id)
            if not job or not self._visible(job.get("tenant_id")):
                return None
            logs = job.get("logs") or []
            since = max(0, min(int(since or 0), len(logs)))
            result = job.get("result")
            return {
                "job_id": job_id,
                "status": job.get("status"),
                "started_at": job.get("started_at"),
                "completed_at": job.get("completed_at"),
                "duration_ms": job.get("duration_ms"),
                "resumed_from": job.get("resumed_from"),
                "current_node": job.get("current_node"),
                "node_status": dict(job.get("node_status") or {}),
                "progress": dict(job.get("progress") or {}),
                "logs": list(logs[since:]),
                "next_since": len(logs),
                "done": job.get("status") in ("SUCCESS", "FAILED"),
                "error": job.get("error"),
                "execution_id": getattr(result, "execution_id", None) if result is not None else None,
            }

    def _run_job_worker(
        self,
        job_id: str,
        workflow: WorkflowGraph,
        initial_df: Optional[pd.DataFrame],
        context: Optional[Dict[str, Any]],
        include_node_outputs: bool = False,
        resume_from: Optional[str] = None
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
                include_node_outputs=include_node_outputs,
                on_event=lambda ev: self._on_progress(job_id, ev),
                resume_from=resume_from
            )
            duration_ms = round((time.time() - start_t) * 1000.0, 2)
            with self.lock:
                if job_id in self.jobs:
                    self.jobs[job_id]["status"] = result.status
                    self.jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
                    self.jobs[job_id]["duration_ms"] = duration_ms
                    self.jobs[job_id]["result"] = result
                    self.jobs[job_id]["logs"] = list(result.logs)
                    self.jobs[job_id]["current_node"] = None
        except Exception as e:
            duration_ms = round((time.time() - start_t) * 1000.0, 2)
            logger.error(f"Job {job_id} failed: {str(e)}")
            with self.lock:
                if job_id in self.jobs:
                    self.jobs[job_id]["status"] = "FAILED"
                    self.jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
                    self.jobs[job_id]["duration_ms"] = duration_ms
                    self.jobs[job_id]["error"] = str(e)
                    self.jobs[job_id]["logs"].append(f"❌ Job failed: {e}")
                    self.jobs[job_id]["current_node"] = None

    def list_jobs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Lists the current tenant's recent execution jobs, newest first."""
        with self.lock:
            sorted_jobs = sorted(
                (j for j in self.jobs.values() if self._visible(j.get("tenant_id"))),
                key=lambda j: j["created_at"],
                reverse=True
            )
            return sorted_jobs[:limit]

    def register_inference_bundle(self, execution_id: str, bundle: Dict[str, Any]):
        """Caches live fitted model, transformers, and schema for instant real-time predictions."""
        with self.lock:
            owner = self.inference_bundle_tenants.get(execution_id, get_current_tenant_id())
            if execution_id in self.inference_bundles and owner != get_current_tenant_id():
                logger.warning(f"Refusing to overwrite inference bundle '{execution_id}' of another tenant")
                return
            self.inference_bundles[execution_id] = bundle
            self.inference_bundle_tenants[execution_id] = get_current_tenant_id()

    def register_job_result(self, execution_id: str, result: Any, duration_ms: float = 0.0, workflow_id: Optional[str] = None):
        """Caches synchronous execution result in memory so subsequent save workbook calls can locate diagnostics."""
        with self.lock:
            existing = self.jobs.get(execution_id)
            if existing and not self._visible(existing.get("tenant_id")):
                logger.warning(f"Refusing to overwrite job '{execution_id}' of another tenant")
                return
            wf_id = workflow_id or (getattr(result, "workflow_id", None) if not isinstance(result, dict) else result.get("workflow_id"))
            self.jobs[execution_id] = {
                "job_id": execution_id,
                "tenant_id": get_current_tenant_id(),
                "workflow_id": wf_id,
                "status": getattr(result, "status", "SUCCESS") if not isinstance(result, dict) else result.get("status", "SUCCESS"),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "started_at": datetime.now(timezone.utc).isoformat(),
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "duration_ms": duration_ms or (getattr(result, "total_duration_ms", 0.0) if not isinstance(result, dict) else result.get("total_duration_ms", 0.0)),
                "trigger_type": "sync_execute",
                "trigger_id": None,
                "result": result,
                "results": result,
                "logs": getattr(result, "logs", []) if not isinstance(result, dict) else result.get("logs", []),
                "error": None
            }
            if wf_id:
                self.workflow_latest_executions[(get_current_tenant_id(), str(wf_id))] = execution_id

    def get_latest_execution_for_workflow(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves the most recent in-memory execution job for a workflow ID (current tenant only)."""
        with self.lock:
            exec_id = self.workflow_latest_executions.get((get_current_tenant_id(), str(workflow_id)))
            if not exec_id:
                return None
            job = self.jobs.get(exec_id)
            if not job or not self._visible(job.get("tenant_id")):
                return None
            return dict(job)

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves status, logs, and outputs for a job (None if it belongs to another tenant)."""
        with self.lock:
            job = self.jobs.get(job_id)
            if not job or not self._visible(job.get("tenant_id")):
                return None
            return dict(job)

    def get_inference_bundle(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves cached fitted model and pipeline transformers by execution ID (current tenant only)."""
        with self.lock:
            if execution_id not in self.inference_bundles:
                return None
            if not self._visible(self.inference_bundle_tenants.get(execution_id)):
                return None
            return self.inference_bundles.get(execution_id)

    def clear_history(self):
        """Clears completed job records and inference bundles from memory."""
        with self.lock:
            self.jobs.clear()
            self.job_futures.clear()
            self.inference_bundles.clear()
            self.inference_bundle_tenants.clear()
            self.workflow_latest_executions.clear()


# Global singleton instance
job_manager = PipelineJobManager()