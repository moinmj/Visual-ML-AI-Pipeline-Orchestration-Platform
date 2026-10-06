"""
Resumable execution state ("resume from the point of failure").

While a workflow runs, the output of every node that finishes successfully is
saved (joblib) in the caller's tenant cache folder:

    <LOCAL_STORAGE_DIR>/t<tenant>/executions/<execution_id>/state/
        manifest.json        node fingerprints, status, step snapshots
        <node_id>.pkl        that node's outputs

When a run is started with resume_from=<failed execution_id>, every node whose
fingerprint is unchanged and that succeeded last time is NOT executed again:
its saved outputs are loaded instead. A node's fingerprint covers its recipe,
its config, its incoming connections AND the fingerprints of all its parents,
so editing a node automatically re-runs that node and everything after it.

State is deleted when a run succeeds (nothing to resume) and expires after
RESUME_STATE_TTL_HOURS. The folder is per tenant, so one tenant can never resume
from (or read) another tenant's run.
"""
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib

from backend.app.core.logging import logger
from backend.app.core.tenant_context import tenant_cache_dir

RESUME_STATE_TTL_HOURS = 72
_MANIFEST = "manifest.json"


def _executions_root() -> Path:
    from backend.app.infrastructure.storage.storage_manager import storage_manager
    return storage_manager.get_absolute_path(f"{tenant_cache_dir()}/executions")


def _safe_id(value: str) -> str:
    return "".join(ch for ch in str(value) if ch.isalnum() or ch in "-_")


def state_dir(execution_id: str) -> Path:
    return _executions_root() / _safe_id(execution_id) / "state"


def _fingerprint_value(value: Any) -> Any:
    """Makes config values hashable; a DataFrame (e.g. a preloaded dataset) becomes a content hash."""
    try:
        import pandas as pd
        if isinstance(value, pd.DataFrame):
            content = pd.util.hash_pandas_object(value, index=True).values.tobytes()
            cols = "|".join(map(str, value.columns))
            return "dataframe:" + hashlib.sha256(content + cols.encode("utf-8")).hexdigest()
    except Exception:
        pass
    if isinstance(value, dict):
        return {str(k): _fingerprint_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_fingerprint_value(v) for v in value]
    return value


def node_fingerprint(node: Any, incoming_edges: List[Any], parent_fingerprints: List[str]) -> str:
    payload = {
        "recipe_id": node.recipe_id,
        "config": _fingerprint_value(node.config or {}),
        "edges": sorted(
            [str(e.source), str(getattr(e, "source_handle", "") or ""), str(getattr(e, "target_handle", "") or "")]
            for e in incoming_edges
        ),
        "parents": sorted(parent_fingerprints),
    }
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_manifest(execution_id: str) -> Optional[Dict[str, Any]]:
    """The saved state of a previous run of the CURRENT tenant, or None."""
    path = state_dir(execution_id) / _MANIFEST
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"[resume] unreadable manifest for {execution_id}: {e}")
        return None


def resume_info(execution_id: str) -> Optional[Dict[str, Any]]:
    """Short, frontend-friendly description of what a resume would reuse."""
    manifest = load_manifest(execution_id)
    if not manifest:
        return None
    nodes = manifest.get("nodes", {})
    reusable = [nid for nid, n in nodes.items() if n.get("status") == "SUCCESS" and n.get("saved")]
    failed = [nid for nid, n in nodes.items() if n.get("status") == "FAILED"]
    not_run = [nid for nid, n in nodes.items() if n.get("status") == "PENDING"]
    return {
        "execution_id": execution_id,
        "status": manifest.get("status"),
        "created_at": manifest.get("created_at"),
        "reusable_nodes": reusable,
        "failed_nodes": failed,
        "not_run_nodes": not_run,
    }


class RunState:
    """Writes the resumable state of the run that is executing now."""

    def __init__(self, execution_id: str, resumed_from: Optional[str] = None):
        self.execution_id = execution_id
        self.dir = state_dir(execution_id)
        self.manifest: Dict[str, Any] = {
            "execution_id": execution_id,
            "resumed_from": resumed_from,
            "status": "RUNNING",
            "created_at": time.time(),
            "nodes": {},
        }
        self.enabled = True
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"[resume] state disabled for {execution_id}: {e}")
            self.enabled = False

    def _write(self):
        if not self.enabled:
            return
        try:
            tmp = self.dir / (_MANIFEST + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.manifest, f, default=str)
            tmp.replace(self.dir / _MANIFEST)
        except Exception as e:
            logger.warning(f"[resume] could not write manifest for {self.execution_id}: {e}")

    def register_nodes(self, fingerprints: Dict[str, str]):
        for nid, fp in fingerprints.items():
            self.manifest["nodes"][nid] = {"fingerprint": fp, "status": "PENDING", "saved": False}
        self._write()

    def node_succeeded(self, node_id: str, outputs: Dict[str, Any], snapshot: Dict[str, Any], summary: Dict[str, Any]):
        entry = self.manifest["nodes"].setdefault(node_id, {})
        entry.update({"status": "SUCCESS", "snapshot": snapshot, "summary": summary, "saved": False})
        if self.enabled:
            try:
                joblib.dump(outputs, self.dir / f"{_safe_id(node_id)}.pkl", compress=3)
                entry["saved"] = True
            except Exception as e:
                # e.g. an output that can't be pickled: this node will simply re-run on resume
                logger.warning(f"[resume] outputs of node {node_id} not saved: {e}")
        self._write()

    def node_reused(self, node_id: str, from_execution_id: str, snapshot: Dict[str, Any], summary: Dict[str, Any]):
        entry = self.manifest["nodes"].setdefault(node_id, {})
        entry.update({"status": "SUCCESS", "snapshot": snapshot, "summary": summary, "saved": False})
        if self.enabled:
            try:
                shutil.copy2(state_dir(from_execution_id) / f"{_safe_id(node_id)}.pkl", self.dir / f"{_safe_id(node_id)}.pkl")
                entry["saved"] = True
            except Exception as e:
                logger.warning(f"[resume] could not carry over node {node_id}: {e}")
        self._write()

    def node_failed(self, node_id: str):
        self.manifest["nodes"].setdefault(node_id, {})["status"] = "FAILED"
        self._write()

    def finish(self, status: str):
        """A successful run has nothing to resume: its saved outputs are removed."""
        self.manifest["status"] = status
        if status == "SUCCESS":
            shutil.rmtree(self.dir, ignore_errors=True)
        else:
            self._write()


def load_node_outputs(execution_id: str, node_id: str) -> Dict[str, Any]:
    return joblib.load(state_dir(execution_id) / f"{_safe_id(node_id)}.pkl")


def purge_expired_states():
    """Best-effort cleanup of the current tenant's resume states older than the TTL."""
    try:
        root = _executions_root()
        if not root.exists():
            return
        cutoff = time.time() - RESUME_STATE_TTL_HOURS * 3600
        for run_dir in root.iterdir():
            sdir = run_dir / "state"
            if sdir.is_dir() and sdir.stat().st_mtime < cutoff:
                shutil.rmtree(sdir, ignore_errors=True)
    except Exception as e:
        logger.debug(f"[resume] purge skipped: {e}")