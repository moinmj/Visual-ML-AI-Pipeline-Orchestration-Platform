"""
Downloadable per-node artifacts.

After a node finishes, everything useful it produced is saved in the caller's tenant
cache folder so it can be downloaded later (even after the run is over):

    <storage>/t<tenant>/executions/<execution_id>/artifacts/<node_id>/
        <name>.parquet   a table      (dataframe, dataframe_train, dataframe_test, forecast_df, ...)
        <name>.json      a result     (metrics, feature_importances, forecasting_summary, ...)
        <name>.joblib    a fitted model

The folder is per tenant, so one tenant can never list or read another tenant's artifacts.
Artifacts expire after NODE_ARTIFACT_TTL_HOURS (see purge_expired_artifacts).
"""
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.core.tenant_context import tenant_cache_dir

_TABLE, _JSON, _MODEL = ".parquet", ".json", ".joblib"
_NAME_OK = re.compile(r"^[A-Za-z0-9_.-]{1,100}$")


def _safe(value: str) -> str:
    return "".join(ch for ch in str(value) if ch.isalnum() or ch in "-_")


def _executions_root() -> Path:
    from backend.app.infrastructure.storage.storage_manager import storage_manager
    return storage_manager.get_absolute_path(f"{tenant_cache_dir()}/executions")


def artifacts_dir(execution_id: str, node_id: str) -> Path:
    return _executions_root() / _safe(execution_id) / "artifacts" / _safe(node_id)


def _looks_like_model(value: Any) -> bool:
    return hasattr(value, "predict") or hasattr(value, "fit") or hasattr(value, "get_forecast") or hasattr(value, "make_future_dataframe")


def save_node_artifacts(execution_id: str, node_id: str, outputs: Dict[str, Any]) -> None:
    """Best-effort: a failure here must never fail the run."""
    if not isinstance(outputs, dict):
        return
    try:
        import joblib
        from backend.app.engine.execution.executor import make_json_safe
    except Exception:
        return
    folder = artifacts_dir(execution_id, node_id)
    for key, value in outputs.items():
        name = re.sub(r"[^A-Za-z0-9_.-]", "_", str(key))[:100]
        if not name:
            continue
        try:
            if isinstance(value, pd.DataFrame):
                if len(value.columns) == 0:
                    continue
                folder.mkdir(parents=True, exist_ok=True)
                value.to_parquet(folder / f"{name}{_TABLE}", index=False)
            elif _looks_like_model(value):
                folder.mkdir(parents=True, exist_ok=True)
                joblib.dump(value, folder / f"{name}{_MODEL}")
            elif isinstance(value, (dict, list)) and value:
                safe = make_json_safe(value)
                folder.mkdir(parents=True, exist_ok=True)
                (folder / f"{name}{_JSON}").write_text(json.dumps(safe, default=str), encoding="utf-8")
        except Exception as e:  # keep going - one bad output must not hide the others
            logger.warning(f"Could not save artifact '{name}' of node '{node_id}' ({execution_id}): {e}")


def _describe(path: Path) -> Dict[str, Any]:
    kind = {_TABLE: "table", _JSON: "json", _MODEL: "model"}[path.suffix]
    info: Dict[str, Any] = {
        "name": path.stem,
        "kind": kind,
        "formats": {"table": ["csv", "json", "parquet"], "json": ["json", "csv"], "model": ["joblib"]}[kind],
        "size_bytes": path.stat().st_size,
    }
    if kind == "table":
        try:
            import pyarrow.parquet as pq
            meta = pq.ParquetFile(path).metadata
            info["rows"], info["columns"] = int(meta.num_rows), int(meta.num_columns)
        except Exception:
            pass
    return info


def list_artifacts(execution_id: str, node_id: str) -> List[Dict[str, Any]]:
    folder = artifacts_dir(execution_id, node_id)
    items: List[Dict[str, Any]] = []
    if folder.is_dir():
        for p in sorted(folder.iterdir()):
            if p.is_file() and p.suffix in (_TABLE, _JSON, _MODEL):
                try:
                    items.append(_describe(p))
                except Exception:
                    continue
    if not any(i["name"] == "dataframe" for i in items):
        # Runs from before artifacts existed still have the node's main table cached.
        legacy = _executions_root() / _safe(execution_id) / f"{_safe(node_id)}.parquet"
        if legacy.is_file() and not any(i["kind"] == "table" for i in items):
            try:
                d = _describe(legacy)
                d["name"] = "dataframe"
                items.insert(0, d)
            except Exception:
                pass
    return items


def find_artifact(execution_id: str, node_id: str, name: str) -> Optional[Path]:
    """Path of one artifact, or None. `name` is validated so it can never point outside the folder."""
    if not _NAME_OK.match(name or ""):
        return None
    folder = artifacts_dir(execution_id, node_id)
    for ext in (_TABLE, _JSON, _MODEL):
        p = folder / f"{name}{ext}"
        if p.is_file():
            return p
    if name == "dataframe":
        legacy = _executions_root() / _safe(execution_id) / f"{_safe(node_id)}.parquet"
        if legacy.is_file():
            return legacy
    return None


def purge_expired_artifacts() -> None:
    """Best-effort cleanup of the current tenant's artifacts older than the TTL."""
    try:
        root = _executions_root()
        if not root.is_dir():
            return
        cutoff = time.time() - settings.NODE_ARTIFACT_TTL_HOURS * 3600
        for exec_dir in root.iterdir():
            adir = exec_dir / "artifacts"
            if adir.is_dir() and adir.stat().st_mtime < cutoff:
                shutil.rmtree(adir, ignore_errors=True)
    except Exception:
        pass