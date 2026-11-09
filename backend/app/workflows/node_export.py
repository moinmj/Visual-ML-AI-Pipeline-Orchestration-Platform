"""
Save a workflow node's result table to Ozone, and look it up again.

Where the file goes (one fixed, predictable key per node of a run):

    <bucket>/<OZONE_PREFIX>/t<tenant_id>/<OZONE_EXPORT_FOLDER>/<execution_id>/<node_id>.csv
    e.g.  s3://my-bucket/ml-workflow/t1/outputs/workflow_exports/3f9c.../node_join.csv

  * The folder sits inside the CALLER'S tenant folder, so one tenant can never write to,
    read or overwrite another tenant's files.
  * Saving the same node of the same run again overwrites the same file (no duplicates).
  * The CSV is byte-for-byte what the "Export CSV" download produces.
  * Ozone is written directly (NOT through storage_manager.save_file): that method silently
    falls back to local disk when Ozone fails, which would report success for a file that
    is not in Ozone. Here a failed upload is an error (HTTP 502).

The column names are not stored anywhere else: they are read from the header row of the
CSV that is in Ozone, so what the API reports is always what is really in the file.
"""
import csv
import io
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
from botocore.exceptions import BotoCoreError, ClientError

from backend.app.core.config import settings
from backend.app.core.exceptions import StorageException, ValidationException
from backend.app.core.logging import logger
from backend.app.infrastructure.storage import ozone_client

DEFAULT_EXPORT_FOLDER = "outputs/workflow_exports"

_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9_-]")
_HEADER_READ_BYTES = 1024 * 1024  # a CSV header is far smaller than this


def _safe_part(value: Any, label: str) -> str:
    """An id from the URL, made safe to use as one part of an object key (no '/', '.', '..', spaces ...)."""
    cleaned = _UNSAFE_CHARS.sub("_", str(value or "").strip())[:100]
    if not cleaned.strip("_"):
        raise ValidationException(f"'{label}' has no usable characters for a storage path.")
    return cleaned


def export_folder() -> str:
    """The configured export folder, normalised to 'a/b' (falls back to the default if unusable)."""
    raw = (getattr(settings, "OZONE_EXPORT_FOLDER", "") or "").strip().strip("/")
    parts = [p for p in raw.split("/") if p]
    if not parts or any(p == ".." for p in parts):
        if raw:
            logger.warning(f"[NodeExport] Ignoring invalid OZONE_EXPORT_FOLDER '{raw}', using '{DEFAULT_EXPORT_FOLDER}'")
        return DEFAULT_EXPORT_FOLDER
    return "/".join(parts)


def _relative_key(execution_id: str, node_id: str) -> str:
    """Key relative to the tenant folder: '<export folder>/<execution_id>/<node_id>.csv'."""
    return f"{export_folder()}/{_safe_part(execution_id, 'execution_id')}/{_safe_part(node_id, 'node_id')}.csv"


def _require_ready(tenant_id: Optional[int]) -> int:
    if tenant_id is None:
        raise ValidationException("No tenant on the session; cannot choose a storage folder.")
    if not ozone_client.ozone_is_configured():
        raise StorageException(
            "Ozone is not configured on the server (OZONE_ENDPOINT / OZONE_ACCESS_KEY / OZONE_SECRET_KEY / OZONE_BUCKET).",
            details={"configured": False},
        )
    return int(tenant_id)


def _describe(full_key: str, columns: List[str]) -> Dict[str, Any]:
    """The part of the response that is the same for save and lookup."""
    return {
        "ozone_path": f"s3://{settings.OZONE_BUCKET}/{full_key}",
        "bucket": settings.OZONE_BUCKET,
        "key": full_key,
        "folder": full_key.rsplit("/", 1)[0] + "/",
        "file_name": full_key.rsplit("/", 1)[-1],
        "file_format": "csv",
        "columns": columns,
        "column_count": len(columns),
    }


def save_node_csv(df: pd.DataFrame, execution_id: str, node_id: str, tenant_id: Optional[int]) -> Dict[str, Any]:
    """
    Writes `df` as CSV into Ozone (blocking - call it from a worker thread) and returns where it went.
    Raises StorageException (HTTP 502) if Ozone is unreachable or rejects the upload.
    """
    tid = _require_ready(tenant_id)
    if len(df.columns) == 0:
        raise ValidationException("The node output has no columns, so there is nothing to save.")

    rel_key = _relative_key(execution_id, node_id)
    full_key = ozone_client.with_prefix(rel_key, tid)
    csv_bytes = df.to_csv(index=False).encode("utf-8")  # same bytes as the Export CSV download

    ozone_path = ozone_client.upload_file_to_ozone(csv_bytes, rel_key, tid, "text/csv")

    info = _describe(full_key, [str(c) for c in df.columns])
    info["ozone_path"] = ozone_path  # exactly what the upload reported
    info["size_bytes"] = len(csv_bytes)
    info["row_count"] = int(len(df))
    info["saved_at"] = datetime.now(timezone.utc).isoformat()
    return info


def _parse_header(data: bytes, complete: bool) -> Optional[List[str]]:
    """
    Column names = the first CSV record. `complete` says whether `data` is the whole file.
    Returns None when the file is longer than `data` and the header may not be fully inside it.
    """
    text = data.decode("utf-8-sig", errors="replace")
    reader = csv.reader(io.StringIO(text, newline=""))
    try:
        header = next(reader)
    except StopIteration:
        return []
    if not complete:
        # The header record is only known to be finished if a further record has started.
        try:
            next(reader)
        except StopIteration:
            return None
    return header


def get_saved_node_export(execution_id: str, node_id: str, tenant_id: Optional[int]) -> Optional[Dict[str, Any]]:
    """
    Looks up the file saved by save_node_csv (blocking - call it from a worker thread).
    Returns None if this node of this run was never saved, otherwise its path and column names.
    """
    tid = _require_ready(tenant_id)
    full_key = ozone_client.with_prefix(_relative_key(execution_id, node_id), tid)

    head = ozone_client.read_ozone_object_head(full_key, tid, max_bytes=_HEADER_READ_BYTES)
    if head is None:
        return None

    try:
        columns = _parse_header(head["head_bytes"], complete=head["size_bytes"] <= len(head["head_bytes"]))
        if columns is None:  # absurdly wide header: read the whole file once
            try:
                whole = ozone_client.get_file_bytes_from_ozone(full_key, tid)
            except (ClientError, BotoCoreError) as e:
                raise StorageException(
                    f"Could not read the file from the datalake (Ozone): {e}",
                    details={"bucket": settings.OZONE_BUCKET, "key": full_key},
                )
            columns = _parse_header(whole, complete=True) or []
    except csv.Error as e:
        raise StorageException(
            f"The saved file in Ozone could not be read as CSV: {e}",
            details={"bucket": settings.OZONE_BUCKET, "key": full_key},
        )

    info = _describe(full_key, columns)
    info["size_bytes"] = head["size_bytes"]
    last_modified = head.get("last_modified")
    info["last_modified"] = last_modified.isoformat() if last_modified else None
    return info