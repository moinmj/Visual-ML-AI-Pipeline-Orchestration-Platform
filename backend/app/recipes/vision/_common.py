"""
Shared helpers for the media recipes (image_classifier, ocr, image_embedding_generator,
video_object_tracking).

Source rules are identical to object_detection: either '<kind>_files' (picked from
unstructured/) or a connected table + '<kind>_column'. kind is "image" (default) or "video".
All path, tenant, extension and size checks reuse object_detection's helpers, so they
stay in ONE place.

Place this file at:  backend/app/recipes/vision/_common.py
"""
import io
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from backend.app.core import media_types as mt
from backend.app.recipes.vision import object_detection as od

MAX_IMAGES_PER_RUN = od.MAX_IMAGES_PER_RUN
FILE_MODE_COLUMN = od.FILE_MODE_COLUMN
MAX_VIDEOS_PER_RUN = 5  # videos are big and slow: keep a run small

_LIMITS = {"image": MAX_IMAGES_PER_RUN, "video": MAX_VIDEOS_PER_RUN}


def max_per_run(kind: str = "image") -> int:
    return _LIMITS[kind]


def image_files(config: Dict[str, Any], kind: str = "image") -> List[str]:
    raw = config.get(f"{kind}_files")
    if not isinstance(raw, (list, tuple)):
        return []
    return [str(f).strip() for f in raw if str(f).strip()]


def source_schema(kind: str = "image") -> Dict[str, Any]:
    """The two 'where are my files' properties."""
    limit = max_per_run(kind)
    return {
        f"{kind}_files": {
            "type": "array",
            "items": {"type": "string"},
            "title": f"{kind.title()} Files",
            "input_type": "file_select",
            "file_kind": kind,
            "maxItems": limit,
            "description": (
                f"Pick {kind}s from your uploaded files (unstructured/). "
                f"Max {limit} per run. Leave empty to use a connected table instead."
            ),
        },
        f"{kind}_column": {
            "type": "string",
            "title": f"{kind.title()} Column",
            "input_type": "column_select",
            "description": f"Only when a table is connected: the column holding the Ozone path of each {kind} (s3://... or a path inside your folder).",
        },
    }


def validate_source(config: Dict[str, Any], kind: str = "image") -> List[str]:
    errors: List[str] = []
    limit = max_per_run(kind)
    raw = config.get(f"{kind}_files")
    if raw is not None and not isinstance(raw, (list, tuple)):
        errors.append(f"{kind.title()} files ('{kind}_files') must be a list of paths.")
    files = image_files(config, kind)
    if len(files) > limit:
        errors.append(f"Too many {kind} files ({len(files)}). The limit is {limit} per run.")
    if not files and not str(config.get(f"{kind}_column") or "").strip():
        errors.append(f"Choose {kind} files ('{kind}_files') or set the {kind} column ('{kind}_column') of a connected table.")
    return errors


def validate_range(config: Dict[str, Any], key: str, lo: float, hi: float, integer: bool = False) -> Optional[str]:
    raw = config.get(key)
    if raw is None:
        return None
    try:
        val = int(raw) if integer else float(raw)
    except (ValueError, TypeError):
        return f"{key} must be {'an integer' if integer else 'a number'}. Received: {repr(raw)}."
    if not lo <= val <= hi:
        return f"{key} must be between {lo} and {hi}. Received: {raw}."
    return None


def require_tenant(config: Dict[str, Any], recipe_name: str, what: str = "images") -> int:
    tenant_id = config.get("tenant_id")
    if tenant_id is None:
        raise PermissionError(f"{recipe_name} needs a tenant context to read {what}.")
    return int(tenant_id)


def resolve_source(inputs: Dict[str, Any], config: Dict[str, Any], recipe_name: str, kind: str = "image") -> Tuple[pd.DataFrame, str, bool]:
    """Returns (dataframe, path_column, from_files)."""
    files = image_files(config, kind)
    table: Optional[pd.DataFrame] = (inputs or {}).get("dataframe")
    limit = max_per_run(kind)
    if files:
        if table is not None:
            raise ValueError(f"Use either the connected table or '{kind.title()} Files', not both. Clear one of them.")
        col = f"{kind}_path"
        df, from_files = pd.DataFrame({col: files}), True
    else:
        if table is None:
            raise ValueError(f"{recipe_name} needs a connected table or '{kind}_files'.")
        col = str(config[f"{kind}_column"]).strip()
        if col not in table.columns:
            raise ValueError(f"{kind.title()} column '{col}' not found. Available columns: {list(table.columns)}")
        df, from_files = table, False
    if len(df) > limit:
        raise ValueError(
            f"Too many {kind}s ({len(df)}). The limit is {limit} per run; "
            "add a Limit node before this one."
        )
    return df, col, from_files


def is_empty(value: Any) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value)) or not str(value).strip()


def open_image(value: Any, tenant_id: int, from_files: bool):
    """Runs every path/size/type check, then returns an RGB PIL image."""
    from PIL import Image

    if is_empty(value):
        raise ValueError("Empty image path")
    data = od._read_image_bytes(str(value), tenant_id, from_unstructured=from_files)
    return Image.open(io.BytesIO(data)).convert("RGB")


def read_media_bytes(value: Any, tenant_id: int, from_files: bool, kind: str) -> Tuple[bytes, str]:
    """
    Same checks as object_detection._read_image_bytes (tenant folder, extension,
    size from one HEAD request) for any media kind. Returns (bytes, extension).
    """
    from backend.app.infrastructure.storage.ozone_client import get_file_bytes_from_ozone

    if is_empty(value):
        raise ValueError(f"Empty {kind} path")
    key = od._unstructured_key(str(value), tenant_id) if from_files else od._ozone_key(str(value), tenant_id)
    ext = mt.extension_of(key)
    if mt.kind_for_extension(ext) != kind:
        raise ValueError(f"Unsupported {kind} type: {ext or 'none'}")
    max_bytes = mt.MEDIA_KINDS[kind]["max_bytes"]
    if od._object_size(key) > max_bytes:
        raise ValueError(f"{kind.title()} is larger than {max_bytes // mt.MB} MB")
    data = get_file_bytes_from_ozone(key, tenant_id)
    if not mt.looks_like(ext, data[:64]):
        raise ValueError(f"File content does not look like a .{ext} file")
    return data, ext


def preview_url(key: Any) -> Optional[str]:
    return None if is_empty(key) else f"/api/v1/datasets/media/preview?key={key}"


def safe_error(exc: Exception) -> str:
    """Never reveal anything about other tenants' objects."""
    return "File is not accessible" if isinstance(exc, PermissionError) else str(exc)[:200]