"""
Shared helpers for the image recipes (image_classifier, ocr).

Image source rules are identical to object_detection: either 'image_files'
(picked from unstructured/) or a connected table + 'image_column'. All path,
tenant, extension and size checks live in object_detection._read_image_bytes,
so they stay in ONE place and are reused here, not copied.

Place this file at:  backend/app/recipes/vision/_common.py
"""
import io
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from backend.app.recipes.vision import object_detection as od

MAX_IMAGES_PER_RUN = od.MAX_IMAGES_PER_RUN
FILE_MODE_COLUMN = od.FILE_MODE_COLUMN


def image_files(config: Dict[str, Any]) -> List[str]:
    raw = config.get("image_files")
    if not isinstance(raw, (list, tuple)):
        return []
    return [str(f).strip() for f in raw if str(f).strip()]


def source_schema() -> Dict[str, Any]:
    """The two 'where are my images' properties, same as object_detection."""
    return {
        "image_files": {
            "type": "array",
            "items": {"type": "string"},
            "title": "Image Files",
            "input_type": "file_select",
            "file_kind": "image",
            "maxItems": MAX_IMAGES_PER_RUN,
            "description": (
                "Pick images from your uploaded files (unstructured/). "
                f"Max {MAX_IMAGES_PER_RUN} per run. Leave empty to use a connected table instead."
            ),
        },
        "image_column": {
            "type": "string",
            "title": "Image Column",
            "input_type": "column_select",
            "description": "Only when a table is connected: the column holding the Ozone path of each image (s3://... or a path inside your folder).",
        },
    }


def validate_source(config: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    raw = config.get("image_files")
    if raw is not None and not isinstance(raw, (list, tuple)):
        errors.append("Image files ('image_files') must be a list of paths.")
    files = image_files(config)
    if len(files) > MAX_IMAGES_PER_RUN:
        errors.append(f"Too many image files ({len(files)}). The limit is {MAX_IMAGES_PER_RUN} per run.")
    if not files and not str(config.get("image_column") or "").strip():
        errors.append("Choose image files ('image_files') or set the image column ('image_column') of a connected table.")
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


def require_tenant(config: Dict[str, Any], recipe_name: str) -> int:
    tenant_id = config.get("tenant_id")
    if tenant_id is None:
        raise PermissionError(f"{recipe_name} needs a tenant context to read images.")
    return int(tenant_id)


def resolve_source(inputs: Dict[str, Any], config: Dict[str, Any], recipe_name: str) -> Tuple[pd.DataFrame, str, bool]:
    """Returns (dataframe, image_column, from_files)."""
    files = image_files(config)
    table: Optional[pd.DataFrame] = (inputs or {}).get("dataframe")
    if files:
        if table is not None:
            raise ValueError("Use either the connected table or 'Image Files', not both. Clear one of them.")
        df, col, from_files = pd.DataFrame({FILE_MODE_COLUMN: files}), FILE_MODE_COLUMN, True
    else:
        if table is None:
            raise ValueError(f"{recipe_name} needs a connected table or 'image_files'.")
        col = str(config["image_column"]).strip()
        if col not in table.columns:
            raise ValueError(f"Image column '{col}' not found. Available columns: {list(table.columns)}")
        df, from_files = table, False
    if len(df) > MAX_IMAGES_PER_RUN:
        raise ValueError(
            f"Too many images ({len(df)}). The limit is {MAX_IMAGES_PER_RUN} per run; "
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


def preview_url(key: Any) -> Optional[str]:
    return None if is_empty(key) else f"/api/v1/datasets/media/preview?key={key}"


def safe_error(exc: Exception) -> str:
    """Never reveal anything about other tenants' objects."""
    return "Image is not accessible" if isinstance(exc, PermissionError) else str(exc)[:200]