"""
Object Detection recipe (Modeling -> Computer Vision & Audio).

Two ways to say which images to process (use one, not both):
  1. Image Files ('image_files'): images picked from the user's uploaded
     unstructured files, e.g. ["unstructured/cat.png"]. No table needed.
  2. A connected table plus 'image_column': one column holds the Ozone path of
     each image (s3://<bucket>/ml-workflow/t<tenant>/... or a path inside the
     tenant folder). This is the original behaviour and is unchanged.

For every image the recipe runs a pretrained YOLO detector and returns a table
with the detected objects (label, confidence, bounding box).

Place this file at:  backend/app/recipes/vision/object_detection.py
Needs:               pip install ultralytics pillow   (pulls in torch)

Security notes (same rules as csv_loader):
  * Images are only read from Ozone, and only from the CALLER'S tenant folder
    (tenant_id is injected from the token by with_tenant_config; it is required).
  * Files picked with 'image_files' must sit directly inside unstructured/.
    Every path is re-checked here, whatever the frontend filtered.
  * Extension and size limits come from core/media_types.py (one place).
  * Only the built-in model names below are accepted. No model path from the
    node config, because loading a .pt file from an arbitrary path can run code.
  * No http(s) URLs are fetched (SSRF).
  * Annotated images go to outputs/object_detection/, never into unstructured/,
    so they do not show up in the image pickers.
"""
import io
import json
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from backend.app.core import media_types as mt
from backend.app.recipes.base.recipe import BaseRecipe

# Built-in detectors (COCO, 80 classes). Larger = more accurate, slower.
SUPPORTED_MODELS = [
    "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
]
DEFAULT_MODEL = "yolov8m.pt"
MAX_IMAGES_PER_RUN = 500
MAX_IMAGE_BYTES = mt.MEDIA_KINDS["image"]["max_bytes"]  # 25 MB per image
OUTPUT_FOLDER = "outputs/object_detection"
FILE_MODE_COLUMN = "image_path"  # column name used when images come from 'image_files'

_MODEL_CACHE: Dict[str, Any] = {}
_MODEL_LOCK = threading.Lock()


def _load_model(model_name: str):
    """Loads a YOLO model once per process and reuses it."""
    with _MODEL_LOCK:
        if model_name not in _MODEL_CACHE:
            try:
                from ultralytics import YOLO
            except ImportError as e:
                raise RuntimeError(
                    "Object Detection needs the 'ultralytics' package. "
                    "Install it with: pip install ultralytics pillow"
                ) from e
            _MODEL_CACHE[model_name] = YOLO(model_name)
        return _MODEL_CACHE[model_name]


def _ozone_key(value: str, tenant_id: int) -> str:
    """
    Turns a cell value into a full Ozone key INSIDE the tenant's folder.
    Accepts 's3://bucket/ml-workflow/t1/img.jpg' or a key relative to the
    tenant folder ('img.jpg' / 'unstructured/img.jpg'). Anything else is refused.
    """
    from backend.app.core.config import settings
    from backend.app.infrastructure.storage.ozone_client import with_prefix, is_in_tenant_prefix

    value = str(value).strip()
    if value.lower().startswith(("http://", "https://")):
        raise ValueError("URLs are not supported; use an Ozone path (s3://...)")
    if value.startswith("s3://"):
        parts = value.split("/", 3)
        if len(parts) < 4 or parts[2] != settings.OZONE_BUCKET:
            raise ValueError("Image is not in this platform's Ozone bucket")
        key = parts[3]
    else:
        if is_in_tenant_prefix(value, tenant_id):
            key = value
        else:
            key = with_prefix(value, tenant_id)
    if not is_in_tenant_prefix(key, tenant_id):
        raise PermissionError("Image is outside this tenant's folder")
    return key


def _unstructured_key(value: str, tenant_id: int) -> str:
    """Like _ozone_key, but the file must sit directly inside this tenant's unstructured/ folder."""
    from backend.app.infrastructure.storage.ozone_client import tenant_prefix

    key = _ozone_key(value, tenant_id)
    folder = tenant_prefix(tenant_id) + mt.UNSTRUCTURED_FOLDER + "/"
    name = key[len(folder):] if key.startswith(folder) else ""
    if not name or "/" in name:
        raise ValueError(f"Files must be picked from {mt.UNSTRUCTURED_FOLDER}/")
    return key


def _object_size(key: str) -> int:
    """Size of an object in bytes (one HEAD request), so oversized files are never downloaded."""
    from backend.app.core.config import settings
    from backend.app.infrastructure.storage.ozone_client import get_s3_client

    try:
        return int(get_s3_client().head_object(Bucket=settings.OZONE_BUCKET, Key=key)["ContentLength"])
    except Exception:
        raise ValueError("Image not found or not readable")


def _read_image_bytes(value: str, tenant_id: int, from_unstructured: bool = False) -> bytes:
    """Checks path, extension and size, then reads the image. Raises with a short reason if any check fails."""
    from backend.app.infrastructure.storage.ozone_client import get_file_bytes_from_ozone

    key = _unstructured_key(value, tenant_id) if from_unstructured else _ozone_key(value, tenant_id)
    ext = mt.extension_of(key)
    if mt.kind_for_extension(ext) != "image":
        raise ValueError(f"Unsupported image type: {ext or 'none'}")
    if _object_size(key) > MAX_IMAGE_BYTES:
        raise ValueError(f"Image is larger than {MAX_IMAGE_BYTES // mt.MB} MB")
    return get_file_bytes_from_ozone(key, tenant_id)


class ObjectDetectionRecipe(BaseRecipe):
    recipe_id = "object_detection"
    name = "Object Detection"
    version = "1.1.0"
    category = "vision_audio"
    media_type = "image"
    description = "Detects and locates objects in images with bounding boxes."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "title": "Image Files",
                    "widget": "file_picker",
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
                    "widget": "column_picker",
                    "description": "Only when a table is connected: the column holding the Ozone path of each image (s3://... or a path inside your folder).",
                },
                "model": {
                    "type": "select",
                    "title": "Detection Model",
                    "widget": "select",
                    "enum": SUPPORTED_MODELS,
                    "default": DEFAULT_MODEL,
                    "description": "Pretrained YOLO model (80 COCO classes). Bigger models are more accurate but slower.",
                },
                "confidence_threshold": {
                    "type": "number",
                    "title": "Confidence Threshold",
                    "widget": "slider",
                    "default": 0.25,
                    "minimum": 0.01,
                    "maximum": 1.0,
                    "description": "Detections below this confidence are dropped.",
                },
                "iou_threshold": {
                    "type": "number",
                    "title": "Overlap (NMS IoU) Threshold",
                    "widget": "slider",
                    "default": 0.7,
                    "minimum": 0.1,
                    "maximum": 1.0,
                    "description": "Boxes overlapping more than this are merged into one.",
                },
                "max_detections": {
                    "type": "integer",
                    "title": "Max Objects per Image",
                    "widget": "number",
                    "default": 100,
                    "minimum": 1,
                    "maximum": 1000,
                },
                "image_size": {
                    "type": "integer",
                    "title": "Inference Image Size (px)",
                    "widget": "number",
                    "default": 640,
                    "minimum": 32,
                    "maximum": 1920,
                },
                "classes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "title": "Only These Classes",
                    "widget": "tags",
                    "description": "Optional. Keep only these labels, e.g. person, car. Empty = all classes.",
                },
                "output_format": {
                    "type": "select",
                    "title": "Output Format",
                    "widget": "select",
                    "enum": ["one_row_per_image", "one_row_per_detection"],
                    "default": "one_row_per_image",
                    "description": "One row per image (detections as JSON) or one row per detected object.",
                },
                "save_annotated_images": {
                    "type": "boolean",
                    "title": "Save Annotated Images",
                    "widget": "toggle",
                    "default": False,
                    "description": f"Saves a copy of each image with boxes drawn into your Ozone folder ({OUTPUT_FOLDER}/<model>_<time>/).",
                },
            },
            # Either 'image_files' or 'image_column' is needed; validate_config checks that.
            "required": [],
        }

    # ------------------------------------------------------------------ validation
    @staticmethod
    def _image_files(config: Dict[str, Any]) -> List[str]:
        raw = config.get("image_files")
        if not isinstance(raw, (list, tuple)):
            return []
        return [str(f).strip() for f in raw if str(f).strip()]

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        errors: List[str] = []

        raw_files = config.get("image_files")
        if raw_files is not None and not isinstance(raw_files, (list, tuple)):
            errors.append("Image files ('image_files') must be a list of paths.")
        files = self._image_files(config)
        if len(files) > MAX_IMAGES_PER_RUN:
            errors.append(f"Too many image files ({len(files)}). The limit is {MAX_IMAGES_PER_RUN} per run.")
        if not files and not str(config.get("image_column") or "").strip():
            errors.append("Choose image files ('image_files') or set the image column ('image_column') of a connected table.")

        model = config.get("model", DEFAULT_MODEL)
        if model not in SUPPORTED_MODELS:
            errors.append(f"Unknown model '{model}'. Choose one of: {', '.join(SUPPORTED_MODELS)}.")

        for key, lo, hi in (("confidence_threshold", 0.01, 1.0), ("iou_threshold", 0.1, 1.0)):
            raw = config.get(key)
            if raw is None:
                continue
            try:
                if not lo <= float(raw) <= hi:
                    errors.append(f"{key} must be between {lo} and {hi}. Received: {raw}.")
            except (ValueError, TypeError):
                errors.append(f"{key} must be a number. Received: {repr(raw)}.")

        for key, lo, hi in (("max_detections", 1, 1000), ("image_size", 32, 1920)):
            raw = config.get(key)
            if raw is None:
                continue
            try:
                if not lo <= int(raw) <= hi:
                    errors.append(f"{key} must be between {lo} and {hi}. Received: {raw}.")
            except (ValueError, TypeError):
                errors.append(f"{key} must be an integer. Received: {repr(raw)}.")

        if config.get("output_format", "one_row_per_image") not in ("one_row_per_image", "one_row_per_detection"):
            errors.append("output_format must be 'one_row_per_image' or 'one_row_per_detection'.")
        return errors

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _parse_classes(raw: Any) -> List[str]:
        if isinstance(raw, str):
            return [c.strip() for c in raw.split(",") if c.strip()]
        if isinstance(raw, (list, tuple)):
            return [str(c).strip() for c in raw if str(c).strip()]
        return []

    @staticmethod
    def _detect(model, image, conf: float, iou: float, max_det: int, imgsz: int, class_ids: Optional[List[int]]):
        results = model.predict(
            image, conf=conf, iou=iou, max_det=max_det, imgsz=imgsz,
            classes=class_ids, verbose=False,
        )
        res = results[0]
        names = res.names
        detections: List[Dict[str, Any]] = []
        for box in res.boxes:
            raw_xy = box.xyxy[0].tolist() if hasattr(box.xyxy[0], "tolist") else list(box.xyxy[0])
            x1, y1, x2, y2 = [round(float(v), 1) for v in raw_xy]
            cls_id = int(box.cls[0])
            detections.append({
                "label": names[cls_id],
                "class_id": cls_id,
                "confidence": round(float(box.conf[0]), 4),
                "x_min": x1, "y_min": y1, "x_max": x2, "y_max": y2,
            })
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections, res

    @staticmethod
    def _save_annotated(res, source_value: str, tenant_id: int, index: int, run_tag: str) -> Optional[str]:
        from PIL import Image
        from backend.app.infrastructure.storage.ozone_client import upload_file_to_ozone

        plotted = res.plot()  # BGR numpy array with boxes drawn
        img = Image.fromarray(plotted[..., ::-1])
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        base = str(source_value).rstrip("/").rsplit("/", 1)[-1].rsplit(".", 1)[0] or "image"
        return upload_file_to_ozone(
            buf.getvalue(), f"{OUTPUT_FOLDER}/{run_tag}/{index}_{base}_detected.jpg", tenant_id, "image/jpeg"
        )

    # ------------------------------------------------------------------ execution
    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        errors = self.validate_config(config)
        if errors:
            raise ValueError("; ".join(errors))

        tenant_id = config.get("tenant_id")
        if tenant_id is None:
            raise PermissionError("Object Detection needs a tenant context to read images.")
        tenant_id = int(tenant_id)

        # Decide where the image list comes from: picked files, or a connected table.
        files = self._image_files(config)
        table: Optional[pd.DataFrame] = (inputs or {}).get("dataframe")
        from_files = bool(files)
        if from_files:
            if table is not None:
                raise ValueError("Use either the connected table or 'Image Files', not both. Clear one of them.")
            df = pd.DataFrame({FILE_MODE_COLUMN: files})
            image_col = FILE_MODE_COLUMN
        else:
            if table is None:
                raise ValueError("ObjectDetection needs a connected table or 'image_files'.")
            df = table
            image_col = str(config["image_column"]).strip()
            if image_col not in df.columns:
                raise ValueError(f"Image column '{image_col}' not found. Available columns: {list(df.columns)}")

        if len(df) > MAX_IMAGES_PER_RUN:
            raise ValueError(
                f"Too many images ({len(df)}). The limit is {MAX_IMAGES_PER_RUN} per run; "
                "add a Limit node before this one."
            )

        model_name = config.get("model", DEFAULT_MODEL)
        conf = float(config.get("confidence_threshold", 0.25))
        iou = float(config.get("iou_threshold", 0.7))
        max_det = int(config.get("max_detections", 100))
        imgsz = int(config.get("image_size", 640))
        out_format = config.get("output_format", "one_row_per_image")
        save_annotated = bool(config.get("save_annotated_images", False))

        model = _load_model(model_name)

        # Map requested class names to YOLO class ids
        wanted = self._parse_classes(config.get("classes"))
        class_ids: Optional[List[int]] = None
        if wanted:
            lookup = {str(v).lower(): int(k) for k, v in model.names.items()}
            unknown = [c for c in wanted if c.lower() not in lookup]
            if unknown:
                raise ValueError(
                    f"Unknown class(es): {unknown}. This model detects: {sorted(lookup)}"
                )
            class_ids = [lookup[c.lower()] for c in wanted]

        from PIL import Image

        # One folder per run (model + UTC time): a new run never overwrites or reuses an old image URL.
        run_tag = f"{model_name.rsplit('.', 1)[0]}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"

        per_image: List[Dict[str, Any]] = []
        for idx, value in enumerate(df[image_col].tolist()):
            row: Dict[str, Any] = {"detections": [], "error": None, "annotated_path": None}
            try:
                if value is None or (isinstance(value, float) and pd.isna(value)) or not str(value).strip():
                    raise ValueError("Empty image path")
                data = _read_image_bytes(str(value), tenant_id, from_unstructured=from_files)
                image = Image.open(io.BytesIO(data)).convert("RGB")
                row["image_width"], row["image_height"] = image.size
                detections, res = self._detect(model, image, conf, iou, max_det, imgsz, class_ids)
                row["detections"] = detections
                if save_annotated:
                    row["annotated_path"] = self._save_annotated(res, str(value), tenant_id, idx, run_tag)
            except PermissionError:
                # Never reveal anything about other tenants' objects
                row["error"] = "Image is not accessible"
            except Exception as e:  # one bad image must not fail the whole run
                row["error"] = str(e)[:200]
            per_image.append(row)

        out = df.copy().reset_index(drop=True)
        out["num_objects"] = [len(r["detections"]) for r in per_image]
        out["detected_labels"] = [", ".join(sorted({d["label"] for d in r["detections"]})) for r in per_image]
        out["max_confidence"] = [
            max((d["confidence"] for d in r["detections"]), default=None) for r in per_image
        ]
        out["image_width"] = [r.get("image_width") for r in per_image]
        out["image_height"] = [r.get("image_height") for r in per_image]
        out["detections"] = [json.dumps(r["detections"]) for r in per_image]
        out["detection_error"] = [r["error"] for r in per_image]
        out["image_preview_url"] = [
            f"/api/v1/datasets/media/preview?key={str(val)}" if val and not (isinstance(val, float) and pd.isna(val)) else None
            for val in df[image_col].tolist()
        ]
        if save_annotated:
            out["annotated_image_path"] = [r["annotated_path"] for r in per_image]
            out["annotated_preview_url"] = [
                f"/api/v1/datasets/media/preview?key={r['annotated_path']}" if r.get("annotated_path") else None
                for r in per_image
            ]

        if out_format == "one_row_per_detection":
            rows = []
            for i, r in enumerate(per_image):
                base = out.drop(columns=["detections"]).iloc[i].to_dict()
                for d in r["detections"]:
                    rows.append({**base, **d})
            det_cols = ["label", "class_id", "confidence", "x_min", "y_min", "x_max", "y_max"]
            out_df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=list(out.columns.drop("detections")) + det_cols)
        else:
            out_df = out

        label_counts: Dict[str, int] = {}
        for r in per_image:
            for d in r["detections"]:
                label_counts[d["label"]] = label_counts.get(d["label"], 0) + 1
        failed = [r for r in per_image if r["error"]]

        annotated_previews = []
        for idx, r in enumerate(per_image):
            orig_val = df[image_col].iloc[idx] if idx < len(df) else None
            annotated_previews.append({
                "original_path": str(orig_val) if orig_val is not None else None,
                "original_preview_url": f"/api/v1/datasets/media/preview?key={orig_val}" if orig_val is not None else None,
                "annotated_path": r.get("annotated_path"),
                "annotated_preview_url": f"/api/v1/datasets/media/preview?key={r['annotated_path']}" if r.get("annotated_path") else None,
                "num_objects": len(r["detections"]),
                "error": r["error"],
            })

        metrics = {
            "task_type": "object_detection",
            "model": model_name,
            "input_source": "image_files" if from_files else "table",
            "images_processed": len(per_image) - len(failed),
            "images_failed": len(failed),
            "total_detections": int(sum(label_counts.values())),
            "images_with_no_objects": int(sum(1 for r in per_image if not r["error"] and not r["detections"])),
            "detections_per_class": dict(sorted(label_counts.items(), key=lambda kv: kv[1], reverse=True)),
            "confidence_threshold": conf,
            "errors_sample": [r["error"] for r in failed[:5]],
            "preview_items": annotated_previews,
        }

        return {
            "dataframe": out_df,
            "metrics": metrics,
            "output_summary": metrics,
            "task_type": "object_detection",
        }

    def to_code(self, config: Dict[str, Any]) -> str:
        model = config.get("model", DEFAULT_MODEL)
        conf = config.get("confidence_threshold", 0.25)
        files = self._image_files(config)
        source = repr(files) if files else f"df[{config.get('image_column', 'image_path')!r}].tolist()"
        return (
            "from ultralytics import YOLO\n\n"
            f"model = YOLO({model!r})\n"
            f"results = model.predict({source}, conf={conf})\n"
            "detections = [\n"
            "    [{'label': r.names[int(b.cls[0])], 'confidence': float(b.conf[0]), 'box': b.xyxy[0].tolist()} for b in r.boxes]\n"
            "    for r in results\n"
            "]"
        )