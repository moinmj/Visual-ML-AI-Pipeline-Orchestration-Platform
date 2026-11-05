"""
Object Removal recipe for PHOTOS (Modeling -> Computer Vision & Audio).

Finds objects (people by default) with a YOLO segmentation model, numbers them 1..N from left to right,
erases the ones you choose and fills the hole with a real inpainting model (LaMa), so the background
behind them is rebuilt from the picture around them. It works on any photo (no fixed-camera needed).

Images come from the image source of _common.py (same rules and security checks as the other image recipes).

Two ways to use it
  1) One shot:  mode = remove_selected, leave 'object_numbers' empty -> every object of the chosen classes is removed.
  2) Pick who:  run 1 with mode = number_objects  -> a numbered preview image + a table (nothing is removed).
                run 2 with mode = keep_selected and object_numbers = 2, 4  -> everybody EXCEPT 2 and 4 is removed
                      (or mode = remove_selected and object_numbers = 1, 3 -> only 1 and 3 are removed).
  The numbers come from the detector, so run 2 must use the SAME model, confidence, overlap, image size and classes
  as run 1 (otherwise the numbering can change).

How the removal works
  * The outline of every object comes from a '-seg' model (a real body shape, not a rectangle).
  * The outlines of the removed objects are grown a little ('mask_padding_px') to cover edges, hair and soft shadows.
  * Objects that stay are protected: their outline is never painted over.
  * LaMa (big-lama) fills the hole. Only a crop around the removed objects is sent to it (with context around), so
    large photos stay sharp outside the hole and it stays fast. The result is blended back with a soft edge.

Place this file at:  backend/app/recipes/vision/object_removal.py
Needs:               pip install ultralytics        (also installs torch / opencv)
                     The LaMa weights (big-lama.pt, about 206 MB) download ONCE from GitHub into
                     ~/.cache/object_removal/ and are checked against a fixed SHA-256 before they are used.
                     No internet on the server? Download the file on another machine, copy it there and set the
                     environment variable OBJECT_REMOVAL_LAMA_PATH to its full path (the SHA-256 is still checked).

Security: only built-in segmentation model names are accepted; the LaMa file is only loaded when its SHA-256 matches;
images are read only from the caller's tenant folder; pixel / count limits below; results go to outputs/object_removal/.

Honest limits: shadows and reflections of removed objects are only partly covered; very large holes (a person filling
most of the photo) give invented, sometimes odd textures; the detector can miss someone (then they stay).
Editing a photo changes it: keep the original and label the result as edited.
"""
import hashlib
import os
import tempfile
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from backend.app.recipes.base.recipe import BaseRecipe
from backend.app.recipes.vision import _common as c
from backend.app.recipes.vision import object_detection as od

SEG_MODELS = ["yolov8n-seg.pt", "yolov8s-seg.pt", "yolov8m-seg.pt", "yolov8l-seg.pt"]  # built-in names only
DEFAULT_MODEL = "yolov8m-seg.pt"  # photos are few, so a bigger (more exact) model is worth it
MODES = ("remove_selected", "keep_selected", "number_objects")
OUTPUT_FORMATS = ("png", "jpg")
OUTPUT_FOLDER = "outputs/object_removal"

MAX_PIXELS = 50_000_000     # refuse photos bigger than this (about 7000 x 7000)
MAX_OBJECT_NUMBERS = 50     # most numbers accepted in 'object_numbers'
CONTEXT_FRAC = 0.5          # context around the removed area given to LaMa, as a share of the area's size ...
CONTEXT_MIN_PX = 48         # ... plus at least this many pixels
FEATHER_PX = 9              # soft outward edge when the filled area is pasted back (odd number)

# LaMa (big-lama), the TorchScript file published with the simple-lama-inpainting project
LAMA_URL = "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"
LAMA_SHA256 = "7ba7aa7ac37a4d41fdbbeba3a2af7ead18058552997e3a3cd1a3b2210c9e6b4c"
LAMA_FILENAME = "big-lama.pt"
LAMA_ENV_PATH = "OBJECT_REMOVAL_LAMA_PATH"

COLORS = {  # BGR
    "listed": (0, 180, 255),   # number_objects mode
    "removed": (0, 0, 255),
    "kept": (0, 200, 0),
}

_verified_weights: Dict[str, bool] = {}  # path -> sha256 already checked in this process


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _lama_weights_path() -> str:
    """Path of the verified LaMa weights. Downloads them once if they are not there yet."""
    override = os.environ.get(LAMA_ENV_PATH)
    path = override or os.path.join(os.path.expanduser("~"), ".cache", "object_removal", LAMA_FILENAME)
    if not os.path.exists(path):
        if override:
            raise RuntimeError(f"{LAMA_ENV_PATH} points to '{override}', but that file does not exist.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".part")
        os.close(fd)
        try:
            with urllib.request.urlopen(LAMA_URL, timeout=120) as resp, open(tmp, "wb") as out:  # nosec: fixed https URL
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            if _sha256(tmp) != LAMA_SHA256:
                raise RuntimeError("The downloaded LaMa file does not match the expected SHA-256 and was discarded.")
            os.replace(tmp, path)
            _verified_weights[path] = True
        except Exception as e:
            raise RuntimeError(
                "Could not download the LaMa weights. Download big-lama.pt on another machine, copy it to the server "
                f"and set {LAMA_ENV_PATH} to its path. ({type(e).__name__})"
            ) from e
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    if not _verified_weights.get(path):
        if _sha256(path) != LAMA_SHA256:
            raise RuntimeError(f"'{path}' does not match the expected SHA-256 of big-lama.pt, so it is not loaded.")
        _verified_weights[path] = True
    return path


class _LamaInpainter:
    """LaMa (big-lama) inpainting. Loaded ONCE per run. Call with a BGR image and a uint8 mask (255 = fill here)."""

    def __init__(self):
        try:
            import torch
        except ImportError as e:
            raise RuntimeError("Object removal needs 'torch'. Install it with: pip install ultralytics") from e
        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = torch.jit.load(_lama_weights_path(), map_location=self.device).eval()

    def __call__(self, bgr, mask):
        import cv2
        import numpy as np

        h, w = mask.shape
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ph, pw = (-h) % 8, (-w) % 8  # LaMa wants sizes that are a multiple of 8
        if ph or pw:
            rgb = np.pad(rgb, ((0, ph), (0, pw), (0, 0)), mode="symmetric")
            mask = np.pad(mask, ((0, ph), (0, pw)), mode="constant")
        img_t = self._torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
        msk_t = self._torch.from_numpy((mask > 0).astype(np.float32))[None, None].to(self.device)
        with self._torch.inference_mode():
            out = self.model(img_t, msk_t)
        out = out[0].permute(1, 2, 0).detach().cpu().numpy()
        out = np.clip(out * 255.0, 0, 255).astype(np.uint8)[:h, :w]
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def _new_model(model_name: str):
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise RuntimeError("Object removal needs 'ultralytics'. Install it with: pip install ultralytics") from e
    return YOLO(model_name)


class ObjectRemovalRecipe(BaseRecipe):
    recipe_id = "object_removal"
    name = "Object Removal (Photos)"
    version = "1.0.0"
    category = "vision_audio"
    media_type = "image"
    description = "Finds and numbers objects (people by default), removes the ones you choose and fills the background in."
    input_types = []  # source node: reads its own images
    output_types = ["dataframe"]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                **c.source_schema("image"),
                "mode": {
                    "type": "select", "title": "Mode", "input_type": "dropdown",
                    "enum": list(MODES), "default": "remove_selected",
                    "description": (
                        "remove_selected = remove the objects in 'Object Numbers' (empty = remove ALL objects of the chosen classes). "
                        "keep_selected = remove everybody EXCEPT the numbers listed. "
                        "number_objects = remove nothing: only returns a preview with a number on every object, so you can choose."
                    ),
                },
                "classes": {
                    "type": "array", "items": {"type": "string"}, "title": "Objects to Find (classes)",
                    "input_type": "tags", "default": ["person"],
                    "description": "Which kinds of objects are numbered and can be removed, e.g. person, car, dog. Empty = every class the model knows.",
                },
                "object_numbers": {
                    "type": "array", "items": {"type": "string"}, "title": "Object Numbers",
                    "input_type": "tags",
                    "description": (
                        "Numbers from the numbered preview (left to right). Use the SAME model, thresholds, image size and classes "
                        "as the run that made the numbers."
                    ),
                },
                "model": {
                    "type": "select", "title": "Segmentation Model", "input_type": "dropdown",
                    "enum": SEG_MODELS, "default": DEFAULT_MODEL,
                    "description": "Gives the real outline of each object. Bigger = more exact outlines, slower.",
                },
                "confidence_threshold": {
                    "type": "number", "title": "Confidence Threshold", "input_type": "slider",
                    "default": 0.25, "minimum": 0.01, "maximum": 1.0,
                    "description": 'Minimum detection confidence score to identify target object for removal.'
                },
                "iou_threshold": {
                    "type": "number", "title": "Overlap (NMS IoU) Threshold", "input_type": "slider",
                    "default": 0.7, "minimum": 0.1, "maximum": 1.0,
                    "description": 'Intersection-Over-Union overlap threshold for inpainting mask.'
                },
                "image_size": {
                    "type": "integer", "title": "Inference Image Size (px)", "input_type": "number",
                    "default": 640, "minimum": 32, "maximum": 1920,
                    "description": "Raise it (for example 1024) when the people in the photo are small.",
                },
                "mask_padding_px": {
                    "type": "integer", "title": "Grow Removed Area (px)", "input_type": "number",
                    "default": 10, "minimum": 0, "maximum": 60,
                    "description": "Grows the outline of each removed object so edges, hair and soft shadows are covered. Too high eats into the background.",
                },
                "inpaint_max_side": {
                    "type": "integer", "title": "Fill Quality (max side, px)", "input_type": "number",
                    "default": 1024, "minimum": 256, "maximum": 2048,
                    "description": "Longest side of the area sent to the fill model. Higher = sharper fill, slower and more memory.",
                },
                "output_image_format": {
                    "type": "select", "title": "Result Image Format", "input_type": "dropdown",
                    "enum": list(OUTPUT_FORMATS), "default": "png",
                    "description": "png = lossless (bigger files), jpg = smaller.",
                },
                "save_numbered_preview": {
                    "type": "boolean", "title": "Also Save Numbered Preview", "input_type": "checkbox",
                    "default": False,
                    "description": "In number_objects mode the numbered preview is always saved. Tick this to get it in the other modes too (removed = red, kept = green).",
                },
            },
            "required": [],
        }

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        errors = c.validate_source(config, "image")
        if config.get("model", DEFAULT_MODEL) not in SEG_MODELS:
            errors.append(f"Unknown model '{config.get('model')}'. Choose one of: {', '.join(SEG_MODELS)}.")
        mode = config.get("mode", "remove_selected")
        if mode not in MODES:
            errors.append(f"mode must be one of: {', '.join(MODES)}.")
        if config.get("output_image_format", "png") not in OUTPUT_FORMATS:
            errors.append("output_image_format must be 'png' or 'jpg'.")
        for err in (
            c.validate_range(config, "confidence_threshold", 0.01, 1.0),
            c.validate_range(config, "iou_threshold", 0.1, 1.0),
            c.validate_range(config, "image_size", 32, 1920, integer=True),
            c.validate_range(config, "mask_padding_px", 0, 60, integer=True),
            c.validate_range(config, "inpaint_max_side", 256, 2048, integer=True),
        ):
            if err:
                errors.append(err)
        try:
            numbers = self._parse_numbers(config.get("object_numbers"))
        except ValueError as e:
            errors.append(str(e))
            numbers = []
        if mode == "keep_selected" and not numbers:
            errors.append("keep_selected needs Object Numbers: without them everybody would be removed.")
        return errors

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _parse_numbers(raw: Any) -> List[int]:
        """'2', '#2', '2, 4' or ['2', 4] -> [2, 4]. Raises ValueError on anything that is not a whole number >= 1."""
        if raw is None or raw == "" or raw == []:
            return []
        items = raw.split(",") if isinstance(raw, str) else list(raw)
        out = set()
        for x in items:
            s = str(x).strip().lstrip("#")
            if not s:
                continue
            if not s.isdigit() or int(s) < 1:
                raise ValueError(f"Object number '{x}' is not a whole number of 1 or more.")
            out.add(int(s))
        if len(out) > MAX_OBJECT_NUMBERS:
            raise ValueError(f"Too many object numbers (max {MAX_OBJECT_NUMBERS}).")
        return sorted(out)

    @staticmethod
    def _decode(data: bytes):
        import cv2
        import numpy as np

        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)  # also applies the photo's EXIF rotation
        if img is None:
            raise ValueError("Could not read the image file")
        h, w = img.shape[:2]
        if h * w > MAX_PIXELS:
            raise ValueError(f"Image is too large ({w}x{h}); the limit is {MAX_PIXELS:,} pixels")
        return img

    @staticmethod
    def _fill(h: int, w: int, polys: List[Any], grow: int = 0):
        """uint8 mask (255 inside the outlines), optionally grown by 'grow' pixels."""
        import cv2
        import numpy as np

        m = np.zeros((h, w), np.uint8)
        if polys:
            cv2.fillPoly(m, polys, 255)
            if grow > 0:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
                m = cv2.dilate(m, k)
        return m

    def _find_objects(self, img, model, cfg: Dict[str, Any], class_ids) -> List[Dict[str, Any]]:
        """Detect with the segmentation model; number the objects 1..N from left to right (then top to bottom)."""
        import numpy as np

        res = model.predict(
            img, conf=cfg["conf"], iou=cfg["iou"], imgsz=cfg["imgsz"], classes=class_ids,
            retina_masks=True, verbose=False,
        )[0]
        boxes = getattr(res, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []
        masks = getattr(res, "masks", None)
        polys = masks.xy if masks is not None and len(masks.xy) == len(boxes) else [None] * len(boxes)
        objs = []
        for cls_id, conf, xy, poly in zip(boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist(), polys):
            objs.append({
                "label": res.names[int(cls_id)],
                "confidence": round(float(conf), 4),
                "x_min": round(float(xy[0]), 1), "y_min": round(float(xy[1]), 1),
                "x_max": round(float(xy[2]), 1), "y_max": round(float(xy[3]), 1),
                "poly": np.asarray(poly, dtype=np.int32) if poly is not None and len(poly) >= 3 else None,
            })
        objs.sort(key=lambda o: (o["x_min"], o["y_min"]))
        for i, o in enumerate(objs, start=1):
            o["number"] = i
        return objs

    @staticmethod
    def _decide(objs: List[Dict[str, Any]], mode: str, numbers: List[int]) -> List[int]:
        """Sets o['action'] ('listed' / 'removed' / 'kept' / 'no_outline') and returns the numbers asked for that do not exist."""
        chosen = set(numbers)
        existing = {o["number"] for o in objs}
        for o in objs:
            if mode == "number_objects":
                o["action"] = "listed"
            elif mode == "keep_selected":
                o["action"] = "kept" if o["number"] in chosen else "removed"
            else:  # remove_selected: empty list = remove everything
                o["action"] = "removed" if (not chosen or o["number"] in chosen) else "kept"
            if o["action"] == "removed" and o["poly"] is None:
                o["action"] = "no_outline"  # the model gave no outline for it: cannot be erased cleanly
        return sorted(chosen - existing)

    @staticmethod
    def _draw_numbers(img, objs: List[Dict[str, Any]]):
        """Preview: every object tinted + outlined, with its number and class."""
        import cv2

        h, w = img.shape[:2]
        scale = max(h, w) / 1000.0
        thick = max(1, int(round(2 * scale)))
        font_scale = max(0.5, 0.8 * scale)
        overlay = img.copy()
        for o in objs:
            if o["poly"] is not None:
                cv2.fillPoly(overlay, [o["poly"]], COLORS.get(o["action"], COLORS["listed"]))
        out = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
        for o in objs:
            color = COLORS.get(o["action"], COLORS["listed"])
            if o["poly"] is not None:
                cv2.polylines(out, [o["poly"]], True, color, thick)
            text = f'{o["number"]} {o["label"]}'
            (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)
            x, y = int(o["x_min"]), max(int(o["y_min"]), th + base + 2)
            cv2.rectangle(out, (x, y - th - base - 2), (x + tw + 4, y + 2), color, -1)
            cv2.putText(out, text, (x + 2, y - base), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thick, cv2.LINE_AA)
        return out

    @staticmethod
    def _inpaint(lama: _LamaInpainter, img, mask, max_side: int):
        """
        Fill the masked area. Only a crop around it (with context) goes to LaMa, so the rest of the photo keeps its
        full resolution, then the filled part is pasted back with a soft outward edge.
        """
        import cv2
        import numpy as np

        h, w = mask.shape
        ys, xs = np.where(mask > 0)
        if len(ys) == 0:
            return img
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
        margin = int(max(x1 - x0, y1 - y0) * CONTEXT_FRAC) + CONTEXT_MIN_PX
        cx0, cy0, cx1, cy1 = max(0, x0 - margin), max(0, y0 - margin), min(w, x1 + margin), min(h, y1 + margin)
        crop, cmask = img[cy0:cy1, cx0:cx1], mask[cy0:cy1, cx0:cx1]
        ch, cw = cmask.shape
        scale = min(1.0, max_side / float(max(ch, cw)))
        if scale < 1.0:
            size = (max(8, int(round(cw * scale))), max(8, int(round(ch * scale))))
            filled = lama(cv2.resize(crop, size, interpolation=cv2.INTER_AREA), cv2.resize(cmask, size, interpolation=cv2.INTER_NEAREST))
            filled = cv2.resize(filled, (cw, ch), interpolation=cv2.INTER_CUBIC)
        else:
            filled = lama(crop, cmask)
        soft = cv2.GaussianBlur(cmask, (FEATHER_PX, FEATHER_PX), 0)
        alpha = (np.maximum(soft, cmask).astype(np.float32) / 255.0)[..., None]  # 1 inside the hole, soft only OUTSIDE it
        out = img.copy()
        out[cy0:cy1, cx0:cx1] = (crop.astype(np.float32) * (1.0 - alpha) + filled.astype(np.float32) * alpha).astype(np.uint8)
        return out

    @staticmethod
    def _save_image(img, fmt: str, tenant_id: int, path_no_ext: str) -> str:
        import cv2
        from backend.app.infrastructure.storage.ozone_client import upload_file_to_ozone

        if fmt == "png":
            ok, buf = cv2.imencode(".png", img)
            mime, ext = "image/png", "png"
        else:
            ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            mime, ext = "image/jpeg", "jpg"
        if not ok:
            raise ValueError("Could not encode the result image")
        return upload_file_to_ozone(buf.tobytes(), f"{path_no_ext}.{ext}", tenant_id, mime)

    def _process_image(self, img, model, lama, cfg: Dict[str, Any], class_ids, tenant_id, idx, source_value, run_tag) -> Dict[str, Any]:
        h, w = img.shape[:2]
        objs = self._find_objects(img, model, cfg, class_ids)
        not_found = self._decide(objs, cfg["mode"], cfg["numbers"])
        base = str(source_value).rstrip("/").rsplit("/", 1)[-1].rsplit(".", 1)[0] or "image"
        stem = f"{OUTPUT_FOLDER}/{run_tag}/{idx}_{base}"
        out: Dict[str, Any] = {
            "image_width": w, "image_height": h, "objects": objs, "numbers_not_found": not_found,
            "result_path": None, "numbered_path": None, "note": None,
        }

        removed = [o for o in objs if o["action"] == "removed"]
        if cfg["mode"] != "number_objects":
            if not objs:
                out["note"] = "No objects of the chosen classes were found, so nothing was removed."
            elif not removed:
                out["note"] = "Nothing matched the selection, so nothing was removed."
            else:
                mask = self._fill(h, w, [o["poly"] for o in removed], cfg["grow"])
                keep_polys = [o["poly"] for o in objs if o["action"] == "kept" and o["poly"] is not None]
                if keep_polys:
                    mask[self._fill(h, w, keep_polys) > 0] = 0  # never paint over an object that stays
                result = self._inpaint(lama, img, mask, cfg["max_side"])
                out["result_path"] = self._save_image(result, cfg["out_fmt"], tenant_id, f"{stem}_removed")
        if cfg["mode"] == "number_objects" or cfg["save_preview"]:
            out["numbered_path"] = self._save_image(self._draw_numbers(img, objs), "jpg", tenant_id, f"{stem}_numbered")
        if cfg["mode"] == "number_objects" and not objs:
            out["note"] = "No objects of the chosen classes were found."
        return out

    # ------------------------------------------------------------------ execution
    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        errors = self.validate_config(config)
        if errors:
            raise ValueError("; ".join(errors))
        tenant_id = c.require_tenant(config, self.name, "images")
        df, img_col, from_files = c.resolve_source(inputs, config, self.name, "image")

        model_name = config.get("model", DEFAULT_MODEL)
        mode = config.get("mode", "remove_selected")
        cfg = {
            "mode": mode,
            "numbers": self._parse_numbers(config.get("object_numbers")),
            "conf": float(config.get("confidence_threshold", 0.25)),
            "iou": float(config.get("iou_threshold", 0.7)),
            "imgsz": int(config.get("image_size", 640)),
            "grow": int(config.get("mask_padding_px", 10)),
            "max_side": int(config.get("inpaint_max_side", 1024)),
            "out_fmt": config.get("output_image_format", "png"),
            "save_preview": bool(config.get("save_numbered_preview", False)),
        }
        classes = od.ObjectDetectionRecipe._parse_classes(config.get("classes", ["person"]))
        run_tag = f"{model_name.rsplit('.', 1)[0]}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"

        model = _new_model(model_name)  # one model for all images of this run
        class_ids = None
        if classes:
            lookup = {str(v).lower(): int(k) for k, v in model.names.items()}
            unknown = [x for x in classes if x.lower() not in lookup]
            if unknown:
                raise ValueError(f"Unknown class(es): {unknown}. This model detects: {sorted(lookup)}")
            class_ids = [lookup[x.lower()] for x in classes]
        lama = _LamaInpainter() if mode != "number_objects" else None  # loaded once, only when something will be removed

        per_image: List[Dict[str, Any]] = []
        for idx, value in enumerate(df[img_col].tolist()):
            row: Dict[str, Any] = {"objects": [], "error": None, "result_path": None, "numbered_path": None,
                                   "numbers_not_found": [], "note": None, "image_width": None, "image_height": None}
            try:
                data, _ext = c.read_media_bytes(value, tenant_id, from_files, "image")
                img = self._decode(data)
                del data
                row.update(self._process_image(img, model, lama, cfg, class_ids, tenant_id, idx, value, run_tag))
            except Exception as e:  # one bad image must not fail the whole run
                row["error"] = c.safe_error(e)
            per_image.append(row)

        # ---- output table: every image appears at least once (even with no objects / an error)
        obj_cols = ["object_number", "label", "confidence", "x_min", "y_min", "x_max", "y_max", "action"]
        rows = []
        for i, r in enumerate(per_image):
            src = df.iloc[i].to_dict()
            base = {
                **src,
                "image_width": r["image_width"], "image_height": r["image_height"],
                "removal_error": r["error"], "note": r["note"],
                "result_image_url": c.preview_url(r["result_path"]) if r["result_path"] else None,
                "numbered_preview_url": c.preview_url(r["numbered_path"]) if r["numbered_path"] else None,
            }
            if r["objects"]:
                for o in r["objects"]:
                    rows.append({**base, "object_number": o["number"], "label": o["label"], "confidence": o["confidence"],
                                 "x_min": o["x_min"], "y_min": o["y_min"], "x_max": o["x_max"], "y_max": o["y_max"],
                                 "action": o["action"]})
            else:
                rows.append({**base, **{k: None for k in obj_cols}})
        out_df = pd.DataFrame(rows)

        failed = [r for r in per_image if r["error"]]
        previews = []
        for value, r in zip(df[img_col].tolist(), per_image):
            by_action: Dict[str, List[int]] = {}
            for o in r["objects"]:
                by_action.setdefault(o["action"], []).append(o["number"])
            extra: Dict[str, Any] = {}
            if r["numbered_path"]:
                extra["numbered_preview"] = r["numbered_path"]
            if r["note"]:
                extra["note"] = r["note"]
            if r["numbers_not_found"]:
                extra["numbers_not_found"] = r["numbers_not_found"]
            previews.append(c.preview_item(
                original=value,
                media=r["result_path"] or r["numbered_path"],  # the edited photo, or the numbered preview in number_objects mode
                error=r["error"],
                num_objects=len(r["objects"]),
                removed_numbers=by_action.get("removed", []),
                kept_numbers=by_action.get("kept", []),
                no_outline_numbers=by_action.get("no_outline", []),
                **extra,
            ))
        all_objs = [o for r in per_image for o in r["objects"]]
        metrics = {
            "task_type": "object_removal",
            "model": model_name,
            "inpainting_model": "big-lama" if lama is not None else None,
            "mode": mode,
            "classes": classes,
            "input_source": "image_files",
            "items_processed": len(per_image) - len(failed),
            "items_failed": len(failed),
            "objects_found": len(all_objs),
            "objects_removed": sum(1 for o in all_objs if o["action"] == "removed"),
            "errors_sample": [r["error"] for r in failed[:5]],
            "preview_items": previews,
        }
        return {"dataframe": out_df, "metrics": metrics, "output_summary": metrics, "task_type": "object_removal"}

    def to_code(self, config: Dict[str, Any]) -> str:
        return (
            "from ultralytics import YOLO\n\n"
            f"model = YOLO({config.get('model', DEFAULT_MODEL)!r})\n"
            "res = model.predict('<image>', retina_masks=True)[0]\n"
            "# res.masks.xy -> outline of each object; fill the removed ones in a mask (255 = remove)\n"
            "# then fill the hole with LaMa (big-lama.pt) or cv2.inpaint for a quick, lower-quality result"
        )