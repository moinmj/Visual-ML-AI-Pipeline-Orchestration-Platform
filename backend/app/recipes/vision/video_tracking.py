"""
Video Frame Extractor + Object Tracking recipe (Modeling -> Computer Vision & Audio).

Reads each video, takes every Nth frame, runs a pretrained YOLO detector with a
tracker (ByteTrack / BoT-SORT) and gives every object an id that stays the same
across frames. Output: one row per tracked object (default) or per detection.

Videos come from 'video_files' (picked from unstructured/; same rules and security checks
as the image recipes, see _common.py). There is no table / column input.

Place this file at:  backend/app/recipes/vision/video_tracking.py
Needs:               pip install ultralytics lapx   (opencv comes with easyocr / ultralytics)

Security: only built-in model/tracker names are accepted; videos are read only from the
caller's tenant folder; limits come from core/media_types.py plus the caps below;
preview frames go to outputs/video_tracking/, never into unstructured/.
"""
import json
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from backend.app.recipes.base.recipe import BaseRecipe
from backend.app.recipes.vision import _common as c
from backend.app.recipes.vision import object_detection as od

SUPPORTED_MODELS = od.SUPPORTED_MODELS
DEFAULT_MODEL = "yolov8n.pt"  # smallest: video means many frames
TRACKERS = ["bytetrack.yaml", "botsort.yaml"]
DEFAULT_TRACKER = "bytetrack.yaml"
MAX_FRAMES_CAP = 1000          # processed frames per video
PREVIEW_FRAMES = 4             # annotated frames saved per video
OUTPUT_FOLDER = "outputs/video_tracking"
OUTPUT_FORMATS = ("one_row_per_track", "one_row_per_detection")


def _new_model(model_name: str):
    """
    A NEW detector per video. Trackers keep state inside the model, so the shared
    cache used by the image recipes must not be used here (two videos or two
    users would mix their object ids).
    """
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise RuntimeError("Video tracking needs 'ultralytics'. Install it with: pip install ultralytics lapx") from e
    return YOLO(model_name)


class VideoObjectTrackingRecipe(BaseRecipe):
    recipe_id = "video_object_tracking"
    name = "Video Frame Extractor + Object Tracking"
    version = "1.0.0"
    category = "vision_audio"
    media_type = "video"
    description = "Extracts frames from video and tracks objects across them."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                **c.source_schema("video"),
                "model": {
                    "type": "select", "title": "Detection Model", "input_type": "dropdown",
                    "enum": SUPPORTED_MODELS, "default": DEFAULT_MODEL,
                    "description": "Pretrained YOLO model (80 COCO classes). Smaller = faster; video needs many frames.",
                },
                "tracker": {
                    "type": "select", "title": "Tracker", "input_type": "dropdown",
                    "enum": TRACKERS, "default": DEFAULT_TRACKER,
                    "description": "bytetrack = fast and simple. botsort = better when the camera moves.",
                },
                "frame_stride": {
                    "type": "integer", "title": "Use Every Nth Frame", "input_type": "number",
                    "default": 5, "minimum": 1, "maximum": 300,
                    "description": "5 = analyse 1 frame out of 5. Lower = more accurate tracking, slower.",
                },
                "max_frames": {
                    "type": "integer", "title": "Max Frames per Video", "input_type": "number",
                    "default": 300, "minimum": 1, "maximum": MAX_FRAMES_CAP,
                    "description": "Stops after this many analysed frames (not counting skipped ones).",
                },
                "confidence_threshold": {
                    "type": "number", "title": "Confidence Threshold", "input_type": "slider",
                    "default": 0.25, "minimum": 0.01, "maximum": 1.0,
                },
                "iou_threshold": {
                    "type": "number", "title": "Overlap (NMS IoU) Threshold", "input_type": "slider",
                    "default": 0.7, "minimum": 0.1, "maximum": 1.0,
                },
                "image_size": {
                    "type": "integer", "title": "Inference Image Size (px)", "input_type": "number",
                    "default": 640, "minimum": 32, "maximum": 1920,
                },
                "classes": {
                    "type": "array", "items": {"type": "string"}, "title": "Only These Classes",
                    "input_type": "tags",
                    "description": "Optional. Track only these labels, e.g. person, car. Empty = all classes.",
                },
                "output_format": {
                    "type": "select", "title": "Output Format", "input_type": "dropdown",
                    "enum": list(OUTPUT_FORMATS), "default": "one_row_per_track",
                    "description": "One row per tracked object (first/last frame, duration) or one row per detection in each frame.",
                },
                "save_preview_frames": {
                    "type": "boolean", "title": "Save Preview Frames", "input_type": "checkbox",
                    "default": False,
                    "description": f"Saves {PREVIEW_FRAMES} annotated frames per video (boxes + track ids) into your Ozone folder ({OUTPUT_FOLDER}/<model>_<time>/).",
                },
            },
            "required": [],
        }

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        errors = c.validate_source(config, "video")
        if config.get("model", DEFAULT_MODEL) not in SUPPORTED_MODELS:
            errors.append(f"Unknown model '{config.get('model')}'. Choose one of: {', '.join(SUPPORTED_MODELS)}.")
        if config.get("tracker", DEFAULT_TRACKER) not in TRACKERS:
            errors.append(f"Unknown tracker '{config.get('tracker')}'. Choose one of: {', '.join(TRACKERS)}.")
        for err in (
            c.validate_range(config, "frame_stride", 1, 300, integer=True),
            c.validate_range(config, "max_frames", 1, MAX_FRAMES_CAP, integer=True),
            c.validate_range(config, "confidence_threshold", 0.01, 1.0),
            c.validate_range(config, "iou_threshold", 0.1, 1.0),
            c.validate_range(config, "image_size", 32, 1920, integer=True),
        ):
            if err:
                errors.append(err)
        if config.get("output_format", "one_row_per_track") not in OUTPUT_FORMATS:
            errors.append("output_format must be 'one_row_per_track' or 'one_row_per_detection'.")
        return errors

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _parse_frame(res, frame_index: int, time_s: Optional[float]) -> List[Dict[str, Any]]:
        boxes = getattr(res, "boxes", None)
        ids = getattr(boxes, "id", None) if boxes is not None else None
        if ids is None:  # nothing tracked in this frame
            return []
        id_l, cls_l = ids.tolist(), boxes.cls.tolist()
        conf_l, xy_l = boxes.conf.tolist(), boxes.xyxy.tolist()
        out = []
        for tid, cls_id, conf, xy in zip(id_l, cls_l, conf_l, xy_l):
            out.append({
                "frame_index": frame_index,
                "time_s": time_s,
                "track_id": int(tid),
                "label": res.names[int(cls_id)],
                "confidence": round(float(conf), 4),
                "x_min": round(float(xy[0]), 1), "y_min": round(float(xy[1]), 1),
                "x_max": round(float(xy[2]), 1), "y_max": round(float(xy[3]), 1),
            })
        return out

    @staticmethod
    def _summarise_tracks(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        by_id: Dict[int, List[Dict[str, Any]]] = {}
        for d in detections:
            by_id.setdefault(d["track_id"], []).append(d)
        tracks = []
        for tid, ds in sorted(by_id.items()):
            first, last = ds[0], ds[-1]
            t0, t1 = first["time_s"], last["time_s"]
            tracks.append({
                "track_id": tid,
                "label": Counter(d["label"] for d in ds).most_common(1)[0][0],
                "first_frame": first["frame_index"], "last_frame": last["frame_index"],
                "first_time_s": t0, "last_time_s": t1,
                "duration_s": round(t1 - t0, 3) if t0 is not None and t1 is not None else None,
                "num_detections": len(ds),
                "avg_confidence": round(sum(d["confidence"] for d in ds) / len(ds), 4),
            })
        return tracks

    def _track_video(self, path: str, cfg: Dict[str, Any], save_preview, tenant_id, vidx, source_value, run_tag) -> Dict[str, Any]:
        import cv2

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError("Could not open the video file")
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            info = {
                "video_width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
                "video_height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
                "fps": round(fps, 3) if fps else None,
            }
            stride, max_frames = cfg["stride"], cfg["max_frames"]
            model = _new_model(cfg["model"])

            class_ids = None
            if cfg["classes"]:
                lookup = {str(v).lower(): int(k) for k, v in model.names.items()}
                unknown = [x for x in cfg["classes"] if x.lower() not in lookup]
                if unknown:
                    raise ValueError(f"Unknown class(es): {unknown}. This model detects: {sorted(lookup)}")
                class_ids = [lookup[x.lower()] for x in cfg["classes"]]

            est = min(max_frames, -(-total // stride)) if total > 0 else 0
            preview_at = {int(i * (est - 1) / max(PREVIEW_FRAMES - 1, 1)) for i in range(PREVIEW_FRAMES)} if est > 1 else {0}

            detections: List[Dict[str, Any]] = []
            previews: List[Dict[str, Any]] = []
            idx = processed = 0
            while processed < max_frames:
                if idx % stride != 0:
                    if not cap.grab():
                        break
                    idx += 1
                    continue
                ok, frame = cap.read()
                if not ok:
                    break
                res = model.track(
                    frame, persist=True, tracker=cfg["tracker"], conf=cfg["conf"], iou=cfg["iou"],
                    imgsz=cfg["imgsz"], classes=class_ids, verbose=False,
                )[0]
                time_s = round(idx / fps, 3) if fps > 0 else None
                detections.extend(self._parse_frame(res, idx, time_s))
                if save_preview and processed in preview_at:
                    previews.append(self._save_preview(res, tenant_id, vidx, source_value, idx, run_tag))
                processed += 1
                idx += 1
        finally:
            cap.release()

        return {**info, "frames_processed": processed, "detections": detections, "previews": previews}

    @staticmethod
    def _save_preview(res, tenant_id: int, vidx: int, source_value: Any, frame_index: int, run_tag: str) -> Dict[str, Any]:
        import cv2
        from backend.app.infrastructure.storage.ozone_client import upload_file_to_ozone

        ok, buf = cv2.imencode(".jpg", res.plot(), [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            raise ValueError("Could not encode preview frame")
        base = str(source_value).rstrip("/").rsplit("/", 1)[-1].rsplit(".", 1)[0] or "video"
        path = upload_file_to_ozone(buf.tobytes(), f"{OUTPUT_FOLDER}/{run_tag}/{vidx}_{base}_f{frame_index}.jpg", tenant_id, "image/jpeg")
        return {"frame_index": frame_index, "media_path": path, "media_preview_url": c.preview_url(path)}

    # ------------------------------------------------------------------ execution
    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        errors = self.validate_config(config)
        if errors:
            raise ValueError("; ".join(errors))
        tenant_id = c.require_tenant(config, self.name, "videos")
        df, video_col, from_files = c.resolve_source(inputs, config, self.name, "video")

        model_name = config.get("model", DEFAULT_MODEL)
        cfg = {
            "model": model_name,
            "tracker": config.get("tracker", DEFAULT_TRACKER),
            "stride": int(config.get("frame_stride", 5)),
            "max_frames": int(config.get("max_frames", 300)),
            "conf": float(config.get("confidence_threshold", 0.25)),
            "iou": float(config.get("iou_threshold", 0.7)),
            "imgsz": int(config.get("image_size", 640)),
            "classes": od.ObjectDetectionRecipe._parse_classes(config.get("classes")),
        }
        out_format = config.get("output_format", "one_row_per_track")
        save_preview = bool(config.get("save_preview_frames", False))
        run_tag = f"{model_name.rsplit('.', 1)[0]}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"

        per_video: List[Dict[str, Any]] = []
        for vidx, value in enumerate(df[video_col].tolist()):
            row: Dict[str, Any] = {"detections": [], "tracks": [], "previews": [], "error": None, "frames_processed": 0}
            tmp_path = None
            try:
                data, ext = c.read_media_bytes(value, tenant_id, from_files, "video")
                fd, tmp_path = tempfile.mkstemp(suffix=f".{ext}")
                with os.fdopen(fd, "wb") as f:  # OpenCV needs a real file path
                    f.write(data)
                del data
                result = self._track_video(tmp_path, cfg, save_preview, tenant_id, vidx, value, run_tag)
                row.update(result)
                row["tracks"] = self._summarise_tracks(result["detections"])
            except Exception as e:  # one bad video must not fail the whole run
                row["error"] = c.safe_error(e)
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            per_video.append(row)

        # ---- output table: every video appears at least once (even with no tracks / an error)
        track_cols = ["track_id", "label", "first_frame", "last_frame", "first_time_s", "last_time_s",
                      "duration_s", "num_detections", "avg_confidence"]
        det_cols = ["frame_index", "time_s", "track_id", "label", "confidence", "x_min", "y_min", "x_max", "y_max"]
        item_cols = track_cols if out_format == "one_row_per_track" else det_cols
        rows = []
        for i, r in enumerate(per_video):
            src = df.iloc[i].to_dict()
            base = {
                **src,
                "video_width": r.get("video_width"), "video_height": r.get("video_height"), "fps": r.get("fps"),
                "frames_processed": r["frames_processed"], "tracking_error": r["error"],
                "video_preview_url": c.preview_url(src[video_col]),
            }
            if save_preview:
                base["preview_frame_urls"] = json.dumps([p["media_preview_url"] for p in r["previews"]])
            items = r["tracks"] if out_format == "one_row_per_track" else r["detections"]
            if items:
                rows.extend({**base, **it} for it in items)
            else:
                rows.append({**base, **{k: None for k in item_cols}})
        out_df = pd.DataFrame(rows)

        failed = [r for r in per_video if r["error"]]
        label_tracks: Dict[str, int] = {}
        for r in per_video:
            for t in r["tracks"]:
                label_tracks[t["label"]] = label_tracks.get(t["label"], 0) + 1
        previews = []
        for value, r in zip(df[video_col].tolist(), per_video):
            per_class = Counter(t["label"] for t in r["tracks"])
            previews.append(c.preview_item(
                original=value,
                media=r["previews"][0]["media_path"] if r["previews"] else None,  # first annotated frame
                error=r["error"],
                frames_processed=r["frames_processed"],
                num_tracks=len(r["tracks"]),
                tracks_per_class=dict(per_class),
                preview_frames=r["previews"],  # every saved frame: {frame_index, media_path, media_preview_url}
            ))
        metrics = {
            "task_type": "video_object_tracking",
            "model": model_name,
            "tracker": cfg["tracker"],
            "input_source": "video_files",
            "items_processed": len(per_video) - len(failed),
            "items_failed": len(failed),
            "frames_processed": int(sum(r["frames_processed"] for r in per_video)),
            "total_tracks": int(sum(label_tracks.values())),
            "tracks_per_class": dict(sorted(label_tracks.items(), key=lambda kv: kv[1], reverse=True)),
            "frame_stride": cfg["stride"],
            "errors_sample": [r["error"] for r in failed[:5]],
            "preview_items": previews,
        }
        return {"dataframe": out_df, "metrics": metrics, "output_summary": metrics, "task_type": "video_object_tracking"}

    def to_code(self, config: Dict[str, Any]) -> str:
        files = c.image_files(config, "video")
        source = repr(files[0]) if files else "'unstructured/<video>'"
        return (
            "from ultralytics import YOLO\n\n"
            f"model = YOLO({config.get('model', DEFAULT_MODEL)!r})\n"
            f"results = model.track({source}, tracker={config.get('tracker', DEFAULT_TRACKER)!r},\n"
            f"                      vid_stride={config.get('frame_stride', 5)}, stream=True)\n"
            "for r in results:\n"
            "    if r.boxes.id is not None:\n"
            "        print(r.boxes.id.tolist(), r.boxes.cls.tolist())"
        )