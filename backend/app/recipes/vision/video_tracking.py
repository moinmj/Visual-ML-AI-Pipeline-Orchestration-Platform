"""
Video Frame Extractor + Object Tracking recipe (Modeling -> Computer Vision & Audio).

Reads each video, takes every Nth frame, runs a pretrained YOLO detector with a
tracker (ByteTrack / BoT-SORT) and gives every object an id that stays the same
across frames. Output: one row per tracked object (default) or per detection.

Videos come from 'video_files' (picked from unstructured/; same rules and security checks
as the image recipes, see _common.py). There is no table / column input.

Annotated video: with 'save_annotated_video' the WHOLE video is written back with the boxes, the
category (class) names and the track ids drawn on it, as an H.264 .mp4 that plays in a browser
<video> tag. Frames that are skipped by 'frame_stride' re-use the boxes of the last analysed frame,
so the video keeps its original frame rate and looks continuous. Every result also carries a
per-category summary ('categories').

STAGE 1 - Gender prediction (new):
  * 'predict_gender' runs a CLIP zero-shot classifier ("a photo of a man" / "a photo of a woman") on the
    crop of every tracked PERSON. One frame is unreliable, so predictions are VOTED per track id:
    the probabilities of up to GENDER_MAX_VOTES frames are averaged. Until GENDER_MIN_VOTES votes exist
    (or when the average is not clear enough) the person is "unknown".
  * 'target_gender' (any / male / female): tracking always follows everybody (this keeps ids stable);
    only the drawn boxes and the output table are filtered. In the TABLE the final voted gender of the
    whole track is used. In the VIDEO the running vote is used, so a person is drawn only from the moment
    the vote says they match the target.
  * Gender here is a PREDICTION from appearance, never a fact. Check privacy / biometric rules for your footage.

STAGE 2 - Target by track id (new):
  * 'target_track_ids' keeps ONLY the listed track ids in the video and in the table. Everybody is still tracked
    internally (this keeps ids stable); only drawing and output are filtered. Works together with 'target_gender'
    (both must match). Typical workflow: run once with 'save_annotated_video', read the ids on the video, run again
    with the id of your person.
  * Track ids are per video (they start again at 1 in every video) and the tracker can swap or lose an id when people
    overlap or leave the frame. Use the ids on ONE video per run. The result reports ids that were never seen.

STAGE 2.2 - Remove the other people (new):
  * 'remove_other_people' writes a SECOND video ('..._cleaned.mp4', no boxes) in which every person who is not a
    target (by 'target_track_ids' and/or 'target_gender') is erased and the scene behind them is filled in.
  * Needs a segmentation model ('removal_model', yolov8*-seg: it gives the real outline of each body, not a box).
    It replaces 'model' for the run. Every frame is analysed (frame stride is forced to 1), so it is slower.
  * Background: for a FIXED camera. A median over up to BG_SAMPLES frames, ignoring the pixels covered by people,
    gives an empty scene that is pasted over the erased people (soft edges; the target's own outline is protected).
    A camera that moves is not supported here (that needs a video inpainting model).
  * Two passes: pass 1 tracks, votes on gender and collects outlines; pass 2 re-reads the video and writes the clean one.
  * Limits: someone who never moves leaves a gap in the background (reported as 'background_gap_ratio'); a person the
    detector misses for a few frames flickers back; a target whose id changes is treated as someone else and erased.
  * Editing footage changes it: keep the original, and label the cleaned video as edited.

Place this file at:  backend/app/recipes/vision/video_tracking.py
Needs:               pip install ultralytics lapx   (opencv comes with easyocr / ultralytics)
                     pip install "av<19"            (only for the annotated video; already needed by speech-to-text)
                     pip install transformers pillow (only for predict_gender / target_gender; torch comes with ultralytics)

Security: only built-in model/tracker names are accepted; videos are read only from the
caller's tenant folder; limits come from core/media_types.py plus the caps below;
preview frames go to outputs/video_tracking/, never into unstructured/.
"""
import json
import os
import tempfile
import warnings
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple

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
MAX_OUTPUT_VIDEO_FRAMES = 3000  # frames written to the annotated video (~100 s at 30 fps); longer is cut
OUTPUT_FOLDER = "outputs/video_tracking"
OUTPUT_FORMATS = ("one_row_per_track", "one_row_per_detection")

# ---- Stage 1: gender prediction (fixed built-in model name, never taken from the user)
GENDER_MODEL = "openai/clip-vit-base-patch32"
GENDER_PROMPTS = ["a photo of a man", "a photo of a woman"]  # index 0 = male, index 1 = female
TARGET_GENDERS = ("any", "male", "female")
MAX_TARGET_IDS = 50       # Stage 2: most track ids accepted in 'target_track_ids'

# ---- Stage 2.2: remove the people who are not the target (fixed camera, median background)
SEG_MODELS = ["yolov8n-seg.pt", "yolov8s-seg.pt", "yolov8m-seg.pt"]  # built-in segmentation models (give body outlines)
DEFAULT_SEG_MODEL = "yolov8n-seg.pt"
BG_SAMPLES = 20           # frames used to build the empty-scene background
BG_BAND_ROWS = 64         # the median is computed in bands of rows, so memory stays small
REMOVE_DILATE_PX = 7      # grow the removed outlines a little: covers edges, hair, shadows
REMOVE_FEATHER_PX = 9     # soft edge (odd number) so the patch blends in
BG_GAP_WARN = 0.02        # more than 2% of the scene never seen without a person -> warn in the result
GENDER_EVERY = 2          # predict on every 2nd analysed frame of a track (saves time)
GENDER_MAX_VOTES = 15     # stop predicting for a track after this many votes
GENDER_MIN_VOTES = 3      # fewer votes than this -> "unknown"
GENDER_MIN_CONF = 0.60    # averaged confidence below this -> "unknown"
GENDER_MIN_CROP_W = 20    # person crops smaller than this (px) are skipped
GENDER_MIN_CROP_H = 40
GENDER_COLORS = {         # BGR
    "male": (255, 128, 0),
    "female": (180, 105, 255),
    "unknown": (160, 160, 160),
    None: (0, 200, 0),
}


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


class _GenderClassifier:
    """CLIP zero-shot male/female classifier for person crops. Loaded ONCE per run (not per frame or video)."""

    def __init__(self):
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as e:
            raise RuntimeError("Gender prediction needs 'transformers'. Install it with: pip install transformers pillow") from e
        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(GENDER_MODEL).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(GENDER_MODEL)

    def p_male(self, crops_bgr: List[Any]) -> List[float]:
        """Probability 'male' (vs 'female') for each BGR crop."""
        import cv2
        from PIL import Image

        images = [Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)) for crop in crops_bgr]
        inputs = self.processor(text=GENDER_PROMPTS, images=images, return_tensors="pt", padding=True).to(self.device)
        with self._torch.no_grad():
            logits = self.model(**inputs).logits_per_image  # shape (n_crops, 2)
            probs = logits.softmax(dim=1)[:, 0]
        return [float(p) for p in probs.tolist()]


class _AnnotatedVideoWriter:
    """Writes BGR frames into an H.264 .mp4 (plays in every browser) with PyAV. 'faststart' puts the index at the
    start of the file, so the browser can begin playing without downloading everything first."""

    def __init__(self, path: str, fps: float):
        try:
            import av
        except ImportError as e:
            raise RuntimeError("The annotated video needs 'av'. Install it with: pip install \"av<19\"") from e
        self._av = av
        self.fps = Fraction(fps if fps and fps > 0 else 25.0).limit_denominator(1001)
        self.container = av.open(path, mode="w", options={"movflags": "faststart"})
        self.stream = None
        self.size = (0, 0)
        self.frames = 0

    def write(self, bgr) -> None:
        import numpy as np

        h, w = bgr.shape[:2]
        if self.stream is None:
            self.size = (w - w % 2, h - h % 2)  # H.264 (yuv420p) needs even sizes
            self.stream = self.container.add_stream("libx264", rate=self.fps)
            self.stream.width, self.stream.height = self.size
            self.stream.pix_fmt = "yuv420p"
            self.stream.options = {"crf": "23", "preset": "veryfast"}
        frame = self._av.VideoFrame.from_ndarray(np.ascontiguousarray(bgr[: self.size[1], : self.size[0]]), format="bgr24")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)
        self.frames += 1

    def close(self) -> None:
        try:
            if self.stream is not None:
                for packet in self.stream.encode():
                    self.container.mux(packet)
        finally:
            self.container.close()


class VideoObjectTrackingRecipe(BaseRecipe):
    recipe_id = "video_object_tracking"
    name = "Video Frame Extractor + Object Tracking"
    version = "1.3.0"
    category = "vision_audio"
    media_type = "video"
    description = "Extracts frames from video and tracks objects across them. Optionally predicts the gender of tracked people."
    input_types = []  # source node: reads its own files, takes no input
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
                "predict_gender": {
                    "type": "boolean", "title": "Predict Gender of People", "input_type": "checkbox",
                    "default": False,
                    "description": (
                        "Predicts male / female for every tracked person with CLIP, voted over many frames per track. "
                        "Adds 'gender' and 'gender_confidence' columns. It is a prediction from appearance and can be wrong; "
                        "small or unclear people stay 'unknown'."
                    ),
                },
                "target_gender": {
                    "type": "select", "title": "Target Gender", "input_type": "dropdown",
                    "enum": list(TARGET_GENDERS), "default": "any",
                    "description": (
                        "any = everybody. male / female = only people predicted as that gender are drawn and kept in the table "
                        "(everybody is still tracked internally). Turns gender prediction on automatically. 'unknown' people are hidden."
                    ),
                },
                "target_track_ids": {
                    "type": "array", "items": {"type": "string"}, "title": "Only These Track IDs",
                    "input_type": "tags",
                    "description": (
                        "Optional. Draw and keep only these track ids, e.g. 3 or 3, 7. Empty = everybody. Run once with "
                        "'Save Annotated Video' to read the ids on the video, then rerun with the id you want. Ids are per video "
                        "(use one video per run) and can change when people overlap or leave the frame. "
                        "Combined with Target Gender, both must match."
                    ),
                },
                "remove_other_people": {
                    "type": "boolean", "title": "Remove Other People (Clean Video)", "input_type": "checkbox",
                    "default": False,
                    "description": (
                        "Saves a second video in which everybody who is NOT a target is erased and the background is filled in. "
                        "Needs Only These Track IDs and/or Target Gender. For a FIXED camera. Analyses every frame (slower; "
                        "raise Max Frames per Video for longer clips). Uses the segmentation model below instead of the Detection Model."
                    ),
                },
                "removal_model": {
                    "type": "select", "title": "Segmentation Model (for removal)", "input_type": "dropdown",
                    "enum": SEG_MODELS, "default": DEFAULT_SEG_MODEL,
                    "description": "Used only when Remove Other People is on. It gives the real outline of each person.",
                },
                "output_format": {
                    "type": "select", "title": "Output Format", "input_type": "dropdown",
                    "enum": list(OUTPUT_FORMATS), "default": "one_row_per_track",
                    "description": "One row per tracked object (first/last frame, duration) or one row per detection in each frame.",
                },
                "save_annotated_video": {
                    "type": "boolean", "title": "Save Annotated Video", "input_type": "checkbox",
                    "default": False,
                    "description": (
                        f"Saves the whole video with boxes, category names and track ids drawn on it (.mp4, plays in the browser) "
                        f"into your Ozone folder ({OUTPUT_FOLDER}/<model>_<time>/). Skipped frames re-use the last boxes, so the "
                        f"video keeps its frame rate. Cut after {MAX_OUTPUT_VIDEO_FRAMES} frames."
                    ),
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
        if config.get("target_gender", "any") not in TARGET_GENDERS:
            errors.append(f"target_gender must be one of: {', '.join(TARGET_GENDERS)}.")
        try:
            ids = self._parse_track_ids(config.get("target_track_ids"))
        except ValueError as e:
            errors.append(str(e))
            ids = []
        if config.get("removal_model", DEFAULT_SEG_MODEL) not in SEG_MODELS:
            errors.append(f"Unknown removal_model '{config.get('removal_model')}'. Choose one of: {', '.join(SEG_MODELS)}.")
        if config.get("remove_other_people") and not ids and config.get("target_gender", "any") == "any":
            errors.append("Remove Other People needs Only These Track IDs and/or a Target Gender: without a target nobody would be kept.")
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

    # ---- Stage 1: gender helpers
    @staticmethod
    def _gender_of(vote: Optional[Dict[str, Any]]) -> Tuple[str, Optional[float]]:
        """Final / running gender of one person track from its votes: (label, confidence)."""
        if not vote or vote["n"] < GENDER_MIN_VOTES:
            return "unknown", None
        p_male = vote["sum"] / vote["n"]
        label, conf = ("male", p_male) if p_male >= 0.5 else ("female", 1.0 - p_male)
        if conf < GENDER_MIN_CONF:
            return "unknown", round(conf, 3)
        return label, round(conf, 3)

    def _update_gender(self, clf: _GenderClassifier, frame, dets: List[Dict[str, Any]], votes: Dict[int, Dict[str, Any]],
                       only_ids: Optional[set] = None) -> None:
        """Adds one round of votes for the people in this frame, then writes the running gender into each detection."""
        h, w = frame.shape[:2]
        todo = []  # (vote record, crop)
        for d in dets:
            if d["label"] != "person":
                continue
            if only_ids and d["track_id"] not in only_ids:  # not the target: no CLIP call needed
                continue
            v = votes.setdefault(d["track_id"], {"sum": 0.0, "n": 0, "seen": 0})
            v["seen"] += 1
            if v["n"] < GENDER_MAX_VOTES and (v["seen"] - 1) % GENDER_EVERY == 0:
                x1, y1 = max(int(d["x_min"]), 0), max(int(d["y_min"]), 0)
                x2, y2 = min(int(d["x_max"]), w), min(int(d["y_max"]), h)
                if x2 - x1 >= GENDER_MIN_CROP_W and y2 - y1 >= GENDER_MIN_CROP_H:
                    todo.append((v, frame[y1:y2, x1:x2]))
        if todo:
            for (v, _), p_male in zip(todo, clf.p_male([crop for _, crop in todo])):
                v["sum"] += p_male
                v["n"] += 1
        for d in dets:
            if d["label"] == "person":
                d["gender"], d["gender_confidence"] = self._gender_of(votes.get(d["track_id"]))
            else:
                d["gender"], d["gender_confidence"] = None, None

    @staticmethod
    def _parse_track_ids(raw: Any) -> List[int]:
        """'3', '#3', '3, 7' or ['3', 7] -> [3, 7]. Raises ValueError on anything that is not a whole number."""
        if raw is None or raw == "" or raw == []:
            return []
        items = raw.split(",") if isinstance(raw, str) else list(raw)
        out = set()
        for x in items:
            s = str(x).strip().lstrip("#")
            if not s:
                continue
            if not s.isdigit():
                raise ValueError(f"Track id '{x}' is not a whole number.")
            out.add(int(s))
        if len(out) > MAX_TARGET_IDS:
            raise ValueError(f"Too many track ids (max {MAX_TARGET_IDS}).")
        return sorted(out)

    @staticmethod
    def _visible(d: Dict[str, Any], target: str, ids: Optional[set] = None) -> bool:
        """A detection is drawn / kept when it passes BOTH filters: the track id filter and the gender filter."""
        if ids and d["track_id"] not in ids:
            return False
        return target == "any" or d.get("gender") == target

    @staticmethod
    def _draw(frame, dets: List[Dict[str, Any]]):
        """Own box drawing (res.plot() would draw every box, including the ones the target filter hides)."""
        import cv2

        for d in dets:
            g = d.get("gender")
            color = GENDER_COLORS.get(g, GENDER_COLORS[None])
            p1, p2 = (int(d["x_min"]), int(d["y_min"])), (int(d["x_max"]), int(d["y_max"]))
            cv2.rectangle(frame, p1, p2, color, 2)
            text = f'{d["label"]} #{d["track_id"]}'
            if g:
                text += f" {g}"
                if d.get("gender_confidence"):
                    text += f' {d["gender_confidence"]:.0%}'
            cv2.putText(frame, text, (p1[0], max(p1[1] - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        return frame

    @staticmethod
    def _annotate(res, frame, shown: List[Dict[str, Any]], custom_draw: bool):
        if custom_draw:  # gender and/or track id filter active: res.plot() would draw the hidden boxes too
            return VideoObjectTrackingRecipe._draw(frame.copy(), shown)
        return res.plot()

    @staticmethod
    def _summarise_tracks(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        by_id: Dict[int, List[Dict[str, Any]]] = {}
        for d in detections:
            by_id.setdefault(d["track_id"], []).append(d)
        tracks = []
        for tid, ds in sorted(by_id.items()):
            first, last = ds[0], ds[-1]
            t0, t1 = first["time_s"], last["time_s"]
            track = {
                "track_id": tid,
                "label": Counter(d["label"] for d in ds).most_common(1)[0][0],
                "first_frame": first["frame_index"], "last_frame": last["frame_index"],
                "first_time_s": t0, "last_time_s": t1,
                "duration_s": round(t1 - t0, 3) if t0 is not None and t1 is not None else None,
                "num_detections": len(ds),
                "avg_confidence": round(sum(d["confidence"] for d in ds) / len(ds), 4),
            }
            if "gender" in first:  # gender mode: the final voted gender is the same on every row of the track
                track["gender"] = first["gender"]
                track["gender_confidence"] = first["gender_confidence"]
            tracks.append(track)
        return tracks

    @staticmethod
    def _summarise_categories(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Per category (class): how many objects (tracks), detections, when it is on screen, average confidence.
        The tracker keeps one id while the detector may change the label (person <-> handbag), so each track is
        counted ONCE, under its most common label: the numbers then agree with 'tracks_per_class' and 'total_tracks'.
        """
        by_id: Dict[int, List[Dict[str, Any]]] = {}
        for d in detections:
            by_id.setdefault(d["track_id"], []).append(d)
        by_label: Dict[str, List[Dict[str, Any]]] = {}
        for ds in by_id.values():
            by_label.setdefault(Counter(d["label"] for d in ds).most_common(1)[0][0], []).extend(ds)
        cats = []
        for label, ds in by_label.items():
            times = [d["time_s"] for d in ds if d["time_s"] is not None]
            cats.append({
                "label": label,
                "num_tracks": len({d["track_id"] for d in ds}),
                "num_detections": len(ds),
                "first_time_s": min(times) if times else None,
                "last_time_s": max(times) if times else None,
                "avg_confidence": round(sum(d["confidence"] for d in ds) / len(ds), 4),
            })
        return sorted(cats, key=lambda x: (-x["num_tracks"], x["label"]))

    @staticmethod
    def _draw_held(res, shown: List[Dict[str, Any]], frame, custom_draw: bool):
        """The boxes of the last analysed frame drawn on a skipped frame (keeps the video's full frame rate)."""
        if custom_draw:
            return VideoObjectTrackingRecipe._draw(frame, shown)
        try:
            return res.plot(img=frame)
        except TypeError:  # an older ultralytics without img=: show the plain frame
            return frame

    # ---- Stage 2.2: remove the other people
    @staticmethod
    def _polygons(res, dets: List[Dict[str, Any]]) -> List[Any]:
        """Body outline (int32 array of x,y points) for each detection, same order as 'dets'; None when missing."""
        import numpy as np

        masks = getattr(res, "masks", None)
        if masks is None or not dets:
            return []
        xy = masks.xy
        if len(xy) != len(dets):  # should not happen; never guess which outline belongs to whom
            return []
        return [np.asarray(p, dtype=np.int32) if len(p) >= 3 else None for p in xy]

    @staticmethod
    def _fill(h: int, w: int, polys: List[Any], dilate: int = 0):
        """uint8 mask (255 inside the outlines), optionally grown by 'dilate' pixels."""
        import cv2
        import numpy as np

        m = np.zeros((h, w), np.uint8)
        if polys:
            cv2.fillPoly(m, polys, 255)
            if dilate > 0:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1))
                m = cv2.dilate(m, k)
        return m

    @staticmethod
    def _build_background(samples: List[Tuple[Any, Any]]) -> Tuple[Any, float]:
        """
        Empty scene for a FIXED camera: the median of each pixel over the sampled frames, ignoring the pixels that
        were covered by a person in that frame. Done in bands of rows to keep memory small.
        Returns (background image, share of pixels that were covered by a person in EVERY sample).
        """
        import numpy as np

        h, w = samples[0][0].shape[:2]
        bg = np.empty_like(samples[0][0])
        gap_pixels = 0
        for y0 in range(0, h, BG_BAND_ROWS):
            y1 = min(y0 + BG_BAND_ROWS, h)
            raw = np.stack([f[y0:y1] for f, _ in samples])                  # (n, rows, w, 3) uint8
            stack = raw.astype(np.float32)
            hidden = np.stack([m[y0:y1] > 0 for _, m in samples])           # (n, rows, w)
            stack[hidden] = np.nan
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)    # all-NaN slices are handled just below
                med = np.nanmedian(stack, axis=0)
            gaps = np.isnan(med)
            if gaps.any():  # covered in every sample: fall back to the plain median (a person may show through)
                med[gaps] = np.median(raw.astype(np.float32), axis=0)[gaps]
                gap_pixels += int(gaps.any(axis=-1).sum())
            bg[y0:y1] = np.clip(med, 0, 255).astype(np.uint8)
        return bg, round(gap_pixels / float(h * w), 4)

    def _write_cleaned_video(self, path: str, polys_by_frame: Dict[int, List[Tuple[int, Any]]], samples, keep_ids: set,
                             n_frames: int, fps: float, tenant_id, vidx, source_value, run_tag) -> Dict[str, Any]:
        """Pass 2: re-read the video and erase every person whose track id is not in keep_ids."""
        import cv2
        import numpy as np
        from backend.app.infrastructure.storage.ozone_client import upload_file_to_ozone

        if not samples:
            return {"cleaned_video": None, "background_gap_ratio": None, "cleaned_frames": 0}
        bg, gap_ratio = self._build_background(samples)
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError("Could not re-open the video for the clean version")
        fd, tmp = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        writer = _AnnotatedVideoWriter(tmp, fps)
        try:
            idx = 0
            try:
                while idx < n_frames and writer.frames < MAX_OUTPUT_VIDEO_FRAMES:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    polys = polys_by_frame.get(idx, [])
                    drop = [p for tid, p in polys if tid not in keep_ids]
                    if drop:
                        h, w = frame.shape[:2]
                        remove = self._fill(h, w, drop, REMOVE_DILATE_PX)
                        keep = self._fill(h, w, [p for tid, p in polys if tid in keep_ids])
                        remove[keep > 0] = 0                       # never cut into the target
                        alpha = cv2.GaussianBlur(remove, (REMOVE_FEATHER_PX, REMOVE_FEATHER_PX), 0).astype(np.float32) / 255.0
                        alpha[keep > 0] = 0                        # the blur must not leak into the target either
                        alpha = alpha[..., None]
                        frame = (frame.astype(np.float32) * (1.0 - alpha) + bg.astype(np.float32) * alpha).astype(np.uint8)
                    writer.write(frame)
                    idx += 1
            finally:
                cap.release()
                writer.close()
            if not writer.frames:
                return {"cleaned_video": None, "background_gap_ratio": gap_ratio, "cleaned_frames": 0}
            with open(tmp, "rb") as f:
                video_bytes = f.read()
            base = str(source_value).rstrip("/").rsplit("/", 1)[-1].rsplit(".", 1)[0] or "video"
            out = upload_file_to_ozone(video_bytes, f"{OUTPUT_FOLDER}/{run_tag}/{vidx}_{base}_cleaned.mp4", tenant_id, "video/mp4")
            return {"cleaned_video": out, "background_gap_ratio": gap_ratio, "cleaned_frames": writer.frames}
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

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
            gender_mode, target = cfg["gender"], cfg["target_gender"]
            ids = cfg.get("target_ids")  # set of track ids or None
            custom_draw = gender_mode or bool(ids)
            clf = cfg.get("gender_clf")
            votes: Dict[int, Dict[str, Any]] = {}  # track_id -> {"sum", "n", "seen"} (fresh for every video)

            class_ids = None
            if cfg["classes"]:
                lookup = {str(v).lower(): int(k) for k, v in model.names.items()}
                unknown = [x for x in cfg["classes"] if x.lower() not in lookup]
                if unknown:
                    raise ValueError(f"Unknown class(es): {unknown}. This model detects: {sorted(lookup)}")
                class_ids = [lookup[x.lower()] for x in cfg["classes"]]

            est = min(max_frames, -(-total // stride)) if total > 0 else 0
            preview_at = {int(i * (est - 1) / max(PREVIEW_FRAMES - 1, 1)) for i in range(PREVIEW_FRAMES)} if est > 1 else {0}

            remove = bool(cfg.get("remove"))
            polys_by_frame: Dict[int, List[Tuple[int, Any]]] = {}  # frame index -> [(track_id, body outline)] of every person
            bg_samples: List[Tuple[Any, Any]] = []                  # (frame, mask of everybody in it) for the background
            bg_every = max(1, est // BG_SAMPLES) if est > 0 else 10
            detections: List[Dict[str, Any]] = []
            previews: List[Dict[str, Any]] = []
            idx = processed = 0
            writer, video_tmp, last_res, last_shown = None, None, None, []
            if cfg.get("save_video"):
                fd, video_tmp = tempfile.mkstemp(suffix=".mp4")
                os.close(fd)
                writer = _AnnotatedVideoWriter(video_tmp, fps)
            while processed < max_frames:
                if idx % stride != 0:
                    if writer is None:
                        if not cap.grab():
                            break
                    else:  # the video needs the pixels of this frame too
                        ok, frame = cap.read()
                        if not ok:
                            break
                        if last_res is not None and writer.frames < MAX_OUTPUT_VIDEO_FRAMES:
                            writer.write(self._draw_held(last_res, last_shown, frame, custom_draw))
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
                dets = self._parse_frame(res, idx, time_s)
                if gender_mode:
                    self._update_gender(clf, frame, dets, votes, ids)
                if remove:
                    polys = self._polygons(res, dets)
                    person_polys = [(d["track_id"], p) for d, p in zip(dets, polys) if d["label"] == "person" and p is not None]
                    polys_by_frame[idx] = person_polys
                    if processed % bg_every == 0 and len(bg_samples) < BG_SAMPLES:
                        fh, fw = frame.shape[:2]
                        bg_samples.append((frame, self._fill(fh, fw, [p for _, p in person_polys], REMOVE_DILATE_PX)))
                detections.extend(dets)  # everybody is kept here; the target filters are applied after the loop
                shown = [d for d in dets if self._visible(d, target, ids)] if custom_draw else dets

                need_preview = save_preview and processed in preview_at
                need_video = writer is not None and writer.frames < MAX_OUTPUT_VIDEO_FRAMES
                if need_preview or need_video:
                    annotated = self._annotate(res, frame, shown, custom_draw)
                    if need_preview:
                        previews.append(self._save_preview(annotated, tenant_id, vidx, source_value, idx, run_tag))
                    if need_video:
                        writer.write(annotated)
                last_res, last_shown = res, shown
                processed += 1
                idx += 1

            # ---- final gender per track: every row of a track gets the same, final voted gender; then apply the target
            if gender_mode:
                final = {tid: self._gender_of(v) for tid, v in votes.items()}
                for d in detections:
                    if d["label"] == "person":
                        d["gender"], d["gender_confidence"] = final.get(d["track_id"], ("unknown", None))
                    else:
                        d["gender"], d["gender_confidence"] = None, None
            seen_ids = {d["track_id"] for d in detections}  # every id the tracker produced, before filtering
            if custom_draw:  # id filter and/or gender filter (a track keeps all its rows or none)
                detections = [d for d in detections if self._visible(d, target, ids)]
            kept_ids = {d["track_id"] for d in detections}
            target_ids_not_seen = sorted(ids - seen_ids) if ids else []
            target_ids_matched = sorted(ids & kept_ids) if ids else []

            # ---- Stage 2.2, pass 2: erase every person who is not a target (needs the FINAL ids and genders from above)
            clean: Dict[str, Any] = {"cleaned_video": None, "background_gap_ratio": None, "cleaned_frames": 0, "removed_ids": []}
            if remove:
                keep_person_ids = {d["track_id"] for d in detections if d["label"] == "person"}
                all_person_ids = {tid for pl in polys_by_frame.values() for tid, _ in pl}
                clean.update(self._write_cleaned_video(
                    path, polys_by_frame, bg_samples, keep_person_ids, processed, fps, tenant_id, vidx, source_value, run_tag,
                ))
                clean["removed_ids"] = sorted(all_person_ids - keep_person_ids)
                polys_by_frame.clear()
                bg_samples.clear()  # free the memory

            output_video, frames_written, truncated = None, 0, False
            if writer is not None:
                writer.close()
                frames_written, truncated = writer.frames, writer.frames >= MAX_OUTPUT_VIDEO_FRAMES
                if frames_written:
                    from backend.app.infrastructure.storage.ozone_client import upload_file_to_ozone

                    with open(video_tmp, "rb") as f:
                        video_bytes = f.read()
                    base = str(source_value).rstrip("/").rsplit("/", 1)[-1].rsplit(".", 1)[0] or "video"
                    output_video = upload_file_to_ozone(
                        video_bytes, f"{OUTPUT_FOLDER}/{run_tag}/{vidx}_{base}_tracked.mp4", tenant_id, "video/mp4"
                    )
        finally:
            cap.release()
            if video_tmp and os.path.exists(video_tmp):
                os.unlink(video_tmp)

        return {**info, "frames_processed": processed, "detections": detections, "previews": previews,
                "output_video": output_video, "video_frames_written": frames_written, "video_truncated": truncated,
                "target_ids_matched": target_ids_matched, "target_ids_not_seen": target_ids_not_seen,
                "cleaned_video": clean["cleaned_video"], "background_gap_ratio": clean["background_gap_ratio"],
                "cleaned_frames": clean["cleaned_frames"], "removed_ids": clean["removed_ids"]}

    @staticmethod
    def _save_preview(img, tenant_id: int, vidx: int, source_value: Any, frame_index: int, run_tag: str) -> Dict[str, Any]:
        """Saves an already annotated BGR frame as a JPEG preview."""
        import cv2
        from backend.app.infrastructure.storage.ozone_client import upload_file_to_ozone

        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
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

        remove_people = bool(config.get("remove_other_people", False))
        # removal needs body outlines (segmentation model) and every frame (a skipped frame would show a ghost)
        model_name = config.get("removal_model", DEFAULT_SEG_MODEL) if remove_people else config.get("model", DEFAULT_MODEL)
        target_gender = config.get("target_gender", "any")
        gender_mode = bool(config.get("predict_gender", False)) or target_gender != "any"
        target_ids = self._parse_track_ids(config.get("target_track_ids"))  # already validated above
        cfg = {
            "target_ids": set(target_ids) or None,
            "model": model_name,
            "tracker": config.get("tracker", DEFAULT_TRACKER),
            "stride": 1 if remove_people else int(config.get("frame_stride", 5)),
            "remove": remove_people,
            "max_frames": int(config.get("max_frames", 300)),
            "conf": float(config.get("confidence_threshold", 0.25)),
            "iou": float(config.get("iou_threshold", 0.7)),
            "imgsz": int(config.get("image_size", 640)),
            "classes": od.ObjectDetectionRecipe._parse_classes(config.get("classes")),
            "save_video": bool(config.get("save_annotated_video", False)),
            "gender": gender_mode,
            "target_gender": target_gender,
            # loaded ONCE for all videos of this run; it keeps no per-video state (the votes live in _track_video)
            "gender_clf": _GenderClassifier() if gender_mode else None,
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
        if gender_mode:
            track_cols += ["gender", "gender_confidence"]
            det_cols += ["gender", "gender_confidence"]
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
        gender_tracks: Dict[str, int] = {}
        for r in per_video:
            for t in r["tracks"]:
                label_tracks[t["label"]] = label_tracks.get(t["label"], 0) + 1
                if gender_mode and t.get("gender"):
                    gender_tracks[t["gender"]] = gender_tracks.get(t["gender"], 0) + 1
        previews = []
        for value, r in zip(df[video_col].tolist(), per_video):
            per_class = Counter(t["label"] for t in r["tracks"])
            extra: Dict[str, Any] = {}
            if r.get("video_truncated"):
                extra["video_truncated"] = True
            if remove_people:
                extra["cleaned_video"] = r.get("cleaned_video")
                extra["removed_track_ids"] = r.get("removed_ids", [])
                extra["background_gap_ratio"] = r.get("background_gap_ratio")
                if (r.get("background_gap_ratio") or 0) > BG_GAP_WARN:
                    extra["background_warning"] = (
                        "Part of the scene was covered by a person in every sampled frame (someone stood still), "
                        "so a faint ghost may remain there."
                    )
            if target_ids:
                extra["target_ids_matched"] = r.get("target_ids_matched", [])
                extra["target_ids_not_seen"] = r.get("target_ids_not_seen", [])
            if gender_mode:
                extra["tracks_per_gender"] = dict(Counter(t["gender"] for t in r["tracks"] if t.get("gender")))
            previews.append(c.preview_item(
                original=value,
                # the annotated video when it was saved, otherwise the first annotated frame
                media=r.get("output_video") or r.get("cleaned_video") or (r["previews"][0]["media_path"] if r["previews"] else None),
                error=r["error"],
                frames_processed=r["frames_processed"],
                num_tracks=len(r["tracks"]),
                tracks_per_class=dict(per_class),
                categories=self._summarise_categories(r["detections"]),  # per category: tracks, detections, first/last time
                preview_frames=r["previews"],  # every saved frame: {frame_index, media_path, media_preview_url}
                **extra,
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
        if remove_people:
            metrics["remove_other_people"] = True
            metrics["removal_model"] = model_name
        if target_ids:
            metrics["target_track_ids"] = target_ids
        if gender_mode:
            metrics["gender_model"] = GENDER_MODEL
            metrics["target_gender"] = target_gender
            metrics["tracks_per_gender"] = gender_tracks
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