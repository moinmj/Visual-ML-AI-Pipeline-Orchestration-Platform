"""
Image Classifier recipe (Modeling -> Computer Vision & Audio).

Runs a pretrained YOLOv8 classification model (ImageNet, 1000 classes) on each
image and returns the predicted label, its confidence and the top-k guesses.

Images come from 'image_files' (picked from unstructured/; same rules and security
checks as object_detection, see _common.py). There is no table / column input.

Place this file at:  backend/app/recipes/vision/image_classifier.py
Needs:               pip install ultralytics pillow   (already needed by object_detection)

Security: only the built-in model names below are accepted (no model path from
the node config, a .pt file can run code); images are read only from the
caller's tenant folder.
"""
import json
from typing import Any, Dict, List, Optional

import pandas as pd

from backend.app.recipes.base.recipe import BaseRecipe
from backend.app.recipes.vision import _common as c
from backend.app.recipes.vision import object_detection as od

SUPPORTED_MODELS = [
    "yolov8n-cls.pt", "yolov8s-cls.pt", "yolov8m-cls.pt", "yolov8l-cls.pt", "yolov8x-cls.pt",
]
DEFAULT_MODEL = "yolov8m-cls.pt"
MAX_TOP_K = 5
UNCERTAIN_LABEL = "uncertain"


class ImageClassifierRecipe(BaseRecipe):
    recipe_id = "image_classifier"
    name = "Image Classifier"
    version = "1.0.0"
    category = "vision_audio"
    media_type = "image"
    description = "Classifies images into categories (pretrained ImageNet model, 1000 classes)."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                **c.source_schema(),
                "model": {
                    "type": "select",
                    "title": "Classification Model",
                    "input_type": "dropdown",
                    "enum": SUPPORTED_MODELS,
                    "default": DEFAULT_MODEL,
                    "description": "Pretrained YOLOv8 classifier (ImageNet, 1000 classes). Bigger = more accurate, slower.",
                },
                "top_k": {
                    "type": "integer",
                    "title": "Top-K Predictions",
                    "input_type": "number",
                    "default": 3,
                    "minimum": 1,
                    "maximum": MAX_TOP_K,
                    "description": "How many of the best guesses to keep per image.",
                },
                "min_confidence": {
                    "type": "number",
                    "title": "Minimum Confidence",
                    "input_type": "slider",
                    "default": 0.0,
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": f"If the best guess is below this, predicted_label is set to '{UNCERTAIN_LABEL}'. 0 = always keep the best guess.",
                },
                "image_size": {
                    "type": "integer",
                    "title": "Inference Image Size (px)",
                    "input_type": "number",
                    "default": 224,
                    "minimum": 32,
                    "maximum": 1280,
                },
            },
            "required": [],
        }

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        errors = c.validate_source(config)
        if config.get("model", DEFAULT_MODEL) not in SUPPORTED_MODELS:
            errors.append(f"Unknown model '{config.get('model')}'. Choose one of: {', '.join(SUPPORTED_MODELS)}.")
        for err in (
            c.validate_range(config, "top_k", 1, MAX_TOP_K, integer=True),
            c.validate_range(config, "min_confidence", 0.0, 1.0),
            c.validate_range(config, "image_size", 32, 1280, integer=True),
        ):
            if err:
                errors.append(err)
        return errors

    @staticmethod
    def _classify(model, image, top_k: int, imgsz: int) -> List[Dict[str, Any]]:
        res = model.predict(image, imgsz=imgsz, verbose=False)[0]
        probs = res.probs
        ids = list(probs.top5)[:top_k]
        confs = probs.top5conf.tolist() if hasattr(probs.top5conf, "tolist") else list(probs.top5conf)
        return [
            {"label": res.names[int(i)], "class_id": int(i), "confidence": round(float(p), 4)}
            for i, p in zip(ids, confs[:top_k])
        ]

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        errors = self.validate_config(config)
        if errors:
            raise ValueError("; ".join(errors))
        tenant_id = c.require_tenant(config, self.name)
        df, image_col, from_files = c.resolve_source(inputs, config, self.name)

        model_name = config.get("model", DEFAULT_MODEL)
        top_k = int(config.get("top_k", 3))
        min_conf = float(config.get("min_confidence", 0.0))
        imgsz = int(config.get("image_size", 224))
        model = od._load_model(model_name)

        per_image: List[Dict[str, Any]] = []
        for value in df[image_col].tolist():
            row: Dict[str, Any] = {"top": [], "error": None}
            try:
                image = c.open_image(value, tenant_id, from_files)
                row["width"], row["height"] = image.size
                row["top"] = self._classify(model, image, top_k, imgsz)
            except Exception as e:  # one bad image must not fail the whole run
                row["error"] = c.safe_error(e)
            per_image.append(row)

        def best(r):
            return r["top"][0] if r["top"] else None

        out = df.copy().reset_index(drop=True)
        labels = []
        for r in per_image:
            b = best(r)
            labels.append(None if b is None else (b["label"] if b["confidence"] >= min_conf else UNCERTAIN_LABEL))
        out["predicted_label"] = labels
        out["predicted_class_id"] = [best(r)["class_id"] if best(r) else None for r in per_image]
        out["confidence"] = [best(r)["confidence"] if best(r) else None for r in per_image]
        out["top_k_predictions"] = [json.dumps(r["top"]) for r in per_image]
        out["image_width"] = [r.get("width") for r in per_image]
        out["image_height"] = [r.get("height") for r in per_image]
        out["classification_error"] = [r["error"] for r in per_image]
        out["image_preview_url"] = [c.preview_url(v) for v in df[image_col].tolist()]

        ok = [r for r in per_image if not r["error"]]
        counts: Dict[str, int] = {}
        for lbl in labels:
            if lbl is not None:
                counts[lbl] = counts.get(lbl, 0) + 1
        confs = [best(r)["confidence"] for r in ok if best(r)]

        previews = []
        for value, lbl, r in zip(df[image_col].tolist(), labels, per_image):
            b = best(r)
            previews.append(c.preview_item(
                original=value,
                text=lbl,
                error=r["error"],
                predicted_label=lbl,
                confidence=b["confidence"] if b else None,
            ))

        metrics = {
            "task_type": "image_classification",
            "model": model_name,
            "input_source": "image_files",
            "items_processed": len(ok),
            "items_failed": len(per_image) - len(ok),
            "predictions_per_class": dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True)),
            "avg_confidence": round(sum(confs) / len(confs), 4) if confs else None,
            "uncertain_count": int(sum(1 for lbl in labels if lbl == UNCERTAIN_LABEL)),
            "errors_sample": [r["error"] for r in per_image if r["error"]][:5],
            "preview_items": previews,
        }
        return {"dataframe": out, "metrics": metrics, "output_summary": metrics, "task_type": "image_classification"}

    def to_code(self, config: Dict[str, Any]) -> str:
        model = config.get("model", DEFAULT_MODEL)
        files = c.image_files(config)
        source = repr(files) if files else "[]"
        return (
            "from ultralytics import YOLO\n\n"
            f"model = YOLO({model!r})\n"
            f"results = model.predict({source}, imgsz={config.get('image_size', 224)})\n"
            "predictions = [\n"
            "    {'label': r.names[r.probs.top1], 'confidence': float(r.probs.top1conf)}\n"
            "    for r in results\n"
            "]"
        )