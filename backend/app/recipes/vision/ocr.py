"""
OCR recipe (Modeling -> Computer Vision & Audio).

Reads printed or handwritten text from images with EasyOCR and returns the
text, a confidence per text block and the box of each block.

Images come from 'image_files' (picked from unstructured/; same rules and security
checks as object_detection, see _common.py). There is no table / column input.

Place this file at:  backend/app/recipes/vision/ocr.py
Needs:               pip install easyocr pillow numpy   (torch is already installed for ultralytics)
Note:                EasyOCR downloads its language models to ~/.EasyOCR the first
                     time a language is used. On a server without internet, copy
                     that folder over once.

Security: only the languages listed below are accepted; images are read only
from the caller's tenant folder; annotated images go to outputs/ocr/.
"""
import io
import json
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from backend.app.recipes.base.recipe import BaseRecipe
from backend.app.recipes.vision import _common as c

# EasyOCR language codes. Note EasyOCR's own rules: Arabic/Urdu/Persian only mix
# with English (and each other); Hindi only with English/Nepali/Marathi; Chinese/Japanese/Korean only with English.
SUPPORTED_LANGUAGES = [
    "en", "ar", "ur", "fa", "hi", "fr", "de", "es", "it", "pt", "ru", "tr", "ch_sim", "ja", "ko",
]
MAX_LANGUAGES = 3
OUTPUT_FOLDER = "outputs/ocr"
OUTPUT_FORMATS = ("one_row_per_image", "one_row_per_text_block")

_READER_CACHE: Dict[Tuple[str, ...], Any] = {}
_READER_LOCK = threading.Lock()


def _get_reader(languages: List[str]):
    """Loads an EasyOCR reader once per language set and reuses it."""
    key = tuple(sorted(languages))
    with _READER_LOCK:
        if key not in _READER_CACHE:
            try:
                import easyocr
            except ImportError as e:
                raise RuntimeError("OCR needs the 'easyocr' package. Install it with: pip install easyocr") from e
            try:
                _READER_CACHE[key] = easyocr.Reader(list(languages))
            except ValueError as e:  # EasyOCR raises this for language combinations it cannot mix
                raise ValueError(f"These languages cannot be used together: {e}") from e
        return _READER_CACHE[key]


def _parse_languages(raw: Any) -> List[str]:
    if raw is None:
        return ["en"]
    if isinstance(raw, str):
        raw = raw.split(",")
    if not isinstance(raw, (list, tuple)):
        return []
    return [str(v).strip() for v in raw if str(v).strip()]


class OCRRecipe(BaseRecipe):
    recipe_id = "ocr"
    name = "OCR"
    version = "1.0.0"
    category = "vision_audio"
    media_type = "image"
    description = "Extracts printed or handwritten text from images and scans."
    input_types = []  # source node: reads its own files, takes no input
    output_types = ["dataframe"]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                **c.source_schema(),
                "languages": {
                    "type": "array",
                    "items": {"type": "string", "enum": SUPPORTED_LANGUAGES},
                    "title": "Languages",
                    "input_type": "multi_select",
                    "default": ["en"],
                    "maxItems": MAX_LANGUAGES,
                    "description": (
                        f"Languages in the image (max {MAX_LANGUAGES}). Arabic/Urdu/Persian can only be "
                        "combined with English; Hindi with English; Chinese/Japanese/Korean with English."
                    ),
                },
                "min_confidence": {
                    "type": "number",
                    "title": "Minimum Confidence",
                    "input_type": "slider",
                    "default": 0.3,
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Text blocks below this confidence are dropped.",
                },
                "output_format": {
                    "type": "select",
                    "title": "Output Format",
                    "input_type": "dropdown",
                    "enum": list(OUTPUT_FORMATS),
                    "default": "one_row_per_image",
                    "description": "One row per image (all text joined) or one row per detected text block.",
                },
                "save_annotated_images": {
                    "type": "boolean",
                    "title": "Save Annotated Images",
                    "input_type": "checkbox",
                    "default": False,
                    "description": f"Saves a copy of each image with text boxes drawn into your Ozone folder ({OUTPUT_FOLDER}/<time>/).",
                },
            },
            "required": [],
        }

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        errors = c.validate_source(config)
        langs = _parse_languages(config.get("languages"))
        if not langs:
            errors.append("Choose at least one language.")
        elif len(langs) > MAX_LANGUAGES:
            errors.append(f"Too many languages ({len(langs)}). The limit is {MAX_LANGUAGES}.")
        unknown = [l for l in langs if l not in SUPPORTED_LANGUAGES]
        if unknown:
            errors.append(f"Unknown language(s): {unknown}. Choose from: {', '.join(SUPPORTED_LANGUAGES)}.")
        err = c.validate_range(config, "min_confidence", 0.0, 1.0)
        if err:
            errors.append(err)
        if config.get("output_format", "one_row_per_image") not in OUTPUT_FORMATS:
            errors.append("output_format must be 'one_row_per_image' or 'one_row_per_text_block'.")
        return errors

    @staticmethod
    def _read_text(reader, image, min_conf: float) -> List[Dict[str, Any]]:
        import numpy as np

        blocks: List[Dict[str, Any]] = []
        for points, text, conf in reader.readtext(np.array(image), detail=1, paragraph=False):
            if float(conf) < min_conf or not str(text).strip():
                continue
            xs = [float(p[0]) for p in points]
            ys = [float(p[1]) for p in points]
            blocks.append({
                "text": str(text).strip(),
                "confidence": round(float(conf), 4),
                "x_min": round(min(xs), 1), "y_min": round(min(ys), 1),
                "x_max": round(max(xs), 1), "y_max": round(max(ys), 1),
            })
        return blocks

    @staticmethod
    def _save_annotated(image, blocks: List[Dict[str, Any]], source_value: str, tenant_id: int, index: int, run_tag: str) -> Optional[str]:
        from PIL import ImageDraw
        from backend.app.infrastructure.storage.ozone_client import upload_file_to_ozone

        canvas = image.copy()
        draw = ImageDraw.Draw(canvas)
        for b in blocks:
            draw.rectangle([b["x_min"], b["y_min"], b["x_max"], b["y_max"]], outline=(255, 0, 0), width=2)
        buf = io.BytesIO()
        canvas.save(buf, format="JPEG", quality=90)
        base = str(source_value).rstrip("/").rsplit("/", 1)[-1].rsplit(".", 1)[0] or "image"
        return upload_file_to_ozone(buf.getvalue(), f"{OUTPUT_FOLDER}/{run_tag}/{index}_{base}_ocr.jpg", tenant_id, "image/jpeg")

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        errors = self.validate_config(config)
        if errors:
            raise ValueError("; ".join(errors))
        tenant_id = c.require_tenant(config, self.name)
        df, image_col, from_files = c.resolve_source(inputs, config, self.name)

        langs = _parse_languages(config.get("languages"))
        min_conf = float(config.get("min_confidence", 0.3))
        out_format = config.get("output_format", "one_row_per_image")
        save_annotated = bool(config.get("save_annotated_images", False))
        reader = _get_reader(langs)
        run_tag = f"{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"

        per_image: List[Dict[str, Any]] = []
        for idx, value in enumerate(df[image_col].tolist()):
            row: Dict[str, Any] = {"blocks": [], "error": None, "annotated_path": None}
            try:
                image = c.open_image(value, tenant_id, from_files)
                row["width"], row["height"] = image.size
                row["blocks"] = self._read_text(reader, image, min_conf)
                if save_annotated:
                    row["annotated_path"] = self._save_annotated(image, row["blocks"], str(value), tenant_id, idx, run_tag)
            except Exception as e:  # one bad image must not fail the whole run
                row["error"] = c.safe_error(e)
            per_image.append(row)

        out = df.copy().reset_index(drop=True)
        out["ocr_text"] = [("\n".join(b["text"] for b in r["blocks"]) if not r["error"] else None) for r in per_image]
        out["num_text_blocks"] = [len(r["blocks"]) for r in per_image]
        out["avg_confidence"] = [
            round(sum(b["confidence"] for b in r["blocks"]) / len(r["blocks"]), 4) if r["blocks"] else None
            for r in per_image
        ]
        out["image_width"] = [r.get("width") for r in per_image]
        out["image_height"] = [r.get("height") for r in per_image]
        out["text_blocks"] = [json.dumps(r["blocks"], ensure_ascii=False) for r in per_image]
        out["ocr_error"] = [r["error"] for r in per_image]
        out["image_preview_url"] = [c.preview_url(v) for v in df[image_col].tolist()]
        if save_annotated:
            out["annotated_image_path"] = [r["annotated_path"] for r in per_image]
            out["annotated_preview_url"] = [c.preview_url(r["annotated_path"]) for r in per_image]

        if out_format == "one_row_per_text_block":
            rows = []
            for i, r in enumerate(per_image):
                base = out.drop(columns=["text_blocks", "ocr_text"]).iloc[i].to_dict()
                for b in r["blocks"]:
                    rows.append({**base, **b})
            block_cols = ["text", "confidence", "x_min", "y_min", "x_max", "y_max"]
            import pandas as pd
            out_df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=list(out.columns.drop(["text_blocks", "ocr_text"])) + block_cols)
        else:
            out_df = out

        failed = [r for r in per_image if r["error"]]
        all_conf = [b["confidence"] for r in per_image for b in r["blocks"]]
        previews = []
        for value, r in zip(df[image_col].tolist(), per_image):
            previews.append(c.preview_item(
                original=value,
                media=r.get("annotated_path"),
                text="\n".join(b["text"] for b in r["blocks"]),
                error=r["error"],
                num_text_blocks=len(r["blocks"]),
            ))

        metrics = {
            "task_type": "ocr",
            "languages": langs,
            "input_source": "image_files",
            "items_processed": len(per_image) - len(failed),
            "items_failed": len(failed),
            "total_text_blocks": len(all_conf),
            "images_with_no_text": int(sum(1 for r in per_image if not r["error"] and not r["blocks"])),
            "avg_confidence": round(sum(all_conf) / len(all_conf), 4) if all_conf else None,
            "min_confidence": min_conf,
            "errors_sample": [r["error"] for r in failed[:5]],
            "preview_items": previews,
        }
        return {"dataframe": out_df, "metrics": metrics, "output_summary": metrics, "task_type": "ocr"}

    def to_code(self, config: Dict[str, Any]) -> str:
        langs = _parse_languages(config.get("languages"))
        files = c.image_files(config)
        source = repr(files) if files else "[]"
        return (
            "import easyocr\n\n"
            f"reader = easyocr.Reader({langs!r})\n"
            f"texts = [reader.readtext(path, detail=0) for path in {source}]"
        )