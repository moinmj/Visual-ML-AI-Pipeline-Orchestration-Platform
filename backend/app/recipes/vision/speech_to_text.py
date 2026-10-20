"""
Speech-to-Text recipe (Modeling -> Computer Vision & Audio).

Transcribes audio recordings with Whisper (via faster-whisper). Returns the full
transcript, the detected language and timed segments.

Audio comes from 'audio_files' (picked from unstructured/; same rules and security checks
as the image/video recipes, see _common.py). There is no table / column input.

Place this file at:  backend/app/recipes/vision/speech_to_text.py
Needs:               pip install faster-whisper "av<19"
                     (decodes wav/mp3/m4a/flac/ogg itself - no ffmpeg install needed).
                     The "av<19" pin matters: faster-whisper 1.2.1 breaks with PyAV 19.
Note:                faster-whisper downloads the model from Hugging Face the first time a
                     model size is used (tiny ~75 MB ... medium ~1.5 GB). On a server without
                     internet, copy ~/.cache/huggingface once from a machine that has it.

Security: only the built-in model sizes below are accepted (no model path from the node config);
audio is read only from the caller's tenant folder; size/type limits come from core/media_types.py;
audio longer than MAX_AUDIO_SECONDS is refused.
"""
import io
import json
import math
import threading
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from backend.app.recipes.base.recipe import BaseRecipe
from backend.app.recipes.vision import _common as c

SUPPORTED_MODELS = ["tiny", "base", "small", "medium"]  # multilingual Whisper sizes
DEFAULT_MODEL = "base"
SUPPORTED_LANGUAGES = [
    "auto", "en", "ur", "hi", "ar", "fa", "fr", "de", "es", "it", "pt", "ru", "tr", "zh", "ja", "ko",
]
TASKS = ("transcribe", "translate")  # translate = speech in any language -> English text
OUTPUT_FORMATS = ("one_row_per_audio", "one_row_per_segment")
MAX_AUDIO_SECONDS = 30 * 60
BEAM_SIZE = 5

_MODEL_CACHE: Dict[str, Tuple[Any, threading.Lock]] = {}
_MODEL_LOCK = threading.Lock()


def _device() -> Tuple[str, str]:
    """(device, compute_type): GPU in half precision when available, else CPU int8."""
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


def _load_model(name: str) -> Tuple[Any, threading.Lock]:
    """Loads a Whisper model once per process. Returns (model, lock); inference runs under the lock."""
    with _MODEL_LOCK:
        if name not in _MODEL_CACHE:
            try:
                from faster_whisper import WhisperModel
            except ImportError as e:
                raise RuntimeError("Speech-to-Text needs 'faster-whisper'. Install it with: pip install faster-whisper") from e
            device, compute_type = _device()
            _MODEL_CACHE[name] = (WhisperModel(name, device=device, compute_type=compute_type), threading.Lock())
        return _MODEL_CACHE[name]


def _transcribe(model_name: str, data: bytes, language: Optional[str], task: str, vad: bool):
    """Returns (segments as dicts, info). Raises ValueError if the audio is too long."""
    model, lock = _load_model(model_name)
    with lock:
        try:
            segments, info = model.transcribe(
                io.BytesIO(data), language=language, task=task, vad_filter=vad, beam_size=BEAM_SIZE,
            )
        except TypeError as e:
            if "metadata_errors" in str(e):  # faster-whisper 1.2.1 + PyAV 19
                raise RuntimeError("faster-whisper does not work with this PyAV version. Run: pip install \"av<19\"") from e
            raise
        duration = float(getattr(info, "duration", 0) or 0)
        if duration > MAX_AUDIO_SECONDS:  # checked before the (lazy) decoding of segments starts
            raise ValueError(f"Audio is longer than {MAX_AUDIO_SECONDS // 60} minutes")
        out = []
        for s in segments:
            lp = getattr(s, "avg_logprob", None)
            out.append({
                "start_s": round(float(s.start), 3),
                "end_s": round(float(s.end), 3),
                "text": str(s.text).strip(),
                "confidence": round(min(1.0, max(0.0, math.exp(lp))), 4) if lp is not None else None,
            })
    return out, info


class SpeechToTextRecipe(BaseRecipe):
    recipe_id = "speech_to_text"
    name = "Speech-to-Text"
    version = "1.0.0"
    category = "vision_audio"
    media_type = "audio"
    description = "Transcribes audio recordings into text."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    def get_schema(self) -> Dict[str, Any]:
        source = c.source_schema("audio")
        return {
            "type": "object",
            "properties": {
                **source,
                "model": {
                    "type": "select", "title": "Model Size", "input_type": "dropdown",
                    "enum": SUPPORTED_MODELS, "default": DEFAULT_MODEL,
                    "description": "Bigger = more accurate, slower. tiny/base are fine for clear speech; small/medium for accents or noise.",
                },
                "language": {
                    "type": "select", "title": "Spoken Language", "input_type": "dropdown",
                    "enum": SUPPORTED_LANGUAGES, "default": "auto",
                    "description": "auto = detect the language. Setting it is faster and more reliable when you know it.",
                },
                "task": {
                    "type": "select", "title": "Task", "input_type": "dropdown",
                    "enum": list(TASKS), "default": "transcribe",
                    "description": "transcribe = text in the spoken language. translate = English text from any language.",
                },
                "vad_filter": {
                    "type": "boolean", "title": "Skip Silence", "input_type": "checkbox", "default": False,
                    "description": "Cuts long silent parts before transcribing. Helps with recordings that have long pauses.",
                },
                "output_format": {
                    "type": "select", "title": "Output Format", "input_type": "dropdown",
                    "enum": list(OUTPUT_FORMATS), "default": "one_row_per_audio",
                    "description": "One row per audio file (full transcript) or one row per timed segment.",
                },
            },
            "required": [],
        }

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        errors = c.validate_source(config, "audio")
        if config.get("model", DEFAULT_MODEL) not in SUPPORTED_MODELS:
            errors.append(f"Unknown model '{config.get('model')}'. Choose one of: {', '.join(SUPPORTED_MODELS)}.")
        if config.get("language", "auto") not in SUPPORTED_LANGUAGES:
            errors.append(f"Unknown language '{config.get('language')}'. Choose one of: {', '.join(SUPPORTED_LANGUAGES)}.")
        if config.get("task", "transcribe") not in TASKS:
            errors.append("task must be 'transcribe' or 'translate'.")
        if config.get("output_format", "one_row_per_audio") not in OUTPUT_FORMATS:
            errors.append("output_format must be 'one_row_per_audio' or 'one_row_per_segment'.")
        return errors

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        errors = self.validate_config(config)
        if errors:
            raise ValueError("; ".join(errors))
        tenant_id = c.require_tenant(config, self.name, "audio files")
        df, audio_col, from_files = c.resolve_source(inputs, config, self.name, "audio")

        model_name = config.get("model", DEFAULT_MODEL)
        language = config.get("language", "auto")
        task = config.get("task", "transcribe")
        vad = bool(config.get("vad_filter", False))
        out_format = config.get("output_format", "one_row_per_audio")
        _load_model(model_name)  # fail the whole run early if faster-whisper / the model is missing

        values = df[audio_col].tolist()
        per_audio: List[Dict[str, Any]] = []
        for value in values:
            row: Dict[str, Any] = {"segments": [], "language": None, "language_probability": None,
                                   "duration_s": None, "error": None}
            try:
                data, _ext = c.read_media_bytes(value, tenant_id, from_files, "audio")
                segments, info = _transcribe(model_name, data, None if language == "auto" else language, task, vad)
                prob = getattr(info, "language_probability", None)
                row.update(
                    segments=segments,
                    language=getattr(info, "language", None),
                    language_probability=round(float(prob), 4) if prob is not None else None,
                    duration_s=round(float(getattr(info, "duration", 0) or 0), 3),
                )
                del data
            except Exception as e:  # one bad audio file must not fail the whole run
                row["error"] = c.safe_error(e)
            per_audio.append(row)

        out = df.copy().reset_index(drop=True)
        out["transcript"] = [
            None if r["error"] else " ".join(s["text"] for s in r["segments"] if s["text"]).strip() for r in per_audio
        ]
        out["detected_language"] = [r["language"] for r in per_audio]
        out["language_probability"] = [r["language_probability"] for r in per_audio]
        out["duration_s"] = [r["duration_s"] for r in per_audio]
        out["num_segments"] = [None if r["error"] else len(r["segments"]) for r in per_audio]
        out["segments"] = [json.dumps(r["segments"], ensure_ascii=False) if not r["error"] else None for r in per_audio]
        out["transcription_error"] = [r["error"] for r in per_audio]
        out["original_preview_url"] = [c.preview_url(v) for v in values]

        if out_format == "one_row_per_segment":
            base_cols = [col for col in out.columns if col not in ("transcript", "segments")]
            seg_cols = ["segment_index", "start_s", "end_s", "text", "confidence"]
            rows = []
            for i, r in enumerate(per_audio):
                base = out.iloc[i][base_cols].to_dict()
                if r["segments"]:
                    rows.extend({**base, "segment_index": k, **s} for k, s in enumerate(r["segments"]))
                else:  # keep every audio file in the output, even with an error or no speech
                    rows.append({**base, **{k: None for k in seg_cols}})
            out_df = pd.DataFrame(rows, columns=base_cols + seg_cols)
        else:
            out_df = out

        failed = [r for r in per_audio if r["error"]]
        previews = []
        for v, r in zip(values, per_audio):
            item = c.preview_item(
                original=v,
                text=" ".join(s["text"] for s in r["segments"] if s["text"]).strip(),  # the transcript
                error=r["error"],
                segments=r["segments"],
                detected_language=r["language"],
                duration_s=r["duration_s"],
                num_segments=len(r["segments"]),
            )
            previews.append(item)
        metrics = {
            "task_type": "speech_to_text",
            "model": model_name,
            "task": task,
            "language": language,
            "input_source": "audio_files",
            "items_processed": len(per_audio) - len(failed),
            "items_failed": len(failed),
            "total_duration_s": round(sum(r["duration_s"] or 0 for r in per_audio), 3),
            "total_segments": int(sum(len(r["segments"]) for r in per_audio)),
            "errors_sample": [r["error"] for r in failed[:5]],
            "preview_items": previews,
        }
        return {"dataframe": out_df, "metrics": metrics, "output_summary": metrics, "task_type": "speech_to_text"}

    def to_code(self, config: Dict[str, Any]) -> str:
        files = c.image_files(config, "audio")
        source = repr(files[0]) if files else "'unstructured/<audio>'"
        language = config.get("language", "auto")
        return (
            "from faster_whisper import WhisperModel\n\n"
            f"model = WhisperModel({config.get('model', DEFAULT_MODEL)!r}, compute_type='int8')\n"
            f"segments, info = model.transcribe({source}, language={None if language == 'auto' else language!r}, "
            f"task={config.get('task', 'transcribe')!r})\n"
            "text = ' '.join(s.text.strip() for s in segments)"
        )