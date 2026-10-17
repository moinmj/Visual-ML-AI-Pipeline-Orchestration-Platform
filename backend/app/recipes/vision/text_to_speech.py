"""
Text-to-Speech recipe (Modeling -> Computer Vision & Audio).

Turns the text in a column into spoken audio (one WAV file per row), saved into the
caller's Ozone folder. The output table gets the path and a preview URL of each file
(columns named annotated_*).

Text comes from the 'text' box (type a message, get one audio file), or from a connected table +
'text_column' (one audio file per row). The typed text is used only when no table is connected.

Place this file at:  backend/app/recipes/vision/text_to_speech.py
Needs:               pip install pyttsx3
                     Linux servers also need the system speech engine:  apt install espeak-ng
                     (Windows uses the built-in SAPI5 voices, nothing else to install.)
Note:                Runs fully offline - the text never leaves the server. The voices are the
                     operating system's, so they sound robotic and differ between machines.
                     Output is always WAV (macOS would produce AIFF and is refused).

Security: nothing from the node config is used as a path or command; text is capped; audio is
written only to outputs/text_to_speech/ inside the caller's tenant folder.
"""
import io
import os
import tempfile
import threading
import wave
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from backend.app.recipes.base.recipe import BaseRecipe
from backend.app.recipes.vision import _common as c

OUTPUT_FOLDER = "outputs/text_to_speech"
MAX_ROWS_PER_RUN = 50
MAX_CHARS_PER_TEXT = 5000
DEFAULT_RATE = 170  # words per minute
DEFAULT_VOLUME = 1.0

# pyttsx3 engines are not thread-safe: one synthesis at a time per process.
_TTS_LOCK = threading.Lock()


class _FatalTTSError(Exception):
    """The speech engine itself is unusable (not just one bad row). Stops the whole run."""


def _synthesize(text: str, rate: int, volume: float, voice_contains: str = "") -> bytes:
    """Speaks `text` into a WAV file and returns its bytes."""
    try:
        import pyttsx3
    except ImportError as e:
        raise _FatalTTSError("Text-to-Speech needs 'pyttsx3'. Install it with: pip install pyttsx3") from e

    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        with _TTS_LOCK:
            try:
                engine = pyttsx3.init()
            except Exception as e:  # e.g. espeak-ng missing on Linux
                raise _FatalTTSError(
                    f"The speech engine is not available on this server ({str(e)[:120]}). "
                    "On Linux install it with: apt install espeak-ng"
                ) from e
            try:
                if voice_contains:
                    needle = voice_contains.lower()
                    voices = engine.getProperty("voices") or []
                    match = next(
                        (v for v in voices if needle in f"{getattr(v, 'name', '')} {getattr(v, 'id', '')}".lower()),
                        None,
                    )
                    if match is None:
                        raise _FatalTTSError(f"No voice matching '{voice_contains}' is installed on this server.")
                    engine.setProperty("voice", match.id)
                engine.setProperty("rate", rate)
                engine.setProperty("volume", volume)
                engine.save_to_file(text, path)
                engine.runAndWait()
            finally:
                engine.stop()
                del engine  # pyttsx3 caches engines; release it so the next call starts clean
        with open(path, "rb") as f:
            data = f.read()
    finally:
        if os.path.exists(path):
            os.unlink(path)
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("The speech engine did not produce a WAV file (empty output or unsupported OS)")
    return data


def _wav_seconds(data: bytes) -> Optional[float]:
    try:
        with wave.open(io.BytesIO(data)) as w:
            return round(w.getnframes() / float(w.getframerate()), 3)
    except Exception:
        return None


class TextToSpeechRecipe(BaseRecipe):
    recipe_id = "text_to_speech"
    name = "Text-to-Speech"
    version = "1.0.0"
    category = "vision_audio"
    description = "Converts text into spoken audio files."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {
                    "type": "textarea", "title": "Text to Speak",
                    "description": (
                        f"Type the message here (max {MAX_CHARS_PER_TEXT} characters). "
                        "Used when no table is connected."
                    ),
                },
                "text_column": {
                    "type": "string", "title": "Text Column (table input only)", "input_type": "column_select",
                    "description": (
                        f"Only when a table is connected: the column holding the text to speak, one audio file "
                        f"per row (max {MAX_CHARS_PER_TEXT} characters per row). Leave empty when you type the text above."
                    ),
                },
                "rate": {
                    "type": "integer", "title": "Speaking Speed (words/min)", "input_type": "slider",
                    "default": DEFAULT_RATE, "minimum": 80, "maximum": 300,
                },
                "volume": {
                    "type": "number", "title": "Volume", "input_type": "slider",
                    "default": DEFAULT_VOLUME, "minimum": 0.1, "maximum": 1.0,
                },
                "voice_contains": {
                    "type": "string", "title": "Voice Name Contains (optional)", "input_type": "text",
                    "description": "Picks the first installed voice whose name contains this text, e.g. 'english' or 'zira'. Empty = system default.",
                },
            },
            "required": [],
        }

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        errors: List[str] = []
        text = config.get("text")
        if text is not None and not isinstance(text, str):
            errors.append("text must be text.")
        elif isinstance(text, str) and len(text.strip()) > MAX_CHARS_PER_TEXT:
            errors.append(f"Text is longer than {MAX_CHARS_PER_TEXT} characters.")
        typed = isinstance(text, str) and bool(text.strip())
        if not typed and not str(config.get("text_column") or "").strip():
            errors.append("Type the text to speak ('text'), or connect a table and choose its text column ('text_column').")
        for err in (
            c.validate_range(config, "rate", 80, 300, integer=True),
            c.validate_range(config, "volume", 0.1, 1.0),
        ):
            if err:
                errors.append(err)
        voice = config.get("voice_contains")
        if voice is not None and (not isinstance(voice, str) or len(voice) > 50):
            errors.append("voice_contains must be text of at most 50 characters.")
        return errors

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        errors = self.validate_config(config)
        if errors:
            raise ValueError("; ".join(errors))
        tenant_id = c.require_tenant(config, self.name, "text")
        table = (inputs or {}).get("dataframe")
        if isinstance(table, (list, tuple, dict)):
            # JSON rows, e.g. {"dataframe": [{"message": "Hi"}]} sent to POST /recipes/{id}/run,
            # which does not convert the request body into a DataFrame.
            try:
                table = pd.DataFrame(list(table) if isinstance(table, tuple) else table)
            except Exception as e:
                raise ValueError(f"inputs.dataframe must be a list of row objects like [{{\"text\": \"Hello\"}}] ({str(e)[:80]})")
        typed = str(config.get("text") or "").strip()
        col = str(config.get("text_column") or "").strip()
        if table is None:
            if not typed:
                raise ValueError(f"{self.name} needs text: type it in 'Text to Speak', or connect a table with a text column.")
            table, col = pd.DataFrame({"text": [typed]}), "text"
        else:
            if not col:
                raise ValueError("A table is connected: choose its text column, or disconnect it to speak the typed text.")
            if col not in table.columns:
                raise ValueError(f"Text column '{col}' not found. Available columns: {list(table.columns)}")
        if len(table) > MAX_ROWS_PER_RUN:
            raise ValueError(
                f"Too many rows ({len(table)}). The limit is {MAX_ROWS_PER_RUN} per run; add a Limit node before this one."
            )

        from backend.app.infrastructure.storage.ozone_client import upload_file_to_ozone

        rate = int(config.get("rate", DEFAULT_RATE))
        volume = float(config.get("volume", DEFAULT_VOLUME))
        voice = str(config.get("voice_contains") or "").strip()
        run_tag = f"{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"

        values = table[col].tolist()
        results: List[Dict[str, Any]] = []
        for idx, value in enumerate(values):
            res: Dict[str, Any] = {"path": None, "seconds": None, "chars": 0, "error": None}
            try:
                if c.is_empty(value):
                    raise ValueError("Empty text")
                text = str(value).strip()
                res["chars"] = len(text)
                if len(text) > MAX_CHARS_PER_TEXT:
                    raise ValueError(f"Text is longer than {MAX_CHARS_PER_TEXT} characters")
                data = _synthesize(text, rate, volume, voice)
                res["seconds"] = _wav_seconds(data)
                res["path"] = upload_file_to_ozone(data, f"{OUTPUT_FOLDER}/{run_tag}/{idx}.wav", tenant_id, "audio/wav")
            except _FatalTTSError:
                raise
            except Exception as e:  # one bad row must not fail the whole run
                res["error"] = c.safe_error(e)
            results.append(res)

        out = table.copy().reset_index(drop=True)
        out["annotated_path"] = [r["path"] for r in results]
        out["annotated_preview_url"] = [c.preview_url(r["path"]) for r in results]
        out["annotated_duration_s"] = [r["seconds"] for r in results]
        out["char_count"] = [r["chars"] for r in results]
        out["tts_error"] = [r["error"] for r in results]

        failed = [r for r in results if r["error"]]
        previews = [
            {
                "text_preview": "" if c.is_empty(v) else str(v)[:100],
                "annotated_path": r["path"],
                "annotated_preview_url": c.preview_url(r["path"]),
                "annotated_duration_s": r["seconds"],
                "error": r["error"],
            }
            for v, r in zip(values, results)
        ]
        metrics = {
            "task_type": "text_to_speech",
            "engine": "pyttsx3",
            "rate": rate,
            "volume": volume,
            "rows_processed": len(results) - len(failed),
            "rows_failed": len(failed),
            "total_characters": int(sum(r["chars"] for r in results)),
            "total_annotated_seconds": round(sum(r["seconds"] or 0 for r in results), 3),
            "errors_sample": [r["error"] for r in failed[:5]],
            "preview_items": previews,
        }
        return {"dataframe": out, "metrics": metrics, "output_summary": metrics, "task_type": "text_to_speech"}

    def to_code(self, config: Dict[str, Any]) -> str:
        typed = str(config.get("text") or "").strip()
        source = repr(typed) if typed else f"df[{config.get('text_column', 'text')!r}].iloc[0]"
        return (
            "import pyttsx3\n\n"
            "engine = pyttsx3.init()\n"
            f"engine.setProperty('rate', {int(config.get('rate', DEFAULT_RATE))})\n"
            f"engine.setProperty('volume', {float(config.get('volume', DEFAULT_VOLUME))})\n"
            f"engine.save_to_file({source}, 'speech.wav')\n"
            "engine.runAndWait()"
        )