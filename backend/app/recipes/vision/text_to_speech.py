"""
Text-to-Speech recipe (Modeling -> Computer Vision & Audio).

Turns the message typed in the 'text' box into one spoken WAV file, saved into the caller's
Ozone folder. The one-row output table gets the path and a preview URL of the file
(columns media_path / media_preview_url). There is no table / column input.

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
                    "description": f"Type the message to turn into speech (max {MAX_CHARS_PER_TEXT} characters).",
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
            "required": ["text"],
        }

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        errors: List[str] = []
        text = config.get("text")
        if text is not None and not isinstance(text, str):
            errors.append("text must be text.")
        elif isinstance(text, str) and len(text.strip()) > MAX_CHARS_PER_TEXT:
            errors.append(f"Text is longer than {MAX_CHARS_PER_TEXT} characters.")
        if not (isinstance(text, str) and text.strip()):
            errors.append("Type the text to speak ('text').")
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
        text = str(config["text"]).strip()

        from backend.app.infrastructure.storage.ozone_client import upload_file_to_ozone

        rate = int(config.get("rate", DEFAULT_RATE))
        volume = float(config.get("volume", DEFAULT_VOLUME))
        voice = str(config.get("voice_contains") or "").strip()
        run_tag = f"{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"

        path: Optional[str] = None
        seconds: Optional[float] = None
        error: Optional[str] = None
        try:
            data = _synthesize(text, rate, volume, voice)
            seconds = _wav_seconds(data)
            path = upload_file_to_ozone(data, f"{OUTPUT_FOLDER}/{run_tag}/0.wav", tenant_id, "audio/wav")
        except _FatalTTSError:
            raise
        except Exception as e:  # report it in the result instead of crashing the node
            error = c.safe_error(e)

        out = pd.DataFrame([{
            "text": text,
            "media_path": path,
            "media_preview_url": c.preview_url(path),
            "duration_s": seconds,
            "char_count": len(text),
            "tts_error": error,
        }])
        metrics = {
            "task_type": "text_to_speech",
            "engine": "pyttsx3",
            "rate": rate,
            "volume": volume,
            "input_source": "typed_text",
            "items_processed": 0 if error else 1,
            "items_failed": 1 if error else 0,
            "total_characters": len(text),
            "total_duration_s": round(seconds or 0, 3),
            "errors_sample": [error] if error else [],
            "preview_items": [
                # the generated audio; the input is typed text, so there are no original_* keys
                c.preview_item(media=path, text=text, error=error, duration_s=seconds)
            ],
        }
        return {"dataframe": out, "metrics": metrics, "output_summary": metrics, "task_type": "text_to_speech"}

    def to_code(self, config: Dict[str, Any]) -> str:
        source = repr(str(config.get("text") or "").strip())
        return (
            "import pyttsx3\n\n"
            "engine = pyttsx3.init()\n"
            f"engine.setProperty('rate', {int(config.get('rate', DEFAULT_RATE))})\n"
            f"engine.setProperty('volume', {float(config.get('volume', DEFAULT_VOLUME))})\n"
            f"engine.save_to_file({source}, 'speech.wav')\n"
            "engine.runAndWait()"
        )