"""Tests for speech_to_text and text_to_speech (models, speech engine and Ozone are mocked,
except test_real_engine_smoke which uses the real engine when it is installed)."""
import io
import json
import threading
import wave
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from backend.app.recipes.base.registry import recipe_registry
from backend.app.recipes.vision import speech_to_text as stt
from backend.app.recipes.vision import text_to_speech as tts
from backend.app.recipes.vision.speech_to_text import SpeechToTextRecipe
from backend.app.recipes.vision.text_to_speech import TextToSpeechRecipe


def _wav_bytes(seconds=0.5, rate=8000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.zeros(int(seconds * rate), dtype=np.int16).tobytes())
    return buf.getvalue()


def _fake_whisper(duration=5.0, language="en", prob=0.98):
    segs = [
        SimpleNamespace(start=0.0, end=2.0, text=" Hello world. ", avg_logprob=-0.1),
        SimpleNamespace(start=2.0, end=5.0, text=" Second part. ", avg_logprob=-0.5),
    ]
    info = SimpleNamespace(language=language, language_probability=prob, duration=duration)
    model = MagicMock()
    model.transcribe.return_value = (iter(segs), info)
    return model


def _stt_patches(model, audio=b"RIFFxxxxWAVE"):
    return (
        patch("backend.app.recipes.vision._common.read_media_bytes", return_value=(audio, "wav")),
        patch("backend.app.recipes.vision.speech_to_text._load_model", return_value=(model, threading.Lock())),
    )


# ------------------------------------------------------------------ registration / validation
def test_registered_and_enabled():
    for rid, media in (("speech_to_text", "audio"), ("text_to_speech", None)):
        metas = [m for m in recipe_registry.list_all() if m.recipe_id == rid]
        assert len(metas) == 1 and metas[0].enabled is True
        assert metas[0].subgroup == "Computer Vision & Audio" and metas[0].media_type == media


def test_validation():
    s, t = SpeechToTextRecipe(), TextToSpeechRecipe()
    assert s.validate_config({"audio_files": ["unstructured/a.wav"]}) == []
    assert s.validate_config({})
    assert s.validate_config({"audio_files": ["a.wav"] * 11})            # over the 10-file limit
    assert s.validate_config({"audio_files": ["a.wav"], "model": "../evil"})
    assert s.validate_config({"audio_files": ["a.wav"], "language": "xx"})
    assert s.validate_config({"audio_files": ["a.wav"], "task": "dance"})
    assert s.validate_config({"audio_files": ["a.wav"], "output_format": "xml"})
    assert t.validate_config({"text": "Hello there"}) == []
    assert t.validate_config({})                                       # no text
    assert t.validate_config({"text": "   "})                          # blank text counts as missing
    assert t.validate_config({"text_column": "msg"})                   # a column is no longer a source
    assert t.validate_config({"text": "x" * (tts.MAX_CHARS_PER_TEXT + 1)})
    assert t.validate_config({"text": 123})
    assert t.validate_config({"text": "Hi", "rate": 10})
    assert t.validate_config({"text": "Hi", "volume": 5})
    assert t.validate_config({"text": "Hi", "voice_contains": "x" * 51})


# ------------------------------------------------------------------ speech to text
def test_stt_per_audio():
    model = _fake_whisper()
    p1, p2 = _stt_patches(model)
    with p1, p2:
        out = SpeechToTextRecipe().execute({}, {"audio_files": ["unstructured/a.wav"], "tenant_id": 1})
    row = out["dataframe"].iloc[0]
    assert row["transcript"] == "Hello world. Second part."
    assert row["detected_language"] == "en" and row["duration_s"] == 5.0 and row["num_segments"] == 2
    segs = json.loads(row["segments"])
    assert segs[0]["start_s"] == 0.0 and segs[1]["end_s"] == 5.0 and 0 < segs[0]["confidence"] <= 1
    assert row["original_preview_url"].endswith("key=unstructured/a.wav")
    m = out["metrics"]
    assert m["items_processed"] == 1 and m["items_failed"] == 0 and m["total_segments"] == 2
    item = m["preview_items"][0]
    assert m["total_duration_s"] == 5.0 and item["text"] == "Hello world. Second part."
    assert item["original_preview_url"].endswith("key=unstructured/a.wav") and item["num_segments"] == 2
    assert item["detected_language"] == "en" and len(item["segments"]) == 2
    assert "media_path" not in item and "media_preview_url" not in item        # no output file


def test_stt_per_segment_and_options_passed_to_model():
    model = _fake_whisper()
    p1, p2 = _stt_patches(model)
    cfg = {"audio_files": ["unstructured/a.wav"], "tenant_id": 1, "output_format": "one_row_per_segment",
           "language": "ur", "task": "translate", "vad_filter": True}
    with p1, p2:
        out = SpeechToTextRecipe().execute({}, cfg)
    df = out["dataframe"]
    assert len(df) == 2 and list(df["segment_index"]) == [0, 1] and list(df["text"]) == ["Hello world.", "Second part."]
    assert "transcript" not in df.columns and "segments" not in df.columns
    kw = model.transcribe.call_args.kwargs
    assert kw["language"] == "ur" and kw["task"] == "translate" and kw["vad_filter"] is True


def test_stt_auto_language_means_none():
    model = _fake_whisper()
    p1, p2 = _stt_patches(model)
    with p1, p2:
        SpeechToTextRecipe().execute({}, {"audio_files": ["unstructured/a.wav"], "tenant_id": 1})
    assert model.transcribe.call_args.kwargs["language"] is None


def test_stt_takes_files_only_and_ignores_a_connected_table():
    assert "audio_column" not in SpeechToTextRecipe().get_schema()["properties"]
    assert SpeechToTextRecipe().validate_config({"audio_column": "rec"})            # a column alone is not a source
    model = _fake_whisper()
    p1, p2 = _stt_patches(model)
    df = pd.DataFrame({"call_id": [7], "rec": ["unstructured/other.wav"]})
    with p1, p2:
        out = SpeechToTextRecipe().execute({"dataframe": df}, {"audio_files": ["unstructured/a.wav"], "tenant_id": 1})
    assert "call_id" not in out["dataframe"].columns and out["metrics"]["input_source"] == "audio_files"
    assert list(out["dataframe"]["audio_path"]) == ["unstructured/a.wav"]


def test_stt_bad_file_does_not_fail_run():
    model = _fake_whisper()
    with patch("backend.app.recipes.vision._common.read_media_bytes", side_effect=ValueError("Unsupported audio type: exe")), \
         patch("backend.app.recipes.vision.speech_to_text._load_model", return_value=(model, threading.Lock())):
        out = SpeechToTextRecipe().execute({}, {"audio_files": ["unstructured/a.wav"], "tenant_id": 1})
    row = out["dataframe"].iloc[0]
    assert row["transcription_error"] == "Unsupported audio type: exe" and pd.isna(row["transcript"])
    assert out["metrics"]["items_failed"] == 1 and out["metrics"]["errors_sample"]
    # segment format still returns one row for the failed file
    with patch("backend.app.recipes.vision._common.read_media_bytes", side_effect=ValueError("bad")), \
         patch("backend.app.recipes.vision.speech_to_text._load_model", return_value=(model, threading.Lock())):
        out2 = SpeechToTextRecipe().execute(
            {}, {"audio_files": ["unstructured/a.wav"], "tenant_id": 1, "output_format": "one_row_per_segment"})
    assert len(out2["dataframe"]) == 1 and out2["dataframe"]["transcription_error"].iloc[0] == "bad"


def test_stt_too_long_audio_is_refused():
    model = _fake_whisper(duration=stt.MAX_AUDIO_SECONDS + 1)
    p1, p2 = _stt_patches(model)
    with p1, p2:
        out = SpeechToTextRecipe().execute({}, {"audio_files": ["unstructured/a.wav"], "tenant_id": 1})
    assert "longer than" in out["dataframe"]["transcription_error"].iloc[0]


def test_stt_silence_gives_empty_transcript_not_error():
    model = MagicMock()
    model.transcribe.return_value = (iter([]), SimpleNamespace(language="en", language_probability=0.5, duration=3.0))
    p1, p2 = _stt_patches(model)
    with p1, p2:
        out = SpeechToTextRecipe().execute({}, {"audio_files": ["unstructured/a.wav"], "tenant_id": 1})
    row = out["dataframe"].iloc[0]
    assert row["transcript"] == "" and row["num_segments"] == 0 and pd.isna(row["transcription_error"])


def test_stt_pyav_mismatch_gives_clear_error():
    model = MagicMock()
    model.transcribe.side_effect = TypeError("open() got an unexpected keyword argument 'metadata_errors'")
    p1, p2 = _stt_patches(model)
    with p1, p2:
        out = SpeechToTextRecipe().execute({}, {"audio_files": ["unstructured/a.wav"], "tenant_id": 1})
    assert 'pip install "av<19"' in out["dataframe"]["transcription_error"].iloc[0]


def test_stt_missing_package_fails_whole_run():
    with patch("backend.app.recipes.vision.speech_to_text._load_model", side_effect=RuntimeError("needs faster-whisper")):
        with pytest.raises(RuntimeError):
            SpeechToTextRecipe().execute({}, {"audio_files": ["unstructured/a.wav"], "tenant_id": 1})


def test_stt_needs_tenant():
    with pytest.raises(PermissionError):
        SpeechToTextRecipe().execute({}, {"audio_files": ["unstructured/a.wav"]})


# ------------------------------------------------------------------ text to speech
def _upload_patch():
    return patch("backend.app.infrastructure.storage.ozone_client.upload_file_to_ozone",
                 side_effect=lambda data, key, tenant, ctype="": f"s3://b/ml-workflow/t{tenant}/{key}")


def test_tts_takes_text_only_no_column_input():
    schema = TextToSpeechRecipe().get_schema()
    assert "text_column" not in schema["properties"] and schema["required"] == ["text"]
    assert schema["properties"]["text"]["type"] == "textarea"
    with pytest.raises(ValueError, match="Type the text"):                 # a connected table does not replace the text
        TextToSpeechRecipe().execute({"dataframe": pd.DataFrame({"msg": ["a"]})}, {"text_column": "msg", "tenant_id": 1})


def test_tts_execute():
    with patch("backend.app.recipes.vision.text_to_speech._synthesize", return_value=_wav_bytes(0.5)) as syn, \
         _upload_patch() as up:
        out = TextToSpeechRecipe().execute({}, {"text": "  Hello there  ", "tenant_id": 3, "rate": 150, "volume": 0.5})
    o = out["dataframe"]
    assert len(o) == 1 and o["text"].iloc[0] == "Hello there" and o["char_count"].iloc[0] == 11
    assert o["media_path"].iloc[0].startswith("s3://b/ml-workflow/t3/outputs/text_to_speech/") and o["media_path"].iloc[0].endswith("/0.wav")
    assert o["duration_s"].iloc[0] == 0.5 and o["media_preview_url"].iloc[0].endswith(o["media_path"].iloc[0])
    assert pd.isna(o["tts_error"].iloc[0])
    assert up.call_args[0][3] == "audio/wav" and up.call_args[0][2] == 3 and up.call_count == 1
    assert syn.call_args[0][0] == "Hello there" and syn.call_args[0][1:3] == (150, 0.5)
    m = out["metrics"]
    assert m["items_processed"] == 1 and m["items_failed"] == 0 and m["total_duration_s"] == 0.5
    assert m["input_source"] == "typed_text" and m["total_characters"] == 11
    item = m["preview_items"][0]
    assert item["text"] == "Hello there" and item["duration_s"] == 0.5
    assert item["media_preview_url"].endswith(o["media_path"].iloc[0]) and item["error"] is None
    assert "original_path" not in item and "original_preview_url" not in item   # the input is text, not a file


def test_tts_a_connected_table_is_ignored_when_text_is_typed():
    with patch("backend.app.recipes.vision.text_to_speech._synthesize", return_value=_wav_bytes()), _upload_patch():
        out = TextToSpeechRecipe().execute(
            {"dataframe": pd.DataFrame({"msg": ["other"]})}, {"text": "typed message", "tenant_id": 1, "text_column": "leftover"})
    assert list(out["dataframe"].columns) == ["text", "media_path", "media_preview_url", "duration_s", "char_count", "tts_error"]
    assert out["dataframe"]["text"].iloc[0] == "typed message"


def test_tts_synthesis_failure_is_reported_without_media_keys():
    with patch("backend.app.recipes.vision.text_to_speech._synthesize", side_effect=ValueError("The speech engine did not produce a WAV file")), \
         _upload_patch() as up:
        out = TextToSpeechRecipe().execute({}, {"text": "hello", "tenant_id": 1})
    o = out["dataframe"]
    assert "WAV" in o["tts_error"].iloc[0] and pd.isna(o["media_path"].iloc[0]) and up.call_count == 0
    assert out["metrics"]["items_failed"] == 1 and out["metrics"]["items_processed"] == 0
    item = out["metrics"]["preview_items"][0]
    assert "WAV" in item["error"] and item["text"] == "hello"
    assert "media_path" not in item and "media_preview_url" not in item        # nothing was generated


def test_tts_engine_unavailable_fails_whole_run():
    with patch("backend.app.recipes.vision.text_to_speech._synthesize", side_effect=tts._FatalTTSError("no engine")), _upload_patch():
        with pytest.raises(tts._FatalTTSError):
            TextToSpeechRecipe().execute({}, {"text": "hello", "tenant_id": 1})


def test_tts_input_errors():
    r = TextToSpeechRecipe()
    with pytest.raises(PermissionError):
        r.execute({}, {"text": "hello"})
    with pytest.raises(ValueError, match="Type the text"):
        r.execute({}, {"tenant_id": 1})
    with pytest.raises(ValueError, match="Type the text"):
        r.execute({}, {"text": "   ", "tenant_id": 1})
    with pytest.raises(ValueError, match="longer than"):
        r.execute({}, {"text": "x" * (tts.MAX_CHARS_PER_TEXT + 1), "tenant_id": 1})


def test_tts_rejects_non_wav_engine_output():
    fake_engine = MagicMock()
    fake_engine.getProperty.return_value = []
    fake_pyttsx3 = MagicMock()
    fake_pyttsx3.init.return_value = fake_engine  # save_to_file writes nothing -> empty file
    with patch.dict("sys.modules", {"pyttsx3": fake_pyttsx3}):
        with pytest.raises(ValueError, match="WAV"):
            tts._synthesize("hello", 170, 1.0)
    fake_engine.stop.assert_called_once()


def test_tts_unknown_voice_is_fatal():
    fake_engine = MagicMock()
    fake_engine.getProperty.return_value = [SimpleNamespace(name="English", id="en")]
    fake_pyttsx3 = MagicMock()
    fake_pyttsx3.init.return_value = fake_engine
    with patch.dict("sys.modules", {"pyttsx3": fake_pyttsx3}):
        with pytest.raises(tts._FatalTTSError, match="No voice"):
            tts._synthesize("hello", 170, 1.0, voice_contains="klingon")


def test_real_engine_smoke():
    """Uses the real speech engine when this machine has one (skipped otherwise)."""
    pyttsx3 = pytest.importorskip("pyttsx3")
    try:
        pyttsx3.init().stop()
    except Exception:
        pytest.skip("no speech engine on this machine")
    data = tts._synthesize("Testing the pipeline.", 170, 1.0)
    assert data[:4] == b"RIFF" and (tts._wav_seconds(data) or 0) > 0.3