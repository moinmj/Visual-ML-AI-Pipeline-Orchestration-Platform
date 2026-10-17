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
    assert t.validate_config({"text_column": "msg"}) == []
    assert t.validate_config({"text": "Hello there"}) == []           # typed message, no table
    assert t.validate_config({})                                       # neither text nor column
    assert t.validate_config({"text": "   "})                          # blank text counts as missing
    assert t.validate_config({"text": "x" * (tts.MAX_CHARS_PER_TEXT + 1)})
    assert t.validate_config({"text": 123})
    assert t.validate_config({"text_column": "msg", "rate": 10})
    assert t.validate_config({"text_column": "msg", "volume": 5})
    assert t.validate_config({"text_column": "msg", "voice_contains": "x" * 51})


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
    assert row["audio_preview_url"].endswith("key=unstructured/a.wav")
    m = out["metrics"]
    assert m["audios_processed"] == 1 and m["audios_failed"] == 0 and m["total_segments"] == 2
    assert m["total_duration_s"] == 5.0 and m["preview_items"][0]["text_preview"].startswith("Hello world.")


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


def test_stt_table_mode_keeps_columns():
    model = _fake_whisper()
    p1, p2 = _stt_patches(model)
    df = pd.DataFrame({"call_id": [7], "rec": ["unstructured/a.wav"]})
    with p1, p2:
        out = SpeechToTextRecipe().execute({"dataframe": df}, {"audio_column": "rec", "tenant_id": 1})
    assert out["dataframe"]["call_id"].iloc[0] == 7 and out["metrics"]["input_source"] == "table"


def test_stt_bad_file_does_not_fail_run():
    model = _fake_whisper()
    with patch("backend.app.recipes.vision._common.read_media_bytes", side_effect=ValueError("Unsupported audio type: exe")), \
         patch("backend.app.recipes.vision.speech_to_text._load_model", return_value=(model, threading.Lock())):
        out = SpeechToTextRecipe().execute({}, {"audio_files": ["unstructured/a.wav"], "tenant_id": 1})
    row = out["dataframe"].iloc[0]
    assert row["transcription_error"] == "Unsupported audio type: exe" and pd.isna(row["transcript"])
    assert out["metrics"]["audios_failed"] == 1 and out["metrics"]["errors_sample"]
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


def test_tts_execute():
    df = pd.DataFrame({"id": [1, 2], "msg": ["Hello there", "Second message"]})
    with patch("backend.app.recipes.vision.text_to_speech._synthesize", return_value=_wav_bytes(0.5)) as syn, \
         _upload_patch() as up:
        out = TextToSpeechRecipe().execute({"dataframe": df}, {"text_column": "msg", "tenant_id": 3, "rate": 150, "volume": 0.5})
    o = out["dataframe"]
    assert list(o["id"]) == [1, 2] and list(o["char_count"]) == [11, 14]
    assert o["audio_path"].iloc[0].startswith("s3://b/ml-workflow/t3/outputs/text_to_speech/") and o["audio_path"].iloc[1].endswith("/1.wav")
    assert o["audio_duration_s"].iloc[0] == 0.5 and o["audio_preview_url"].iloc[0].endswith(o["audio_path"].iloc[0])
    assert up.call_args[0][3] == "audio/wav" and up.call_args[0][2] == 3
    assert syn.call_args[0][1:3] == (150, 0.5)
    m = out["metrics"]
    assert m["rows_processed"] == 2 and m["rows_failed"] == 0 and m["total_audio_seconds"] == 1.0
    assert m["preview_items"][0]["text_preview"] == "Hello there"


def test_tts_typed_text_without_a_table():
    with patch("backend.app.recipes.vision.text_to_speech._synthesize", return_value=_wav_bytes(0.5)) as syn, _upload_patch() as up:
        out = TextToSpeechRecipe().execute({}, {"text": "  Hello from the text box  ", "tenant_id": 2, "text_column": "leftover"})
    o = out["dataframe"]
    assert len(o) == 1 and o["text"].iloc[0] == "Hello from the text box"
    assert o["audio_path"].iloc[0].startswith("s3://b/ml-workflow/t2/outputs/text_to_speech/") and o["audio_duration_s"].iloc[0] == 0.5
    assert syn.call_args[0][0] == "Hello from the text box" and up.call_count == 1
    assert out["metrics"]["preview_items"][0]["audio_preview_url"].endswith(o["audio_path"].iloc[0])


def test_tts_bad_rows_do_not_fail_run():
    df = pd.DataFrame({"msg": ["ok", "", None, "x" * (tts.MAX_CHARS_PER_TEXT + 1)]})
    with patch("backend.app.recipes.vision.text_to_speech._synthesize", return_value=_wav_bytes()) as syn, _upload_patch():
        out = TextToSpeechRecipe().execute({"dataframe": df}, {"text_column": "msg", "tenant_id": 1})
    o = out["dataframe"]
    assert pd.isna(o["tts_error"].iloc[0]) and o["tts_error"].iloc[1] == "Empty text" and o["tts_error"].iloc[2] == "Empty text"
    assert "longer than" in o["tts_error"].iloc[3]
    assert pd.isna(o["audio_path"].iloc[1]) and syn.call_count == 1
    assert out["metrics"]["rows_failed"] == 3 and out["metrics"]["rows_processed"] == 1


def test_tts_accepts_json_rows_from_run_endpoint():
    rows = [{"id": 1, "msg": "Hello there"}, {"id": 2, "msg": "Second"}]
    with patch("backend.app.recipes.vision.text_to_speech._synthesize", return_value=_wav_bytes()), _upload_patch():
        out = TextToSpeechRecipe().execute({"dataframe": rows}, {"text_column": "msg", "tenant_id": 1})
    assert list(out["dataframe"]["id"]) == [1, 2] and out["metrics"]["rows_processed"] == 2
    with pytest.raises(ValueError, match="must be a list of row objects"):
        TextToSpeechRecipe().execute({"dataframe": {"msg": ["a", "b,"], "x": ["only one"]}}, {"text_column": "msg", "tenant_id": 1})


def test_tts_engine_unavailable_fails_whole_run():
    df = pd.DataFrame({"msg": ["a", "b"]})
    with patch("backend.app.recipes.vision.text_to_speech._synthesize", side_effect=tts._FatalTTSError("no engine")), _upload_patch():
        with pytest.raises(tts._FatalTTSError):
            TextToSpeechRecipe().execute({"dataframe": df}, {"text_column": "msg", "tenant_id": 1})


def test_tts_input_errors():
    r = TextToSpeechRecipe()
    with pytest.raises(PermissionError):
        r.execute({"dataframe": pd.DataFrame({"msg": ["a"]})}, {"text_column": "msg"})
    with pytest.raises(ValueError, match="needs text"):
        r.execute({}, {"text_column": "msg", "tenant_id": 1})
    with pytest.raises(ValueError, match="choose its text column"):
        r.execute({"dataframe": pd.DataFrame({"msg": ["a"]})}, {"text": "typed but a table is connected", "tenant_id": 1})
    with pytest.raises(ValueError, match="not found"):
        r.execute({"dataframe": pd.DataFrame({"msg": ["a"]})}, {"text_column": "nope", "tenant_id": 1})
    with pytest.raises(ValueError, match="Too many rows"):
        r.execute({"dataframe": pd.DataFrame({"msg": ["a"] * (tts.MAX_ROWS_PER_RUN + 1)})}, {"text_column": "msg", "tenant_id": 1})


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