"""Tests for the image_classifier and ocr recipes (models and Ozone are mocked)."""
import io
import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from PIL import Image

from backend.app.recipes.base.registry import recipe_registry
from backend.app.recipes.vision import _common as common
from backend.app.recipes.vision.image_classifier import ImageClassifierRecipe, UNCERTAIN_LABEL
from backend.app.recipes.vision.image_embedding import ImageEmbeddingRecipe
from backend.app.recipes.vision.object_detection import ObjectDetectionRecipe
from backend.app.recipes.vision.ocr import OCRRecipe
from backend.app.recipes.vision.speech_to_text import SpeechToTextRecipe
from backend.app.recipes.vision.video_tracking import VideoObjectTrackingRecipe


def _jpeg_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (100, 60), color="white").save(buf, format="JPEG")
    return buf.getvalue()


def test_registered_and_enabled():
    for rid in ("image_classifier", "ocr", "object_detection"):
        metas = [m for m in recipe_registry.list_all() if m.recipe_id == rid]
        assert len(metas) == 1 and metas[0].enabled is True
        assert metas[0].subgroup == "Computer Vision & Audio"
        assert metas[0].media_type == "image"


FILE_RECIPES = [
    (ObjectDetectionRecipe, "image_files", "image_column"),
    (ImageClassifierRecipe, "image_files", "image_column"),
    (OCRRecipe, "image_files", "image_column"),
    (ImageEmbeddingRecipe, "image_files", "image_column"),
    (VideoObjectTrackingRecipe, "video_files", "video_column"),
    (SpeechToTextRecipe, "audio_files", "audio_column"),
]


@pytest.mark.parametrize("recipe_cls, files_key, column_key", FILE_RECIPES)
def test_media_recipes_take_files_only_no_column_input(recipe_cls, files_key, column_key):
    recipe = recipe_cls()
    props = recipe.get_schema()["properties"]
    assert files_key in props and props[files_key]["input_type"] == "file_select"
    assert column_key not in props
    assert not [k for k in props if k.endswith("_column")]
    # a column on its own is no longer a valid source ...
    assert recipe.validate_config({column_key: "img"})
    # ... and a connected table does not replace the files
    with pytest.raises(ValueError):
        recipe.execute({"dataframe": pd.DataFrame({"img": ["unstructured/a.jpg"]})}, {column_key: "img", "tenant_id": 1})


def test_preview_item_leaves_out_files_that_do_not_exist():
    only_input = common.preview_item(original="unstructured/a.jpg", text="cat", predicted_label="cat")
    assert only_input["original_path"] == "unstructured/a.jpg" and only_input["text"] == "cat"
    assert only_input["predicted_label"] == "cat" and only_input["error"] is None
    assert "media_path" not in only_input and "media_preview_url" not in only_input   # no null media keys

    only_output = common.preview_item(media="s3://b/ml-workflow/t1/outputs/x.wav", text="hello")
    assert only_output["media_preview_url"].endswith("key=s3://b/ml-workflow/t1/outputs/x.wav")
    assert "original_path" not in only_output and "original_preview_url" not in only_output

    both = common.preview_item(original="unstructured/a.jpg", media="s3://b/x.jpg", error="boom")
    assert both["original_path"] and both["media_path"] and both["error"] == "boom"
    assert "media_path" not in common.preview_item(original="a", media="  ")       # blank counts as none
    assert "media_path" not in common.preview_item(original="a", media=float("nan"))


def test_validation():
    clf, ocr = ImageClassifierRecipe(), OCRRecipe()
    assert clf.validate_config({"image_files": ["unstructured/a.jpg"]}) == []
    assert any("image" in e.lower() for e in clf.validate_config({}))
    assert clf.validate_config({"image_files": ["a.jpg"], "model": "evil.pt"})
    assert clf.validate_config({"image_files": ["a.jpg"], "top_k": 9})
    assert ocr.validate_config({"image_files": ["unstructured/a.jpg"], "languages": ["ar", "en"]}) == []
    assert ocr.validate_config({"image_files": ["a.jpg"], "languages": ["xx"]})
    assert ocr.validate_config({"image_files": ["a.jpg"], "languages": ["en", "fr", "de", "es"]})
    assert ocr.validate_config({"image_files": ["a.jpg"], "min_confidence": 2})


def test_classifier_execute_files_mode():
    probs = MagicMock()
    probs.top5 = [3, 1, 2, 0, 4]
    probs.top5conf = [0.9, 0.05, 0.03, 0.01, 0.01]
    res = MagicMock(probs=probs, names={0: "a", 1: "cat", 2: "c", 3: "dog", 4: "e"})
    model = MagicMock()
    model.predict.return_value = [res]
    with patch("backend.app.recipes.vision.object_detection._read_image_bytes", return_value=_jpeg_bytes()), \
         patch("backend.app.recipes.vision.object_detection._load_model", return_value=model):
        out = ImageClassifierRecipe().execute({}, {"image_files": ["unstructured/dog.jpg"], "tenant_id": 1, "top_k": 2})
    df = out["dataframe"]
    assert df["predicted_label"].iloc[0] == "dog"
    assert df["confidence"].iloc[0] == 0.9
    assert len(json.loads(df["top_k_predictions"].iloc[0])) == 2
    assert df["image_preview_url"].iloc[0].endswith("key=unstructured/dog.jpg")
    assert out["metrics"]["items_processed"] == 1
    item = out["metrics"]["preview_items"][0]
    assert item["original_preview_url"].endswith("key=unstructured/dog.jpg")
    assert item["text"] == "dog" and item["predicted_label"] == "dog" and item["confidence"] == 0.9
    assert "media_path" not in item and "media_preview_url" not in item        # a classifier makes no output file


def test_classifier_uncertain_and_bad_image():
    probs = MagicMock(top5=[1, 0, 2, 3, 4], top5conf=[0.2, 0.1, 0.1, 0.1, 0.1])
    model = MagicMock()
    model.predict.return_value = [MagicMock(probs=probs, names={i: f"c{i}" for i in range(5)})]

    def read(value, tenant_id, from_unstructured=False):
        if value.endswith("bad.exe"):
            raise ValueError("Unsupported image type: exe")
        return _jpeg_bytes()

    with patch("backend.app.recipes.vision.object_detection._read_image_bytes", side_effect=read), \
         patch("backend.app.recipes.vision.object_detection._load_model", return_value=model):
        out = ImageClassifierRecipe().execute(
            {}, {"image_files": ["unstructured/a.jpg", "unstructured/bad.exe"], "tenant_id": 1, "min_confidence": 0.5})
    o = out["dataframe"]
    assert o["predicted_label"].iloc[0] == UNCERTAIN_LABEL
    assert o["classification_error"].iloc[1] == "Unsupported image type: exe"  # bad file does not fail the run
    assert out["metrics"]["items_failed"] == 1
    assert out["metrics"]["preview_items"][1]["error"] == "Unsupported image type: exe"


def test_classifier_needs_tenant_and_ignores_a_connected_table():
    with pytest.raises(PermissionError):
        ImageClassifierRecipe().execute({}, {"image_files": ["unstructured/a.jpg"]})
    probs = MagicMock(top5=[1, 0, 2, 3, 4], top5conf=[0.9, 0.1, 0.0, 0.0, 0.0])
    model = MagicMock()
    model.predict.return_value = [MagicMock(probs=probs, names={i: f"c{i}" for i in range(5)})]
    with patch("backend.app.recipes.vision.object_detection._read_image_bytes", return_value=_jpeg_bytes()), \
         patch("backend.app.recipes.vision.object_detection._load_model", return_value=model):
        out = ImageClassifierRecipe().execute(
            {"dataframe": pd.DataFrame({"x": [1]})}, {"image_files": ["unstructured/a.jpg"], "tenant_id": 1})
    assert list(out["dataframe"]["image_path"]) == ["unstructured/a.jpg"] and "x" not in out["dataframe"].columns


def _fake_reader():
    reader = MagicMock()
    reader.readtext.return_value = [
        ([[10, 10], [60, 10], [60, 30], [10, 30]], "Hello", 0.95),
        ([[10, 40], [60, 40], [60, 55], [10, 55]], "noise", 0.1),
    ]
    return reader


def test_ocr_execute_and_formats():
    with patch("backend.app.recipes.vision.object_detection._read_image_bytes", return_value=_jpeg_bytes()), \
         patch("backend.app.recipes.vision.ocr._get_reader", return_value=_fake_reader()):
        out = OCRRecipe().execute({}, {"image_files": ["unstructured/s.jpg"], "tenant_id": 1, "min_confidence": 0.3})
        assert out["dataframe"]["ocr_text"].iloc[0] == "Hello"
        assert out["dataframe"]["num_text_blocks"].iloc[0] == 1
        item = out["metrics"]["preview_items"][0]
        assert item["text"] == "Hello" and item["num_text_blocks"] == 1
        assert "media_path" not in item and "media_preview_url" not in item       # no annotated image was asked for

        out2 = OCRRecipe().execute({}, {"image_files": ["unstructured/s.jpg"], "tenant_id": 1, "output_format": "one_row_per_text_block"})
        assert list(out2["dataframe"]["text"]) == ["Hello"]
        assert out2["dataframe"]["x_max"].iloc[0] == 60.0


def test_ocr_saves_annotated_image():
    with patch("backend.app.recipes.vision.object_detection._read_image_bytes", return_value=_jpeg_bytes()), \
         patch("backend.app.recipes.vision.ocr._get_reader", return_value=_fake_reader()), \
         patch("backend.app.infrastructure.storage.ozone_client.upload_file_to_ozone", return_value="s3://b/ml-workflow/t1/outputs/ocr/x.jpg") as up:
        out = OCRRecipe().execute({}, {"image_files": ["unstructured/s.jpg"], "tenant_id": 1, "save_annotated_images": True})
    assert up.call_args[0][1].startswith("outputs/ocr/")
    assert out["dataframe"]["annotated_preview_url"].iloc[0].endswith("x.jpg")
    item = out["metrics"]["preview_items"][0]
    assert item["media_path"].endswith("x.jpg") and item["media_preview_url"].endswith("x.jpg")