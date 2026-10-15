"""Tests for the image_classifier and ocr recipes (models and Ozone are mocked)."""
import io
import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from PIL import Image

from backend.app.recipes.base.registry import recipe_registry
from backend.app.recipes.vision.image_classifier import ImageClassifierRecipe, UNCERTAIN_LABEL
from backend.app.recipes.vision.ocr import OCRRecipe


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
    assert out["metrics"]["images_processed"] == 1


def test_classifier_uncertain_and_bad_image():
    probs = MagicMock(top5=[1, 0, 2, 3, 4], top5conf=[0.2, 0.1, 0.1, 0.1, 0.1])
    model = MagicMock()
    model.predict.return_value = [MagicMock(probs=probs, names={i: f"c{i}" for i in range(5)})]
    df = pd.DataFrame({"img": ["unstructured/a.jpg", ""]})
    with patch("backend.app.recipes.vision.object_detection._read_image_bytes", return_value=_jpeg_bytes()), \
         patch("backend.app.recipes.vision.object_detection._load_model", return_value=model):
        out = ImageClassifierRecipe().execute({"dataframe": df}, {"image_column": "img", "tenant_id": 1, "min_confidence": 0.5})
    o = out["dataframe"]
    assert o["predicted_label"].iloc[0] == UNCERTAIN_LABEL
    assert o["classification_error"].iloc[1] == "Empty image path"  # bad row does not fail the run
    assert out["metrics"]["images_failed"] == 1


def test_classifier_needs_tenant_and_rejects_both_sources():
    with pytest.raises(PermissionError):
        ImageClassifierRecipe().execute({}, {"image_files": ["unstructured/a.jpg"]})
    with pytest.raises(ValueError):
        ImageClassifierRecipe().execute({"dataframe": pd.DataFrame({"x": [1]})}, {"image_files": ["unstructured/a.jpg"], "tenant_id": 1})


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
        assert out["metrics"]["preview_items"][0]["text_preview"] == "Hello"

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
    