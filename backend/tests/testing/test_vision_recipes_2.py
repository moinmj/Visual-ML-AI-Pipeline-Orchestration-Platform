"""Tests for image_embedding_generator and video_object_tracking (models and Ozone are mocked)."""
import io
import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from backend.app.recipes.base.registry import recipe_registry
from backend.app.recipes.vision.image_embedding import ImageEmbeddingRecipe
from backend.app.recipes.vision.video_tracking import VideoObjectTrackingRecipe


class L(list):
    def tolist(self):
        return list(self)


def _jpeg_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (80, 60), color="white").save(buf, format="JPEG")
    return buf.getvalue()


def _video_bytes(frames=20):
    import cv2
    fd, path = tempfile.mkstemp(suffix=".avi")
    os.close(fd)
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 64))
    for _ in range(frames):
        w.write(np.zeros((64, 64, 3), dtype=np.uint8))
    w.release()
    data = open(path, "rb").read()
    os.unlink(path)
    return data


def test_registered_and_enabled():
    for rid, media in (("image_embedding_generator", "image"), ("video_object_tracking", "video")):
        metas = [m for m in recipe_registry.list_all() if m.recipe_id == rid]
        assert len(metas) == 1 and metas[0].enabled is True
        assert metas[0].subgroup == "Computer Vision & Audio" and metas[0].media_type == media


def test_validation():
    emb, vid = ImageEmbeddingRecipe(), VideoObjectTrackingRecipe()
    assert emb.validate_config({"image_files": ["unstructured/a.jpg"]}) == []
    assert emb.validate_config({"image_files": ["a.jpg"], "model": "evil"})
    assert emb.validate_config({"image_files": ["a.jpg"], "output_format": "xml"})
    assert vid.validate_config({"video_files": ["unstructured/a.mp4"]}) == []
    assert vid.validate_config({})
    assert vid.validate_config({"video_files": ["a.mp4"] * 6})          # over the 5-video limit
    assert vid.validate_config({"video_files": ["a.mp4"], "tracker": "x.yaml"})
    assert vid.validate_config({"video_files": ["a.mp4"], "frame_stride": 0})


def _fake_embedder(dim=4):
    fake = MagicMock()
    fake.encode.side_effect = lambda imgs: np.tile(np.arange(1, dim + 1, dtype="float32"), (len(imgs), 1))
    return fake


def test_embedding_json_and_columns():
    files = ["unstructured/a.jpg", "unstructured/bad.exe"]

    def read(value, tenant_id, from_unstructured=False):
        if value.endswith("bad.exe"):
            raise ValueError("Unsupported image type: exe")
        return _jpeg_bytes()

    with patch("backend.app.recipes.vision.object_detection._read_image_bytes", side_effect=read), \
         patch("backend.app.recipes.vision.image_embedding._load_embedder", return_value=_fake_embedder()):
        out = ImageEmbeddingRecipe().execute({}, {"image_files": files, "tenant_id": 1})
        o = out["dataframe"]
        vec = json.loads(o["embedding"].iloc[0])
        assert len(vec) == 4 and abs(np.linalg.norm(vec) - 1.0) < 1e-4      # normalized
        assert pd.isna(o["embedding"].iloc[1]) and o["embedding_error"].iloc[1] == "Unsupported image type: exe"
        assert out["metrics"]["embedding_dim"] == 4 and out["metrics"]["items_failed"] == 1
        items = out["metrics"]["preview_items"]
        assert items[0]["original_path"] == "unstructured/a.jpg" and items[0]["embedding_dim"] == 4
        assert "media_path" not in items[0] and "media_preview_url" not in items[0]   # embeddings make no file

        out2 = ImageEmbeddingRecipe().execute(
            {}, {"image_files": files, "tenant_id": 1, "output_format": "columns", "normalize": False})
        o2 = out2["dataframe"]
        assert [f"emb_{i}" for i in range(4)] == [c for c in o2.columns if c.startswith("emb_")]
        assert o2["emb_3"].iloc[0] == 4.0 and np.isnan(o2["emb_0"].iloc[1])


def _fake_tracking_model():
    def track(frame, **kw):
        boxes = MagicMock()
        boxes.id = L([1, 2])
        boxes.cls = L([0, 0])
        boxes.conf = L([0.9, 0.8])
        boxes.xyxy = L([[1, 2, 30, 40], [5, 6, 20, 25]])
        res = MagicMock(boxes=boxes, names={0: "person"})
        res.plot.return_value = np.zeros((64, 64, 3), dtype=np.uint8)
        return [res]
    model = MagicMock()
    model.names = {0: "person"}
    model.track.side_effect = track
    return model


def test_video_tracking_per_track_and_per_detection():
    cfg = {"video_files": ["unstructured/v.avi"], "tenant_id": 1, "frame_stride": 2}
    with patch("backend.app.recipes.vision._common.read_media_bytes", return_value=(_video_bytes(20), "avi")), \
         patch("backend.app.recipes.vision.video_tracking._new_model", return_value=_fake_tracking_model()):
        out = VideoObjectTrackingRecipe().execute({}, cfg)
        df = out["dataframe"]
        assert sorted(df["track_id"]) == [1, 2]
        assert (df["first_frame"] == 0).all() and (df["last_frame"] == 18).all()
        assert (df["num_detections"] == 10).all()
        assert out["metrics"]["frames_processed"] == 10 and out["metrics"]["tracks_per_class"] == {"person": 2}
        assert df["video_preview_url"].iloc[0].endswith("key=unstructured/v.avi")

        out2 = VideoObjectTrackingRecipe().execute({}, {**cfg, "output_format": "one_row_per_detection", "classes": "person"})
        assert len(out2["dataframe"]) == 20


def test_video_max_frames_and_preview_frames():
    cfg = {"video_files": ["unstructured/v.avi"], "tenant_id": 1, "frame_stride": 1, "max_frames": 5, "save_preview_frames": True}
    with patch("backend.app.recipes.vision._common.read_media_bytes", return_value=(_video_bytes(20), "avi")), \
         patch("backend.app.recipes.vision.video_tracking._new_model", return_value=_fake_tracking_model()), \
         patch("backend.app.infrastructure.storage.ozone_client.upload_file_to_ozone", return_value="s3://b/ml-workflow/t1/outputs/video_tracking/x.jpg") as up:
        out = VideoObjectTrackingRecipe().execute({}, cfg)
    assert out["metrics"]["frames_processed"] == 5
    assert 1 <= up.call_count <= 4 and up.call_args[0][1].startswith("outputs/video_tracking/")
    assert out["metrics"]["preview_items"][0]["preview_frames"]
    assert "preview_frame_urls" in out["dataframe"].columns
    item = out["metrics"]["preview_items"][0]
    assert item["media_path"] and item["media_preview_url"]                 # first annotated frame
    assert item["original_preview_url"].endswith("key=unstructured/v.avi")


def test_video_without_preview_frames_has_no_media_keys():
    cfg = {"video_files": ["unstructured/v.avi"], "tenant_id": 1, "frame_stride": 2}
    with patch("backend.app.recipes.vision._common.read_media_bytes", return_value=(_video_bytes(6), "avi")), \
         patch("backend.app.recipes.vision.video_tracking._new_model", return_value=_fake_tracking_model()):
        out = VideoObjectTrackingRecipe().execute({}, cfg)
    item = out["metrics"]["preview_items"][0]
    assert "media_path" not in item and "media_preview_url" not in item
    assert item["original_path"] == "unstructured/v.avi"


def test_video_bad_file_does_not_fail_run_and_unknown_class():
    with patch("backend.app.recipes.vision._common.read_media_bytes", return_value=(b"not a video", "avi")), \
         patch("backend.app.recipes.vision.video_tracking._new_model", return_value=_fake_tracking_model()):
        out = VideoObjectTrackingRecipe().execute({}, {"video_files": ["unstructured/v.avi"], "tenant_id": 1})
    assert out["metrics"]["items_failed"] == 1 and out["dataframe"]["tracking_error"].iloc[0]
    with patch("backend.app.recipes.vision._common.read_media_bytes", return_value=(_video_bytes(4), "avi")), \
         patch("backend.app.recipes.vision.video_tracking._new_model", return_value=_fake_tracking_model()):
        out = VideoObjectTrackingRecipe().execute({}, {"video_files": ["unstructured/v.avi"], "tenant_id": 1, "classes": ["unicorn"]})
    assert "Unknown class" in out["dataframe"]["tracking_error"].iloc[0]


def test_video_needs_tenant():
    with pytest.raises(PermissionError):
        VideoObjectTrackingRecipe().execute({}, {"video_files": ["unstructured/v.avi"]})