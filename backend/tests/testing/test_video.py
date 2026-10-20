"""
Tests the ANNOTATED VIDEO output and the per-category summary of video_tracking.py.

Place at:   backend/tests/test_video_output.py            (run from the repo root)
Run:        pytest backend/tests/test_video_output.py -v -s
Needs:      pip install pytest numpy pillow opencv-python-headless "av<19"

Real: your recipe, OpenCV (reads the source video) and PyAV (H.264 encoder + the decoder used to check the result).
Fake: the YOLO model (no download / GPU) and Ozone storage.
"""
import io
import os
import tempfile

import numpy as np
import pytest

av = pytest.importorskip("av")
cv2 = pytest.importorskip("cv2")

from backend.app.core.config import settings  # noqa: E402
from backend.app.infrastructure.storage import ozone_client as oz  # noqa: E402
from backend.app.recipes.vision import object_detection as od  # noqa: E402
from backend.app.recipes.vision import video_tracking as vt  # noqa: E402

TENANT = 1
FPS, W, H, SRC_FRAMES = 30, 320, 240, 90
BOX = (50, 50, 150, 150)  # the fake detector's box (x1, y1, x2, y2)


# ---------------------------------------------------------------- helpers
def make_source_video() -> bytes:
    """A real 3-second grey H.264 .mp4 that OpenCV can read."""
    path = tempfile.mkstemp(suffix=".mp4")[1]
    c = av.open(path, "w")
    s = c.add_stream("libx264", rate=FPS)
    s.width, s.height, s.pix_fmt = W, H, "yuv420p"
    for _ in range(SRC_FRAMES):
        f = av.VideoFrame.from_ndarray(np.full((H, W, 3), 90, dtype=np.uint8), format="bgr24")
        for p in s.encode(f):
            c.mux(p)
    for p in s.encode():
        c.mux(p)
    c.close()
    data = open(path, "rb").read()
    os.unlink(path)
    return data


def decode(data: bytes):
    c = av.open(io.BytesIO(data))
    stream = c.streams.video[0]
    frames = [f.to_ndarray(format="bgr24") for f in c.decode(video=0)]
    info = {"codec": stream.codec_context.name, "w": stream.width, "h": stream.height,
            "fmt": stream.codec_context.pix_fmt, "fps": float(stream.average_rate)}
    c.close()
    return frames, info


def is_red(frame, x=100, y=100) -> bool:
    b, g, r = (int(v) for v in frame[y, x])
    return r > 170 and g < 90 and b < 90


class _T:  # tiny stand-in for the tensors ultralytics returns
    def __init__(self, v):
        self.v = v

    def tolist(self):
        return self.v


class _Boxes:
    def __init__(self, ids, cls, conf, xyxy):
        self.id, self.cls, self.conf, self.xyxy = _T(ids), _T(cls), _T(conf), _T(xyxy)


class _Res:
    names = {0: "person", 1: "car"}

    def __init__(self, frame, with_boxes=True):
        self.orig_img = frame
        self.boxes = _Boxes([1, 2], [0, 1], [0.9, 0.8], [list(BOX), [200, 60, 260, 120]]) if with_boxes else _Boxes(None, [], [], [])
        if not with_boxes:
            self.boxes.id = None

    def plot(self, img=None):
        img = (self.orig_img if img is None else img).copy()
        cv2.rectangle(img, BOX[:2], BOX[2:], (0, 0, 255), -1)  # solid red so the test can find it after H.264
        return img


class _FakeModel:
    names = _Res.names
    with_boxes = True

    def track(self, frame, **kwargs):
        return [_Res(frame, self.with_boxes)]


@pytest.fixture
def setup(monkeypatch):
    store, prefix = {}, oz.tenant_prefix(TENANT)
    store[prefix + "unstructured/clip.mp4"] = make_source_video()

    def upload(data, key, tenant_id, content_type):
        full = oz.tenant_prefix(tenant_id) + key
        store[full] = data
        return f"s3://{settings.OZONE_BUCKET}/{full}"

    monkeypatch.setattr(oz, "upload_file_to_ozone", upload)
    monkeypatch.setattr(oz, "get_file_bytes_from_ozone", lambda key, tenant_id: store[key])
    monkeypatch.setattr(od, "_object_size", lambda key: len(store.get(key, b"")) or 1000)
    model = _FakeModel()
    monkeypatch.setattr(vt, "_new_model", lambda name: model)
    return store, model


def run(**cfg):
    config = {"tenant_id": TENANT, "video_files": ["unstructured/clip.mp4"], "frame_stride": 5, "max_frames": 10, **cfg}
    rec = vt.VideoObjectTrackingRecipe()
    assert not rec.validate_config(config)
    return rec.execute({}, config)


def video_bytes(store, item):
    return store[item["media_path"].split(f"s3://{settings.OZONE_BUCKET}/")[1]]


# ---------------------------------------------------------------- tests
def test_annotated_video_is_a_real_continuous_browser_video(setup):
    store, _ = setup
    result = run(save_annotated_video=True)
    it = result["metrics"]["preview_items"][0]

    assert it["media_path"].endswith("_tracked.mp4") and it["media_preview_url"].endswith("_tracked.mp4")
    data = video_bytes(store, it)
    frames, info = decode(data)

    # stride 5, 10 analysed frames -> source frames 0..45 -> 46 frames in the video (not 10): it is continuous
    assert len(frames) == 46, len(frames)
    assert info["codec"] == "h264" and info["fmt"] == "yuv420p"
    assert (info["w"], info["h"]) == (W, H)
    assert abs(info["fps"] - FPS) < 0.5  # original frame rate, so it plays at real speed
    assert data.index(b"moov") < data.index(b"mdat"), "no faststart: the browser must download everything before playing"

    # frame 0 and 5 are analysed frames; frames 1-4 and 41-44 were SKIPPED and still show the held boxes
    for i in (0, 3, 5, 44):
        assert is_red(frames[i]), f"frame {i}: boxes missing"
    assert not is_red(frames[3], x=300, y=200), "the rest of the picture must stay untouched"


def test_without_the_option_no_video_is_written(setup):
    store, _ = setup
    result = run(save_preview_frames=True)
    it = result["metrics"]["preview_items"][0]
    assert it["media_path"].endswith(".jpg") and len(it["preview_frames"]) == 4  # the old behaviour: first frame
    assert not any(k.endswith(".mp4") and "outputs" in k for k in store)


def test_video_wins_over_frames_for_media_path_but_frames_are_kept(setup):
    result = run(save_annotated_video=True, save_preview_frames=True)
    it = result["metrics"]["preview_items"][0]
    assert it["media_path"].endswith(".mp4")
    assert len(it["preview_frames"]) == 4 and all(f["media_path"].endswith(".jpg") for f in it["preview_frames"])


def test_categories_summary(setup):
    result = run()
    it = result["metrics"]["preview_items"][0]
    cats = {c["label"]: c for c in it["categories"]}
    assert set(cats) == {"person", "car"}
    assert cats["person"]["num_tracks"] == 1 and cats["person"]["num_detections"] == 10
    assert cats["person"]["first_time_s"] == 0.0 and cats["person"]["last_time_s"] == 1.5  # frame 45 / 30 fps
    assert cats["person"]["avg_confidence"] == 0.9 and cats["car"]["avg_confidence"] == 0.8
    assert it["tracks_per_class"] == {"person": 1, "car": 1}  # existing field untouched


def test_long_video_is_cut_and_flagged(setup, monkeypatch):
    store, _ = setup
    monkeypatch.setattr(vt, "MAX_OUTPUT_VIDEO_FRAMES", 20)
    it = run(save_annotated_video=True)["metrics"]["preview_items"][0]
    frames, _ = decode(video_bytes(store, it))
    assert len(frames) == 20 and it["video_truncated"] is True


def test_video_is_written_even_when_nothing_is_detected(setup):
    store, model = setup
    model.with_boxes = False
    it = run(save_annotated_video=True)["metrics"]["preview_items"][0]
    frames, _ = decode(video_bytes(store, it))
    assert len(frames) >= 10 and it["categories"] == [] and it["error"] is None


def test_odd_frame_size_is_cropped_to_even(tmp_path):
    w = vt._AnnotatedVideoWriter(str(tmp_path / "odd.mp4"), 25)
    for _ in range(5):
        w.write(np.full((241, 321, 3), 128, dtype=np.uint8))
    w.close()
    frames, info = decode((tmp_path / "odd.mp4").read_bytes())
    assert (info["w"], info["h"]) == (320, 240) and len(frames) == 5


def test_unknown_fps_falls_back_to_25(tmp_path):
    w = vt._AnnotatedVideoWriter(str(tmp_path / "x.mp4"), 0.0)
    w.write(np.zeros((48, 64, 3), dtype=np.uint8))
    w.close()
    assert decode((tmp_path / "x.mp4").read_bytes())[1]["fps"] == 25.0