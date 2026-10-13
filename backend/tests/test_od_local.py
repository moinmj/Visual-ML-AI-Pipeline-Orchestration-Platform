"""
Local test for the Object Detection recipe. Does NOT touch .env.

Run from the project root (the folder that contains backend/ and .env):
    python test_od_local.py

Put a few .jpg/.png files in a folder called  test_images  next to this file.
"""
import os

# Override the DB for this process only (real env vars beat .env in pydantic-settings).
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./data/platform.db"

import glob
import pandas as pd

# 1) Local stand-in for Ozone: read the image from ./test_images instead.
#    The recipe's own tenant/path checks still run before this is called.
from backend.app.infrastructure.storage import ozone_client


def fake_get_file_bytes(key, tenant_id=None):
    path = os.path.join("test_images", os.path.basename(key))
    with open(path, "rb") as f:
        return f.read()


ozone_client.get_file_bytes_from_ozone = fake_get_file_bytes

# 2) Is the recipe registered and does it override the "planned" entry?
from backend.app.recipes.base.registry import recipe_registry
from backend.app.core.config import settings

meta = recipe_registry.get_metadata("object_detection")
print(f"object_detection -> enabled={meta.enabled}, group={meta.group!r}, subgroup={meta.subgroup!r}")
print("enabled recipes with this id in list_all():",
      [m.enabled for m in recipe_registry.list_all() if m.recipe_id == "object_detection"])

# 3) Run it on the local images.
files = sorted(glob.glob("test_images/*.*"))
if not files:
    raise SystemExit("Put some images in ./test_images first.")

tenant_id = 1
prefix = f"s3://{settings.OZONE_BUCKET}/{settings.OZONE_PREFIX}/t{tenant_id}/"
df = pd.DataFrame({"id": range(len(files)),
                   "img": [prefix + os.path.basename(p) for p in files]})

recipe = recipe_registry.get("object_detection")
out = recipe.execute(
    {"dataframe": df},
    {
        "image_column": "img",
        "tenant_id": tenant_id,
        "confidence_threshold": 0.3,
        "model": "yolov8m.pt",   # <-- add this
    },
)

pd.set_option("display.width", 200)
print(out["dataframe"][["img", "num_objects", "detected_labels", "max_confidence", "detection_error"]])
print(out["metrics"])