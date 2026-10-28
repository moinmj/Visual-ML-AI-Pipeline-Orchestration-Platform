"""
Reproduces exactly what the video recipe does to read a video from Ozone,
but prints the REAL error instead of "Image not found or not readable".

Run from the repo root (the folder that contains `backend/`), so .env is loaded:

    python check_video_key.py 1 unstructured/scenewalking.mp4

Args: <tenant_id> <video path as given to the recipe>
"""
import sys
import traceback

tenant_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
value = sys.argv[2] if len(sys.argv) > 2 else "unstructured/scenewalking.mp4"

from backend.app.core.config import settings
from backend.app.infrastructure.storage import ozone_client as oz
from backend.app.recipes.vision import object_detection as od
from backend.app.recipes.vision import _common as c

print("endpoint :", settings.OZONE_ENDPOINT)
print("bucket   :", repr(settings.OZONE_BUCKET))
print("prefix   :", repr(settings.OZONE_PREFIX), "-> tenant folder:", oz.tenant_prefix(tenant_id))

# 1) the key the recipe builds
try:
    key = od._unstructured_key(value, tenant_id)
    print("key built:", repr(key))
except Exception:
    print("KEY BUILD FAILED:")
    traceback.print_exc()
    sys.exit(1)

# 2) the HEAD request, with the real exception shown
try:
    r = oz.get_s3_client().head_object(Bucket=settings.OZONE_BUCKET, Key=key)
    print("HEAD ok  : ContentLength =", r["ContentLength"])
except Exception as e:
    print("HEAD FAILED:", type(e).__name__)
    print("  ", e)
    resp = getattr(e, "response", None)
    if resp:
        print("   error code:", resp.get("Error", {}).get("Code"), "| http:", resp.get("ResponseMetadata", {}).get("HTTPStatusCode"))

# 3) what the app can actually see in that folder
try:
    folder = oz.tenant_prefix(tenant_id) + "unstructured/"
    resp = oz.get_s3_client().list_objects_v2(Bucket=settings.OZONE_BUCKET, Prefix=folder, MaxKeys=20)
    print(f"LIST {folder!r}:")
    for o in resp.get("Contents", []):
        print("   ", repr(o["Key"]), o["Size"])
    if not resp.get("Contents"):
        print("    (nothing listed)")
except Exception as e:
    print("LIST FAILED:", type(e).__name__, e)

# 4) the exact call the video recipe makes
try:
    data, ext = c.read_media_bytes(value, tenant_id, True, "video")
    print(f"read_media_bytes ok: {len(data)} bytes, .{ext}")
except Exception as e:
    print("read_media_bytes FAILED:", type(e).__name__, e)