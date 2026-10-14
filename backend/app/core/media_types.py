"""
Shared rules for unstructured data (images, audio, video).

Used by the unstructured upload (datasets/unstructured.py) and, later, by the
recipes, so "what counts as an image / how big may it be" is defined in
exactly one place.

Place this file at:  backend/app/core/media_types.py
"""
from typing import Dict, Optional

MB = 1024 * 1024

# kind -> allowed extensions and max size per file.
# SVG is deliberately NOT allowed (it can carry scripts).
MEDIA_KINDS: Dict[str, Dict] = {
    "image": {
        "extensions": ("jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"),
        "max_bytes": 25 * MB,
    },
    "audio": {
        "extensions": ("wav", "mp3", "m4a", "flac", "ogg"),
        "max_bytes": 200 * MB,
    },
    "video": {
        "extensions": ("mp4", "avi", "mov", "mkv", "webm"),
        "max_bytes": 500 * MB,
    },
}

CONTENT_TYPES: Dict[str, str] = {
    "jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "bmp": "image/bmp",
    "webp": "image/webp", "tif": "image/tiff", "tiff": "image/tiff",
    "wav": "audio/wav", "mp3": "audio/mpeg", "m4a": "audio/mp4", "flac": "audio/flac", "ogg": "audio/ogg",
    "mp4": "video/mp4", "avi": "video/x-msvideo", "mov": "video/quicktime",
    "mkv": "video/x-matroska", "webm": "video/webm",
}

# Where unstructured uploads live, inside the tenant folder:
#   ml-workflow/t<tenant_id>/unstructured/<file>      (one flat folder, type = extension)
UNSTRUCTURED_FOLDER = "unstructured"

# Limits for one upload request
MAX_FILES_PER_UPLOAD = 200
MAX_REQUEST_BYTES = 1024 * MB  # 1 GB of files in total per request


def extension_of(name: str) -> str:
    name = (name or "").strip()
    return name.rsplit(".", 1)[-1].lower() if "." in name else ""


def kind_for_extension(ext: str) -> Optional[str]:
    ext = (ext or "").lower().lstrip(".")
    for kind, rule in MEDIA_KINDS.items():
        if ext in rule["extensions"]:
            return kind
    return None


def content_type_for(ext: str) -> str:
    return CONTENT_TYPES.get((ext or "").lower().lstrip("."), "application/octet-stream")


def looks_like(ext: str, head: bytes) -> bool:
    """
    Cheap check that the first bytes match the extension, so a renamed file
    (e.g. an .exe called song.mp3) is refused. Images are verified with
    Pillow instead, so they are not handled here.
    """
    ext = (ext or "").lower()
    if ext == "wav":
        return head[:4] == b"RIFF" and head[8:12] == b"WAVE"
    if ext == "avi":
        return head[:4] == b"RIFF" and head[8:12] == b"AVI "
    if ext == "mp3":
        return head[:3] == b"ID3" or (len(head) > 1 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0)
    if ext == "flac":
        return head[:4] == b"fLaC"
    if ext == "ogg":
        return head[:4] == b"OggS"
    if ext in ("m4a", "mp4", "mov"):
        return head[4:8] == b"ftyp" or head[4:8] in (b"moov", b"mdat", b"wide", b"free")
    if ext in ("mkv", "webm"):
        return head[:4] == b"\x1a\x45\xdf\xa3"
    return True