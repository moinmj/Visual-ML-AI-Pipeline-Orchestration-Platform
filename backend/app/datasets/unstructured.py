"""
Unstructured data upload (images, audio, video).

Called by the EXISTING upload route (POST /datasets/upload) when the request
carries data_type=unstructured. Structured uploads are not touched.

What it does
  * Stores each file, as is, in ONE flat folder inside the caller's tenant folder:
        ml-workflow/t<tenant_id>/unstructured/<file>
    The file type is decided only by the extension (no sub-folders per type).
  * Creates NO dataset record and changes NO database table.
  * If a name is already taken, the new file is stored as name_2.ext, name_3.ext ...
    (nothing is ever overwritten) and the response says so.
  * Bad files are skipped and listed with the reason; the rest are stored.
    If storage itself fails, everything stored by this request is removed.

Place this file at:  backend/app/datasets/unstructured.py
"""
import io
import re
from typing import Any, Dict, List, Literal, Optional, Set

from fastapi import UploadFile
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from backend.app.core import media_types as mt
from backend.app.core.exceptions import StorageException, ValidationException
from backend.app.core.logging import logger
from backend.app.infrastructure.storage.storage_manager import storage_manager


class UploadedFileInfo(BaseModel):
    file_name: str = Field(..., description="Name the user uploaded")
    stored_as: str = Field(..., description="Name it was stored under")
    path: str = Field(..., description="Path inside the user's folder, e.g. unstructured/cat.png")
    media_type: str = Field(..., description="image, audio or video")
    size_bytes: int
    renamed: bool = False  # True when stored_as differs from the uploaded name (cleaned up or de-duplicated)
    width: Optional[int] = None
    height: Optional[int] = None
    preview_url: Optional[str] = Field(None, description="Direct URL to preview/stream this media file")


class SkippedFile(BaseModel):
    file_name: str
    reason: str


class UnstructuredUploadResponse(BaseModel):
    data_type: Literal["unstructured"] = "unstructured"
    folder: str = Field(mt.UNSTRUCTURED_FOLDER + "/", description="Folder the files were stored in")
    uploaded: List[UploadedFileInfo] = Field(default_factory=list)
    skipped: List[SkippedFile] = Field(default_factory=list)


def _clean_name(filename: str) -> str:
    """Drops any path, replaces unsafe characters, lower-cases the extension. Result is 'stem.ext'."""
    base = (filename or "file").replace("\\", "/").rsplit("/", 1)[-1]
    ext = mt.extension_of(base)
    stem = base.rsplit(".", 1)[0] if "." in base else base
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")[:100] or "file"
    return f"{stem}.{ext}"


def _key_exists(rel_key: str, tenant_id: int) -> bool:
    """Exact check (one HEAD request), so it stays correct however many files the folder holds."""
    from botocore.exceptions import BotoCoreError, ClientError
    from backend.app.core.config import settings
    from backend.app.infrastructure.storage.ozone_client import get_s3_client, with_prefix

    try:
        get_s3_client().head_object(Bucket=settings.OZONE_BUCKET, Key=with_prefix(rel_key, tenant_id))
        return True
    except ClientError as e:
        if str(e.response.get("Error", {}).get("Code")) in ("404", "NoSuchKey", "NotFound"):
            return False
        raise StorageException(f"Could not check the datalake (Ozone): {e}")
    except BotoCoreError as e:
        raise StorageException(f"Could not reach the datalake (Ozone): {e}")


def _unique_name(clean: str, taken: Set[str], tenant_id: int) -> str:
    stem, _, ext = clean.rpartition(".")
    candidate, n = clean, 1
    while candidate.lower() in taken or _key_exists(f"{mt.UNSTRUCTURED_FOLDER}/{candidate}", tenant_id):
        n += 1
        if n > 1000:
            raise ValidationException(f"Too many files already named '{clean}'.")
        candidate = f"{stem}_{n}.{ext}"
    taken.add(candidate.lower())
    return candidate


def _inspect(kind: str, ext: str, data: bytes) -> Dict[str, Any]:
    """Checks the content really is what the extension says. Raises ValueError if not."""
    if kind == "image":
        from PIL import Image
        try:
            with Image.open(io.BytesIO(data)) as im:
                im.verify()  # catches corrupt / truncated images
            with Image.open(io.BytesIO(data)) as im:
                width, height = im.size
        except Exception:
            raise ValueError("unreadable or corrupt image")
        return {"width": width, "height": height}
    if not mt.looks_like(ext, data[:16]):
        raise ValueError(f"content does not look like a .{ext} file")
    return {}


class UnstructuredService:
    async def upload(self, tenant_id: int, files: List[UploadFile]) -> UnstructuredUploadResponse:
        if tenant_id is None:
            raise ValidationException("A tenant is required to upload files.")
        if not files:
            raise ValidationException("No files were uploaded.")
        if len(files) > mt.MAX_FILES_PER_UPLOAD:
            raise ValidationException(
                f"Too many files ({len(files)}). The limit is {mt.MAX_FILES_PER_UPLOAD} per upload."
            )

        from backend.app.infrastructure.storage.ozone_client import upload_file_to_ozone

        taken: Set[str] = set()
        uploaded: List[UploadedFileInfo] = []
        skipped: List[SkippedFile] = []
        stored_paths: List[str] = []  # full s3:// paths, removed again if storage fails
        total_bytes = 0

        try:
            for f in files:
                original = f.filename or "file"
                ext = mt.extension_of(original)
                kind = mt.kind_for_extension(ext)
                if kind is None:
                    reason = f"Unsupported file type '.{ext}'" if ext else "File has no extension"
                    skipped.append(SkippedFile(file_name=original, reason=reason))
                    continue

                data = await f.read()
                size = len(data)
                if size == 0:
                    skipped.append(SkippedFile(file_name=original, reason="File is empty"))
                    continue
                limit = mt.MEDIA_KINDS[kind]["max_bytes"]
                if size > limit:
                    skipped.append(SkippedFile(file_name=original, reason=f"Larger than {limit // mt.MB} MB for {kind} files"))
                    continue
                if total_bytes + size > mt.MAX_REQUEST_BYTES:
                    skipped.append(SkippedFile(file_name=original, reason=f"Upload limit of {mt.MAX_REQUEST_BYTES // mt.MB} MB per request reached"))
                    continue

                try:
                    extra = await run_in_threadpool(_inspect, kind, ext, data)
                except Exception as e:
                    skipped.append(SkippedFile(file_name=original, reason=f"Not a valid {kind} file ({str(e)[:80]})"))
                    continue

                clean = _clean_name(original)
                stored_name = await run_in_threadpool(_unique_name, clean, taken, tenant_id)
                rel_key = f"{mt.UNSTRUCTURED_FOLDER}/{stored_name}"
                s3_path = await run_in_threadpool(
                    upload_file_to_ozone, data, rel_key, tenant_id, mt.content_type_for(ext)
                )
                stored_paths.append(s3_path)
                total_bytes += size
                uploaded.append(UploadedFileInfo(
                    file_name=original, stored_as=stored_name, path=rel_key, media_type=kind,
                    size_bytes=size, renamed=(stored_name != original),
                    preview_url=f"/api/v1/datasets/media/preview?key={rel_key}",
                    **extra,
                ))
                del data  # free memory before the next file
        except Exception:
            # Storage (or something unexpected) failed: don't leave a half-uploaded batch behind.
            for p in stored_paths:
                try:
                    storage_manager.delete_file(p, tenant_id)
                except Exception as cleanup_err:
                    logger.warning(f"[unstructured] cleanup of {p} failed: {cleanup_err}")
            raise

        if not uploaded:
            raise ValidationException(
                "None of the files could be stored.",
                errors=[s.model_dump() for s in skipped],
            )

        logger.info(f"[unstructured] tenant {tenant_id}: stored {len(uploaded)} file(s), skipped {len(skipped)}")
        return UnstructuredUploadResponse(uploaded=uploaded, skipped=skipped)


unstructured_service = UnstructuredService()