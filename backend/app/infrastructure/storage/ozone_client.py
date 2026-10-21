from typing import Optional

import boto3
from botocore.client import Config
from botocore.exceptions import BotoCoreError, ClientError
from backend.app.core.config import settings
from backend.app.core.logging import logger


# ---------------------------------------------------------------------------
# Folder (key prefix) handling
#
# Ozone/S3 has no real folders: "ml-workflow/" is just the start of every key.
# Everything this platform writes lives under
#
#       <OZONE_PREFIX>/t<tenant_id>/...        e.g. ml-workflow/t1/datasets/<id>_sales.csv
#
# so each tenant gets its own folder inside the platform folder. Browse, upload,
# read and delete are all scoped to the CALLER'S tenant folder, which means the
# rest of the bucket (landing/, warehouse/, files/ ...) and every other tenant's
# folder are never listed, registered, read or deleted by this app.
# ---------------------------------------------------------------------------

def get_prefix() -> str:
    """Root folder inside the bucket, normalised to 'name/' (or '' = whole bucket)."""
    raw = (getattr(settings, "OZONE_PREFIX", "") or "").strip().strip("/")
    return f"{raw}/" if raw else ""


def tenant_ml_workflow_prefix(tenant_id: int) -> str:
    """Folder for platform workflow files: 'ml-workflow/t<tenant_id>/'."""
    return f"{get_prefix()}t{int(tenant_id)}/"


def tenant_root_prefix(tenant_id: int) -> str:
    """Root folder for all tenant data in Ozone: 't<tenant_id>/'."""
    return f"t{int(tenant_id)}/"


def tenant_prefix(tenant_id: int) -> str:
    """Folder for one tenant inside ml-workflow (default, backwards compatible)."""
    return tenant_ml_workflow_prefix(tenant_id)


def _clean_relative_key(key: str) -> str:
    key = (key or "").strip().lstrip("/")
    if any(part == ".." for part in key.split("/")):
        raise ValueError("Object keys must not contain '..'")
    return key


def with_prefix(key: str, tenant_id: int) -> str:
    """Puts a key under the tenant's workflow folder. Idempotent: a key already inside it is unchanged."""
    key = _clean_relative_key(key)
    prefix = tenant_prefix(tenant_id)
    return key if key.startswith(prefix) else prefix + key


def is_in_tenant_prefix(key: str, tenant_id: int, scope: str = "any") -> bool:
    """
    True only if the full object key lives inside THIS tenant's folders.
    If scope is 'ml_workflow': checks 'ml-workflow/t<tenant_id>/'.
    If scope is 'all_tenant_data': checks 't<tenant_id>/'.
    If scope is 'any': checks either 'ml-workflow/t<tenant_id>/' or 't<tenant_id>/'.
    """
    key = (key or "").strip()
    if not key or any(part == ".." for part in key.split("/")):
        return False
    ml_prefix = tenant_ml_workflow_prefix(tenant_id)
    root_prefix = tenant_root_prefix(tenant_id)
    if scope == "ml_workflow":
        return key.startswith(ml_prefix)
    if scope == "all_tenant_data":
        return key.startswith(root_prefix)
    return key.startswith(ml_prefix) or key.startswith(root_prefix)


def _client_config() -> Config:
    """
    boto3/botocore >= 1.36 adds CRC32 integrity checksums (aws-chunked uploads)
    to every PutObject by default. The Apache Ozone S3 gateway does not support
    that and answers 500 Internal Server Error, so checksums are only sent when
    an operation strictly requires them (the pre-1.36 behaviour).
    """
    base = dict(
        signature_version="s3v4",
        s3={"addressing_style": "path"},
        connect_timeout=3,
        read_timeout=8,
        retries={"max_attempts": 1},
    )
    try:
        return Config(
            **base,
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        )
    except TypeError:
        # Older botocore (< 1.36) doesn't know these options and doesn't send the checksums anyway.
        return Config(**base)


def get_s3_client():
    """
    Returns boto3 S3 client configured for the Apache Ozone S3 gateway.
    """
    return boto3.client(
        "s3",
        endpoint_url=settings.OZONE_ENDPOINT,
        aws_access_key_id=settings.OZONE_ACCESS_KEY,
        aws_secret_access_key=settings.OZONE_SECRET_KEY,
        config=_client_config(),
        region_name=settings.OZONE_REGION,
    )


def upload_file_to_ozone(file_bytes: bytes, ozone_key: str, tenant_id: int, content_type: str = "application/octet-stream") -> str:
    """
    Uploads file bytes to the Ozone bucket under the tenant's folder
    (e.g. 'datasets/x.csv' for tenant 1 is stored as 'ml-workflow/t1/datasets/x.csv').
    Returns the full s3://bucket/<full-key> path, which is what gets saved in the DB.
    Raises on failure (callers should not silently store a dataset row for a failed upload).
    """
    bucket = settings.OZONE_BUCKET
    full_key = with_prefix(ozone_key, tenant_id)
    s3 = get_s3_client()
    try:
        s3.put_object(Bucket=bucket, Key=full_key, Body=file_bytes, ContentType=content_type)
    except (ClientError, BotoCoreError) as e:
        # Returned to the caller as a clear 502 instead of a bare 500.
        from backend.app.core.exceptions import StorageException
        logger.error(f"[Ozone] Upload of '{full_key}' to bucket '{bucket}' failed: {e}")
        raise StorageException(
            f"Could not store the file in the datalake (Ozone): {e}",
            details={"bucket": bucket, "key": full_key},
        )
    ozone_path = f"s3://{bucket}/{full_key}"
    logger.info(f"[Ozone] Uploaded object to '{ozone_path}'")
    return ozone_path


def get_file_bytes_from_ozone(ozone_key: str, tenant_id: Optional[int] = None) -> bytes:
    """
    Retrieves full file bytes for an Ozone object.
    Takes the FULL key (as parsed from a stored s3:// path).
    When tenant_id is given, the key must be inside that tenant's folder or
    PermissionError is raised - API code paths always pass it.
    """
    if tenant_id is not None and not is_in_tenant_prefix(ozone_key, tenant_id):
        raise PermissionError(f"Key '{ozone_key}' is outside this tenant's folder")
    bucket = settings.OZONE_BUCKET
    s3 = get_s3_client()
    response = s3.get_object(Bucket=bucket, Key=ozone_key)
    data = response["Body"].read()
    logger.info(f"[Ozone] Read {len(data)} bytes for key='{ozone_key}'")
    return data


def delete_file_from_ozone(ozone_key: str, tenant_id: Optional[int] = None) -> bool:
    """
    Deletes an object key (FULL key) from the Ozone bucket.
    When tenant_id is given, the key must be inside that tenant's folder.
    """
    if tenant_id is not None and not is_in_tenant_prefix(ozone_key, tenant_id):
        logger.warning(f"[Ozone] Refused to delete '{ozone_key}': outside tenant {tenant_id}'s folder")
        return False
    bucket = settings.OZONE_BUCKET
    try:
        s3 = get_s3_client()
        s3.delete_object(Bucket=bucket, Key=ozone_key)
        logger.info(f"[Ozone] Deleted key '{ozone_key}' from bucket '{bucket}'")
        return True
    except Exception as e:
        logger.warning(f"[Ozone] Could not delete key '{ozone_key}': {e}")
        return False


MAX_SCANNED_KEYS = 50000  # safety net: one browse call never walks more keys than this


def _parent_folders(key: str, base_prefix: str) -> list:
    """
    The folders a key sits in, below base_prefix (the tenant folder itself is not returned).
    'ml-workflow/t1/outputs/ocr/x.jpg' with base 'ml-workflow/t1/' -> ['ml-workflow/t1/outputs/', 'ml-workflow/t1/outputs/ocr/']
    A placeholder key that ends with '/' is itself a folder.
    """
    rel = key[len(base_prefix):] if key.startswith(base_prefix) else ""
    parts = [p for p in (rel.rstrip("/").split("/") if key.endswith("/") else rel.split("/")[:-1]) if p]
    return [base_prefix + "/".join(parts[:i]) + "/" for i in range(1, len(parts) + 1)]


def list_ozone_objects(
    tenant_id: int,
    prefix: str = "",
    scope: str = "ml_workflow",
    folder: str = "all",
    media_type: Optional[str] = None,
    max_keys: int = 1000,
    include_folders: bool = False,
    meta: Optional[dict] = None,
):
    """
    Lists objects inside the caller's tenant space in Ozone.
    - scope="all_tenant_data": the WHOLE tenant space = the platform folder 'ml-workflow/t<tenant_id>/'
      (datasets, unstructured, outputs ...) PLUS the bucket-level folder 't<tenant_id>/'. 'relative_path' is the full key.
    - scope="ml_workflow": lists under 'ml-workflow/t<tenant_id>/', with optional
      folder='datasets' (structured data) or folder='unstructured' (unstructured media).
    - media_type: optional filter by 'image', 'audio', 'video', 'tabular', 'other', 'folder' or 'all'.
    - include_folders: also return the folders (media_type='folder'), including empty ones.
    - max_keys: the most files returned. meta['truncated'] is set to True when there were more.
    Other tenants and the rest of the bucket are never listed (the prefixes end with '/').
    """
    from backend.app.core import media_types as mt

    bucket = settings.OZONE_BUCKET
    clean_prefix = (prefix or "").strip().lstrip("/")
    wanted = (media_type or "").lower().strip()
    want_files = wanted != "folder"
    want_folders = wanted == "folder" or (include_folders and wanted in ("", "all"))

    if scope == "all_tenant_data":
        roots = list(dict.fromkeys([tenant_ml_workflow_prefix(tenant_id), tenant_root_prefix(tenant_id)]))
        scans = [(root, f"{root}{clean_prefix}") for root in roots]
    else:
        base_prefix = tenant_ml_workflow_prefix(tenant_id)
        folder_part = {"datasets": "datasets/", "unstructured": "unstructured/"}.get(folder, "")
        scans = [(base_prefix, f"{base_prefix}{folder_part}{clean_prefix}")]

    s3 = get_s3_client()
    items, folders = [], {}
    scanned, truncated = 0, False
    for base_prefix, full_prefix in scans:
        token = None
        while not truncated:
            kwargs = {"Bucket": bucket, "Prefix": full_prefix, "MaxKeys": 1000}
            if token:
                kwargs["ContinuationToken"] = token
            response = s3.list_objects_v2(**kwargs)
            for obj in response.get("Contents", []):
                key = obj["Key"]
                scanned += 1
                if want_folders:
                    for f in _parent_folders(key, base_prefix):
                        if f.startswith(full_prefix):
                            folders.setdefault(f, base_prefix)
                if key.endswith("/") or not want_files:
                    continue  # 'folder' placeholder objects are folders, not files

                ext = mt.extension_of(key)
                kind = mt.kind_for_extension(ext)
                if ext in ("csv", "tsv", "xlsx", "xls", "json", "parquet"):
                    detected_kind = "tabular"
                elif kind:
                    detected_kind = kind
                else:
                    detected_kind = "other"

                if wanted not in ("", "all") and detected_kind != wanted:
                    continue

                if len(items) >= max_keys:
                    truncated = True  # a matching file exists beyond the limit
                    break
                items.append({
                    "key": key,
                    "relative_path": key if scope == "all_tenant_data" else key[len(base_prefix):],
                    "file_name": key.rsplit("/", 1)[-1],
                    "file_format": ext or "unknown",
                    "size_bytes": obj["Size"],
                    "media_type": detected_kind,
                    "last_modified": obj["LastModified"].isoformat() if obj.get("LastModified") else None,
                })
            if truncated:
                break
            if scanned >= MAX_SCANNED_KEYS:
                truncated = True
                break
            if not response.get("IsTruncated"):
                break
            token = response.get("NextContinuationToken")
        if truncated:
            break

    if meta is not None:
        meta["truncated"] = truncated

    folder_items = [{
        "key": fkey,
        "relative_path": fkey if scope == "all_tenant_data" else fkey[len(folders[fkey]):],
        "file_name": fkey.rstrip("/").rsplit("/", 1)[-1],
        "file_format": "folder",
        "size_bytes": 0,
        "media_type": "folder",
        "last_modified": None,
    } for fkey in sorted(folders)]
    return folder_items + items