from typing import Optional

import boto3
from botocore.client import Config
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


def tenant_prefix(tenant_id: int) -> str:
    """Folder for one tenant, e.g. 'ml-workflow/t1/'. The trailing slash matters: it keeps t1 from matching t10."""
    return f"{get_prefix()}t{int(tenant_id)}/"


def _clean_relative_key(key: str) -> str:
    key = (key or "").strip().lstrip("/")
    if any(part == ".." for part in key.split("/")):
        raise ValueError("Object keys must not contain '..'")
    return key


def with_prefix(key: str, tenant_id: int) -> str:
    """Puts a key under the tenant's folder. Idempotent: a key already inside it is unchanged."""
    key = _clean_relative_key(key)
    prefix = tenant_prefix(tenant_id)
    return key if key.startswith(prefix) else prefix + key


def is_in_tenant_prefix(key: str, tenant_id: int) -> bool:
    """True only if the full object key lives inside THIS tenant's folder."""
    key = key or ""
    return key.startswith(tenant_prefix(tenant_id)) and ".." not in key.split("/")


def get_s3_client():
    """
    Returns boto3 S3 client configured for the Apache Ozone S3 gateway.
    """
    return boto3.client(
        "s3",
        endpoint_url=settings.OZONE_ENDPOINT,
        aws_access_key_id=settings.OZONE_ACCESS_KEY,
        aws_secret_access_key=settings.OZONE_SECRET_KEY,
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
            connect_timeout=5,
            read_timeout=30,
            retries={"max_attempts": 2},
        ),
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
    s3.put_object(Bucket=bucket, Key=full_key, Body=file_bytes, ContentType=content_type)
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


def list_ozone_objects(tenant_id: int, prefix: str = "", max_keys: int = 1000):
    """
    Lists objects inside the CALLER'S tenant folder only (the datalake browse option),
    so a dataset can be registered against one without re-uploading it.
    `prefix` narrows further within that folder ('' = the whole tenant folder).
    Returns a list of {"key": <full key>, "size_bytes": int, "last_modified": iso-str}.
    The returned key is the FULL key, which is what /ozone/register expects.
    """
    bucket = settings.OZONE_BUCKET
    full_prefix = with_prefix(prefix, tenant_id) if prefix else tenant_prefix(tenant_id)
    s3 = get_s3_client()
    items = []
    continuation_token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": full_prefix, "MaxKeys": min(max_keys - len(items), 1000)}
        if continuation_token:
            kwargs["ContinuationToken"] = continuation_token
        response = s3.list_objects_v2(**kwargs)
        for obj in response.get("Contents", []):
            if obj["Key"].endswith("/"):
                continue  # skip 'folder' placeholder objects
            items.append({
                "key": obj["Key"],
                "size_bytes": obj["Size"],
                "last_modified": obj["LastModified"].isoformat() if obj.get("LastModified") else None,
            })
        if not response.get("IsTruncated") or len(items) >= max_keys:
            break
        continuation_token = response.get("NextContinuationToken")
    return items