import boto3
from botocore.client import Config
from backend.app.core.config import settings
from backend.app.core.logging import logger


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


def upload_file_to_ozone(file_bytes: bytes, ozone_key: str, content_type: str = "application/octet-stream") -> str:
    """
    Uploads file bytes to the Ozone bucket. Returns the s3://bucket/key path.
    Raises on failure (callers should not silently store a dataset row for a failed upload).
    """
    bucket = settings.OZONE_BUCKET
    s3 = get_s3_client()
    s3.put_object(Bucket=bucket, Key=ozone_key, Body=file_bytes, ContentType=content_type)
    ozone_path = f"s3://{bucket}/{ozone_key}"
    logger.info(f"[Ozone] Uploaded object to '{ozone_path}'")
    return ozone_path


def get_file_bytes_from_ozone(ozone_key: str) -> bytes:
    """
    Retrieves full file bytes for an Ozone object.
    """
    bucket = settings.OZONE_BUCKET
    s3 = get_s3_client()
    response = s3.get_object(Bucket=bucket, Key=ozone_key)
    data = response["Body"].read()
    logger.info(f"[Ozone] Read {len(data)} bytes for key='{ozone_key}'")
    return data


def delete_file_from_ozone(ozone_key: str) -> bool:
    """
    Deletes an object key from the Ozone bucket.
    """
    bucket = settings.OZONE_BUCKET
    try:
        s3 = get_s3_client()
        s3.delete_object(Bucket=bucket, Key=ozone_key)
        logger.info(f"[Ozone] Deleted key '{ozone_key}' from bucket '{bucket}'")
        return True
    except Exception as e:
        logger.warning(f"[Ozone] Could not delete key '{ozone_key}': {e}")
        return False


def list_ozone_objects(prefix: str = "", max_keys: int = 1000):
    """
    Lists existing objects already in the Ozone bucket (the datalake browse option),
    so a dataset can be registered against one without re-uploading it.
    Returns a list of {"key": str, "size_bytes": int, "last_modified": iso-str}.
    """
    bucket = settings.OZONE_BUCKET
    s3 = get_s3_client()
    items = []
    continuation_token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": min(max_keys - len(items), 1000)}
        if continuation_token:
            kwargs["ContinuationToken"] = continuation_token
        response = s3.list_objects_v2(**kwargs)
        for obj in response.get("Contents", []):
            items.append({
                "key": obj["Key"],
                "size_bytes": obj["Size"],
                "last_modified": obj["LastModified"].isoformat() if obj.get("LastModified") else None,
            })
        if not response.get("IsTruncated") or len(items) >= max_keys:
            break
        continuation_token = response.get("NextContinuationToken")
    return items