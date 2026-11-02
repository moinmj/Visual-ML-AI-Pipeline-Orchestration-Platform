import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone
from backend.app.infrastructure.storage import ozone_client as oc

def _s3(keys):
    s3 = MagicMock()
    s3.list_objects_v2.return_value = {"Contents": [{"Key": k, "Size": 1, "LastModified": datetime.now(timezone.utc)} for k in keys], "IsTruncated": False}
    return s3

KEYS = ["landing/db/a.csv", "files/x.png", "ml-workflow/t3/datasets/y.csv", "warehouse/z.parquet"]

def test_bucket_scope_off_by_default():
    with patch.object(oc.settings, "OZONE_ALLOW_BUCKET_BROWSE", False, create=True):
        with pytest.raises(ValueError):
            oc.list_ozone_objects(1, scope="bucket")

def test_bucket_scope_lists_everything_with_folders():
    s3 = _s3(KEYS)
    with patch.object(oc.settings, "OZONE_ALLOW_BUCKET_BROWSE", True, create=True), patch.object(oc, "get_s3_client", return_value=s3):
        out = oc.list_ozone_objects(1, scope="bucket", include_folders=True)
    assert s3.list_objects_v2.call_args.kwargs["Prefix"] == ""
    files = [o["key"] for o in out if o["media_type"] != "folder"]
    folders = [o["key"] for o in out if o["media_type"] == "folder"]
    assert sorted(files) == sorted(KEYS)
    assert {"landing/", "landing/db/", "files/", "ml-workflow/t3/datasets/"} <= set(folders)
    assert all(o["relative_path"] == o["key"] for o in out)

def test_tenant_scopes_unchanged():
    s3 = _s3(["ml-workflow/t1/datasets/a.csv"])
    with patch.object(oc.settings, "OZONE_PREFIX", "ml-workflow", create=True), patch.object(oc, "get_s3_client", return_value=s3):
        oc.list_ozone_objects(1, scope="ml_workflow")
    assert s3.list_objects_v2.call_args.kwargs["Prefix"] == "ml-workflow/t1/"