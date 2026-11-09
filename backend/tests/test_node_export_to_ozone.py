"""
Save a node's result table to Ozone (POST .../save-to-ozone) and look it up again
(GET .../saved-output). Ozone is replaced by an in-memory fake S3, so no network is needed.
"""
import io
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest
from botocore.exceptions import ClientError
from httpx import ASGITransport, AsyncClient

from backend.app.core.config import settings
from backend.app.core.exceptions import StorageException, ValidationException
from backend.app.core.security import TokenData, get_current_user
from backend.app.infrastructure.storage import ozone_client
from backend.app.main import app
from backend.app.workflows import node_export

ROLES = ["Tenant Admin", "Data Scientist", "ML Engineer"]
EXEC_ID = "3f9c2d1e-aaaa-bbbb-cccc-123456789abc"
NODE_ID = "node_join"
DF = pd.DataFrame({"customer_id": [1, 2, 3], "name": ["Alice", "Bob", "Zoë"], "amount": [10.5, 20.0, None]})


class FakeS3:
    """Just enough of the boto3 S3 client: put_object / get_object (with Range) over a dict."""

    def __init__(self, honour_range=True, fail_put=False):
        self.objects = {}
        self.honour_range = honour_range
        self.fail_put = fail_put
        self.put_calls = []

    def put_object(self, Bucket, Key, Body, ContentType=None, **_):
        if self.fail_put:
            raise ClientError({"Error": {"Code": "InternalError", "Message": "gateway down"}}, "PutObject")
        self.put_calls.append({"Bucket": Bucket, "Key": Key, "ContentType": ContentType})
        self.objects[(Bucket, Key)] = bytes(Body)

    def get_object(self, Bucket, Key, Range=None, **_):
        if (Bucket, Key) not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey", "Message": "missing"}}, "GetObject")
        data = self.objects[(Bucket, Key)]
        resp = {"LastModified": datetime(2026, 11, 7, 10, 0, 0, tzinfo=timezone.utc)}
        if Range and self.honour_range:
            if not data:
                raise ClientError({"Error": {"Code": "InvalidRange", "Message": "empty"}}, "GetObject")
            end = int(Range.split("-")[1])
            part = data[: end + 1]
            resp.update(Body=io.BytesIO(part), ContentLength=len(part),
                        ContentRange=f"bytes 0-{len(part) - 1}/{len(data)}")
        else:  # gateway ignores Range and sends everything
            resp.update(Body=io.BytesIO(data), ContentLength=len(data))
        return resp


@pytest.fixture
def s3(monkeypatch):
    fake = FakeS3()
    for name, value in {
        "OZONE_ENDPOINT": "http://ozone.test:9878", "OZONE_ACCESS_KEY": "ak", "OZONE_SECRET_KEY": "sk",
        "OZONE_BUCKET": "test-bucket", "OZONE_PREFIX": "ml-workflow", "OZONE_EXPORT_FOLDER": "outputs/workflow_exports",
    }.items():
        monkeypatch.setattr(settings, name, value, raising=False)
    monkeypatch.setattr(ozone_client, "get_s3_client", lambda: fake)
    return fake


def _as(tenant_id, roles=ROLES):
    app.dependency_overrides[get_current_user] = lambda: TokenData(f"user-t{tenant_id}", tenant_id, roles, ["*"])


@pytest.fixture(autouse=True)
def _restore_override():
    saved = app.dependency_overrides.get(get_current_user)
    yield
    if saved is not None:
        app.dependency_overrides[get_current_user] = saved
    else:
        app.dependency_overrides.pop(get_current_user, None)


@pytest.fixture
def node_df():
    with patch("backend.app.workflows.router._get_execution_node_dataframe", new=AsyncMock(return_value=DF)) as m:
        yield m


def _client():
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


SAVE = f"/api/v1/workflows/{EXEC_ID}/nodes/{NODE_ID}/save-to-ozone"
LOOKUP = f"/api/v1/workflows/{EXEC_ID}/nodes/{NODE_ID}/saved-output"
EXPECTED_KEY = f"ml-workflow/t1/outputs/workflow_exports/{EXEC_ID}/{NODE_ID}.csv"


# ----------------------------------------------------------------- HTTP: save

@pytest.mark.asyncio
async def test_save_writes_the_csv_into_the_tenant_folder(s3, node_df):
    _as(1)
    async with _client() as c:
        r = await c.post(SAVE)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "SUCCESS"
    assert body["ozone_path"] == f"s3://test-bucket/{EXPECTED_KEY}"
    assert body["key"] == EXPECTED_KEY
    assert body["folder"] == f"ml-workflow/t1/outputs/workflow_exports/{EXEC_ID}/"
    assert body["file_name"] == f"{NODE_ID}.csv"
    assert body["columns"] == ["customer_id", "name", "amount"]
    assert body["row_count"] == 3 and body["column_count"] == 3
    assert s3.put_calls == [{"Bucket": "test-bucket", "Key": EXPECTED_KEY, "ContentType": "text/csv"}]
    assert body["size_bytes"] == len(s3.objects[("test-bucket", EXPECTED_KEY)])


@pytest.mark.asyncio
async def test_saved_file_is_identical_to_the_export_csv_download(s3, node_df):
    _as(1)
    async with _client() as c:
        download = await c.get(f"/api/v1/workflows/{EXEC_ID}/nodes/{NODE_ID}/export-csv")
        await c.post(SAVE)
    assert download.status_code == 200
    assert s3.objects[("test-bucket", EXPECTED_KEY)] == download.content


@pytest.mark.asyncio
async def test_save_respects_limit(s3, node_df):
    _as(1)
    async with _client() as c:
        r = await c.post(SAVE + "?limit=2")
    assert r.json()["row_count"] == 2
    assert s3.objects[("test-bucket", EXPECTED_KEY)].decode().count("\n") == 3  # header + 2 rows


@pytest.mark.asyncio
async def test_saving_twice_overwrites_one_file(s3, node_df):
    _as(1)
    async with _client() as c:
        await c.post(SAVE)
        await c.post(SAVE)
    assert len(s3.objects) == 1


@pytest.mark.asyncio
async def test_save_404_when_node_has_no_table(s3):
    _as(1)
    with patch("backend.app.workflows.router._get_execution_node_dataframe", new=AsyncMock(return_value=None)):
        async with _client() as c:
            r = await c.post(SAVE)
    assert r.status_code == 404
    assert s3.put_calls == []


@pytest.mark.asyncio
async def test_save_needs_export_permission(s3, node_df):
    _as(1, roles=["Business User"])
    async with _client() as c:
        assert (await c.post(SAVE)).status_code == 403
        assert (await c.get(LOOKUP)).status_code == 403
    assert s3.put_calls == []


@pytest.mark.asyncio
async def test_ozone_failure_is_a_502_and_nothing_is_written_locally(s3, node_df, tmp_path):
    s3.fail_put = True
    _as(1)
    with patch.object(settings, "LOCAL_STORAGE_DIR", str(tmp_path), create=True):
        async with _client() as c:
            r = await c.post(SAVE)
    assert r.status_code == 502
    assert r.json()["error"]["code"] == "STORAGE_ERROR"
    assert list(tmp_path.rglob("*.csv")) == []  # no silent local fallback


@pytest.mark.asyncio
async def test_ozone_not_configured_is_a_502(monkeypatch, node_df):
    monkeypatch.setattr(settings, "OZONE_BUCKET", None, raising=False)
    _as(1)
    async with _client() as c:
        r = await c.post(SAVE)
    assert r.status_code == 502
    assert "not configured" in r.json()["error"]["message"]


# ----------------------------------------------------------- HTTP: look it up

@pytest.mark.asyncio
async def test_lookup_returns_path_and_column_names_after_save(s3, node_df):
    _as(1)
    async with _client() as c:
        saved = (await c.post(SAVE)).json()
        r = await c.get(LOOKUP)
    assert r.status_code == 200, r.text
    got = r.json()
    assert got["ozone_path"] == saved["ozone_path"] == f"s3://test-bucket/{EXPECTED_KEY}"
    assert got["columns"] == ["customer_id", "name", "amount"]
    assert got["column_count"] == 3
    assert got["size_bytes"] == saved["size_bytes"]
    assert got["last_modified"].startswith("2026-11-07")


@pytest.mark.asyncio
async def test_lookup_404_when_never_saved(s3):
    _as(1)
    async with _client() as c:
        r = await c.get(LOOKUP)
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_tenants_cannot_see_each_others_saves(s3, node_df):
    async with _client() as c:
        _as(1)
        await c.post(SAVE)
        _as(2)
        assert (await c.get(LOOKUP)).status_code == 404  # tenant 2 looks in ml-workflow/t2/...
        saved_by_2 = (await c.post(SAVE)).json()
    assert saved_by_2["key"].startswith("ml-workflow/t2/")
    assert len(s3.objects) == 2  # two separate files, nothing overwritten


# ------------------------------------------------------------- unit: key safety

@pytest.mark.parametrize("bad", ["..", "...", "", "   ", "/", "../.."])
def test_unusable_ids_are_rejected(bad):
    with pytest.raises(ValidationException):
        node_export._relative_key(bad, "node")
    with pytest.raises(ValidationException):
        node_export._relative_key("exec", bad)


def test_ids_cannot_escape_the_folder():
    # every character outside [A-Za-z0-9_-] (slashes, dots, spaces, ?) becomes '_'
    key = node_export._relative_key("a/../b", "n ode?.csv")
    assert key == "outputs/workflow_exports/a____b/n_ode__csv.csv"


def test_invalid_export_folder_setting_falls_back(monkeypatch):
    monkeypatch.setattr(settings, "OZONE_EXPORT_FOLDER", "../other", raising=False)
    assert node_export.export_folder() == "outputs/workflow_exports"
    monkeypatch.setattr(settings, "OZONE_EXPORT_FOLDER", "/my/folder/", raising=False)
    assert node_export.export_folder() == "my/folder"


def test_zero_columns_is_rejected(s3):
    with pytest.raises(ValidationException):
        node_export.save_node_csv(pd.DataFrame(), EXEC_ID, NODE_ID, 1)


# ------------------------------------------------------- unit: reading headers

def _put(s3, text: str):
    s3.objects[("test-bucket", EXPECTED_KEY)] = text.encode("utf-8")


def test_header_with_commas_quotes_and_unicode(s3):
    _put(s3, 'id,"Name, full","He said ""hi""",Größe\n1,a,b,c\n')
    assert node_export.get_saved_node_export(EXEC_ID, NODE_ID, 1)["columns"] == ["id", "Name, full", 'He said "hi"', "Größe"]


def test_header_only_file_and_windows_line_endings(s3):
    _put(s3, "a,b,c\r\n")
    assert node_export.get_saved_node_export(EXEC_ID, NODE_ID, 1)["columns"] == ["a", "b", "c"]


def test_non_string_column_names_round_trip(s3):
    df = pd.DataFrame([[1, 2]], columns=[0, 1])
    info = node_export.save_node_csv(df, EXEC_ID, NODE_ID, 1)
    assert info["columns"] == ["0", "1"]
    assert node_export.get_saved_node_export(EXEC_ID, NODE_ID, 1)["columns"] == ["0", "1"]


def test_big_file_only_the_start_is_read(s3, monkeypatch):
    monkeypatch.setattr(node_export, "_HEADER_READ_BYTES", 64)
    _put(s3, "a,b,c\n" + "1,2,3\n" * 1000)
    info = node_export.get_saved_node_export(EXEC_ID, NODE_ID, 1)
    assert info["columns"] == ["a", "b", "c"]
    assert info["size_bytes"] == len(s3.objects[("test-bucket", EXPECTED_KEY)])  # total size, not the 64 bytes read


def test_header_longer_than_the_read_window_falls_back_to_whole_file(s3, monkeypatch):
    monkeypatch.setattr(node_export, "_HEADER_READ_BYTES", 16)
    cols = [f"column_number_{i}" for i in range(10)]
    _put(s3, ",".join(cols) + "\n1,2,3,4,5,6,7,8,9,10\n")
    assert node_export.get_saved_node_export(EXEC_ID, NODE_ID, 1)["columns"] == cols


def test_gateway_that_ignores_range_still_works(s3, monkeypatch):
    s3.honour_range = False
    monkeypatch.setattr(node_export, "_HEADER_READ_BYTES", 64)
    _put(s3, "a,b,c\n" + "1,2,3\n" * 1000)
    info = node_export.get_saved_node_export(EXEC_ID, NODE_ID, 1)
    assert info["columns"] == ["a", "b", "c"]
    assert info["size_bytes"] == len(s3.objects[("test-bucket", EXPECTED_KEY)])


def test_empty_object_gives_no_columns(s3):
    _put(s3, "")
    info = node_export.get_saved_node_export(EXEC_ID, NODE_ID, 1)
    assert info["columns"] == [] and info["size_bytes"] == 0


def test_read_error_other_than_missing_is_a_storage_error(s3):
    s3.get_object = MagicMock(side_effect=ClientError({"Error": {"Code": "InternalError", "Message": "boom"}}, "GetObject"))
    with pytest.raises(StorageException):
        node_export.get_saved_node_export(EXEC_ID, NODE_ID, 1)


def test_head_read_refuses_keys_outside_the_tenant(s3):
    with pytest.raises(PermissionError):
        ozone_client.read_ozone_object_head("ml-workflow/t2/outputs/x.csv", 1)