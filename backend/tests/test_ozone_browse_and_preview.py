import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from backend.app.infrastructure.storage import ozone_client
from backend.app.datasets.service import dataset_service
from backend.app.datasets.schemas import OzoneObjectResponse, DatasetResponse
from backend.app.core.security import create_access_token


def test_ozone_prefixes():
    # 1. ml_workflow prefix
    assert ozone_client.tenant_ml_workflow_prefix(1) == "ml-workflow/t1/"
    # 2. tenant root prefix
    assert ozone_client.tenant_root_prefix(1) == "t1/"
    # 3. is_in_tenant_prefix
    assert ozone_client.is_in_tenant_prefix("ml-workflow/t1/datasets/data.csv", 1) is True
    assert ozone_client.is_in_tenant_prefix("t1/raw/img.jpg", 1) is True
    # other tenant is blocked
    assert ozone_client.is_in_tenant_prefix("ml-workflow/t2/datasets/data.csv", 1) is False
    assert ozone_client.is_in_tenant_prefix("t2/raw/img.jpg", 1) is False
    # path traversal is blocked
    assert ozone_client.is_in_tenant_prefix("ml-workflow/t1/../t2/secret.csv", 1) is False


def test_browse_ozone_dataset_response_shape():
    mock_s3 = MagicMock()
    mock_s3.list_objects_v2.return_value = {
        "Contents": [
            {
                "Key": "ml-workflow/t1/unstructured/test_cat.png",
                "Size": 204800,
                "LastModified": datetime(2026, 10, 14, 5, 0, 0, tzinfo=timezone.utc),
            },
            {
                "Key": "ml-workflow/t1/datasets/sales.csv",
                "Size": 1024,
                "LastModified": datetime(2026, 10, 14, 5, 0, 0, tzinfo=timezone.utc),
            },
        ],
        "IsTruncated": False,
    }

    with patch.object(ozone_client, "get_s3_client", return_value=mock_s3):
        # 1. Browse unstructured
        items = dataset_service.browse_ozone(tenant_id=1, scope="ml_workflow", folder="unstructured")
        assert len(items) == 2
        cat = next(i for i in items if i["file_name"] == "test_cat.png")

        # Must validate against OzoneObjectResponse and DatasetResponse!
        validated = OzoneObjectResponse(**cat)
        assert isinstance(validated, DatasetResponse)
        assert validated.id == "ml-workflow/t1/unstructured/test_cat.png"
        assert validated.name == "test_cat"
        assert validated.file_format == "png"
        assert validated.file_size_bytes == 204800
        assert validated.storage_path.endswith("ml-workflow/t1/unstructured/test_cat.png")
        assert validated.row_count == 0
        assert validated.column_count == 0
        assert validated.quality_score == 100.0
        assert validated.preview_url == "/api/v1/datasets/media/preview?key=ml-workflow/t1/unstructured/test_cat.png"
        assert validated.media_type == "image"

        # 2. Scope: all_tenant_data
        dataset_service.browse_ozone(tenant_id=1, scope="all_tenant_data")
        call_kwargs = mock_s3.list_objects_v2.call_args[1]
        assert call_kwargs["Prefix"] == "t1/"


@pytest.mark.asyncio
async def test_media_preview_route_security():
    from httpx import AsyncClient, ASGITransport
    from backend.app.main import app

    token_t1 = create_access_token(sub="user1", tenant_id=1)
    token_t2 = create_access_token(sub="user2", tenant_id=2)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Mock reading bytes from Ozone
        fake_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDRfake_png_data"
        with patch.object(ozone_client, "get_file_bytes_from_ozone", return_value=fake_bytes):
            # 1. Tenant 1 requests their own image -> 200 OK
            res = await client.get(
                "/api/v1/datasets/media/preview?key=unstructured/test_cat.png",
                headers={"Authorization": f"Bearer {token_t1}"},
            )
            assert res.status_code == 200
            assert res.headers["content-type"] == "image/png"
            assert res.content == fake_bytes

            # 2. Tenant 2 requests Tenant 1's image -> 403 Forbidden
            res_denied = await client.get(
                "/api/v1/datasets/media/preview?key=ml-workflow/t1/unstructured/test_cat.png",
                headers={"Authorization": f"Bearer {token_t2}"},
            )
            assert res_denied.status_code == 403

            # 3. Direct browser tag with ?token= query parameter -> 200 OK
            res_query = await client.get(
                f"/api/v1/datasets/media/preview?key=unstructured/test_cat.png&token={token_t1}"
            )
            assert res_query.status_code == 200
            assert res_query.content == fake_bytes


def test_object_detection_preview_urls():
    import pandas as pd
    from backend.app.recipes.vision.object_detection import ObjectDetectionRecipe

    recipe = ObjectDetectionRecipe()
    df = pd.DataFrame({"img": ["unstructured/cat.png", "t1/images/dog.jpg"]})

    # Mock detection and ozone read
    fake_img_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb"
    with patch("backend.app.recipes.vision.object_detection._read_image_bytes") as mock_read, \
         patch("backend.app.recipes.vision.object_detection._load_model") as mock_model, \
         patch("backend.app.infrastructure.storage.ozone_client.upload_file_to_ozone") as mock_upload:

        from PIL import Image
        import io
        img = Image.new("RGB", (100, 100), color="red")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        mock_read.return_value = buf.getvalue()

        # Mock YOLO model
        import numpy as np
        model_instance = MagicMock()
        mock_res = MagicMock()
        mock_res.names = {0: "cat"}
        mock_res.plot.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        box = MagicMock()
        box.xyxy = [[10.0, 10.0, 50.0, 50.0]]
        box.cls = [0]
        box.conf = [0.95]
        mock_res.boxes = [box]
        model_instance.predict.return_value = [mock_res]
        mock_model.return_value = model_instance

        mock_upload.return_value = "s3://bucket/ml-workflow/t1/outputs/object_detection/0_cat_detected.jpg"

        out = recipe.execute(
            {"dataframe": df},
            {
                "image_column": "img",
                "tenant_id": 1,
                "save_annotated_images": True,
            }
        )

        out_df = out["dataframe"]
        # Both original image preview URL and annotated preview URL are present
        assert "image_preview_url" in out_df.columns
        assert out_df["image_preview_url"].iloc[0] == "/api/v1/datasets/media/preview?key=unstructured/cat.png"
        assert out_df["image_preview_url"].iloc[1] == "/api/v1/datasets/media/preview?key=t1/images/dog.jpg"

        assert "annotated_preview_url" in out_df.columns
        assert "/api/v1/datasets/media/preview?key=" in out_df["annotated_preview_url"].iloc[0]

        # Check preview_items in metrics
        assert "preview_items" in out["metrics"]
        assert len(out["metrics"]["preview_items"]) == 2
        first_item = out["metrics"]["preview_items"][0]
        assert first_item["original_path"] == "unstructured/cat.png"
        assert first_item["original_preview_url"] == "/api/v1/datasets/media/preview?key=unstructured/cat.png"
        assert first_item["annotated_preview_url"] is not None

