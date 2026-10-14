from typing import Annotated, List, Literal, Optional
from backend.app.datasets.unstructured import UnstructuredUploadResponse, unstructured_service
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, Query, Request, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.security import TokenData, get_current_user
from backend.app.infrastructure.database.session import get_db
from backend.app.datasets.schemas import (
    DatasetResponse,
    DatasetPreviewResponse,
    DatasetProfileResponse,
    OzoneObjectResponse,
)
from backend.app.datasets.service import dataset_service

# Every route here requires a valid Bearer token. The tenant is ALWAYS taken from
# the token (user.tenant_id) - never from the path, query or body - so a caller can
# only ever list, preview, register or delete their own tenant's data.
router = APIRouter(prefix="/datasets", tags=["Datasets"])

CurrentUser = Annotated[TokenData, Depends(get_current_user)]


@router.post("/upload", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
async def upload_structured_dataset(
    user: CurrentUser,
    file: UploadFile = File(..., description="ONE CSV, XLSX, JSON or Parquet file"),
    name: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
):
    """
    **Structured upload** — one CSV / XLSX / JSON / Parquet file.

    The file is stored in `ml-workflow/t<tenant_id>/datasets/` and a dataset record is created.
    """
    return await dataset_service.upload_dataset(
        db, tenant_id=user.tenant_id, file=file, name=name, description=description
    )


@router.post(
    "/upload/unstructured",
    response_model=UnstructuredUploadResponse,
    status_code=status.HTTP_201_CREATED,
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["files"],
                        "properties": {
                            "files": {
                                "type": "array",
                                "items": {"type": "string", "format": "binary"},
                                "description": "One or more image, audio or video files (PNG, JPG, MP4, MP3 …)",
                            }
                        },
                    }
                }
            },
        }
    },
)
async def upload_unstructured_files(
    request: Request,
    user: CurrentUser,
):
    """
    **Unstructured upload** — images, audio or video files.

    Select one or more files using the **files** picker above.
    Files are stored flat in `ml-workflow/t<tenant_id>/unstructured/`. No dataset record is created.
    Each response item includes a `preview_url` you can use to render the uploaded image in the browser.
    """
    form = await request.form()
    raw = form.getlist("files")
    real_files = [f for f in raw if hasattr(f, "read") and getattr(f, "filename", None)]
    if not real_files:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No files were provided. Use the 'files' field to attach one or more image/audio/video files.",
        )
    return await unstructured_service.upload(tenant_id=user.tenant_id, files=real_files)


@router.get("/ozone/browse", response_model=List[OzoneObjectResponse])
async def browse_ozone_datasets(
    user: CurrentUser,
    scope: Literal["all_tenant_data", "ml_workflow"] = Query(
        "ml_workflow",
        description="Option 1: 'all_tenant_data' (get all data of that tenant in Ozone). "
                    "Option 2: 'ml_workflow' (get data from ml-workflow/t<tenant_id>/).",
    ),
    folder: Literal["all", "datasets", "unstructured"] = Query(
        "all",
        description="When scope='ml_workflow': 'datasets' for structured data, "
                    "'unstructured' for unstructured media (images/audio/video), or 'all'.",
    ),
    media_type: Optional[str] = Query(
        None,
        description="Optional filter by media type: 'image', 'audio', 'video', 'tabular', or omit for all",
    ),
    prefix: str = Query("", description="Optional key prefix to filter by, relative to the selected scope/folder"),
):
    """
    Lists files sitting in the caller's tenant folder in Ozone.
    Returns objects matching the DatasetResponse JSON shape so the frontend can consume them directly.
    """
    return dataset_service.browse_ozone(
        tenant_id=user.tenant_id,
        scope=scope,
        folder=folder,
        media_type=media_type,
        prefix=prefix,
    )


@router.get("/media/preview")
async def preview_media_file(
    request: Request,
    key: str = Query(..., description="Key or s3:// path of the media file to preview"),
    token: Optional[str] = Query(None, description="Bearer token for direct browser <img> rendering"),
):
    """
    Streams raw media file bytes (e.g. image) from Ozone for previewing in the browser.
    Accepts Bearer auth header or ?token=<jwt> query parameter.
    Validates tenant scoping to ensure strict tenant isolation.
    """
    from backend.app.core.config import settings
    from backend.app.core import media_types as mt
    from backend.app.core.security import _decode_token
    from backend.app.infrastructure.storage.ozone_client import (
        is_in_tenant_prefix, with_prefix, get_file_bytes_from_ozone
    )

    auth_header = request.headers.get("Authorization")
    raw_token = None
    if auth_header and auth_header.startswith("Bearer "):
        raw_token = auth_header.split(" ", 1)[1].strip()
    elif token:
        raw_token = token.strip()

    if not raw_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials were not provided",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        user = _decode_token(raw_token)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid credentials: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )

    import re
    clean_key = str(key).strip().lstrip("/")
    if clean_key.startswith("s3://"):
        parts = clean_key.split("/", 3)
        if len(parts) < 4 or parts[2] != settings.OZONE_BUCKET:
            raise HTTPException(status_code=400, detail="Object is not in this platform's Ozone bucket")
        full_key = parts[3]
    elif clean_key.startswith("ml-workflow/") or re.match(r"^t\d+/", clean_key):
        full_key = clean_key
    else:
        full_key = with_prefix(clean_key, user.tenant_id)

    if not is_in_tenant_prefix(full_key, user.tenant_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: file is outside this tenant's folder"
        )

    try:
        data = get_file_bytes_from_ozone(full_key, tenant_id=user.tenant_id)
    except PermissionError:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"File not found: {clean_key} ({e})")

    ext = mt.extension_of(full_key)
    content_type = mt.content_type_for(ext)
    return Response(
        content=data,
        media_type=content_type,
        headers={
            "Cache-Control": "public, max-age=86400",
            "Content-Disposition": f'inline; filename="{full_key.rsplit("/", 1)[-1]}"',
        },
    )


@router.post("/ozone/register", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
async def register_ozone_dataset(
    user: CurrentUser,
    ozone_key: str = Form(..., description="Object key returned by /ozone/browse"),
    name: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db)
):
    """
    Registers a file that's already in the caller's Ozone tenant folder as a dataset,
    without re-uploading it.
    """
    return await dataset_service.register_from_ozone(
        db, tenant_id=user.tenant_id, ozone_key=ozone_key, name=name, description=description
    )


@router.get("/", response_model=List[DatasetResponse])
async def list_datasets(
    user: CurrentUser,
    response: Response,
    search: Optional[str] = Query(
        None, max_length=200,
        description="Find datasets by ID, name, file name or description (case-insensitive, partial match). "
                    "Several words must all match, e.g. 'walmart 2024'.",
    ),
    file_format: Optional[str] = Query(None, description="Only this format: csv, xlsx, json or parquet"),
    sort: str = Query("newest", pattern="^(newest|oldest|name|name_desc|largest|rows)$",
                      description="newest (default), oldest, name, name_desc, largest, rows"),
    limit: Optional[int] = Query(None, ge=1, le=500, description="Max results to return (omit for all)"),
    offset: int = Query(0, ge=0, description="Skip this many results (for paging)"),
    db: AsyncSession = Depends(get_db),
):
    """
    List / search the caller's tenant's datasets (only those stored in its Ozone folder).

    Without parameters this returns every dataset, newest first (unchanged behaviour).
    The total number of matches (before limit/offset) is in the X-Total-Count header.
    """
    datasets, total = await dataset_service.list_datasets(
        db, tenant_id=user.tenant_id, search=search, file_format=file_format,
        sort=sort, limit=limit, offset=offset,
    )
    response.headers["X-Total-Count"] = str(total)
    return datasets


@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(dataset_id: str, user: CurrentUser, db: AsyncSession = Depends(get_db)):
    """
    Get dataset metadata by ID (404 if it belongs to another tenant).
    """
    return await dataset_service.get_dataset(db, dataset_id, user.tenant_id)


@router.get("/{dataset_id}/preview", response_model=DatasetPreviewResponse)
async def get_dataset_preview(
    dataset_id: str,
    user: CurrentUser,
    limit: int = Query(10, ge=1, le=1000, description="Number of rows to return per page"),
    offset: int = Query(0, ge=0, description="Row offset index for pagination"),
    page: Optional[int] = Query(None, ge=1, description="Optional 1-indexed page number (e.g. page=2 with limit=100 sets offset=100)"),
    db: AsyncSession = Depends(get_db)
):
    """
    Preview rows of the dataset with full pagination support (limit, offset, and page).
    """
    actual_offset = offset
    if page is not None and offset == 0:
        actual_offset = (page - 1) * limit

    return await dataset_service.get_preview(
        db, dataset_id, tenant_id=user.tenant_id, limit=limit, offset=actual_offset
    )


@router.get("/{dataset_id}/profile", response_model=DatasetProfileResponse)
async def get_dataset_profile(dataset_id: str, user: CurrentUser, db: AsyncSession = Depends(get_db)):
    """
    Retrieve the full statistical and quality profile of a dataset.
    """
    dataset = await dataset_service.get_dataset(db, dataset_id, user.tenant_id)
    return {
        "id": dataset.id,
        "name": dataset.name,
        "profile": dataset.profile or {}
    }


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(dataset_id: str, user: CurrentUser, db: AsyncSession = Depends(get_db)):
    """
    Delete a dataset and its physical file from storage.
    """
    await dataset_service.delete_dataset(db, dataset_id, user.tenant_id)
    return None