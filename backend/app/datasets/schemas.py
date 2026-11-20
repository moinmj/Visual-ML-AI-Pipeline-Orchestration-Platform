from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, ForwardRef, Literal
from datetime import datetime


from backend.app.workflows.schemas import UTCDateTime


class DatasetBase(BaseModel):
    name: str = Field(..., description="User-friendly name of the dataset")
    description: Optional[str] = Field(None, description="Optional dataset description")


class DatasetCreate(DatasetBase):
    pass


class DatasetResponse(DatasetBase):
    id: str
    file_name: str
    file_format: str
    file_size_bytes: int
    storage_path: str
    row_count: int
    column_count: int
    quality_score: float
    created_at: UTCDateTime
    updated_at: UTCDateTime

    model_config = {
        "from_attributes": True
    }


class OzoneObjectResponse(DatasetResponse):
    media_type: Optional[str] = Field(None, description="Inferred media category: image, audio, video, tabular, other, folder")
    preview_url: Optional[str] = Field(None, description="Direct URL to preview/stream image or media file")
    key: Optional[str] = Field(None, description="Full Ozone key")
    relative_path: Optional[str] = Field(None, description="Path relative to tenant root or ml-workflow root")
    scope: Optional[str] = Field(None, description="all_tenant_data or ml_workflow")
    dataset_id: Optional[str] = Field(
        None,
        description="UUID of the dataset record for this file if it is already registered, otherwise null. "
                    "Use this (not 'id', which is the storage key) in every API that takes a dataset_id.",
    )
    can_register: bool = Field(
        False,
        description="True when this is a tabular file inside the caller's own tenant folders and can be "
                    "registered with POST /datasets/ozone/register.",
    )


class OzoneGroupNode(BaseModel):
    """
    One folder in the grouped browse view. A folder either holds files (`items`) or sub-folders (`children`).
    Example: 'Data lake' -> 'Database'; 'ML workflow' -> 'Structured datasets' / 'Unstructured data'.
    """
    id: str = Field(..., description="Stable id for the frontend: data_lake, database, ml_workflow, structured_datasets, unstructured_data, other")
    label: str = Field(..., description="Display name of the folder")
    path: Optional[str] = Field(None, description="Ozone key prefix this folder maps to")
    count: int = Field(0, description="Number of files in this folder, including its sub-folders (Ozone 'folder' entries are not counted)")
    total_size_bytes: int = Field(0, description="Total size of the files counted in `count`")
    sub_groups: Optional[Dict[str, int]] = Field(None, description="File counts by media_type (image/audio/video/tabular/other); only set on leaf folders")
    items: List[OzoneObjectResponse] = Field(default_factory=list, description="Files directly in this folder (leaf folders only)")
    children: List["OzoneGroupNode"] = Field(default_factory=list, description="Sub-folders")


OzoneGroupNode.model_rebuild()


class OzoneBrowseGroupedResponse(BaseModel):
    source: str = Field("bucket", description="Where the folders live: the Ozone bucket. Data lake and ML workflow are both in it.")
    scope: str = Field(..., description="The scope that was browsed (ml_workflow, landing, all_tenant_data, bucket). It decides which folders are returned.")
    total: int = Field(..., description="Files across all folders")
    truncated: bool = Field(False, description="True when more files exist than `limit`; raise `limit` or narrow with `prefix`")
    groups: List[OzoneGroupNode]


class OzoneSourceNode(BaseModel):
    """One data source folder in landing (e.g. 'u5_annual_survey'). Tables are loaded in a second call."""
    id: str = Field(..., description="Raw folder name. Send this as `source` to get the tables of this data source.")
    label: str = Field(..., description="Display name (the folder name without its 'u<n>_' prefix)")
    path: str = Field(..., description="Ozone key prefix of the source folder")
    table_count: int = Field(0, description="Number of tables in this source")
    file_count: int = Field(0, description="Number of files in this source")
    total_size_bytes: int = Field(0, description="Total size of the files in this source")
    last_modified: Optional[str] = Field(None, description="Newest file time in this source (ISO 8601)")


class OzoneLandingSourcesResponse(BaseModel):
    """landing_view=sources: the list of data sources only (no tables, no files)."""
    view: Literal["sources"] = "sources"
    source: str = Field("bucket", description="Where the data lives: the Ozone bucket")
    scope: str = "landing"
    total_sources: int
    total_tables: int
    total_files: int
    truncated: bool = False
    sources: List[OzoneSourceNode]


class OzoneTableNode(BaseModel):
    """One table of a data source (landing_view=tables). Carries the dataset fields so the UI needs no digging into `items`."""
    id: str = Field(..., description="Table name (the folder under the data source)")
    label: str = Field(..., description="Display name of the table")
    path: Optional[str] = Field(None, description="Ozone key prefix of the table folder")
    count: int = Field(0, description="Number of files in this table")
    total_size_bytes: int = Field(0, description="Total size of the files in this table")
    sub_groups: Optional[Dict[str, int]] = Field(None, description="File counts by media_type")
    last_modified: Optional[str] = Field(None, description="Newest file time in this table (ISO 8601)")
    dataset_id: Optional[str] = Field(None, description="Dataset UUID to use for this table: the newest file that is already registered. null = nothing registered yet (see can_register).")
    dataset_ids: List[str] = Field(default_factory=list, description="Dataset UUIDs of ALL registered files of this table, newest first (a table can hold several files)")
    is_registered: bool = Field(False, description="True when at least one file of this table is registered as a dataset")
    can_register: bool = Field(False, description="True when at least one file of this table is not registered yet")
    latest_file_key: Optional[str] = Field(None, description="Ozone key of the newest file of this table (what POST /datasets/ozone/register takes)")
    items: List[OzoneObjectResponse] = Field(default_factory=list, description="The files of this table, newest first; each has its own dataset_id")


class OzoneLandingTableNode(BaseModel):
    """One table of a data source (landing_view=tables). dataset_id sits here, on the table, not inside any `items`."""
    id: str = Field(..., description="Table name (the folder under the data source)")
    label: str = Field(..., description="Display name of the table")
    path: str = Field(..., description="Ozone key prefix of the table folder")
    count: int = Field(0, description="Number of files in this table")
    total_size_bytes: int = Field(0, description="Total size of the files in this table")
    dataset_id: Optional[str] = Field(None, description="Dataset UUID of the table's newest file; null if nothing is registered yet")


class OzoneLandingTablesResponse(BaseModel):
    """landing_view=tables&source=...: the tables of ONE data source, each with its files."""
    view: Literal["tables"] = "tables"
    source: OzoneSourceNode
    scope: str = "landing"
    total_tables: int
    total_files: int
    truncated: bool = False
    tables: List[OzoneLandingTableNode] = Field(..., description="One node per table, each with its dataset_id")


class ColumnSchemaItem(BaseModel):
    name: str = Field(..., description="Column name")
    data_type: str = Field(..., description="Inferred semantic data type: numeric, categorical, text, datetime, boolean")
    raw_type: str = Field(..., description="Underlying Pandas / NumPy dtype (e.g. int64, float64, object)")


class DatasetPreviewResponse(BaseModel):
    id: str
    name: str
    columns: List[str]
    column_types: Dict[str, str] = Field(default_factory=dict, description="Dictionary mapping each column name to its inferred data type")
    columns_schema: List[ColumnSchemaItem] = Field(default_factory=list, description="List of columns with rich type descriptors")
    inferred_target_column: Optional[str] = Field(None, description="Inferred target label column for supervised tasks (e.g. Weekly_Sales)")
    total_rows: int
    limit: int = Field(10, description="Page size limit applied to rows")
    offset: int = Field(0, description="Row start offset for pagination")
    preview_rows: List[Dict[str, Any]]


class DatasetProfileResponse(BaseModel):
    id: str
    name: str
    profile: Dict[str, Any]