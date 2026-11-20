import re
import uuid
import pandas as pd
from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete, func, or_
from sqlalchemy.exc import IntegrityError
from fastapi import UploadFile

from backend.app.datasets.models import Dataset
try:
    from backend.app.infrastructure.storage.storage_manager import storage_manager
except ImportError:
    from backend.app.infrastructure.storage import storage_manager
from backend.app.profiling.profiler import DataProfiler
from backend.app.core.exceptions import NotFoundException, ValidationException
from backend.app.core.logging import logger


TABULAR_FORMATS = {"csv", "xlsx", "xls", "json", "parquet"}
# Files saved as datasets/{dataset_id}_{name}: the UUID prefix is the dataset id.
UUID_PREFIX = re.compile(r"^([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})_")
# Landing files are named {timestamp}_{uuid} or just {uuid}: nothing a user can read.
_MACHINE_NAME = re.compile(r"^(\d{8,14}_)?[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
_SKIP_FOLDERS = {"landing", "db", "ml-workflow", "datasets", "unstructured"}


_SOURCE_PREFIX = re.compile(r"^u\d+_")


def landing_source_label(source: str) -> str:
    """'u5_annual_survey' -> 'annual_survey' (display only; the raw folder name stays the id)."""
    return _SOURCE_PREFIX.sub("", source) or source


def ozone_display_name(key: str, stem: str, file_name: str) -> str:
    """
    Readable name for the UI. `key` and `file_name` are NOT changed; only `name` is.
      ml-workflow/t4/datasets/05033d74-..._sales.csv        -> "sales"            (UUID prefix removed)
      landing/db/t4/u5_postgresql/retail_sales/2026..._uuid.parquet -> "retail_sales" (the table folder)
    """
    name = UUID_PREFIX.sub("", stem) or stem
    if _MACHINE_NAME.match(name):
        # Name is just a timestamp/uuid: the folder it sits in is the real name (the table / source).
        for seg in reversed(key.strip("/").split("/")[:-1]):
            low = seg.lower()
            if seg == file_name or low.rsplit(".", 1)[-1] in TABULAR_FORMATS and "." in seg:
                continue  # e.g. retail_sales/x.parquet/x.parquet
            if low in _SKIP_FOLDERS or re.fullmatch(r"t\d+", low):
                break     # reached the tenant/root folders: no useful name up here
            return seg
    return name


class DatasetService:
    @staticmethod
    async def upload_dataset(
        db: AsyncSession,
        tenant_id: int,
        file: UploadFile,
        name: Optional[str] = None,
        description: Optional[str] = None
    ) -> Dataset:
        # Validate format
        file_name = file.filename or "dataset.csv"
        ext = file_name.split(".")[-1].lower()
        if ext not in ["csv", "xlsx", "xls", "json", "parquet"]:
            raise ValidationException(f"Unsupported file format: '.{ext}'. Allowed: CSV, XLSX, JSON, Parquet.")

        content = await file.read()
        file_size = len(content)
        if file_size == 0:
            raise ValidationException("The uploaded file is empty.")

        dataset_id = str(uuid.uuid4())
        dataset_name = name or file_name.rsplit(".", 1)[0]
        storage_rel_path = f"datasets/{dataset_id}_{file_name}"

        # 1. Save file to storage (returns the local path, or an s3://bucket/key
        #    Ozone path when STORAGE_BACKEND="ozone" -- must be used from here on,
        #    not the raw key we asked it to save under).
        storage_path = await storage_manager.save_file(content, storage_rel_path, tenant_id)

        # 2. Read DataFrame and perform automatic statistical profiling
        try:
            df = storage_manager.read_dataframe(storage_path, tenant_id)
            profile_data = DataProfiler.profile_dataframe(df)
        except Exception as e:
            # Clean up on failure
            storage_manager.delete_file(storage_path, tenant_id)
            logger.error(f"Error parsing uploaded tabular file: {e}")
            raise ValidationException(f"Failed to parse and profile dataset: {str(e)}")

        # 3. Create DB Record
        dataset = Dataset(
            id=dataset_id,
            tenant_id=tenant_id,
            name=dataset_name,
            description=description,
            file_name=file_name,
            file_format=ext,
            file_size_bytes=file_size,
            storage_path=storage_path,
            row_count=profile_data["row_count"],
            column_count=profile_data["column_count"],
            quality_score=profile_data["quality_score"],
            profile=profile_data
        )

        db.add(dataset)
        await db.commit()
        await db.refresh(dataset)
        logger.info(f"Dataset successfully created: {dataset.id} ({dataset.name}) with {dataset.row_count} rows.")
        return dataset

    @staticmethod
    async def list_datasets(
        db: AsyncSession,
        tenant_id: int,
        search: Optional[str] = None,
        file_format: Optional[str] = None,
        sort: str = "newest",
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> Tuple[List[Dataset], int]:
        """
        The tenant's datasets (only those stored in its Ozone folder), optionally
        searched, filtered, sorted and paged. Returns (page_of_datasets, total_matches).

        search: case-insensitive "contains" match on dataset ID, name, file name and
        description; several words must ALL match (e.g. "sales 2024").
        """
        from backend.app.core.config import settings
        from backend.app.infrastructure.storage.ozone_client import tenant_prefix
        folder_path = f"s3://{settings.OZONE_BUCKET}/{tenant_prefix(tenant_id)}"
        local_prefix = f"t{int(tenant_id)}/"

        query = (
            select(Dataset)
            .where(Dataset.tenant_id == tenant_id)
            .where(or_(
                Dataset.storage_path.startswith(folder_path, autoescape=True),
                Dataset.storage_path.startswith(local_prefix, autoescape=True),
                ~Dataset.storage_path.startswith("s3://")
            ))
        )

        for term in (search or "").split():
            term = term.lower()
            query = query.where(or_(
                func.lower(Dataset.id).contains(term, autoescape=True),
                func.lower(Dataset.name).contains(term, autoescape=True),
                func.lower(Dataset.file_name).contains(term, autoescape=True),
                func.lower(func.coalesce(Dataset.description, "")).contains(term, autoescape=True),
            ))

        if file_format:
            query = query.where(func.lower(Dataset.file_format) == file_format.lower().lstrip("."))

        total = (await db.execute(select(func.count()).select_from(query.subquery()))).scalar() or 0

        order = {
            "newest": Dataset.created_at.desc(),
            "oldest": Dataset.created_at.asc(),
            "name": func.lower(Dataset.name).asc(),
            "name_desc": func.lower(Dataset.name).desc(),
            "largest": Dataset.file_size_bytes.desc(),
            "rows": Dataset.row_count.desc(),
        }.get(sort, Dataset.created_at.desc())
        query = query.order_by(order, Dataset.id)

        if offset:
            query = query.offset(offset)
        if limit:
            query = query.limit(limit)

        result = await db.execute(query)
        return list(result.scalars().all()), total

    @staticmethod
    def browse_ozone(
        tenant_id: int,
        scope: str = "ml_workflow",
        folder: str = "all",
        media_type: Optional[str] = None,
        prefix: str = "",
        include_folders: bool = False,
        limit: int = 1000,
        meta: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Lists objects in the caller's tenant space in Ozone, formatted identically
        to DatasetResponse so frontend dataset list components can render them.
        """
        from datetime import datetime, timezone
        from backend.app.core.config import settings

        try:
            raw_objects = storage_manager.list_ozone_objects(
                tenant_id=tenant_id,
                prefix=prefix,
                scope=scope,
                folder=folder,
                media_type=media_type,
                max_keys=limit,
                include_folders=include_folders,
                meta=meta,
            )
        except ValueError as e:
            raise ValidationException(str(e))

        now_utc = datetime.now(timezone.utc)
        results = []
        for obj in raw_objects:
            key = obj["key"]
            file_name = obj["file_name"]
            ext = obj["file_format"]
            m_type = obj.get("media_type")
            is_folder = m_type == "folder"
            stem = file_name if is_folder or "." not in file_name else file_name.rsplit(".", 1)[0]

            preview_url = None
            if m_type == "image":
                preview_url = f"/api/v1/datasets/media/preview?key={key}"

            last_mod = obj.get("last_modified")
            if last_mod:
                try:
                    dt = datetime.fromisoformat(last_mod.replace("Z", "+00:00"))
                except Exception:
                    dt = now_utc
            else:
                dt = now_utc

            folder_label = (
                f"ml-workflow/{folder}" if scope == "ml_workflow"
                else "landing/db" if scope == "landing"
                else "bucket" if scope == "bucket"
                else "all_tenant_data"
            )
            results.append({
                "id": key,
                "name": stem if is_folder else ozone_display_name(key, stem, file_name),
                "description": f"Ozone {'folder' if is_folder else 'object'} [{folder_label}]",
                "file_name": file_name,
                "file_format": ext,
                "file_size_bytes": obj.get("size_bytes", 0),
                "storage_path": f"s3://{settings.OZONE_BUCKET}/{key}",
                "row_count": 0,
                "column_count": 0,
                "quality_score": 100.0,
                "created_at": dt,
                "updated_at": dt,
                "media_type": m_type,
                "preview_url": preview_url,
                "key": key,
                "relative_path": obj.get("relative_path", key),
                "scope": scope,
            })
        return results

    @staticmethod
    def list_landing_sources(tenant_id: int) -> Dict[str, Any]:
        """Step 1: every data source in landing/db/t<id>/ with table/file counts. No files are returned."""
        from backend.app.infrastructure.storage.ozone_client import tenant_landing_prefix

        meta: Dict[str, Any] = {}
        try:
            raw = storage_manager.list_landing_sources(tenant_id=tenant_id, meta=meta)
        except ValueError as e:
            raise ValidationException(str(e))
        landing = tenant_landing_prefix(tenant_id)
        sources = [{
            "id": r["name"],
            "label": landing_source_label(r["name"]),
            "path": f"{landing}{r['name']}/",
            "table_count": r["table_count"],
            "file_count": r["file_count"],
            "total_size_bytes": r["total_size_bytes"],
            "last_modified": r["last_modified"],
        } for r in raw]
        return {
            "view": "sources",
            "source": "bucket",
            "scope": "landing",
            "total_sources": len(sources),
            "total_tables": sum(x["table_count"] for x in sources),
            "total_files": sum(x["file_count"] for x in sources),
            "truncated": bool(meta.get("truncated")),
            "sources": sources,
        }

    @staticmethod
    def resolve_landing_source(tenant_id: int, source: str) -> List[str]:
        """
        Finds the real source folder for what the caller sent. Accepts the id ('u5_annual_survey')
        or the label ('annual_survey'). Returns the matching raw folder names: [] = none, 1 = found,
        2+ = the label is ambiguous (e.g. u5_x and u6_x) and the caller must send the id.
        """
        names = storage_manager.list_landing_source_names(tenant_id=tenant_id)
        if source in names:
            return [source]
        return [n for n in names if landing_source_label(n) == source]

    @staticmethod
    def group_landing_tables(
        tenant_id: int,
        source: str,
        items: List[Dict[str, Any]],
        truncated: bool = False,
    ) -> Dict[str, Any]:
        """
        Step 2: the files of ONE data source grouped by table (the folder right under the source).
        landing/db/t4/<source>/<table>/<file>  ->  table node '<table>' holding that file.
        """
        from backend.app.infrastructure.storage.ozone_client import tenant_landing_prefix

        landing = tenant_landing_prefix(tenant_id)
        src_path = f"{landing}{source}/"
        tables: Dict[str, Dict[str, Any]] = {}
        newest: Dict[str, str] = {}  # table name -> updated_at of the file currently reported
        newest_overall: Optional[str] = None
        for it in items:
            if it.get("media_type") == "folder":
                continue
            key = (it.get("key") or "").lstrip("/")
            if not key.startswith(src_path):
                continue
            parts = key[len(src_path):].split("/")
            name = parts[0] if len(parts) > 1 else parts[0].rsplit(".", 1)[0]
            node = tables.setdefault(name, {
                "id": name, "label": name, "path": f"{src_path}{name}/" if len(parts) > 1 else src_path,
                "count": 0, "total_size_bytes": 0, "dataset_id": None, "can_register": True, "key": None,
            })
            node["count"] += 1
            node["total_size_bytes"] += int(it.get("file_size_bytes") or 0)
            stamp = str(it.get("updated_at") or "")
            if stamp and (newest_overall is None or stamp > newest_overall):
                newest_overall = stamp
            if node["key"] is None or stamp > newest[name]:
                newest[name] = stamp
                node["key"] = key
                node["dataset_id"] = it.get("dataset_id")
                node["can_register"] = bool(it.get("can_register", True))

        nodes = sorted(tables.values(), key=lambda n: n["label"].lower())
        total_files = sum(n["count"] for n in nodes)
        return {
            "view": "tables",
            "source": {
                "id": source,
                "label": landing_source_label(source),
                "path": src_path,
                "table_count": len(nodes),
                "file_count": total_files,
                "total_size_bytes": sum(n["total_size_bytes"] for n in nodes),
                "last_modified": newest_overall,
            },
            "scope": "landing",
            "total_tables": len(nodes),
            "total_files": total_files,
            "truncated": truncated,
            "tables": nodes,
        }

    @staticmethod
    def group_ozone_items(
        tenant_id: int,
        scope: str,
        items: List[Dict[str, Any]],
        folder: str = "all",
        truncated: bool = False,
    ) -> Dict[str, Any]:
        """
        Turns the flat browse list into folders for the frontend. The scope you browse decides which
        folders come back, so picking a scope is never meaningless:

            scope=ml_workflow  ->  ML workflow
                                     Structured datasets   (.../datasets/)
                                     Unstructured data     (.../unstructured/)
                                     Other                 (e.g. outputs/; only when it has files)
            scope=landing      ->  Data lake
                                     Database              (landing/db/t<id>/)
            scope=all_tenant_data / bucket -> Data lake + ML workflow (+ Other for anything else)

        With scope=ml_workflow, folder=datasets / unstructured shows only that sub-folder.
        Grouping is by key prefix only; no extra Ozone calls. Items keep every field they already have.
        """
        from backend.app.infrastructure.storage.ozone_client import (
            tenant_landing_prefix,
            tenant_ml_workflow_prefix,
        )

        landing = tenant_landing_prefix(tenant_id)
        ml = tenant_ml_workflow_prefix(tenant_id)

        def leaf(node_id: str, label: str, path: Optional[str]) -> Dict[str, Any]:
            return {"id": node_id, "label": label, "path": path, "count": 0, "total_size_bytes": 0,
                    "sub_groups": {}, "items": [], "children": []}

        database = leaf("database", "Database", landing)
        structured = leaf("structured_datasets", "Structured datasets", f"{ml}datasets/")
        unstructured = leaf("unstructured_data", "Unstructured data", f"{ml}unstructured/")
        ml_other = leaf("ml_other", "Other", ml)       # inside ML workflow, e.g. outputs/
        other = leaf("other", "Other", None)           # anywhere else in the bucket

        for it in items:
            key = (it.get("key") or "").lstrip("/")
            if key.startswith(landing):
                target = database
            elif key.startswith(f"{ml}datasets/"):
                target = structured
            elif key.startswith(f"{ml}unstructured/"):
                target = unstructured
            elif key.startswith(ml):
                target = ml_other
            else:
                target = other
            target["items"].append(it)
            if it.get("media_type") != "folder":   # folder entries are listed but not counted
                target["count"] += 1
                target["total_size_bytes"] += int(it.get("file_size_bytes") or 0)
                kind = it.get("media_type") or "other"
                target["sub_groups"][kind] = target["sub_groups"].get(kind, 0) + 1

        def parent(node_id: str, label: str, path: Optional[str], children: List[Dict[str, Any]]) -> Dict[str, Any]:
            return {"id": node_id, "label": label, "path": path,
                    "count": sum(c["count"] for c in children),
                    "total_size_bytes": sum(c["total_size_bytes"] for c in children),
                    "sub_groups": None, "items": [], "children": children}

        # ML workflow children: honour folder=datasets / unstructured; Other only when it has files.
        if folder == "datasets":
            ml_children = [structured]
        elif folder == "unstructured":
            ml_children = [unstructured]
        else:
            ml_children = [structured, unstructured]
        if ml_other["items"]:
            ml_children.append(ml_other)

        data_lake = parent("data_lake", "Data lake", landing, [database])
        ml_workflow = parent("ml_workflow", "ML workflow", ml, ml_children)

        if scope == "landing":
            groups = [data_lake]
        elif scope == "ml_workflow":
            groups = [ml_workflow]
        else:  # all_tenant_data, bucket
            groups = [data_lake, ml_workflow]
            if other["items"]:
                groups.append(other)

        return {
            "source": "bucket",
            "scope": scope,
            "total": sum(g["count"] for g in groups),
            "truncated": truncated,
            "groups": groups,
        }

    @staticmethod
    async def register_from_ozone(
        db: AsyncSession,
        tenant_id: int,
        ozone_key: str,
        name: Optional[str] = None,
        description: Optional[str] = None
    ) -> Dataset:
        """
        Registers an existing Ozone object as a dataset, without re-uploading it.
        Reads it once to compute the same profile/schema an upload would get.
        """
        ozone_key = (ozone_key or "").strip().lstrip("/")   # bucket-scope keys can start with "/"
        ext = ozone_key.rsplit(".", 1)[-1].lower() if "." in ozone_key else ""
        if ext not in ["csv", "xlsx", "xls", "json", "parquet"]:
            raise ValidationException(f"Unsupported file format: '.{ext}'. Allowed: CSV, XLSX, JSON, Parquet.")

        from backend.app.core.config import settings
        from backend.app.infrastructure.storage.ozone_client import is_in_tenant_prefix, tenant_prefix
        if not is_in_tenant_prefix(ozone_key, tenant_id):
            raise ValidationException(
                f"Only files under '{tenant_prefix(tenant_id)}', 't{tenant_id}/' or 'landing/db/t{tenant_id}/' can be registered."
            )
        storage_path = f"s3://{settings.OZONE_BUCKET}/{ozone_key}"

        # Idempotent: the same file in the same tenant is registered once. Done after the
        # tenant-prefix check and before the (expensive) read, and always scoped to the tenant.
        existing = (await db.execute(
            select(Dataset).where(Dataset.tenant_id == tenant_id, Dataset.storage_path == storage_path)
        )).scalars().first()
        if existing:
            return existing

        file_name = ozone_key.rsplit("/", 1)[-1]
        dataset_name = name or file_name.rsplit(".", 1)[0]

        # Files saved as datasets/{dataset_id}_{name} keep their own id: if the UUID in the
        # file name is not used by any dataset row yet, reuse it, so the id the file already
        # carries is the dataset id. Otherwise (no UUID prefix, or id already taken) a new id.
        dataset_id = str(uuid.uuid4())
        m = UUID_PREFIX.match(file_name)
        if m:
            candidate = m.group(1).lower()
            taken = (await db.execute(select(Dataset.id).where(Dataset.id == candidate))).first()
            if not taken:
                dataset_id = candidate

        try:
            df = storage_manager.read_dataframe(storage_path, tenant_id)
            profile_data = DataProfiler.profile_dataframe(df)
        except Exception as e:
            logger.error(f"Error reading/profiling Ozone object '{ozone_key}': {e}")
            raise ValidationException(f"Failed to read and profile '{ozone_key}': {str(e)}")

        from backend.app.infrastructure.storage.ozone_client import get_s3_client
        file_size = get_s3_client().head_object(Bucket=settings.OZONE_BUCKET, Key=ozone_key)["ContentLength"]

        dataset = Dataset(
            id=dataset_id,
            tenant_id=tenant_id,
            name=dataset_name,
            description=description,
            file_name=file_name,
            file_format=ext,
            file_size_bytes=file_size,
            storage_path=storage_path,
            row_count=profile_data["row_count"],
            column_count=profile_data["column_count"],
            quality_score=profile_data["quality_score"],
            profile=profile_data
        )
        try:
            db.add(dataset)
            await db.commit()
        except IntegrityError:
            # The reused id was taken by a row this session could not see: fall back to a new id.
            await db.rollback()
            dataset.id = str(uuid.uuid4())
            db.add(dataset)
            await db.commit()
        await db.refresh(dataset)
        logger.info(f"Dataset registered from Ozone: {dataset.id} ({dataset.name}) -> {storage_path}")
        return dataset

    @staticmethod
    async def annotate_ozone_items(db: AsyncSession, tenant_id: int, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Adds `dataset_id` (UUID of the registered dataset for the file, else None) and
        `can_register` to each browse item. 'id' / 'key' are left untouched.
        One query for the caller's tenant; matching is by exact storage_path.
        """
        from backend.app.core.config import settings
        from backend.app.infrastructure.storage.ozone_client import is_in_tenant_prefix

        rows = (await db.execute(
            select(Dataset.id, Dataset.storage_path).where(Dataset.tenant_id == tenant_id)
        )).all()
        id_by_path = {path: ds_id for ds_id, path in rows}
        known_ids = {str(ds_id).lower(): ds_id for ds_id, _ in rows}

        for it in items:
            key = (it.get("key") or "").lstrip("/")
            clean_path = f"s3://{settings.OZONE_BUCKET}/{key}"
            ds_id = id_by_path.get(it.get("storage_path")) or id_by_path.get(clean_path)
            if not ds_id:
                # Fallback: the file is named {dataset_id}_{name}. Use that id only if a dataset
                # row with exactly that id exists for this tenant, so the id always works in the APIs.
                m = UUID_PREFIX.match(it.get("file_name") or "")
                if m:
                    ds_id = known_ids.get(m.group(1).lower())
            it["dataset_id"] = ds_id
            it["can_register"] = bool(
                key
                and it.get("media_type") != "folder"
                and it.get("file_format") in TABULAR_FORMATS
                and is_in_tenant_prefix(key, tenant_id)
            )
        return items

    @staticmethod
    async def get_dataset(db: AsyncSession, dataset_id: str, tenant_id: int) -> Dataset:
        # A dataset that belongs to another tenant is reported as "not found" (404),
        # so its existence is never revealed to the caller.
        result = await db.execute(
            select(Dataset).where(Dataset.id == dataset_id, Dataset.tenant_id == tenant_id)
        )
        dataset = result.scalar_one_or_none()
        if not dataset:
            raise NotFoundException("Dataset", dataset_id)
        return dataset

    @staticmethod
    async def get_preview(db: AsyncSession, dataset_id: str, tenant_id: int, limit: int = 10, offset: int = 0) -> Dict[str, Any]:
        dataset = await DatasetService.get_dataset(db, dataset_id, tenant_id)
        df = storage_manager.read_dataframe(dataset.storage_path, tenant_id)

        total_rows = int(len(df))
        start_idx = max(0, offset)
        end_idx = min(total_rows, start_idx + limit)
        preview_df = df.iloc[start_idx:end_idx].replace({float("nan"): None, float("inf"): None, float("-inf"): None})

        from backend.app.profiling.profiler import DataProfiler
        column_types: Dict[str, str] = {}
        columns_schema: List[Dict[str, str]] = []
        for col in df.columns:
            inferred = DataProfiler._infer_column_type(df[col])
            raw_dtype = str(df[col].dtype)
            column_types[str(col)] = inferred
            columns_schema.append({
                "name": str(col),
                "data_type": inferred,
                "raw_type": raw_dtype
            })

        inferred_target = (dataset.profile or {}).get("inferred_target_column")
        if not inferred_target:
            profile_res = DataProfiler.profile_dataframe(df)
            inferred_target = profile_res.get("inferred_target_column")

        return {
            "id": dataset.id,
            "name": dataset.name,
            "columns": list(df.columns),
            "column_types": column_types,
            "columns_schema": columns_schema,
            "inferred_target_column": inferred_target,
            "total_rows": total_rows,
            "limit": limit,
            "offset": start_idx,
            "preview_rows": preview_df.to_dict(orient="records")
        }

    @staticmethod
    async def delete_dataset(db: AsyncSession, dataset_id: str, tenant_id: int) -> bool:
        dataset = await DatasetService.get_dataset(db, dataset_id, tenant_id)
        storage_manager.delete_file(dataset.storage_path, tenant_id)
        await db.delete(dataset)
        await db.commit()
        logger.info(f"Deleted dataset {dataset_id}")
        return True


dataset_service = DatasetService()