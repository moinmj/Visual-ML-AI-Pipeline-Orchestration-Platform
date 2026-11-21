"""
Zip/Archive Unpacker Recipe (Ingestion & Orchestration -> File & Web Ingestion).

Unpacks and parses compressed archive packages (ZIP, TAR, TAR.GZ, TGZ, GZ),
discovers matching inner data files (CSV, Parquet, JSON, JSONL), validates against
path traversal vulnerabilities (Zip Slip), and unions the contained records into a
consolidated DataFrame or provides a comprehensive member manifest.
Features zip/tar auto-detection, multi-table concatenation with member provenance tracking,
and realistic in-memory simulation fallback.
"""

import base64
import fnmatch
import gzip
import io
import os
import re
import tarfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from backend.app.core.exceptions import ExecutionException, ValidationException
from backend.app.core.logging import logger
from backend.app.recipes.base.recipe import BaseRecipe, RecipePort


class ArchiveUnpackerRecipe(BaseRecipe):
    recipe_id = "archive_unpacker"
    name = "Zip/Archive Unpacker"
    version = "1.0.0"
    category = "ingestion"
    group = "Ingestion & Orchestration"
    subgroup = "File & Web Ingestion"
    description = "Unpacks ZIP/TAR archives and loads the contained data files into a DataFrame."
    input_types: List[str] = []
    output_types = ["dataframe"]

    inputs: List[RecipePort] = []

    outputs = [
        RecipePort(
            id="output",
            label="Output Table",
            type="dataframe",
            description="Unpacked and parsed archive records dataset or member manifest",
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "sections": [
                {
                    "id": "source",
                    "title": "Archive Location & Filtering",
                    "description": "Configure archive file path, member pattern, and output mode",
                },
                {
                    "id": "parsing",
                    "title": "Format & Delimiters",
                    "description": "Inner file format parsing, CSV delimiters, and text encoding",
                },
                {
                    "id": "runtime_limits",
                    "title": "Limits & Simulation",
                    "description": "Member extraction limits, metadata columns, and mock archive injection",
                },
            ],
            "properties": {
                "archive_path": {
                    "type": "string",
                    "title": "Archive File Path",
                    "default": "./data/archives/daily_export.zip",
                    "input_type": "text",
                    "section": "source",
                    "description": "Path to ZIP, TAR, TAR.GZ, or TGZ archive file on disk.",
                },
                "file_pattern": {
                    "type": "string",
                    "title": "Contained File Pattern",
                    "default": "*.csv;*.parquet",
                    "input_type": "text",
                    "section": "source",
                    "description": "Glob pattern(s) to match member files inside the archive (e.g. '*.csv', '*.parquet').",
                },
                "output_mode": {
                    "type": "string",
                    "title": "Output Mode",
                    "enum": ["parse_data", "archive_manifest"],
                    "default": "parse_data",
                    "input_type": "dropdown",
                    "section": "source",
                    "description": "'parse_data' parses table rows; 'archive_manifest' returns a table of archive contents.",
                },
                "file_format": {
                    "type": "string",
                    "title": "Inner File Format",
                    "enum": ["auto", "csv", "parquet", "json", "jsonl"],
                    "default": "auto",
                    "input_type": "dropdown",
                    "section": "parsing",
                    "description": "Data format of files inside the archive. 'auto' infers from member extension.",
                },
                "delimiter": {
                    "type": "string",
                    "title": "CSV Delimiter",
                    "enum": [",", "\t", ";", "|"],
                    "default": ",",
                    "input_type": "dropdown",
                    "section": "parsing",
                    "description": "Field separator for delimited text members.",
                    "visible_when": {
                        "field": "file_format",
                        "equals": "csv",
                    },
                },
                "encoding": {
                    "type": "string",
                    "title": "Text Encoding",
                    "enum": ["utf-8", "latin-1", "ascii"],
                    "default": "utf-8",
                    "input_type": "dropdown",
                    "section": "parsing",
                    "description": "Text decoding encoding for CSV and JSON members.",
                },
                "max_files": {
                    "type": "integer",
                    "title": "Max Members to Extract",
                    "default": 20,
                    "minimum": 1,
                    "maximum": 500,
                    "input_type": "number",
                    "section": "runtime_limits",
                    "description": "Upper limit of member files to parse in a single pipeline run.",
                },
                "include_metadata_columns": {
                    "type": "boolean",
                    "title": "Include Archive Metadata Columns",
                    "default": True,
                    "input_type": "checkbox",
                    "section": "runtime_limits",
                    "description": "Appends '_archive_path', '_member_name', and '_member_size' to parsed rows.",
                },
                "simulation_fallback": {
                    "type": "boolean",
                    "title": "Simulation Fallback",
                    "default": True,
                    "input_type": "checkbox",
                    "section": "runtime_limits",
                    "description": "Generate realistic synthetic archive dataset if file path does not exist.",
                },
                "mock_archive_base64": {
                    "type": "string",
                    "title": "Mock Archive Base64",
                    "default": "",
                    "input_type": "textarea",
                    "section": "runtime_limits",
                    "description": "Optional Base64-encoded ZIP/TAR bytes for in-memory extraction tests.",
                },
            },
            "required": ["archive_path"],
        }

    def _matches_pattern(self, name: str, patterns_str: str) -> bool:
        if not patterns_str or patterns_str == "*":
            return True
        patterns = [p.strip() for p in patterns_str.split(";") if p.strip()]
        for pat in patterns:
            if fnmatch.fnmatch(name, pat) or fnmatch.fnmatch(os.path.basename(name), pat):
                return True
            try:
                if re.search(pat, name, re.IGNORECASE):
                    return True
            except Exception:
                pass
        return False

    def _is_safe_path(self, path_str: str) -> bool:
        """Protects against Zip Slip directory traversal attacks."""
        norm = os.path.normpath(path_str)
        return not (norm.startswith("..") or os.path.isabs(norm) or "/../" in norm or "\\..\\" in norm)

    def _extract_members_from_buffer(
        self,
        bio: io.BytesIO,
        archive_name: str,
        file_pattern: str,
        max_files: int,
    ) -> List[Dict[str, Any]]:
        members: List[Dict[str, Any]] = []

        # 1. Try ZIP
        if zipfile.is_zipfile(bio):
            bio.seek(0)
            with zipfile.ZipFile(bio, "r") as zf:
                for info in zf.infolist():
                    if info.is_dir() or not self._is_safe_path(info.filename):
                        continue
                    if self._matches_pattern(info.filename, file_pattern):
                        content = zf.read(info.filename)
                        dt = datetime(*info.date_time, tzinfo=timezone.utc).isoformat() if info.date_time else datetime.now(timezone.utc).isoformat()
                        members.append({
                            "member_name": info.filename,
                            "filename": os.path.basename(info.filename),
                            "size_bytes": info.file_size,
                            "compressed_size": info.compress_size,
                            "modified_at": dt,
                            "bytes": content,
                            "archive_name": archive_name,
                        })
                        if len(members) >= max_files:
                            break
            return members

        # 2. Try TAR (plain, gz, bz2)
        bio.seek(0)
        try:
            with tarfile.open(fileobj=bio, mode="r:*") as tf:
                for info in tf.getmembers():
                    if not info.isfile() or not self._is_safe_path(info.name):
                        continue
                    if self._matches_pattern(info.name, file_pattern):
                        extracted_f = tf.extractfile(info)
                        content = extracted_f.read() if extracted_f else b""
                        dt = datetime.fromtimestamp(info.mtime, tz=timezone.utc).isoformat()
                        members.append({
                            "member_name": info.name,
                            "filename": os.path.basename(info.name),
                            "size_bytes": info.size,
                            "compressed_size": len(content),
                            "modified_at": dt,
                            "bytes": content,
                            "archive_name": archive_name,
                        })
                        if len(members) >= max_files:
                            break
            return members
        except (tarfile.TarError, gzip.BadGzipFile):
            pass

        # 3. Try plain GZIP single file
        bio.seek(0)
        try:
            with gzip.GzipFile(fileobj=bio, mode="rb") as gz:
                content = gz.read()
                clean_name = re.sub(r"\.gz$", "", archive_name, flags=re.IGNORECASE)
                if self._matches_pattern(clean_name, file_pattern):
                    members.append({
                        "member_name": clean_name,
                        "filename": os.path.basename(clean_name),
                        "size_bytes": len(content),
                        "compressed_size": bio.getbuffer().nbytes,
                        "modified_at": datetime.now(timezone.utc).isoformat(),
                        "bytes": content,
                        "archive_name": archive_name,
                    })
            return members
        except Exception:
            pass

        raise ExecutionException(f"Unsupported or corrupted archive format in '{archive_name}'.")

    def _generate_synthetic_archive_members(self) -> List[Dict[str, Any]]:
        now_ts = datetime.now(timezone.utc)
        members: List[Dict[str, Any]] = []

        # Synthetic member 1: sales_part1.csv
        df1 = pd.DataFrame([
            {"transaction_id": "TRX-1001", "store": "Store-East", "revenue": 1420.50, "items_sold": 12},
            {"transaction_id": "TRX-1002", "store": "Store-West", "revenue": 890.25, "items_sold": 8},
        ])
        bio1 = io.StringIO()
        df1.to_csv(bio1, index=False)
        b1 = bio1.getvalue().encode("utf-8")

        members.append({
            "member_name": "data/sales_part1.csv",
            "filename": "sales_part1.csv",
            "size_bytes": len(b1),
            "compressed_size": len(b1),
            "modified_at": now_ts.isoformat(),
            "bytes": b1,
            "archive_name": "simulated_lakehouse_export.zip",
        })

        # Synthetic member 2: sales_part2.csv
        df2 = pd.DataFrame([
            {"transaction_id": "TRX-1003", "store": "Store-Central", "revenue": 2150.00, "items_sold": 19},
        ])
        bio2 = io.StringIO()
        df2.to_csv(bio2, index=False)
        b2 = bio2.getvalue().encode("utf-8")

        members.append({
            "member_name": "data/sales_part2.csv",
            "filename": "sales_part2.csv",
            "size_bytes": len(b2),
            "compressed_size": len(b2),
            "modified_at": now_ts.isoformat(),
            "bytes": b2,
            "archive_name": "simulated_lakehouse_export.zip",
        })

        return members

    def _parse_member_bytes(
        self,
        raw_bytes: bytes,
        filename: str,
        file_format: str,
        delimiter: str,
        encoding: str,
    ) -> pd.DataFrame:
        fmt = file_format.lower()
        if fmt == "auto":
            lower = filename.lower()
            if lower.endswith(".parquet") or lower.endswith(".pq"):
                fmt = "parquet"
            elif lower.endswith(".jsonl"):
                fmt = "jsonl"
            elif lower.endswith(".json"):
                fmt = "json"
            else:
                fmt = "csv"

        bio = io.BytesIO(raw_bytes)
        try:
            if fmt == "parquet":
                return pd.read_parquet(bio)
            elif fmt == "jsonl":
                return pd.read_json(bio, lines=True, encoding=encoding)
            elif fmt == "json":
                return pd.read_json(bio, encoding=encoding)
            else:
                return pd.read_csv(bio, sep=delimiter, encoding=encoding)
        except Exception as e:
            logger.warning(f"Error reading member '{filename}' with format '{fmt}': {e}")
            try:
                bio.seek(0)
                return pd.read_csv(bio, sep=None, engine="python", encoding=encoding)
            except Exception:
                raise ExecutionException(f"Failed parsing inner archive member '{filename}': {e}")

    @staticmethod
    def _ensure_default_archive_exists(archive_path: str) -> None:
        """Bootstraps default sample archive on disk if testing out-of-the-box on a fresh environment."""
        if archive_path in ("./data/archives/daily_export.zip", "./data/archives/sample_archive.zip"):
            p = Path(archive_path)
            if not p.exists():
                try:
                    p.parent.mkdir(parents=True, exist_ok=True)
                    csv1 = (
                        "order_id,customer_id,product_category,amount,status,order_date\n"
                        "ORD-5001,CUST-101,Electronics,299.99,Completed,2026-11-18\n"
                        "ORD-5002,CUST-102,Apparel,49.50,Completed,2026-11-18\n"
                        "ORD-5003,CUST-103,Home & Kitchen,120.00,Shipped,2026-11-19\n"
                        "ORD-5004,CUST-101,Electronics,899.00,Processing,2026-11-20\n"
                        "ORD-5005,CUST-104,Books,24.99,Completed,2026-11-20\n"
                    )
                    csv2 = (
                        "order_id,customer_id,product_category,amount,status,order_date\n"
                        "ORD-5006,CUST-105,Sports,145.00,Completed,2026-11-21\n"
                        "ORD-5007,CUST-102,Apparel,79.99,Completed,2026-11-21\n"
                        "ORD-5008,CUST-106,Electronics,450.00,Processing,2026-11-21\n"
                    )
                    with zipfile.ZipFile(p, "w") as zf:
                        zf.writestr("daily_orders_part1.csv", csv1.strip())
                        zf.writestr("daily_orders_part2.csv", csv2.strip())
                    logger.info(f"Bootstrapped default sample archive at '{archive_path}'")
                except Exception as e:
                    logger.warning(f"Could not bootstrap default archive at '{archive_path}': {e}")

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        t0 = time.time()
        archive_path = str(config.get("archive_path", "./data/archives/daily_export.zip")).strip()
        if not archive_path:
            raise ValidationException("Parameter 'archive_path' cannot be empty.")

        file_pattern = str(config.get("file_pattern", "*.csv;*.parquet")).strip()
        output_mode = str(config.get("output_mode", "parse_data")).strip().lower()
        file_format = str(config.get("file_format", "auto")).strip().lower()
        delimiter = str(config.get("delimiter", ","))
        encoding = str(config.get("encoding", "utf-8")).strip().lower()
        max_files = max(1, min(int(config.get("max_files", 20)), 500))
        include_meta = bool(config.get("include_metadata_columns", True))
        simulation_fallback = bool(config.get("simulation_fallback", True))
        mock_b64 = str(config.get("mock_archive_base64", "")).strip()

        # Check if upstream DataFrame provides an archive path
        in_df = inputs.get("dataframe") if inputs else None
        if isinstance(in_df, pd.DataFrame) and not in_df.empty:
            for col in in_df.columns:
                if any(k in col.lower() for k in ("archive", "zip", "tar", "path", "file", "url")):
                    first_val = str(in_df[col].dropna().iloc[0]).strip() if not in_df[col].dropna().empty else ""
                    if first_val:
                        archive_path = first_val
                        break

        members: List[Dict[str, Any]] = []
        is_simulation = False

        # 1. Base64 mock archive injected
        if mock_b64:
            try:
                archive_bytes = base64.b64decode(mock_b64)
                bio = io.BytesIO(archive_bytes)
                members = self._extract_members_from_buffer(
                    bio=bio,
                    archive_name="mock_archive.zip",
                    file_pattern=file_pattern,
                    max_files=max_files,
                )
            except Exception as e:
                raise ValidationException(f"Failed decoding or extracting 'mock_archive_base64': {e}")

        # 2. Remote HTTP / HTTPS URL archive
        elif archive_path.startswith("http://") or archive_path.startswith("https://"):
            try:
                import requests
                headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
                resp = requests.get(archive_path, headers=headers, timeout=15, stream=True)
                resp.raise_for_status()
                bio = io.BytesIO(resp.content)
                url_filename = archive_path.split("?")[0].split("/")[-1] or "remote_archive.zip"
                members = self._extract_members_from_buffer(
                    bio=bio,
                    archive_name=url_filename,
                    file_pattern=file_pattern,
                    max_files=max_files,
                )
            except Exception as e:
                logger.warning(f"Error downloading remote archive '{archive_path}': {e}")
                if not simulation_fallback:
                    raise ExecutionException(f"Failed downloading archive from '{archive_path}': {e}")
                members = self._generate_synthetic_archive_members()
                is_simulation = True

        # 3. Disk file reading
        else:
            self._ensure_default_archive_exists(archive_path)
            if os.path.exists(archive_path) and os.path.isfile(archive_path):
                try:
                    with open(archive_path, "rb") as f:
                        content = f.read()
                    bio = io.BytesIO(content)
                    members = self._extract_members_from_buffer(
                        bio=bio,
                        archive_name=os.path.basename(archive_path),
                        file_pattern=file_pattern,
                        max_files=max_files,
                    )
                except Exception as e:
                    logger.warning(f"Error reading archive file '{archive_path}': {e}")
                    if not simulation_fallback:
                        raise ExecutionException(f"Failed opening archive '{archive_path}': {e}")
                    members = self._generate_synthetic_archive_members()
                    is_simulation = True

            else:
                # File path missing
                if not simulation_fallback:
                    raise ExecutionException(f"Archive file not found at path '{archive_path}'.")
                members = self._generate_synthetic_archive_members()
                is_simulation = True

        total_bytes = sum(m.get("size_bytes", 0) for m in members)

        # Branch A: Output mode is archive_manifest
        if output_mode == "archive_manifest":
            manifest_rows = []
            for m in members:
                manifest_rows.append({
                    "member_name": m["member_name"],
                    "filename": m["filename"],
                    "size_bytes": m["size_bytes"],
                    "compressed_size": m.get("compressed_size", m["size_bytes"]),
                    "modified_at": m["modified_at"],
                    "archive_name": m["archive_name"],
                })
            manifest_df = pd.DataFrame(manifest_rows)
            if manifest_df.empty:
                manifest_df = pd.DataFrame(columns=["member_name", "filename", "size_bytes", "compressed_size", "modified_at", "archive_name"])

            manifest_df.columns = [str(c) for c in manifest_df.columns]
            elapsed = round((time.time() - t0) * 1000, 2)

            summary = {
                "archive_path": archive_path,
                "members_count": int(len(manifest_df)),
                "total_uncompressed_bytes": int(total_bytes),
                "is_simulation": bool(is_simulation),
            }

            metrics = {
                "title": "Zip / Archive Ingestion Metrics",
                "tag": "ARCHIVE",
                "archive_path": archive_path,
                "members_found": int(len(manifest_df)),
                "total_bytes": int(total_bytes),
                "delivery_status": "Simulated Benchmark" if is_simulation else "Live Archive File",
                "is_simulation": bool(is_simulation),
                "output_mode": "archive_manifest",
                "execution_time_ms": float(elapsed),
            }
            return {
                "dataframe": manifest_df,
                "output": manifest_df,
                "output_summary": summary,
                "metrics": metrics,
            }

        # Branch B: Output mode is parse_data
        parsed_dfs: List[pd.DataFrame] = []
        for m in members:
            try:
                df_part = self._parse_member_bytes(
                    raw_bytes=m["bytes"],
                    filename=m["filename"],
                    file_format=file_format,
                    delimiter=delimiter,
                    encoding=encoding,
                )
            except Exception as e:
                logger.warning(f"Error parsing member '{m['member_name']}': {e}")
                continue

            if not df_part.empty:
                if include_meta:
                    df_part["_archive_path"] = archive_path if not is_simulation else m["archive_name"]
                    df_part["_member_name"] = m["member_name"]
                    df_part["_member_size"] = m["size_bytes"]
                parsed_dfs.append(df_part)

        combined_df = pd.concat(parsed_dfs, ignore_index=True) if parsed_dfs else pd.DataFrame()

        if not combined_df.empty:
            combined_df.columns = [str(c) for c in combined_df.columns]
        else:
            combined_df = pd.DataFrame(columns=["_archive_path", "_member_name", "_member_size"])

        elapsed = round((time.time() - t0) * 1000, 2)

        summary = {
            "archive_path": archive_path,
            "members_count": int(len(members)),
            "total_uncompressed_bytes": int(total_bytes),
            "total_records": int(len(combined_df)),
            "is_simulation": bool(is_simulation),
        }

        metrics = {
            "title": "Zip / Archive Ingestion Metrics",
            "tag": "ARCHIVE",
            "archive_path": archive_path,
            "members_parsed": int(len(members)),
            "total_rows": int(len(combined_df)),
            "total_bytes": int(total_bytes),
            "delivery_status": "Simulated Benchmark" if is_simulation else "Live Archive File",
            "is_simulation": bool(is_simulation),
            "output_mode": "parse_data",
            "execution_time_ms": float(elapsed),
        }

        return {
            "dataframe": combined_df,
            "output": combined_df,
            "output_summary": summary,
            "metrics": metrics,
        }
