"""
File System Watcher Recipe (Ingestion & Orchestration -> File & Web Ingestion).

Watches a local directory or network mount for incoming/updated data files (CSV, Parquet, JSON, JSONL),
discovering them via glob/regex, ordering them by modification time or size, and parsing them into a unified
DataFrame or outputting a comprehensive file manifest.
Features in-flight file stabilization (min_age_seconds), recursive folder traversal, post-ingestion archiving,
and deterministic simulation fallback when paths are empty or unmounted.
"""

import fnmatch
import io
import os
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from backend.app.core.exceptions import ExecutionException, ValidationException
from backend.app.core.logging import logger
from backend.app.recipes.base.recipe import BaseRecipe, RecipePort


class FileSystemWatcherRecipe(BaseRecipe):
    recipe_id = "file_system_watcher"
    name = "File System Watcher"
    version = "1.0.0"
    category = "ingestion"
    group = "Ingestion & Orchestration"
    subgroup = "File & Web Ingestion"
    description = "Watches a folder on a file system and loads new or changed files into a DataFrame."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    inputs = [
        RecipePort(
            id="input",
            label="Trigger / Input Dataset",
            type="dataframe",
            required=False,
            max_connections=1,
            description="Optional upstream dataset for batch IDs or execution trigger",
        )
    ]

    outputs = [
        RecipePort(
            id="output",
            label="Output Table",
            type="dataframe",
            description="Ingested file system dataset or file metadata manifest",
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "directory_path": {
                    "type": "string",
                    "title": "Directory Path",
                    "default": "./data/incoming",
                    "input_type": "text",
                    "description": "Local or mounted file system directory path to scan for data files.",
                },
                "file_pattern": {
                    "type": "string",
                    "title": "File Pattern",
                    "default": "*.csv",
                    "input_type": "text",
                    "description": "Glob or regex pattern to filter files (e.g. '*.csv', 'events_*.parquet').",
                },
                "recursive": {
                    "type": "boolean",
                    "title": "Recursive Scan",
                    "default": False,
                    "input_type": "checkbox",
                    "description": "Scan nested subdirectories recursively when set to true.",
                },
                "output_mode": {
                    "type": "string",
                    "title": "Output Mode",
                    "enum": ["parse_data", "file_manifest"],
                    "default": "parse_data",
                    "input_type": "select",
                    "description": "'parse_data' loads row records; 'file_manifest' outputs file paths and metadata.",
                },
                "file_format": {
                    "type": "string",
                    "title": "File Format",
                    "enum": ["auto", "csv", "parquet", "json", "jsonl"],
                    "default": "auto",
                    "input_type": "select",
                    "description": "File parsing format. 'auto' automatically detects format by extension.",
                },
                "delimiter": {
                    "type": "string",
                    "title": "CSV Delimiter",
                    "enum": [",", "\t", ";", "|"],
                    "default": ",",
                    "input_type": "select",
                    "description": "Field separator for delimited text files.",
                },
                "encoding": {
                    "type": "string",
                    "title": "Encoding",
                    "enum": ["utf-8", "latin-1", "ascii"],
                    "default": "utf-8",
                    "input_type": "select",
                    "description": "Character encoding used to decode text files.",
                },
                "max_files": {
                    "type": "integer",
                    "title": "Max Files to Read",
                    "default": 20,
                    "minimum": 1,
                    "maximum": 1000,
                    "input_type": "number",
                    "description": "Maximum number of matched files to process in one execution.",
                },
                "sort_by": {
                    "type": "string",
                    "title": "Sort Files By",
                    "enum": ["modified_desc", "modified_asc", "name_asc", "size_desc"],
                    "default": "modified_desc",
                    "input_type": "select",
                    "description": "Ordering criteria applied before selecting up to max_files.",
                },
                "min_age_seconds": {
                    "type": "integer",
                    "title": "Minimum File Age (Seconds)",
                    "default": 0,
                    "minimum": 0,
                    "maximum": 86400,
                    "input_type": "number",
                    "description": "Ignore files modified more recently than this to prevent reading partial writes.",
                },
                "include_metadata_columns": {
                    "type": "boolean",
                    "title": "Include Metadata Columns",
                    "default": True,
                    "input_type": "checkbox",
                    "description": "Appends '_file_path', '_file_name', and '_modified_time' to parsed rows.",
                },
                "action_after_read": {
                    "type": "string",
                    "title": "Action After Read",
                    "enum": ["none", "move", "delete"],
                    "default": "none",
                    "input_type": "select",
                    "description": "Action performed on scanned files after successful ingestion.",
                },
                "archive_directory": {
                    "type": "string",
                    "title": "Archive Directory",
                    "default": "./data/archive",
                    "input_type": "text",
                    "description": "Target folder when action_after_read is 'move'.",
                },
                "simulation_fallback": {
                    "type": "boolean",
                    "title": "Simulation Fallback",
                    "default": True,
                    "input_type": "checkbox",
                    "description": "Generate synthetic batch records if directory is empty or missing.",
                },
            },
            "required": ["directory_path"],
        }

    def _matches_pattern(self, name: str, pattern: str) -> bool:
        if not pattern or pattern == "*":
            return True
        if fnmatch.fnmatch(name, pattern):
            return True
        try:
            return bool(re.search(pattern, name, re.IGNORECASE))
        except Exception:
            return False

    def _scan_directory(
        self,
        directory_path: str,
        file_pattern: str,
        recursive: bool,
        min_age_seconds: int,
        sort_by: str,
        max_files: int,
    ) -> List[Dict[str, Any]]:
        p = Path(directory_path)
        if not p.exists() or not p.is_dir():
            return []

        now_ts = time.time()
        matched: List[Dict[str, Any]] = []

        scanner = p.rglob("*") if recursive else p.glob("*")
        for entry in scanner:
            if entry.is_file():
                if self._matches_pattern(entry.name, file_pattern):
                    try:
                        stat = entry.stat()
                        age = now_ts - stat.st_mtime
                        if age >= min_age_seconds:
                            matched.append({
                                "path": str(entry.resolve()),
                                "filename": entry.name,
                                "size_bytes": stat.st_size,
                                "mtime": stat.st_mtime,
                                "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                                "obj": entry,
                            })
                    except (OSError, PermissionError) as err:
                        logger.warning(f"Error accessing file {entry}: {err}")

        # Sorting
        if sort_by == "modified_desc":
            matched.sort(key=lambda x: x["mtime"], reverse=True)
        elif sort_by == "modified_asc":
            matched.sort(key=lambda x: x["mtime"])
        elif sort_by == "name_asc":
            matched.sort(key=lambda x: x["filename"].lower())
        elif sort_by == "size_desc":
            matched.sort(key=lambda x: x["size_bytes"], reverse=True)

        return matched[:max_files]

    def _generate_synthetic_data(self, file_pattern: str, max_files: int) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        now_ts = datetime.now(timezone.utc)
        ext = "csv"
        if "parquet" in file_pattern.lower():
            ext = "parquet"
        elif "jsonl" in file_pattern.lower():
            ext = "jsonl"
        elif "json" in file_pattern.lower():
            ext = "json"

        simulated_files: List[Dict[str, Any]] = []
        frames: List[pd.DataFrame] = []
        count = min(max(1, max_files), 3)

        for i in range(1, count + 1):
            fname = f"sample_dataset_{now_ts.strftime('%Y%m%d')}_{i:03d}.{ext}"
            fpath = f"/virtual/data/{fname}"
            rows = [
                {"record_id": f"REC-{100 * i + j}", "category": f"CAT-{(j % 3) + 1}", "value": round(15.2 * j + i, 2), "active": True}
                for j in range(1, 6)
            ]
            part_df = pd.DataFrame(rows)
            part_df["_file_path"] = fpath
            part_df["_file_name"] = fname
            part_df["_modified_time"] = now_ts.isoformat()
            frames.append(part_df)

            simulated_files.append({
                "path": fpath,
                "filename": fname,
                "size_bytes": 1024 * i,
                "mtime": time.time(),
                "modified_at": now_ts.isoformat(),
            })

        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return combined, simulated_files

    def _parse_file(
        self,
        filepath: str,
        file_format: str,
        delimiter: str,
        encoding: str,
    ) -> pd.DataFrame:
        fmt = file_format.lower()
        if fmt == "auto":
            lower = filepath.lower()
            if lower.endswith(".parquet") or lower.endswith(".pq"):
                fmt = "parquet"
            elif lower.endswith(".jsonl"):
                fmt = "jsonl"
            elif lower.endswith(".json"):
                fmt = "json"
            else:
                fmt = "csv"

        try:
            if fmt == "parquet":
                return pd.read_parquet(filepath)
            elif fmt == "jsonl":
                return pd.read_json(filepath, lines=True, encoding=encoding)
            elif fmt == "json":
                return pd.read_json(filepath, encoding=encoding)
            else:
                return pd.read_csv(filepath, sep=delimiter, encoding=encoding)
        except Exception as e:
            logger.warning(f"Failed parsing file '{filepath}' as '{fmt}': {e}")
            try:
                return pd.read_csv(filepath, sep=None, engine="python", encoding=encoding)
            except Exception:
                raise ExecutionException(f"Failed to parse file '{filepath}' as {fmt}: {e}")

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.time()
        dir_path = str(config.get("directory_path", "./data/incoming")).strip()
        if not dir_path:
            raise ValidationException("Parameter 'directory_path' cannot be empty.")

        file_pattern = str(config.get("file_pattern", "*.csv")).strip()
        recursive = bool(config.get("recursive", False))
        output_mode = str(config.get("output_mode", "parse_data")).strip().lower()
        file_format = str(config.get("file_format", "auto")).strip().lower()
        delimiter = str(config.get("delimiter", ","))
        encoding = str(config.get("encoding", "utf-8")).strip().lower()
        max_files = max(1, min(int(config.get("max_files", 20)), 1000))
        sort_by = str(config.get("sort_by", "modified_desc")).strip().lower()
        min_age_seconds = max(0, int(config.get("min_age_seconds", 0)))
        include_meta = bool(config.get("include_metadata_columns", True))
        action_after = str(config.get("action_after_read", "none")).strip().lower()
        archive_dir = str(config.get("archive_directory", "./data/archive")).strip()
        simulation_fallback = bool(config.get("simulation_fallback", True))

        if action_after not in ("none", "move", "delete"):
            raise ValidationException(f"Unsupported action_after_read '{action_after}'. Must be 'none', 'move', or 'delete'.")

        scanned = self._scan_directory(
            directory_path=dir_path,
            file_pattern=file_pattern,
            recursive=recursive,
            min_age_seconds=min_age_seconds,
            sort_by=sort_by,
            max_files=max_files,
        )

        is_simulation = False
        files_to_process = scanned

        if not files_to_process:
            if not simulation_fallback:
                raise ExecutionException(f"Directory '{dir_path}' contains no matching files for pattern '{file_pattern}'.")
            synth_df, simulated_files = self._generate_synthetic_data(file_pattern, max_files)
            files_to_process = simulated_files
            is_simulation = True

        # Branch A: Output mode is file_manifest
        if output_mode == "file_manifest":
            rows = []
            for item in files_to_process:
                rows.append({
                    "filename": item["filename"],
                    "path": item["path"],
                    "size_bytes": item["size_bytes"],
                    "modified_at": item["modified_at"],
                    "action_after_read": action_after,
                })
            manifest_df = pd.DataFrame(rows)
            if manifest_df.empty:
                manifest_df = pd.DataFrame(columns=["filename", "path", "size_bytes", "modified_at", "action_after_read"])

            elapsed = round((time.time() - t0) * 1000, 2)
            metrics = {
                "execution_time_ms": elapsed,
                "directory_path": dir_path,
                "files_found": len(files_to_process),
                "is_simulation": is_simulation,
                "output_mode": "file_manifest",
                "action_after_read": action_after,
            }
            return {"dataframe": manifest_df, "metrics": metrics}

        # Branch B: Output mode is parse_data
        if is_simulation:
            combined_df = synth_df
            if not include_meta:
                meta_cols = ["_file_path", "_file_name", "_modified_time"]
                combined_df = combined_df.drop(columns=[c for c in meta_cols if c in combined_df.columns])
        else:
            dfs: List[pd.DataFrame] = []
            for item in files_to_process:
                df_part = self._parse_file(
                    filepath=item["path"],
                    file_format=file_format,
                    delimiter=delimiter,
                    encoding=encoding,
                )
                if not df_part.empty:
                    if include_meta:
                        df_part["_file_path"] = item["path"]
                        df_part["_file_name"] = item["filename"]
                        df_part["_modified_time"] = item["modified_at"]
                    dfs.append(df_part)

                # Post-read file actions
                if action_after == "delete":
                    try:
                        os.remove(item["path"])
                    except Exception as e:
                        logger.warning(f"Could not delete {item['path']}: {e}")
                elif action_after == "move":
                    try:
                        os.makedirs(archive_dir, exist_ok=True)
                        shutil.move(item["path"], os.path.join(archive_dir, item["filename"]))
                    except Exception as e:
                        logger.warning(f"Could not move {item['path']} to {archive_dir}: {e}")

            combined_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

        elapsed = round((time.time() - t0) * 1000, 2)
        metrics = {
            "execution_time_ms": elapsed,
            "directory_path": dir_path,
            "files_processed": len(files_to_process),
            "total_rows": len(combined_df),
            "is_simulation": is_simulation,
            "output_mode": "parse_data",
            "action_after_read": action_after,
            "archive_directory": archive_dir if action_after == "move" else None,
        }

        return {"dataframe": combined_df, "metrics": metrics}
