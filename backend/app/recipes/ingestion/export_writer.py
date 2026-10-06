import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

from backend.app.recipes.base.recipe import BaseRecipe, RecipePort
from backend.app.core.config import settings
from backend.app.core.logging import logger


SUPPORTED_FORMATS = ["csv", "parquet", "excel", "json", "tsv"]
FORMAT_EXTENSIONS = {
    "csv": ".csv",
    "parquet": ".parquet",
    "excel": ".xlsx",
    "json": ".json",
    "tsv": ".tsv"
}


def _format_size(size_bytes: int) -> str:
    """Formats raw bytes into human-readable representation."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


class ExportWriterRecipe(BaseRecipe):
    recipe_id = "export_writer"
    name = "Export / Write Operator"
    version = "1.0.0"
    category = "ingestion"
    description = (
        "Writes and exports the pipeline dataset to files (CSV, Parquet, Excel, JSON, TSV) "
        "into storage or custom paths, and passes the table through for downstream nodes (n8n sink pattern)."
    )
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    inputs = [
        RecipePort(
            id="input",
            label="Input Table",
            type="dataframe",
            required=True,
            max_connections=1,
            description="Upstream dataset to export"
        )
    ]
    outputs = [
        RecipePort(
            id="output",
            label="Passthrough Output",
            type="dataframe",
            required=False,
            max_connections=1,
            description="Passthrough dataset for downstream nodes"
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_format": {
                    "type": "string",
                    "title": "File Format",
                    "enum": ["csv", "parquet", "excel", "json", "tsv"],
                    "default": "csv",
                    "input_type": "dropdown",
                    "ui:widget": "select",
                    "widget": "dropdown",
                    "description": "Format of the exported file (CSV, Parquet, Excel .xlsx, JSON, or TSV)."
                },
                "file_name": {
                    "type": "string",
                    "title": "File Name",
                    "default": "exported_dataset",
                    "input_type": "text",
                    "ui:widget": "text",
                    "widget": "text",
                    "description": "Name of the exported file (extension is automatically appended if omitted)."
                },
                "destination": {
                    "type": "string",
                    "title": "Destination",
                    "enum": ["export_store", "custom_path"],
                    "default": "export_store",
                    "input_type": "dropdown",
                    "ui:widget": "select",
                    "widget": "dropdown",
                    "description": "Save in platform's standard export store, or specify a custom relative path."
                },
                "custom_file_path": {
                    "type": "string",
                    "title": "Custom File Path",
                    "default": "",
                    "input_type": "text",
                    "ui:widget": "text",
                    "widget": "text",
                    "description": "Relative file path when using 'custom_path' (e.g. 'exports/my_export.parquet')."
                },
                "include_index": {
                    "type": "boolean",
                    "title": "Include DataFrame Index",
                    "default": False,
                    "input_type": "boolean",
                    "ui:widget": "switch",
                    "widget": "switch",
                    "description": "Whether to write the DataFrame row index as a column in the output file."
                },
                "delimiter": {
                    "type": "string",
                    "title": "Delimiter (CSV / TSV)",
                    "enum": [",", ";", "\t", "|"],
                    "default": ",",
                    "input_type": "dropdown",
                    "ui:widget": "select",
                    "widget": "dropdown",
                    "description": "Field separator for delimited text exports."
                },
                "compression": {
                    "type": "string",
                    "title": "Compression",
                    "enum": ["none", "gzip", "snappy"],
                    "default": "none",
                    "input_type": "dropdown",
                    "ui:widget": "select",
                    "widget": "dropdown",
                    "description": "Compression algorithm (supported for CSV and Parquet)."
                },
                "excel_sheet_name": {
                    "type": "string",
                    "title": "Excel Sheet Name",
                    "default": "Sheet1",
                    "input_type": "text",
                    "ui:widget": "text",
                    "widget": "text",
                    "description": "Name of the worksheet for Excel (.xlsx) exports."
                },
                "json_orient": {
                    "type": "string",
                    "title": "JSON Orientation",
                    "enum": ["records", "split", "index", "columns"],
                    "default": "records",
                    "input_type": "dropdown",
                    "ui:widget": "select",
                    "widget": "dropdown",
                    "description": "Layout format for JSON export ('records' is standard list-of-dicts n8n shape)."
                },
                "overwrite": {
                    "type": "boolean",
                    "title": "Overwrite Existing File",
                    "default": True,
                    "input_type": "boolean",
                    "ui:widget": "switch",
                    "widget": "switch",
                    "description": "Whether to overwrite the file if a file with the same name already exists."
                }
            },
            "required": ["file_format"]
        }

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        errors = []
        fmt = str(config.get("file_format", "csv")).lower().strip()
        if fmt not in SUPPORTED_FORMATS:
            errors.append(f"Unsupported format '{fmt}'. Supported: {', '.join(SUPPORTED_FORMATS)}.")

        dest = str(config.get("destination", "export_store")).lower().strip()
        if dest == "custom_path":
            custom_path = (config.get("custom_file_path") or "").strip()
            if not custom_path:
                errors.append("Custom file path must be specified when destination is 'custom_path'.")
            elif ".." in custom_path:
                errors.append("Path traversal ('..') is not permitted in custom file path.")

        return errors

    def _resolve_target_path(self, config: Dict[str, Any], fmt: str) -> Path:
        """Determines the full absolute filesystem path for the export target."""
        base_dir = Path(settings.LOCAL_STORAGE_DIR)
        dest = str(config.get("destination", "export_store")).lower().strip()
        ext = FORMAT_EXTENSIONS.get(fmt, f".{fmt}")

        if dest == "custom_path":
            custom_raw = (config.get("custom_file_path") or "").strip()
            # Clean and sanitize path
            clean_rel = custom_raw.replace("\\", "/").lstrip("/")
            target = base_dir / clean_rel
            if not target.suffix:
                target = target.with_suffix(ext)
        else:
            # Standard platform export store
            raw_name = (config.get("file_name") or "exported_dataset").strip()
            # Sanitize filename (remove illegal characters)
            safe_name = re.sub(r'[\\/*?:"<>|]', "_", raw_name)
            if not safe_name.lower().endswith(ext):
                safe_name = f"{safe_name}{ext}"
            export_dir = base_dir / "exports"
            export_dir.mkdir(parents=True, exist_ok=True)
            target = export_dir / safe_name

        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        df: pd.DataFrame = inputs.get("dataframe")
        if df is None:
            raise ValueError("ExportWriterRecipe expects 'dataframe' in inputs.")

        fmt = str(config.get("file_format", "csv")).lower().strip()
        if fmt not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported file format '{fmt}'. Must be one of: {SUPPORTED_FORMATS}")

        target_path = self._resolve_target_path(config, fmt)
        overwrite = config.get("overwrite", True)

        if target_path.exists() and not overwrite:
            raise ValueError(f"Target file '{target_path.name}' already exists and overwrite is disabled.")

        include_index = bool(config.get("include_index", False))
        compression = str(config.get("compression", "none")).lower().strip()
        comp_arg = None if compression in ["none", ""] else compression

        # Write to format
        try:
            if fmt == "csv":
                sep = str(config.get("delimiter", ","))
                df.to_csv(target_path, sep=sep, index=include_index, compression=comp_arg, encoding="utf-8")
            elif fmt == "tsv":
                df.to_csv(target_path, sep="\t", index=include_index, compression=comp_arg, encoding="utf-8")
            elif fmt == "parquet":
                parquet_comp = "snappy" if comp_arg in [None, "none", "snappy"] else comp_arg
                df.to_parquet(target_path, index=include_index, compression=parquet_comp)
            elif fmt == "excel":
                sheet = str(config.get("excel_sheet_name", "Sheet1")).strip() or "Sheet1"
                df.to_excel(target_path, index=include_index, sheet_name=sheet)
            elif fmt == "json":
                orient = str(config.get("json_orient", "records")).strip()
                df.to_json(target_path, orient=orient, indent=2, force_ascii=False)
        except Exception as e:
            raise ValueError(f"Failed to export DataFrame to {fmt.upper()} at '{target_path}': {str(e)}") from e

        file_size = target_path.stat().st_size if target_path.exists() else 0
        file_size_formatted = _format_size(file_size)

        base_dir = Path(settings.LOCAL_STORAGE_DIR).resolve()
        try:
            rel_path = str(target_path.resolve().relative_to(base_dir)).replace("\\", "/")
        except Exception:
            rel_path = str(target_path).replace("\\", "/")

        logger.info(f"ExportWriterRecipe: Successfully exported {len(df)} rows to '{target_path.name}' ({file_size_formatted})")

        return {
            "dataframe": df,  # Passthrough
            "feature_names": list(df.columns),
            "output_summary": {
                "row_count": len(df),
                "columns": list(df.columns),
                "file_name": target_path.name,
                "file_format": fmt,
                "file_path": rel_path,
                "absolute_path": str(target_path),
                "file_size_bytes": file_size,
                "file_size_formatted": file_size_formatted
            },
            "metrics": {
                "file_name": target_path.name,
                "file_format": fmt,
                "file_path": rel_path,
                "file_size_bytes": file_size,
                "file_size_formatted": file_size_formatted,
                "row_count": len(df),
                "column_count": len(df.columns)
            }
        }

    def to_code(self, config: Dict[str, Any]) -> str:
        fmt = str(config.get("file_format", "csv")).lower().strip()
        fname = (config.get("file_name") or "exported_dataset").strip()
        ext = FORMAT_EXTENSIONS.get(fmt, f".{fmt}")
        full_fname = fname if fname.endswith(ext) else f"{fname}{ext}"
        include_index = bool(config.get("include_index", False))

        if fmt == "csv":
            sep = config.get("delimiter", ",")
            return f"df.to_csv('{full_fname}', sep='{sep}', index={include_index})"
        elif fmt == "tsv":
            return f"df.to_csv('{full_fname}', sep='\\t', index={include_index})"
        elif fmt == "parquet":
            return f"df.to_parquet('{full_fname}', index={include_index})"
        elif fmt == "excel":
            sheet = config.get("excel_sheet_name", "Sheet1")
            return f"df.to_excel('{full_fname}', sheet_name='{sheet}', index={include_index})"
        elif fmt == "json":
            orient = config.get("json_orient", "records")
            return f"df.to_json('{full_fname}', orient='{orient}', indent=2)"
        return f"# Export to {fmt}\ndf.to_{fmt}('{full_fname}')"
