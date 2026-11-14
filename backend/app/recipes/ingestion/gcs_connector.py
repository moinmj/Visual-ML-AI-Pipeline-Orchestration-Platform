"""
Google Cloud Storage (GCS) Connector Recipe (Ingestion & Orchestration -> Cloud Storage Connectors).

Ingests analytical datasets (Parquet, CSV, JSON, JSONL) from Google Cloud Storage (GCS) buckets
into the pipeline as a DataFrame.
Supports 'gs://' URIs, bucket and blob path pairs, Service Account JSON authentication,
file format auto-detection, benchmark simulation fallback, and row limiting.
"""

import io
import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backend.app.core.exceptions import ExecutionException, ValidationException
from backend.app.core.logging import logger
from backend.app.recipes.base.recipe import BaseRecipe, RecipePort


class GCSConnectorRecipe(BaseRecipe):
    recipe_id = "gcs_connector"
    name = "Google Cloud Storage Connector"
    version = "1.0.0"
    category = "ingestion"
    group = "Ingestion & Orchestration"
    subgroup = "Cloud Storage Connectors"
    description = "Loads CSV, Parquet, JSON, or JSONL files from a Google Cloud Storage bucket into a DataFrame."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    inputs: List[RecipePort] = []

    outputs = [
        RecipePort(
            id="output",
            label="GCS Dataset",
            type="dataframe",
            description="DataFrame containing parsed Google Cloud Storage records",
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "sections": [
                {
                    "id": "source",
                    "title": "GCS Storage Source",
                    "description": "Configure GCS URI, bucket, and object path",
                },
                {
                    "id": "format",
                    "title": "File Format & Parsing",
                    "description": "Format specification and CSV delimiter options",
                },
                {
                    "id": "connection",
                    "title": "Authentication & Access",
                    "description": "Google Cloud service account credentials and project ID",
                },
                {
                    "id": "runtime_limits",
                    "title": "Limits & Performance",
                    "description": "Row limits and execution simulation options",
                },
            ],
            "properties": {
                "source_mode": {
                    "type": "string",
                    "title": "Source Mode",
                    "enum": ["gcs_uri", "bucket_and_blob"],
                    "default": "gcs_uri",
                    "input_type": "dropdown",
                    "section": "source",
                    "description": "Specify file location via full 'gs://' URI or individual bucket and blob path.",
                },
                "gcs_uri": {
                    "type": "string",
                    "title": "GCS URI",
                    "default": "gs://analytics-lakehouse-bucket/raw/events/telemetry_data.parquet",
                    "input_type": "text",
                    "section": "source",
                    "description": "Full GCS URI (e.g. 'gs://my-bucket/path/to/data.parquet').",
                    "visible_when": {
                        "field": "source_mode",
                        "equals": "gcs_uri",
                    },
                },
                "bucket_name": {
                    "type": "string",
                    "title": "Bucket Name",
                    "default": "analytics-lakehouse-bucket",
                    "input_type": "text",
                    "section": "source",
                    "description": "GCS bucket name (used when source_mode is 'bucket_and_blob').",
                    "visible_when": {
                        "field": "source_mode",
                        "equals": "bucket_and_blob",
                    },
                },
                "blob_path": {
                    "type": "string",
                    "title": "Blob Path",
                    "default": "raw/events/telemetry_data.parquet",
                    "input_type": "text",
                    "section": "source",
                    "description": "Blob object path inside the bucket (e.g. 'raw/events/telemetry_data.parquet').",
                    "visible_when": {
                        "field": "source_mode",
                        "equals": "bucket_and_blob",
                    },
                },
                "file_format": {
                    "type": "string",
                    "title": "File Format",
                    "enum": ["auto", "parquet", "csv", "json", "jsonl"],
                    "default": "auto",
                    "input_type": "dropdown",
                    "section": "format",
                    "description": "Dataset file format. 'auto' infers format from the file extension.",
                },
                "delimiter": {
                    "type": "string",
                    "title": "CSV Delimiter",
                    "default": ",",
                    "enum": [",", "\t", ";", "|"],
                    "input_type": "dropdown",
                    "section": "format",
                    "description": "Delimiter separating columns when reading CSV files.",
                    "visible_when": {
                        "field": "file_format",
                        "equals": "csv",
                    },
                },
                "project_id": {
                    "type": "string",
                    "title": "GCP Project ID",
                    "default": "",
                    "input_type": "text",
                    "section": "connection",
                    "description": "Google Cloud Project ID (leave blank to infer from credentials).",
                },
                "service_account_json": {
                    "type": "string",
                    "title": "Service Account Key JSON",
                    "default": "",
                    "input_type": "password",
                    "section": "connection",
                    "description": "Service account key JSON contents (leave blank to use default credentials).",
                },
                "credential_id": {
                    "type": "string",
                    "title": "Stored GCP Credential",
                    "default": "",
                    "input_type": "credential_dropdown",
                    "provider": "gcp",
                    "section": "connection",
                    "description": "Select stored GCP service account credentials from platform settings.",
                },
                "limit": {
                    "type": "integer",
                    "title": "Row Limit",
                    "default": 1000,
                    "input_type": "number",
                    "section": "runtime_limits",
                    "description": "Maximum number of rows to retrieve (1 to 1,000,000).",
                },
                "simulation_mode": {
                    "type": "boolean",
                    "title": "Simulation / Demo Mode",
                    "default": False,
                    "input_type": "checkbox",
                    "section": "runtime_limits",
                    "description": "When enabled or when GCS is unreachable, returns realistic Lakehouse benchmark telemetry records.",
                },
            },
            "required": ["source_mode", "file_format"],
        }

    def _parse_gcs_uri(self, uri: str) -> Tuple[str, str]:
        """Extracts bucket and blob name from a gs:// URI."""
        uri = (uri or "").strip()
        if uri.startswith("gs://"):
            parts = uri[5:].split("/", 1)
            bucket = parts[0]
            blob = parts[1] if len(parts) > 1 else ""
            return bucket, blob
        return "", ""

    def _infer_format(self, path_or_uri: str, format_param: str) -> str:
        """Infers format from file extension or respects explicit parameter."""
        fmt = (format_param or "auto").strip().lower()
        if fmt != "auto" and fmt in ["parquet", "csv", "json", "jsonl"]:
            return fmt
        low = path_or_uri.lower()
        if low.endswith(".parquet") or ".parquet" in low:
            return "parquet"
        if low.endswith(".csv") or ".csv" in low:
            return "csv"
        if low.endswith(".jsonl") or ".ndjson" in low:
            return "jsonl"
        if low.endswith(".json"):
            return "json"
        return "parquet"

    def _generate_benchmark_data(self, config: Dict[str, Any], bucket: str, blob: str) -> List[Dict[str, Any]]:
        """Generates realistic telemetry benchmark data for demo or fallback execution."""
        limit = max(1, min(int(config.get("limit", 100) or 100), 5000))
        sensors = ["TEMP_SENSOR_A", "FLOW_METER_B", "VOLTAGE_C", "PRESSURE_D", "CURRENT_E"]
        regions = ["us-central1", "us-east4", "europe-west3", "asia-southeast1"]
        records: List[Dict[str, Any]] = []

        for i in range(limit):
            records.append({
                "record_id": f"gcs_rec_{1000000 + i}",
                "device_id": f"GCP_NODE_{100 + (i % 25):04d}",
                "sensor_type": sensors[i % len(sensors)],
                "metric_value": round(25.0 + ((i * 4.3) % 120.0), 3),
                "battery_pct": max(10, 100 - (i % 90)),
                "status_code": 200 if i % 20 != 0 else 500,
                "gcp_region": regions[i % len(regions)],
                "gcs_bucket": bucket or "analytics-lakehouse-bucket",
                "blob_path": blob or "raw/events/telemetry_data.parquet",
                "partition_year": 2026,
                "partition_month": (i % 12) + 1,
                "timestamp": f"2026-04-{((i % 28) + 1):02d}T10:{((i % 60)):02d}:00Z",
            })
        return records

    def _read_blob_bytes(self, config: Dict[str, Any], bucket: str, blob: str) -> bytes:
        """Downloads raw blob bytes from Google Cloud Storage."""
        from google.cloud import storage
        from google.oauth2 import service_account

        sa_json = str(config.get("service_account_json", "")).strip()
        project_id = str(config.get("project_id", "")).strip() or None

        if sa_json:
            info = json.loads(sa_json)
            creds = service_account.Credentials.from_service_account_info(info)
            client = storage.Client(credentials=creds, project=project_id or creds.project_id)
        else:
            client = storage.Client(project=project_id)

        gcs_bucket = client.bucket(bucket)
        gcs_blob = gcs_bucket.blob(blob)
        return gcs_blob.download_as_bytes()

    def execute(
        self,
        inputs: Dict[str, Any],
        config: Dict[str, Any],
        context: Optional[Any] = None,
    ) -> Dict[str, Any]:
        start_time = time.time()
        limit = max(1, min(int(config.get("limit", 1000) or 1000), 1000000))
        source_mode = str(config.get("source_mode", "gcs_uri")).strip().lower()
        simulation_mode = bool(config.get("simulation_mode", False))
        delimiter = str(config.get("delimiter", ",") or ",")

        if source_mode == "gcs_uri":
            uri = str(config.get("gcs_uri", "")).strip()
            bucket, blob = self._parse_gcs_uri(uri)
        else:
            bucket = str(config.get("bucket_name", "")).strip()
            blob = str(config.get("blob_path", "")).strip()
            uri = f"gs://{bucket}/{blob}" if bucket and blob else ""

        format_type = self._infer_format(blob or uri, str(config.get("file_format", "auto")))
        records: List[Dict[str, Any]] = []
        delivery_status = "SIMULATED"
        status_msg = ""

        # Attempt Live GCS Ingestion if not simulation mode
        if not simulation_mode and bucket and blob:
            try:
                blob_bytes = self._read_blob_bytes(config, bucket, blob)
                bio = io.BytesIO(blob_bytes)
                if format_type == "parquet":
                    df_live = pd.read_parquet(bio)
                elif format_type == "csv":
                    df_live = pd.read_csv(bio, sep=delimiter)
                elif format_type == "jsonl":
                    df_live = pd.read_json(bio, lines=True)
                else:
                    df_live = pd.read_json(bio)

                if limit and len(df_live) > limit:
                    df_live = df_live.iloc[:limit]

                records = df_live.to_dict(orient="records")
                delivery_status = "LIVE"
                status_msg = f"Successfully read {len(records)} records from gs://{bucket}/{blob} (format: {format_type})."
            except ImportError:
                status_msg = "google-cloud-storage driver not installed. Using benchmark simulation fallback."
                logger.info(status_msg)
            except Exception as e:
                logger.warning(f"GCS ingestion failed: {e}")
                status_msg = f"GCS ingestion failed ({str(e)}). Using benchmark simulation fallback."

        # Fallback to realistic benchmark data
        if not records:
            records = self._generate_benchmark_data(config, bucket, blob)
            if not status_msg:
                status_msg = f"Loaded {len(records)} benchmark records for GCS '{uri or 'analytics-bucket'}'."

        df_out = pd.DataFrame(records)
        duration_ms = int((time.time() - start_time) * 1000)
        col_count = len(df_out.columns)

        summary = {
            "title": "Google Cloud Storage Connector Summary",
            "message": status_msg,
            "status": delivery_status,
            "gcs_uri": uri,
            "bucket": bucket,
            "blob_path": blob,
            "format": format_type,
            "rows_retrieved": len(records),
            "columns_count": col_count,
            "duration_ms": duration_ms,
        }

        metrics = {
            "rows_retrieved": len(records),
            "columns_count": col_count,
            "duration_ms": duration_ms,
            "delivery_status": delivery_status,
        }

        return {
            "dataframe": df_out,
            "output": df_out,
            "output_summary": summary,
            "metrics": metrics,
            "status": delivery_status,
            "rows_retrieved": len(records),
        }
