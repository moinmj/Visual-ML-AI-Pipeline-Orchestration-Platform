"""
Amazon S3 Data Lake Connector Recipe (Ingestion & Orchestration -> Data Lake & Object Storage).

Reads analytical datasets (Parquet, CSV, JSON, JSONL) from Amazon S3 or S3-compatible object stores
(MinIO, Apache Ozone, Ceph) into the pipeline as a DataFrame.
Supports S3 URIs, bucket/key pairs, IAM/access keys, custom endpoint URLs, format auto-detection,
benchmark simulation fallback, and code generation.
"""

import os
import re
import io
import time
from typing import Dict, Any, List, Optional, Tuple

try:
    import pandas as pd
except ImportError:
    pd = None

from backend.app.recipes.base.recipe import BaseRecipe, RecipePort
from backend.app.core.logging import logger
from backend.app.core.config import settings


class S3DataLakeConnectorRecipe(BaseRecipe):
    recipe_id = "s3_data_lake_connector"
    name = "Amazon S3 Data Lake Connector"
    version = "1.0.0"
    category = "ingestion"
    group = "Ingestion & Orchestration"
    subgroup = "Data Lake & Object Storage"
    description = "Ingests Parquet, CSV, or JSON datasets from Amazon S3 and S3-compatible Data Lakes (MinIO, Ozone)."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    inputs: List[RecipePort] = []

    outputs = [
        RecipePort(
            id="output",
            label="S3 Lake Dataset",
            type="dataframe",
            description="DataFrame containing parsed S3 Data Lake records"
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "sections": [
                {"id": "source", "title": "S3 Source & Location", "description": "Configure S3 URI or Bucket and Object Key"},
                {"id": "connection", "title": "AWS Authentication & Region", "description": "Configure AWS credentials and endpoint"},
                {"id": "runtime_limits", "title": "Limits & Performance", "description": "Row limits and simulation mode"}
            ],
            "properties": {
                # ── 1. S3 Source & Location ──
                "source_mode": {
                    "type": "string",
                    "title": "Source Mode",
                    "input_type": "dropdown",
                    "enum": ["s3_uri", "bucket_and_key"],
                    "default": "s3_uri",
                    "section": "source",
                    "description": "Specify file location via full 's3://' URI or individual bucket and file key."
                },
                "s3_uri": {
                    "type": "string",
                    "title": "S3 URI",
                    "input_type": "text",
                    "default": "s3://lakehouse-analytics-bucket/raw/events/telemetry_data.parquet",
                    "section": "source",
                    "description": "Full S3 URI (e.g. 's3://my-bucket/path/to/data.parquet' or 's3://my-bucket/logs/2026/').",
                    "visible_when": {
                        "field": "source_mode",
                        "equals": "s3_uri"
                    }
                },
                "bucket_name": {
                    "type": "string",
                    "title": "Bucket Name",
                    "input_type": "text",
                    "default": "lakehouse-analytics-bucket",
                    "section": "source",
                    "description": "S3 bucket name (used when source_mode is 'bucket_and_key').",
                    "visible_when": {
                        "field": "source_mode",
                        "equals": "bucket_and_key"
                    }
                },
                "file_key": {
                    "type": "string",
                    "title": "Object Key / Path",
                    "input_type": "text",
                    "default": "raw/events/telemetry_data.parquet",
                    "section": "source",
                    "description": "Object key inside the bucket (e.g. 'raw/events/data.parquet').",
                    "visible_when": {
                        "field": "source_mode",
                        "equals": "bucket_and_key"
                    }
                },
                "file_format": {
                    "type": "string",
                    "title": "File Format",
                    "input_type": "dropdown",
                    "enum": ["auto", "parquet", "csv", "json", "jsonl"],
                    "default": "auto",
                    "section": "source",
                    "description": "Dataset file format. 'auto' infers format from the file extension."
                },

                # ── 2. AWS Authentication & Region ──
                "credential_id": {
                    "type": "string",
                    "title": "Connected S3 / AWS Credential",
                    "input_type": "credential_dropdown",
                    "provider": "s3",
                    "section": "connection",
                    "description": "Select stored AWS S3 credentials from platform settings."
                },
                "aws_access_key_id": {
                    "type": "string",
                    "title": "AWS Access Key ID",
                    "input_type": "text",
                    "default": "",
                    "section": "connection",
                    "description": "AWS access key (leave blank to use environment credentials or IAM role)."
                },
                "aws_secret_access_key": {
                    "type": "string",
                    "title": "AWS Secret Access Key",
                    "input_type": "password",
                    "default": "",
                    "section": "connection",
                    "description": "AWS secret access key."
                },
                "aws_session_token": {
                    "type": "string",
                    "title": "Session Token (Optional)",
                    "input_type": "password",
                    "default": "",
                    "section": "connection",
                    "description": "Temporary session token for STS / assumed roles."
                },
                "region_name": {
                    "type": "string",
                    "title": "AWS Region",
                    "input_type": "text",
                    "default": "us-east-1",
                    "section": "connection",
                    "description": "AWS Region (e.g. 'us-east-1', 'eu-west-1', 'ap-south-1')."
                },
                "endpoint_url": {
                    "type": "string",
                    "title": "Custom Endpoint URL (Optional)",
                    "input_type": "text",
                    "default": "",
                    "section": "connection",
                    "description": "Custom S3 endpoint for MinIO, Apache Ozone, LocalStack, or Ceph (e.g. 'http://minio:9000')."
                },

                # ── 3. Limits & Performance ──
                "limit": {
                    "type": "integer",
                    "title": "Row Limit",
                    "input_type": "number",
                    "default": 1000,
                    "section": "runtime_limits",
                    "description": "Maximum number of rows to retrieve (1 to 1,000,000)."
                },
                "simulation_mode": {
                    "type": "boolean",
                    "title": "Simulation / Demo Mode",
                    "input_type": "checkbox",
                    "default": False,
                    "section": "runtime_limits",
                    "description": "When enabled or when S3 is unreachable, returns realistic Lakehouse benchmark telemetry records."
                }
            },
            "required": ["source_mode", "file_format"]
        }

    def _parse_s3_uri(self, uri: str) -> Tuple[str, str]:
        uri = uri.strip()
        if uri.startswith("s3://"):
            parts = uri[5:].split("/", 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ""
            return bucket, key
        return "", ""

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        errors: List[str] = []
        mode = str(config.get("source_mode", "s3_uri")).strip().lower()

        if not config.get("simulation_mode"):
            if mode == "s3_uri":
                uri = str(config.get("s3_uri", "")).strip()
                if not uri:
                    errors.append("S3 URI is required in 's3_uri' mode.")
                elif not uri.startswith("s3://"):
                    errors.append("S3 URI must start with 's3://'.")
            elif mode == "bucket_and_key":
                if not str(config.get("bucket_name", "")).strip():
                    errors.append("Bucket name is required in 'bucket_and_key' mode.")
                if not str(config.get("file_key", "")).strip():
                    errors.append("Object key / file path is required in 'bucket_and_key' mode.")

        limit = config.get("limit")
        if limit is not None:
            try:
                l_val = int(limit)
                if l_val <= 0:
                    errors.append("Limit must be greater than 0.")
            except (ValueError, TypeError):
                errors.append("Limit must be an integer.")

        return errors

    def _infer_format(self, key_or_uri: str, format_param: str) -> str:
        fmt = format_param.strip().lower()
        if fmt != "auto" and fmt in ["parquet", "csv", "json", "jsonl"]:
            return fmt
        low = key_or_uri.lower()
        if low.endswith(".parquet") or ".parquet" in low:
            return "parquet"
        if low.endswith(".csv") or ".csv" in low:
            return "csv"
        if low.endswith(".jsonl") or ".ndjson" in low:
            return "jsonl"
        if low.endswith(".json"):
            return "json"
        return "parquet"

    def _generate_benchmark_data(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        limit = max(1, min(int(config.get("limit", 100)), 5000))
        uri = str(config.get("s3_uri", "s3://lakehouse-analytics-bucket/raw/events/telemetry.parquet"))
        bucket, key = self._parse_s3_uri(uri)
        if not bucket:
            bucket = str(config.get("bucket_name", "lakehouse-analytics-bucket"))
            key = str(config.get("file_key", "raw/events/telemetry.parquet"))

        sensors = ["THERMAL_01", "VIBRATION_02", "PRESSURE_03", "OPTICAL_04", "ACOUSTIC_05"]
        zones = ["us-east-1a", "us-east-1b", "eu-central-1a", "ap-south-1a"]
        records: List[Dict[str, Any]] = []

        for i in range(limit):
            records.append({
                "record_id": f"s3_rec_{1000000 + i}",
                "device_id": f"EDGE_GW_{100 + (i % 30):04d}",
                "sensor_type": sensors[i % len(sensors)],
                "metric_value": round(20.0 + ((i * 3.7) % 110.0), 3),
                "battery_pct": max(5, 100 - (i % 95)),
                "status_code": 200 if i % 15 != 0 else 503,
                "availability_zone": zones[i % len(zones)],
                "lake_bucket": bucket,
                "object_path": key,
                "partition_year": 2026,
                "partition_month": (i % 12) + 1,
                "timestamp": f"2026-03-{((i % 28) + 1):02d}T08:{((i % 60)):02d}:00Z"
            })
        return records

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        start_time = time.time()
        limit = max(1, min(int(config.get("limit", 1000)), 1000000))
        mode = str(config.get("source_mode", "s3_uri")).strip().lower()
        simulation_mode = bool(config.get("simulation_mode", False))

        if mode == "s3_uri":
            uri = str(config.get("s3_uri", "")).strip()
            bucket, key = self._parse_s3_uri(uri)
        else:
            bucket = str(config.get("bucket_name", "")).strip()
            key = str(config.get("file_key", "")).strip()
            uri = f"s3://{bucket}/{key}"

        format_type = self._infer_format(key or uri, str(config.get("file_format", "auto")))
        records: List[Dict[str, Any]] = []
        delivery_status = "SIMULATED"
        status_msg = ""

        # Live Execution Attempt with boto3
        if not simulation_mode and bucket and key:
            try:
                import boto3
                boto_kwargs: Dict[str, Any] = {
                    "region_name": str(config.get("region_name", "us-east-1")).strip()
                }
                ak = str(config.get("aws_access_key_id", "")).strip()
                sk = str(config.get("aws_secret_access_key", "")).strip()
                st = str(config.get("aws_session_token", "")).strip()
                if ak and sk:
                    boto_kwargs["aws_access_key_id"] = ak
                    boto_kwargs["aws_secret_access_key"] = sk
                    if st:
                        boto_kwargs["aws_session_token"] = st

                endpoint = str(config.get("endpoint_url", "")).strip()
                if endpoint:
                    boto_kwargs["endpoint_url"] = endpoint

                s3_client = boto3.client("s3", **boto_kwargs)
                response = s3_client.get_object(Bucket=bucket, Key=key)
                body_bytes = response["Body"].read()

                bio = io.BytesIO(body_bytes)
                if format_type == "parquet":
                    df_live = pd.read_parquet(bio)
                elif format_type == "csv":
                    df_live = pd.read_csv(bio)
                elif format_type == "jsonl":
                    df_live = pd.read_json(bio, lines=True)
                else:
                    df_live = pd.read_json(bio)

                if limit and len(df_live) > limit:
                    df_live = df_live.iloc[:limit]

                records = df_live.to_dict(orient="records")
                delivery_status = "LIVE"
                status_msg = f"Successfully read {len(records)} records from s3://{bucket}/{key} (format: {format_type})."
            except ImportError:
                raise ValidationException("boto3 or pyarrow driver not installed. Please install 'boto3' and 'pyarrow' to connect to S3.")
            except Exception as e:
                logger.warning(f"S3 ingestion failed: {e}")
                raise ValidationException(f"Amazon S3 Data Lake ingestion failed: {str(e)}")

        # Simulation Mode (ONLY when simulation_mode is True)
        if simulation_mode:
            records = self._generate_benchmark_data(config)
            status_msg = f"Loaded {len(records)} benchmark records for S3 Data Lake '{uri}'."

        df_out = pd.DataFrame(records) if pd is not None else records
        duration_ms = int((time.time() - start_time) * 1000)
        col_count = len(df_out.columns) if hasattr(df_out, "columns") else (len(records[0].keys()) if records else 0)

        summary = {
            "title": "Amazon S3 Data Lake Connector Summary",
            "message": status_msg,
            "status": delivery_status,
            "s3_uri": uri,
            "bucket": bucket,
            "key": key,
            "format": format_type,
            "rows_retrieved": len(records),
            "columns_count": col_count,
            "duration_ms": duration_ms
        }

        metrics = {
            "rows_retrieved": len(records),
            "columns_count": col_count,
            "duration_ms": duration_ms,
            "delivery_status": delivery_status
        }

        return {
            "dataframe": df_out,
            "output": df_out,
            "output_summary": summary,
            "metrics": metrics
        }

    def to_code(self, config: Dict[str, Any]) -> str:
        uri = config.get("s3_uri", "s3://lakehouse-analytics-bucket/raw/events/data.parquet")
        fmt = config.get("file_format", "auto")
        region = config.get("region_name", "us-east-1")

        return (
            f"# Generated Amazon S3 Data Lake Ingestion Script\n"
            f"import boto3\n"
            f"import io\n"
            f"import pandas as pd\n\n"
            f"s3_uri = '{uri}'\n"
            f"# If using direct pandas S3 support:\n"
            f"df = pd.read_{fmt if fmt != 'auto' else 'parquet'}(s3_uri)\n"
            f"print(f'Successfully ingested {{len(df)}} rows from {{s3_uri}}.')\n"
        )
