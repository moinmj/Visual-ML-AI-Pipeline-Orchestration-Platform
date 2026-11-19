"""
API Response Endpoint Recipe.
Sub-group: Output
Category: Ingestion & Orchestration (ingestion)

Exposes data from the upstream node as a live HTTP API endpoint that returns
structured JSON payloads for frontend UI, webhooks, or external integrations.
"""
from typing import Dict, Any, List, Optional, Union
import json
import os
import secrets
import time
from pathlib import Path
import numpy as np
import pandas as pd

from backend.app.recipes.base.recipe import BaseRecipe, RecipePort
from backend.app.core.config import settings
from backend.app.core.logging import logger


def generate_unique_api_key() -> str:
    """Generates a cryptographically random, unique API secret key for endpoints."""
    return f"crt-sec-{secrets.token_hex(12)}"


# In-memory storage for rapid endpoint lookup by slug
_ENDPOINT_PAYLOAD_CACHE: Dict[str, Dict[str, Any]] = {}


def _safe_cache_filename(key: str) -> str:
    """Sanitizes composite keys (e.g. 'wf-id:slug') into OS-safe filenames (colon is forbidden on Windows)."""
    return key.replace(":", "__").replace("/", "_").replace("\\", "_")


def get_cached_endpoint_payload(endpoint_key: str) -> Optional[Dict[str, Any]]:
    """Retrieves cached API response payload by slug or composite key."""
    clean_key = endpoint_key.strip().lower()
    if clean_key in _ENDPOINT_PAYLOAD_CACHE:
        return _ENDPOINT_PAYLOAD_CACHE[clean_key]

    # Check on-disk persistence fallback
    try:
        cache_dir = Path(settings.LOCAL_STORAGE_DIR) / "endpoints"
        safe_name = _safe_cache_filename(clean_key)
        cache_file = cache_dir / f"{safe_name}.json"
        if not cache_file.exists():
            cache_file = cache_dir / f"{clean_key}.json"

        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                _ENDPOINT_PAYLOAD_CACHE[clean_key] = data
                return data
    except Exception as ex:
        logger.warning(f"Failed to read on-disk endpoint cache for '{clean_key}': {ex}")

    return None


def store_cached_endpoint_payload(endpoint_key: str, payload_data: Dict[str, Any]) -> None:
    """Stores API response payload in memory and on disk for instant serving."""
    clean_key = endpoint_key.strip().lower()
    _ENDPOINT_PAYLOAD_CACHE[clean_key] = payload_data

    try:
        cache_dir = Path(settings.LOCAL_STORAGE_DIR) / "endpoints"
        cache_dir.mkdir(parents=True, exist_ok=True)
        safe_name = _safe_cache_filename(clean_key)
        cache_file = cache_dir / f"{safe_name}.json"
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(payload_data, f, indent=2, default=str)
    except Exception as ex:
        logger.warning(f"Failed to write on-disk endpoint cache for '{clean_key}': {ex}")


class ApiResponseEndpointRecipe(BaseRecipe):
    recipe_id = "api_response_endpoint"
    name = "API Response Endpoint"
    version = "1.0.0"
    category = "ingestion"
    group = "Ingestion & Orchestration"
    subgroup = "Output"
    description = (
        "Exposes data from the upstream node as a live HTTP REST endpoint that returns "
        "structured JSON payloads for frontend UI apps, dashboard widgets, webhooks, or external integrations."
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
            description="Dataset or predictions from upstream node to expose via HTTP endpoint"
        )
    ]

    outputs = [
        RecipePort(
            id="output",
            label="API Spec & Preview",
            type="dataframe",
            required=False,
            description="Summary table of endpoint URLs, methods, formats, and sample JSON payload"
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "validate_columns": False,
            "x-no-column-validation": True,
            "properties": {
                "endpoint_path": {
                    "type": "string",
                    "title": "Endpoint Path Slug",
                    "default": "flow-data",
                    "description": "Unique URL path identifier (e.g. 'flow-data' or 'customer-scores').",
                    "input_type": "text"
                },
                "http_method": {
                    "type": "string",
                    "title": "Allowed HTTP Method",
                    "enum": [
                        "Both (GET & POST)",
                        "GET",
                        "POST"
                    ],
                    "default": "Both (GET & POST)",
                    "description": "Select which HTTP method is accepted by this endpoint: GET, POST, or both.",
                    "input_type": "dropdown"
                },
                "response_format": {
                    "type": "string",
                    "title": "JSON Response Format",
                    "enum": [
                        "records",
                        "object",
                        "single_record",
                        "split"
                    ],
                    "default": "records",
                    "description": "'records': Array of JSON objects [{}]. 'object': Envelope {'status', 'data', 'count'}. 'single_record': First row as object. 'split': {'columns', 'data'}.",
                    "input_type": "dropdown"
                },
                "record_limit": {
                    "type": "integer",
                    "title": "Default Record Limit",
                    "default": 100,
                    "minimum": 1,
                    "maximum": 10000,
                    "description": "Maximum number of records to return per request (can be overridden with ?limit=N).",
                    "input_type": "number"
                },
                "include_metadata": {
                    "type": "boolean",
                    "title": "Include Response Metadata",
                    "default": True,
                    "description": "Includes record count, execution timestamp, and column types in response envelope.",
                    "input_type": "checkbox"
                },
                "http_status_code": {
                    "type": "integer",
                    "title": "HTTP Status Code",
                    "enum": [200, 201, 202],
                    "default": 200,
                    "description": "HTTP status code returned to the client.",
                    "input_type": "dropdown"
                },
                "allow_cors": {
                    "type": "boolean",
                    "title": "Enable CORS Headers",
                    "default": True,
                    "description": "Sets 'Access-Control-Allow-Origin: *' so web browsers and frontend widgets can call it directly.",
                    "input_type": "checkbox"
                },
                "auth_mode": {
                    "type": "string",
                    "title": "Authentication Requirement",
                    "enum": ["None (Public)", "Bearer Token / Session", "API Key"],
                    "default": "None (Public)",
                    "description": "Access control requirement for external callers.",
                    "input_type": "dropdown"
                },
                "api_key": {
                    "type": "string",
                    "title": "API Secret Key",
                    "default": generate_unique_api_key(),
                    "description": "Unique secret key required when Authentication Requirement is set to 'API Key'. Callers provide this via 'X-API-Key' header.",
                    "input_type": "password"
                },
                "manual_session_id": {
                    "type": "string",
                    "title": "Session ID (Optional Override)",
                    "default": "",
                    "description": "Optional session ID for URL testing. If left blank, your active logged-in session ID will be used automatically.",
                    "input_type": "text"
                },
                "cache_duration_seconds": {
                    "type": "integer",
                    "title": "Cache Duration (Seconds)",
                    "default": 300,
                    "minimum": 0,
                    "maximum": 86400,
                    "description": "Number of seconds responses are served from cache before requiring live pipeline refresh.",
                    "input_type": "number"
                }
            }
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        # 1. Retrieve upstream DataFrame
        df = inputs.get("dataframe")
        if df is None:
            df = inputs.get("input")
        if df is None and context and isinstance(context, dict):
            df = context.get("dataframe")

        if df is None:
            # Fallback for isolated standalone node canvas test
            df = pd.DataFrame([
                {"id": 1, "status": "active", "label": "Sample Flow Output 1", "score": 0.95},
                {"id": 2, "status": "active", "label": "Sample Flow Output 2", "score": 0.88},
                {"id": 3, "status": "pending", "label": "Sample Flow Output 3", "score": 0.72},
            ])

        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        orig_rows = len(df)

        # 2. Parse configuration
        endpoint_path = str(config.get("endpoint_path", "flow-data")).strip().strip("/")
        if not endpoint_path:
            endpoint_path = "flow-data"

        response_format = str(config.get("response_format", "records")).lower()
        record_limit = int(config.get("record_limit", 100))
        include_metadata = bool(config.get("include_metadata", True))
        status_code = int(config.get("http_status_code", 200))
        allow_cors = bool(config.get("allow_cors", True))
        auth_mode = str(config.get("auth_mode", "None (Public)"))
        raw_key = str(config.get("api_key", "")).strip()
        if not raw_key or raw_key == "crt-sec-9a8b7c6d5e":
            api_key = generate_unique_api_key()
        else:
            api_key = raw_key
        http_method = str(config.get("http_method", "Both (GET & POST)")).strip()
        cache_duration = int(config.get("cache_duration_seconds", 300))

        # Context-aware resolution (workflow, execution, session)
        workflow_id = None
        execution_id = None
        session_id = None

        if context and isinstance(context, dict):
            workflow_id = context.get("workflow_id")
            execution_id = context.get("execution_id")
            session_id = context.get("session_id") or context.get("auth_session_id")

        try:
            from backend.app.core.tenant_context import (
                get_current_workflow_id,
                get_current_execution_id,
                get_current_session_id,
            )
            workflow_id = workflow_id or get_current_workflow_id()
            execution_id = execution_id or get_current_execution_id()
            session_id = session_id or get_current_session_id()
        except Exception:
            pass

        # Allow explicit manual session override if provided in node config
        manual_sid = str(config.get("manual_session_id", "")).strip()
        if manual_sid:
            session_id = manual_sid

        # Apply record limit
        sliced_df = df.head(record_limit) if record_limit > 0 else df

        # Sanitize DataFrame for clean JSON serialization
        clean_df = sliced_df.replace({np.nan: None})

        # 3. Format JSON payload according to format selection
        raw_records = clean_df.to_dict(orient="records")
        for r in raw_records:
            for k, v in list(r.items()):
                if hasattr(v, "isoformat"):
                    r[k] = v.isoformat()
                elif isinstance(v, (np.integer, np.floating)):
                    r[k] = v.item()

        if response_format == "single_record":
            data_payload: Any = raw_records[0] if raw_records else {}
        elif response_format == "split":
            data_payload = {
                "columns": list(clean_df.columns),
                "data": clean_df.values.tolist()
            }
        elif response_format == "object":
            data_payload = {
                "status": "success",
                "count": len(raw_records),
                "total_records": orig_rows,
                "data": raw_records
            }
        else:  # "records"
            data_payload = raw_records

        # Full envelope for endpoint caching
        endpoint_envelope = {
            "endpoint_path": endpoint_path,
            "http_method": http_method,
            "status_code": status_code,
            "format": response_format,
            "allow_cors": allow_cors,
            "auth_mode": auth_mode,
            "api_key": api_key,
            "session_id": session_id,
            "auth_session_id": session_id,
            "cache_duration_seconds": cache_duration,
            "updated_at": time.time(),
            "columns": list(df.columns),
            "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "total_records": orig_rows,
            "returned_records": len(raw_records),
            "data": data_payload
        }

        # 4. Cache payload for live endpoint retrieval
        store_cached_endpoint_payload(endpoint_path, endpoint_envelope)
        if workflow_id:
            composite_key = f"{workflow_id}:{endpoint_path}"
            store_cached_endpoint_payload(composite_key, endpoint_envelope)
            store_cached_endpoint_payload(str(workflow_id), endpoint_envelope)

        # Base Domains
        backend_base = getattr(settings, "BACKEND_PUBLIC_URL", "https://ml-crt.aiiotgeeks.com").rstrip("/")
        frontend_base = getattr(settings, "FRONTEND_PUBLIC_URL", "https://crt.aiiotgeeks.com").rstrip("/")

        # 5. Build Live Clickable URLs (Clean Production URLs — No Secrets in Query Parameters)
        api_v1_path = f"/api/v1/workflows/endpoints/{endpoint_path}"
        backend_live_url = f"{backend_base}{api_v1_path}"

        # Frontend Canvas URL with resolved workflow_id and executionId
        wf_slug = str(workflow_id).strip() if workflow_id else "default-flow"
        if execution_id:
            frontend_canvas_url = f"{frontend_base}/workflows/canvas?workflow_id={wf_slug}&executionId={execution_id}"
        else:
            frontend_canvas_url = f"{frontend_base}/workflows/canvas?workflow_id={wf_slug}"

        # Clean Production Spec Rows
        spec_rows = [
            {"Property": "Endpoint Path", "Configuration / Value": f"/{endpoint_path}", "Category": "Route"},
            {"Property": "Backend API URL (Clickable)", "Configuration / Value": backend_live_url, "Category": "Route"},
            {"Property": "Frontend Canvas URL", "Configuration / Value": frontend_canvas_url, "Category": "Route"},
            {"Property": "Allowed HTTP Method", "Configuration / Value": http_method, "Category": "Access"},
            {"Property": "Auth Requirement", "Configuration / Value": auth_mode, "Category": "Access"},
        ]

        if auth_mode == "Bearer Token / Session":
            spec_rows.append({
                "Property": "Header Usage",
                "Configuration / Value": "Authorization: Bearer <your_session_token> (or X-Session-Id: <token>)",
                "Category": "Access"
            })
        elif auth_mode == "API Key":
            spec_rows.append({
                "Property": "Header Usage",
                "Configuration / Value": "X-API-Key: <your_api_key>",
                "Category": "Access"
            })

        spec_rows.extend([
            {"Property": "Response Format", "Configuration / Value": response_format, "Category": "Payload"},
            {"Property": "HTTP Status Code", "Configuration / Value": str(status_code), "Category": "Payload"},
            {"Property": "Served Records", "Configuration / Value": f"{len(raw_records)} / {orig_rows}", "Category": "Payload"},
            {"Property": "CORS Enabled", "Configuration / Value": "Yes (Access-Control-Allow-Origin: *)" if allow_cors else "Disabled", "Category": "Access"},
            {"Property": "Cache Duration", "Configuration / Value": f"{cache_duration} seconds", "Category": "Caching"},
        ])

        spec_df = pd.DataFrame(spec_rows)

        # Sample snippet for curl / frontend integration
        verb = "POST" if http_method == "POST" else "GET"
        if auth_mode == "API Key":
            curl_example = f"curl -X {verb} \"{backend_live_url}\" -H \"X-API-Key: <your_api_key>\""
            fetch_example = f"fetch('{backend_live_url}', {{ method: '{verb}', headers: {{ 'X-API-Key': '<your_api_key>' }} }}).then(r => r.json()).then(console.log);"
        elif auth_mode == "Bearer Token / Session":
            curl_example = f"curl -X {verb} \"{backend_live_url}\" -H \"Authorization: Bearer <your_session_token>\""
            fetch_example = f"fetch('{backend_live_url}', {{ method: '{verb}', headers: {{ 'Authorization': 'Bearer <your_session_token>' }} }}).then(r => r.json()).then(console.log);"
        else:
            curl_example = f"curl -X {verb} \"{backend_live_url}?limit=10\""
            fetch_example = f"fetch('{backend_live_url}', {{ method: '{verb}' }}).then(r => r.json()).then(console.log);"

        return {
            "dataframe": spec_df,
            "json_payload": data_payload,
            "metrics": {
                "endpoint_path": endpoint_path,
                "http_method": http_method,
                "status_code": status_code,
                "total_upstream_rows": orig_rows,
                "served_rows": len(raw_records),
                "columns_count": len(df.columns),
                "format": response_format,
                "api_v1_url": backend_live_url,
                "auth_mode": auth_mode,
            },
            "output_summary": {
                "title": f"API Endpoint: /{endpoint_path}",
                "endpoint_url": backend_live_url,
                "frontend_canvas_url": frontend_canvas_url,
                "http_methods": http_method,
                "auth_mode": auth_mode,
                "curl_command": curl_example,
                "javascript_fetch": fetch_example,
                "preview_sample": raw_records[:3] if raw_records else []
            }
        }

    def to_code(self, config: Dict[str, Any]) -> str:
        ep = config.get("endpoint_path", "flow-data")
        fmt = config.get("response_format", "records")
        lim = config.get("record_limit", 100)
        return (
            f"# API Response Endpoint Recipe\n"
            f"# Serves upstream DataFrame as live JSON via REST\n"
            f"# Endpoint: GET /api/v1/workflows/endpoints/{ep}\n"
            f"# Frontend: GET /api/ui/flow/{{workflow_id}}/endpoints/{ep}\n"
            f"formatted_json = dataframe.head({lim}).to_dict(orient='{fmt if fmt in ['records', 'split'] else 'records'}')\n"
        )
