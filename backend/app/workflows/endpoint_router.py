"""
Flow API Endpoints Router.

Provides live HTTP REST endpoints for the 'API Response Endpoint' recipe.
Supports public or authenticated access, dynamic JSON response formatting,
pagination (limit & offset), caching, CORS, and flexible URL patterns:
  - Frontend flow URLs: /api/ui/flow/{workflow_id}/endpoints/{endpoint_path}
                        /api/ui/flow/{workflow_id}/endpoint
                        /ui/flow/{workflow_id}/endpoint
  - Standard REST URLs: /api/v1/workflows/endpoints/{endpoint_path}
                        /api/v1/workflows/{workflow_id}/endpoints/{endpoint_path}
                        /api/v1/workflows/{workflow_id}/endpoint
"""
import json
import time
from typing import Dict, Any, List, Optional, Union
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from fastapi.responses import JSONResponse, Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.infrastructure.database.session import get_db
from backend.app.workflows.models import Workflow
from backend.app.recipes.ingestion.api_response_endpoint import (
    get_cached_endpoint_payload,
    store_cached_endpoint_payload,
)


endpoint_router = APIRouter(tags=["Flow API Endpoints"])


async def _check_endpoint_auth(request: Request, auth_mode: str, expected_api_key: Optional[str] = None) -> None:
    """
    Validates authentication against the configured requirement of the node.
    - 'None (Public)': open to all clients.
    - 'Bearer Token / Session': requires valid session cookie, header, or bearer token.
    - 'API Key': requires X-API-Key header or ?api_key= parameter.
    """
    mode = (auth_mode or "None (Public)").strip()

    if mode == "None (Public)":
        return

    if mode == "API Key":
        api_key = (
            request.headers.get("X-API-Key")
            or request.headers.get("x-api-key")
            or request.query_params.get("api_key")
            or request.query_params.get("key")
        )
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required: Provide a valid API key via 'X-API-Key' header or '?api_key=' parameter.",
            )
        if expected_api_key and expected_api_key.strip() and api_key.strip() != expected_api_key.strip():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden: The provided API key is invalid.",
            )
        return

    if mode == "Bearer Token / Session":
        from backend.app.core.security import _extract_sid, resolve_session
        sid, _ = _extract_sid(request)
        if not sid:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required: Provide a valid session cookie ('sid'), 'X-Session-Id' header, or 'Authorization: Bearer <token>'.",
            )
        try:
            await resolve_session(sid)
        except Exception as ex:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid or expired authentication session: {ex}",
            )


def _format_payload(
    raw_data: Any,
    effective_format: str,
    limit: Optional[int],
    offset: int,
    total_records: int,
    cached_payload: Dict[str, Any]
) -> Any:
    """Applies pagination and response format transformation."""
    data = raw_data

    # If data is a list of records
    if isinstance(data, list):
        if offset > 0:
            data = data[offset:]
        if limit is not None and limit > 0:
            data = data[:limit]

        if effective_format == "single_record":
            return data[0] if data else {}
        elif effective_format == "split":
            cols = cached_payload.get("columns", list(data[0].keys()) if data else [])
            rows = [[row.get(c) for c in cols] for row in data]
            return {"columns": cols, "data": rows}
        elif effective_format == "object":
            return {
                "status": "success",
                "count": len(data),
                "total_records": total_records,
                "offset": offset,
                "data": data,
            }
        else:  # "records"
            return data

    # If data is an envelope dict {"status": ..., "data": [...]}
    if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
        inner_list = data["data"]
        if offset > 0:
            inner_list = inner_list[offset:]
        if limit is not None and limit > 0:
            inner_list = inner_list[:limit]

        if effective_format == "single_record":
            return inner_list[0] if inner_list else {}
        elif effective_format == "records":
            return inner_list
        elif effective_format == "split":
            cols = cached_payload.get("columns", list(inner_list[0].keys()) if inner_list else [])
            rows = [[row.get(c) for c in cols] for row in inner_list]
            return {"columns": cols, "data": rows}
        else:
            return {
                **data,
                "count": len(inner_list),
                "offset": offset,
                "data": inner_list,
            }

    return data


async def _resolve_endpoint_response(
    request: Request,
    endpoint_path: Optional[str] = None,
    workflow_id: Optional[str] = None,
    limit: Optional[int] = None,
    offset: int = 0,
    response_format: Optional[str] = None,
    refresh: bool = False,
    download: bool = False,
    db: Optional[AsyncSession] = None,
) -> Response:
    """Core handler that resolves, authenticates, formats, and serves endpoint data."""
    clean_slug = (endpoint_path or "").strip().strip("/").lower()
    clean_wfid = (workflow_id or "").strip()

    cached: Optional[Dict[str, Any]] = None

    # Step 1: Cache retrieval (unless refresh requested)
    if not refresh:
        if clean_wfid and clean_slug:
            cached = get_cached_endpoint_payload(f"{clean_wfid}:{clean_slug}")
        if not cached and clean_slug:
            cached = get_cached_endpoint_payload(clean_slug)
        if not cached and clean_wfid:
            cached = get_cached_endpoint_payload(clean_wfid)

    # Step 2: Dynamic workflow pipeline resolution on cache-miss or refresh
    if cached is None and clean_wfid and db is not None:
        try:
            stmt = select(Workflow).where(Workflow.id == clean_wfid)
            res = await db.execute(stmt)
            wf = res.scalar_one_or_none()
            if wf:
                nodes = wf.nodes or []
                # Find an api_response_endpoint node
                target_node = None
                for n in nodes:
                    rec_id = n.get("recipe_id") or n.get("type")
                    if rec_id in [
                        "api_response_endpoint",
                        "api_endpoint",
                        "flow_endpoint",
                        "api_response",
                        "webhook_response",
                    ]:
                        cfg = n.get("config", {})
                        cfg_slug = str(cfg.get("endpoint_path", "")).strip().strip("/").lower()
                        if not clean_slug or cfg_slug == clean_slug:
                            target_node = n
                            break

                if target_node:
                    from backend.app.engine.execution.step_executor import NodeStepExecutor
                    await NodeStepExecutor.execute_node_step(
                        target_node_id=target_node["id"],
                        nodes=nodes,
                        edges=wf.edges or [],
                        node_configs=wf.node_configs or {},
                        session_id=clean_wfid,
                        sample_size=limit or 500,
                        auto_run_ancestors=True,
                        db=db,
                    )
                    # Recheck cache after execution
                    if clean_slug:
                        cached = get_cached_endpoint_payload(f"{clean_wfid}:{clean_slug}") or get_cached_endpoint_payload(clean_slug)
                    if not cached:
                        cached = get_cached_endpoint_payload(clean_wfid)
        except Exception as ex:
            logger.warning(f"Error during dynamic endpoint pipeline execution: {ex}")

    # Fallback: check cache one last time
    if cached is None:
        if clean_slug:
            cached = get_cached_endpoint_payload(clean_slug)
        if not cached and clean_wfid:
            cached = get_cached_endpoint_payload(clean_wfid)

    if cached is None:
        identifier = f"'{clean_slug}'" if clean_slug else f"for workflow '{clean_wfid}'"
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"API Endpoint {identifier} not found or has not been executed yet. "
                "Ensure your workflow contains an 'API Response Endpoint' node and execute or test it."
            ),
        )

    # Step 2.5: Validate HTTP method (GET vs POST)
    allowed_method = str(cached.get("http_method", "Both (GET & POST)")).strip().upper()
    req_method = request.method.upper()
    if req_method not in ("OPTIONS", "HEAD"):
        if allowed_method == "GET" and req_method != "GET":
            raise HTTPException(
                status_code=status.HTTP_405_METHOD_NOT_ALLOWED,
                detail=f"Method Not Allowed: Endpoint is configured to accept GET requests only, but received {req_method}.",
                headers={"Allow": "GET"},
            )
        elif allowed_method == "POST" and req_method != "POST":
            raise HTTPException(
                status_code=status.HTTP_405_METHOD_NOT_ALLOWED,
                detail=f"Method Not Allowed: Endpoint is configured to accept POST requests only, but received {req_method}.",
                headers={"Allow": "POST"},
            )

    # Step 3: Validate authentication
    auth_mode = cached.get("auth_mode", "None (Public)")
    expected_api_key = cached.get("api_key")
    await _check_endpoint_auth(request, auth_mode, expected_api_key=expected_api_key)

    # Step 4: Extract and format payload
    raw_data = cached.get("data")
    effective_format = (response_format or cached.get("format", "records")).lower()
    total_records = cached.get("total_records", len(raw_data) if isinstance(raw_data, list) else 1)
    status_code = int(cached.get("status_code", 200))
    allow_cors = bool(cached.get("allow_cors", True))
    cache_duration = int(cached.get("cache_duration_seconds", 300))

    final_payload = _format_payload(
        raw_data=raw_data,
        effective_format=effective_format,
        limit=limit,
        offset=offset,
        total_records=total_records,
        cached_payload=cached,
    )

    # Serialize JSON
    content_str = json.dumps(final_payload, default=str)

    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "X-Total-Count": str(total_records),
        "X-Returned-Count": str(len(final_payload) if isinstance(final_payload, list) else 1),
    }

    if cache_duration > 0:
        headers["Cache-Control"] = f"public, max-age={cache_duration}"
    else:
        headers["Cache-Control"] = "no-cache, no-store, must-revalidate"

    if allow_cors:
        headers["Access-Control-Allow-Origin"] = "*"
        headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS, HEAD"
        headers["Access-Control-Allow-Headers"] = "*"

    if download:
        filename = f"{clean_slug or clean_wfid or 'flow-data'}.json"
        headers["Content-Disposition"] = f'attachment; filename="{filename}"'

    return Response(
        content=content_str,
        status_code=status_code,
        headers=headers,
        media_type="application/json",
    )


# ==============================================================================
# ROUTE DEFINITIONS
# ==============================================================================

# ── 1. Frontend Flow URL Patterns (/api/ui/flow/..., /ui/flow/...) ───────────

@endpoint_router.get("/ui/flow/{workflow_id}/endpoints/{endpoint_path}")
@endpoint_router.post("/ui/flow/{workflow_id}/endpoints/{endpoint_path}")
@endpoint_router.get("/api/ui/flow/{workflow_id}/endpoints/{endpoint_path}")
@endpoint_router.post("/api/ui/flow/{workflow_id}/endpoints/{endpoint_path}")
@endpoint_router.get("/api/v1/ui/flow/{workflow_id}/endpoints/{endpoint_path}")
@endpoint_router.post("/api/v1/ui/flow/{workflow_id}/endpoints/{endpoint_path}")
async def serve_flow_endpoint_with_path(
    request: Request,
    workflow_id: str,
    endpoint_path: str,
    limit: Optional[int] = Query(None, ge=1, le=10000, description="Max records to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    format: Optional[str] = Query(None, description="Override format: records, object, single_record, split"),
    refresh: bool = Query(False, description="Force refresh from upstream pipeline"),
    download: bool = Query(False, description="Download as JSON file"),
    db: AsyncSession = Depends(get_db),
):
    """Serves the flow data endpoint for a specific workflow and path slug."""
    return await _resolve_endpoint_response(
        request=request,
        endpoint_path=endpoint_path,
        workflow_id=workflow_id,
        limit=limit,
        offset=offset,
        response_format=format,
        refresh=refresh,
        download=download,
        db=db,
    )


@endpoint_router.get("/ui/flow/{workflow_id}/endpoint")
@endpoint_router.post("/ui/flow/{workflow_id}/endpoint")
@endpoint_router.get("/api/ui/flow/{workflow_id}/endpoint")
@endpoint_router.post("/api/ui/flow/{workflow_id}/endpoint")
@endpoint_router.get("/api/v1/ui/flow/{workflow_id}/endpoint")
@endpoint_router.post("/api/v1/ui/flow/{workflow_id}/endpoint")
async def serve_flow_endpoint_default(
    request: Request,
    workflow_id: str,
    limit: Optional[int] = Query(None, ge=1, le=10000, description="Max records to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    format: Optional[str] = Query(None, description="Override format: records, object, single_record, split"),
    refresh: bool = Query(False, description="Force refresh from upstream pipeline"),
    download: bool = Query(False, description="Download as JSON file"),
    db: AsyncSession = Depends(get_db),
):
    """Serves the default flow data endpoint for a specific workflow."""
    return await _resolve_endpoint_response(
        request=request,
        endpoint_path=None,
        workflow_id=workflow_id,
        limit=limit,
        offset=offset,
        response_format=format,
        refresh=refresh,
        download=download,
        db=db,
    )


# ── 2. Standard API & Workflow Route Patterns ────────────────────────────────

@endpoint_router.get("/workflows/endpoints/{endpoint_path}")
@endpoint_router.post("/workflows/endpoints/{endpoint_path}")
@endpoint_router.get("/endpoints/{endpoint_path}")
@endpoint_router.post("/endpoints/{endpoint_path}")
async def serve_endpoint_by_slug(
    request: Request,
    endpoint_path: str,
    limit: Optional[int] = Query(None, ge=1, le=10000, description="Max records to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    format: Optional[str] = Query(None, description="Override format: records, object, single_record, split"),
    refresh: bool = Query(False, description="Force refresh from upstream pipeline"),
    download: bool = Query(False, description="Download as JSON file"),
    db: AsyncSession = Depends(get_db),
):
    """Serves an API endpoint by its path slug."""
    return await _resolve_endpoint_response(
        request=request,
        endpoint_path=endpoint_path,
        workflow_id=None,
        limit=limit,
        offset=offset,
        response_format=format,
        refresh=refresh,
        download=download,
        db=db,
    )


@endpoint_router.get("/workflows/{workflow_id}/endpoints/{endpoint_path}")
@endpoint_router.post("/workflows/{workflow_id}/endpoints/{endpoint_path}")
async def serve_workflow_endpoint_by_slug(
    request: Request,
    workflow_id: str,
    endpoint_path: str,
    limit: Optional[int] = Query(None, ge=1, le=10000, description="Max records to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    format: Optional[str] = Query(None, description="Override format: records, object, single_record, split"),
    refresh: bool = Query(False, description="Force refresh from upstream pipeline"),
    download: bool = Query(False, description="Download as JSON file"),
    db: AsyncSession = Depends(get_db),
):
    """Serves an API endpoint by workflow ID and path slug."""
    return await _resolve_endpoint_response(
        request=request,
        endpoint_path=endpoint_path,
        workflow_id=workflow_id,
        limit=limit,
        offset=offset,
        response_format=format,
        refresh=refresh,
        download=download,
        db=db,
    )


@endpoint_router.get("/workflows/{workflow_id}/endpoint")
@endpoint_router.post("/workflows/{workflow_id}/endpoint")
async def serve_workflow_default_endpoint(
    request: Request,
    workflow_id: str,
    limit: Optional[int] = Query(None, ge=1, le=10000, description="Max records to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    format: Optional[str] = Query(None, description="Override format: records, object, single_record, split"),
    refresh: bool = Query(False, description="Force refresh from upstream pipeline"),
    download: bool = Query(False, description="Download as JSON file"),
    db: AsyncSession = Depends(get_db),
):
    """Serves the primary API endpoint for a workflow."""
    return await _resolve_endpoint_response(
        request=request,
        endpoint_path=None,
        workflow_id=workflow_id,
        limit=limit,
        offset=offset,
        response_format=format,
        refresh=refresh,
        download=download,
        db=db,
    )


# ── 3. CORS Preflight Handlers ───────────────────────────────────────────────

@endpoint_router.options("/ui/flow/{workflow_id}/endpoints/{endpoint_path}")
@endpoint_router.options("/api/ui/flow/{workflow_id}/endpoints/{endpoint_path}")
@endpoint_router.options("/api/v1/ui/flow/{workflow_id}/endpoints/{endpoint_path}")
@endpoint_router.options("/ui/flow/{workflow_id}/endpoint")
@endpoint_router.options("/api/ui/flow/{workflow_id}/endpoint")
@endpoint_router.options("/api/v1/ui/flow/{workflow_id}/endpoint")
@endpoint_router.options("/workflows/endpoints/{endpoint_path}")
@endpoint_router.options("/endpoints/{endpoint_path}")
@endpoint_router.options("/workflows/{workflow_id}/endpoints/{endpoint_path}")
@endpoint_router.options("/workflows/{workflow_id}/endpoint")
async def endpoint_options_preflight():
    """Handles CORS preflight OPTIONS requests."""
    return Response(
        status_code=status.HTTP_204_NO_CONTENT,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS, HEAD",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Max-Age": "86400",
        },
    )
