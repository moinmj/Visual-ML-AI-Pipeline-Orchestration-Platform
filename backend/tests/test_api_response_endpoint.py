import pytest
import json
import pandas as pd
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock
from fastapi import Request, HTTPException

from backend.app.recipes.ingestion.api_response_endpoint import (
    ApiResponseEndpointRecipe,
    get_cached_endpoint_payload,
    store_cached_endpoint_payload,
    generate_unique_api_key,
)
from backend.app.recipes.base.registry import recipe_registry
from backend.app.workflows.endpoint_router import (
    endpoint_router,
    _resolve_endpoint_response,
)


def make_mock_request(method: str = "GET", headers: dict = None, query_params: dict = None) -> MagicMock:
    """Creates a mock Request for direct fast handler testing without server network overhead."""
    req = MagicMock(spec=Request)
    req.method = method.upper()
    hdr_dict = {k.lower(): v for k, v in (headers or {}).items()}
    req.headers = hdr_dict
    req.query_params = query_params or {}
    req.cookies = {}
    return req


# ─────────────────────────────────────────────────────────────────────────────
# 1. RECIPE METADATA & SCHEMA TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_api_endpoint_recipe_metadata():
    """Verify recipe registration, category, group, subgroup, and ports."""
    recipe = ApiResponseEndpointRecipe()
    assert recipe.recipe_id == "api_response_endpoint"
    assert recipe.name == "API Response Endpoint"
    assert recipe.category == "ingestion"
    assert recipe.group == "Ingestion & Orchestration"
    assert recipe.subgroup == "Output"

    # Input & Output Ports
    assert len(recipe.inputs) == 1
    assert recipe.inputs[0].id == "input"
    assert recipe.inputs[0].type == "dataframe"

    assert len(recipe.outputs) == 1
    assert recipe.outputs[0].id == "output"
    assert recipe.outputs[0].type == "dataframe"


def test_api_endpoint_schema_properties():
    """Verify parameters schema and ensure all properties define explicit input_type."""
    recipe = ApiResponseEndpointRecipe()
    schema = recipe.get_schema()
    props = schema["properties"]

    expected_keys = [
        "endpoint_path",
        "http_method",
        "response_format",
        "record_limit",
        "include_metadata",
        "http_status_code",
        "allow_cors",
        "auth_mode",
        "api_key",
        "cache_duration_seconds",
    ]
    for key in expected_keys:
        assert key in props, f"Missing expected schema property '{key}'"
        assert "input_type" in props[key], f"Property '{key}' missing input_type"

    assert props["endpoint_path"]["input_type"] == "text"
    assert props["http_method"]["input_type"] == "dropdown"
    assert props["response_format"]["input_type"] == "dropdown"
    assert props["api_key"]["input_type"] == "password"
    assert props["record_limit"]["input_type"] == "number"
    assert props["allow_cors"]["input_type"] == "checkbox"
    assert props["auth_mode"]["input_type"] == "dropdown"


def test_api_endpoint_registry_and_aliases():
    """Verify registration in global registry and all expected aliases."""
    r = recipe_registry.get("api_response_endpoint")
    assert r is not None
    assert r.recipe_id == "api_response_endpoint"

    for alias in ["api_endpoint", "webhook_response", "api_response", "flow_endpoint"]:
        ra = recipe_registry.get(alias)
        assert ra is not None
        assert ra.recipe_id == "api_response_endpoint"


# ─────────────────────────────────────────────────────────────────────────────
# 2. RECIPE EXECUTION & PAYLOAD FORMATS
# ─────────────────────────────────────────────────────────────────────────────

def test_api_endpoint_execution_records_format():
    """Verify execution with 'records' JSON format."""
    recipe = ApiResponseEndpointRecipe()
    sample_df = pd.DataFrame([
        {"id": 101, "customer": "Acme Corp", "mrr": 4500.0, "status": "active"},
        {"id": 102, "customer": "Globex Inc", "mrr": 1200.0, "status": "active"},
        {"id": 103, "customer": "Soylent Corp", "mrr": 890.0, "status": "churned"},
    ])

    res = recipe.execute(
        inputs={"dataframe": sample_df},
        config={
            "endpoint_path": "acme-customers",
            "response_format": "records",
            "record_limit": 50,
            "allow_cors": True,
            "http_status_code": 200,
        },
        context={"workflow_id": "wf-test-001"}
    )

    assert "dataframe" in res
    assert "json_payload" in res
    assert "metrics" in res
    assert "output_summary" in res

    spec_df = res["dataframe"]
    assert isinstance(spec_df, pd.DataFrame)
    assert len(spec_df) >= 8

    json_payload = res["json_payload"]
    assert isinstance(json_payload, list)
    assert len(json_payload) == 3
    assert json_payload[0]["customer"] == "Acme Corp"


def test_api_endpoint_execution_object_format():
    """Verify execution with envelope 'object' format."""
    recipe = ApiResponseEndpointRecipe()
    sample_df = pd.DataFrame([
        {"item": "Server", "latency_ms": 24},
        {"item": "Database", "latency_ms": 12},
    ])

    res = recipe.execute(
        inputs={"input": sample_df},
        config={
            "endpoint_path": "system-metrics",
            "response_format": "object",
            "include_metadata": True,
        }
    )

    payload = res["json_payload"]
    assert isinstance(payload, dict)
    assert payload["status"] == "success"
    assert payload["count"] == 2
    assert payload["total_records"] == 2
    assert len(payload["data"]) == 2


def test_api_endpoint_execution_single_record_and_split():
    """Verify single_record and split output formats."""
    recipe = ApiResponseEndpointRecipe()
    sample_df = pd.DataFrame([
        {"kpi": "Conversion Rate", "value": 0.042},
        {"kpi": "Bounce Rate", "value": 0.28},
    ])

    # Single Record
    res_single = recipe.execute(
        inputs={"dataframe": sample_df},
        config={"endpoint_path": "kpi-lead", "response_format": "single_record"}
    )
    assert isinstance(res_single["json_payload"], dict)
    assert res_single["json_payload"]["kpi"] == "Conversion Rate"

    # Split format
    res_split = recipe.execute(
        inputs={"dataframe": sample_df},
        config={"endpoint_path": "kpi-split", "response_format": "split"}
    )
    assert "columns" in res_split["json_payload"]
    assert "data" in res_split["json_payload"]
    assert res_split["json_payload"]["columns"] == ["kpi", "value"]


def test_api_endpoint_to_code():
    """Verify exportable Python code snippet generation."""
    recipe = ApiResponseEndpointRecipe()
    code = recipe.to_code({"endpoint_path": "custom-api", "response_format": "records", "record_limit": 25})
    assert "api_response_endpoint" in code.lower() or "endpoint" in code.lower()
    assert "custom-api" in code


# ─────────────────────────────────────────────────────────────────────────────
# 3. HTTP ENDPOINT ROUTER & URL FORMATS
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_http_endpoint_serving_via_test_client():
    """Verify live HTTP retrieval for various endpoint URL patterns."""
    # Seed cache
    sample_data = [
        {"id": 1, "product": "Widget A", "price": 19.99},
        {"id": 2, "product": "Widget B", "price": 29.99},
        {"id": 3, "product": "Widget C", "price": 39.99},
    ]
    envelope = {
        "endpoint_path": "widget-inventory",
        "status_code": 200,
        "format": "records",
        "allow_cors": True,
        "auth_mode": "None (Public)",
        "cache_duration_seconds": 300,
        "total_records": 3,
        "returned_records": 3,
        "data": sample_data,
        "columns": ["id", "product", "price"],
    }

    store_cached_endpoint_payload("widget-inventory", envelope)
    store_cached_endpoint_payload("wf-prod-42:widget-inventory", envelope)
    store_cached_endpoint_payload("wf-prod-42", envelope)

    # 1. Flow URL with workflow and slug
    r1 = await _resolve_endpoint_response(
        make_mock_request("GET"),
        endpoint_path="widget-inventory",
        workflow_id="wf-prod-42",
    )
    assert r1.status_code == 200
    assert r1.headers.get("access-control-allow-origin") == "*"
    data1 = json.loads(r1.body.decode("utf-8"))
    assert len(data1) == 3

    # 2. Flow URL default (by workflow_id alone)
    r2 = await _resolve_endpoint_response(
        make_mock_request("GET"),
        endpoint_path=None,
        workflow_id="wf-prod-42",
    )
    assert r2.status_code == 200
    data2 = json.loads(r2.body.decode("utf-8"))
    assert len(data2) == 3

    # 3. Standard API URL (by endpoint_path slug alone)
    r3 = await _resolve_endpoint_response(
        make_mock_request("GET"),
        endpoint_path="widget-inventory",
        workflow_id=None,
    )
    assert r3.status_code == 200
    data3 = json.loads(r3.body.decode("utf-8"))
    assert data3[0]["product"] == "Widget A"

    # 4. Pagination query params: limit=1, offset=1
    r4 = await _resolve_endpoint_response(
        make_mock_request("GET"),
        endpoint_path="widget-inventory",
        workflow_id="wf-prod-42",
        limit=1,
        offset=1,
    )
    assert r4.status_code == 200
    data4 = json.loads(r4.body.decode("utf-8"))
    assert len(data4) == 1
    assert data4[0]["product"] == "Widget B"

    # 5. Format override: format="object"
    r5 = await _resolve_endpoint_response(
        make_mock_request("GET"),
        endpoint_path="widget-inventory",
        workflow_id="wf-prod-42",
        response_format="object",
    )
    assert r5.status_code == 200
    data5 = json.loads(r5.body.decode("utf-8"))
    assert isinstance(data5, dict)
    assert data5["status"] == "success"
    assert data5["count"] == 3


@pytest.mark.asyncio
async def test_http_endpoint_auth_enforcement():
    """Verify authentication modes (API Key and Session) are properly enforced."""
    # Seed API Key protected payload
    apiKey_envelope = {
        "endpoint_path": "secure-data",
        "status_code": 200,
        "format": "records",
        "allow_cors": True,
        "auth_mode": "API Key",
        "api_key": "my-secret-key",
        "cache_duration_seconds": 60,
        "total_records": 1,
        "data": [{"secret": "xyz123"}],
    }
    store_cached_endpoint_payload("secure-data", apiKey_envelope)

    # Calling without API key returns 401
    with pytest.raises(HTTPException) as exc1:
        await _resolve_endpoint_response(make_mock_request("GET"), endpoint_path="secure-data")
    assert exc1.value.status_code == 401

    # Calling with valid X-API-Key header returns 200
    req_auth = make_mock_request("GET", headers={"X-API-Key": "my-secret-key"})
    r_auth = await _resolve_endpoint_response(req_auth, endpoint_path="secure-data")
    assert r_auth.status_code == 200
    assert json.loads(r_auth.body.decode("utf-8"))[0]["secret"] == "xyz123"

    # Calling with ?api_key= query param returns 200
    req_param = make_mock_request("GET", query_params={"api_key": "my-secret-key"})
    r_param = await _resolve_endpoint_response(req_param, endpoint_path="secure-data")
    assert r_param.status_code == 200
    assert json.loads(r_param.body.decode("utf-8"))[0]["secret"] == "xyz123"

    # Seed Session protected payload
    session_envelope = {
        "endpoint_path": "session-protected-data",
        "status_code": 200,
        "format": "records",
        "allow_cors": True,
        "auth_mode": "Bearer Token / Session",
        "session_id": "test-valid-session-sid-987",
        "cache_duration_seconds": 60,
        "total_records": 1,
        "data": [{"session_content": "success"}],
    }
    store_cached_endpoint_payload("session-protected-data", session_envelope)

    # Calling without session header returns 401
    with pytest.raises(HTTPException) as exc2:
        await _resolve_endpoint_response(make_mock_request("GET"), endpoint_path="session-protected-data")
    assert exc2.value.status_code == 401

    # Calling with Authorization: Bearer <session_id> returns 200
    req_bearer = make_mock_request("GET", headers={"Authorization": "Bearer test-valid-session-sid-987"})
    r_sess_bearer = await _resolve_endpoint_response(req_bearer, endpoint_path="session-protected-data")
    assert r_sess_bearer.status_code == 200
    assert json.loads(r_sess_bearer.body.decode("utf-8"))[0]["session_content"] == "success"

    # Calling with X-Session-Id header returns 200
    req_hdr = make_mock_request("GET", headers={"X-Session-Id": "test-valid-session-sid-987"})
    r_sess_hdr = await _resolve_endpoint_response(req_hdr, endpoint_path="session-protected-data")
    assert r_sess_hdr.status_code == 200
    assert json.loads(r_sess_hdr.body.decode("utf-8"))[0]["session_content"] == "success"


@pytest.mark.asyncio
async def test_http_endpoint_404_on_missing():
    """Verify clean 404 response when an endpoint has not been created or executed."""
    with pytest.raises(HTTPException) as exc:
        await _resolve_endpoint_response(make_mock_request("GET"), endpoint_path="non-existent-endpoint-slug-999")
    assert exc.value.status_code == 404
    assert "not found" in exc.value.detail.lower()


@pytest.mark.asyncio
async def test_http_endpoint_method_restrictions():
    """Verify HTTP method restriction: GET-only, POST-only, and Both."""
    # 1. GET only
    get_envelope = {
        "endpoint_path": "get-only-endpoint",
        "http_method": "GET",
        "status_code": 200,
        "format": "records",
        "allow_cors": True,
        "auth_mode": "None (Public)",
        "cache_duration_seconds": 60,
        "total_records": 1,
        "data": [{"type": "get_only"}],
    }
    store_cached_endpoint_payload("get-only-endpoint", get_envelope)

    r_get = await _resolve_endpoint_response(make_mock_request("GET"), endpoint_path="get-only-endpoint")
    assert r_get.status_code == 200
    assert json.loads(r_get.body.decode("utf-8"))[0]["type"] == "get_only"

    with pytest.raises(HTTPException) as exc_get:
        await _resolve_endpoint_response(make_mock_request("POST"), endpoint_path="get-only-endpoint")
    assert exc_get.value.status_code == 405
    assert "Method Not Allowed" in exc_get.value.detail

    # 2. POST only
    post_envelope = {
        "endpoint_path": "post-only-endpoint",
        "http_method": "POST",
        "status_code": 200,
        "format": "records",
        "allow_cors": True,
        "auth_mode": "None (Public)",
        "cache_duration_seconds": 60,
        "total_records": 1,
        "data": [{"type": "post_only"}],
    }
    store_cached_endpoint_payload("post-only-endpoint", post_envelope)

    with pytest.raises(HTTPException) as exc_post:
        await _resolve_endpoint_response(make_mock_request("GET"), endpoint_path="post-only-endpoint")
    assert exc_post.value.status_code == 405
    assert "Method Not Allowed" in exc_post.value.detail

    r_post_ok = await _resolve_endpoint_response(make_mock_request("POST"), endpoint_path="post-only-endpoint")
    assert r_post_ok.status_code == 200
    assert json.loads(r_post_ok.body.decode("utf-8"))[0]["type"] == "post_only"

    # 3. Both (GET & POST)
    both_envelope = {
        "endpoint_path": "both-methods-endpoint",
        "http_method": "Both (GET & POST)",
        "status_code": 200,
        "format": "records",
        "allow_cors": True,
        "auth_mode": "None (Public)",
        "cache_duration_seconds": 60,
        "total_records": 1,
        "data": [{"type": "both"}],
    }
    store_cached_endpoint_payload("both-methods-endpoint", both_envelope)

    r_both_get = await _resolve_endpoint_response(make_mock_request("GET"), endpoint_path="both-methods-endpoint")
    assert r_both_get.status_code == 200
    r_both_post = await _resolve_endpoint_response(make_mock_request("POST"), endpoint_path="both-methods-endpoint")
    assert r_both_post.status_code == 200


@pytest.mark.asyncio
async def test_http_endpoint_custom_api_key_forbidden():
    """Verify that when a custom api_key is configured, wrong keys yield 403 Forbidden."""
    custom_envelope = {
        "endpoint_path": "custom-secret-endpoint",
        "http_method": "Both (GET & POST)",
        "status_code": 200,
        "format": "records",
        "allow_cors": True,
        "auth_mode": "API Key",
        "api_key": "crt-sec-9a8b7c6d5e",
        "cache_duration_seconds": 60,
        "total_records": 1,
        "data": [{"status": "confidential"}],
    }
    store_cached_endpoint_payload("custom-secret-endpoint", custom_envelope)

    # 1. Missing key -> 401
    with pytest.raises(HTTPException) as exc1:
        await _resolve_endpoint_response(make_mock_request("GET"), endpoint_path="custom-secret-endpoint")
    assert exc1.value.status_code == 401

    # 2. Wrong key in header -> 403 Forbidden
    with pytest.raises(HTTPException) as exc2:
        await _resolve_endpoint_response(
            make_mock_request("GET", headers={"X-API-Key": "incorrect-secret"}),
            endpoint_path="custom-secret-endpoint"
        )
    assert exc2.value.status_code == 403
    assert "invalid" in exc2.value.detail.lower()

    # 3. Wrong key in param -> 403 Forbidden
    with pytest.raises(HTTPException) as exc3:
        await _resolve_endpoint_response(
            make_mock_request("GET", query_params={"api_key": "bad-key"}),
            endpoint_path="custom-secret-endpoint"
        )
    assert exc3.value.status_code == 403
    assert "invalid" in exc3.value.detail.lower()

    # 4. Correct key in header -> 200 OK
    r_ok_header = await _resolve_endpoint_response(
        make_mock_request("GET", headers={"X-API-Key": "crt-sec-9a8b7c6d5e"}),
        endpoint_path="custom-secret-endpoint"
    )
    assert r_ok_header.status_code == 200
    assert json.loads(r_ok_header.body.decode("utf-8"))[0]["status"] == "confidential"

    # 5. Correct key in param -> 200 OK
    r_ok_param = await _resolve_endpoint_response(
        make_mock_request("GET", query_params={"api_key": "crt-sec-9a8b7c6d5e"}),
        endpoint_path="custom-secret-endpoint"
    )
    assert r_ok_param.status_code == 200
    assert json.loads(r_ok_param.body.decode("utf-8"))[0]["status"] == "confidential"
