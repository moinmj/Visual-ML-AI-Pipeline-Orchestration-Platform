import pytest
import pandas as pd
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock
from fastapi.testclient import TestClient

from backend.app.recipes.ingestion.api_response_endpoint import (
    ApiResponseEndpointRecipe,
    get_cached_endpoint_payload,
    store_cached_endpoint_payload,
)
from backend.app.recipes.base.registry import recipe_registry
from backend.app.main import app
from backend.app.infrastructure.database.session import get_db

@asynccontextmanager
async def _dummy_lifespan(app):
    yield

app.router.lifespan_context = _dummy_lifespan

async def _mock_get_db():
    yield AsyncMock()

app.dependency_overrides[get_db] = _mock_get_db


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
    assert props["api_key"]["input_type"] == "text"
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

def test_http_endpoint_serving_via_test_client():
    """Verify live HTTP retrieval via TestClient for various endpoint URL patterns."""
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

    client = TestClient(app)

    # 1. Frontend Flow URL: /api/ui/flow/{workflow_id}/endpoints/{endpoint_path}
    r1 = client.get("/api/ui/flow/wf-prod-42/endpoints/widget-inventory")
    assert r1.status_code == 200
    assert r1.headers.get("access-control-allow-origin") == "*"
    assert len(r1.json()) == 3

    # 2. Frontend Flow URL default: /api/ui/flow/{workflow_id}/endpoint
    r2 = client.get("/api/ui/flow/wf-prod-42/endpoint")
    assert r2.status_code == 200
    assert len(r2.json()) == 3

    # 3. Standard API URL: /api/v1/workflows/endpoints/{endpoint_path}
    r3 = client.get("/api/v1/workflows/endpoints/widget-inventory")
    assert r3.status_code == 200
    assert r3.json()[0]["product"] == "Widget A"

    # 4. Standard API URL with workflow ID: /api/v1/workflows/{workflow_id}/endpoints/{endpoint_path}
    r4 = client.get("/api/v1/workflows/wf-prod-42/endpoints/widget-inventory")
    assert r4.status_code == 200
    assert len(r4.json()) == 3

    # 5. Direct slug: /endpoints/{endpoint_path}
    r5 = client.get("/endpoints/widget-inventory")
    assert r5.status_code == 200

    # 6. Pagination query params: ?limit=1&offset=1
    r6 = client.get("/api/ui/flow/wf-prod-42/endpoints/widget-inventory?limit=1&offset=1")
    assert r6.status_code == 200
    data6 = r6.json()
    assert len(data6) == 1
    assert data6[0]["product"] == "Widget B"

    # 7. Format override via query param: ?format=object
    r7 = client.get("/api/ui/flow/wf-prod-42/endpoints/widget-inventory?format=object")
    assert r7.status_code == 200
    data7 = r7.json()
    assert isinstance(data7, dict)
    assert data7["status"] == "success"
    assert data7["count"] == 3

    # 8. CORS OPTIONS preflight
    r8 = client.options("/api/ui/flow/wf-prod-42/endpoints/widget-inventory")
    assert r8.status_code == 204
    assert r8.headers.get("access-control-allow-origin") == "*"
    assert "GET" in r8.headers.get("access-control-allow-methods", "")


def test_http_endpoint_auth_enforcement():
    """Verify authentication modes (API Key and Session) are properly enforced."""
    # Seed API Key protected payload
    apiKey_envelope = {
        "endpoint_path": "secure-data",
        "status_code": 200,
        "format": "records",
        "allow_cors": True,
        "auth_mode": "API Key",
        "cache_duration_seconds": 60,
        "total_records": 1,
        "data": [{"secret": "xyz123"}],
    }
    store_cached_endpoint_payload("secure-data", apiKey_envelope)

    client = TestClient(app)

    # Calling without API key returns 401
    r_unauth = client.get("/endpoints/secure-data")
    assert r_unauth.status_code == 401

    # Calling with valid X-API-Key header returns 200
    r_auth = client.get("/endpoints/secure-data", headers={"X-API-Key": "my-secret-key"})
    assert r_auth.status_code == 200
    assert r_auth.json()[0]["secret"] == "xyz123"

    # Calling with ?api_key= query param returns 200
    r_param = client.get("/endpoints/secure-data?api_key=my-secret-key")
    assert r_param.status_code == 200
    assert r_param.json()[0]["secret"] == "xyz123"


def test_http_endpoint_404_on_missing():
    """Verify clean 404 response when an endpoint has not been created or executed."""
    client = TestClient(app)
    r = client.get("/endpoints/non-existent-endpoint-slug-999")
    assert r.status_code == 404
    assert "not found" in r.json()["detail"].lower()


def test_http_endpoint_method_restrictions():
    """Verify HTTP method restriction: GET-only, POST-only, and Both."""
    client = TestClient(app)

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

    r_get = client.get("/endpoints/get-only-endpoint")
    assert r_get.status_code == 200
    assert r_get.json()[0]["type"] == "get_only"

    r_post = client.post("/endpoints/get-only-endpoint")
    assert r_post.status_code == 405
    assert "Method Not Allowed" in r_post.json()["detail"]

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

    r_get_fail = client.get("/endpoints/post-only-endpoint")
    assert r_get_fail.status_code == 405
    assert "Method Not Allowed" in r_get_fail.json()["detail"]

    r_post_ok = client.post("/endpoints/post-only-endpoint")
    assert r_post_ok.status_code == 200
    assert r_post_ok.json()[0]["type"] == "post_only"

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

    assert client.get("/endpoints/both-methods-endpoint").status_code == 200
    assert client.post("/endpoints/both-methods-endpoint").status_code == 200


def test_http_endpoint_custom_api_key_forbidden():
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

    client = TestClient(app)

    # 1. Missing key -> 401
    r_no_key = client.get("/endpoints/custom-secret-endpoint")
    assert r_no_key.status_code == 401

    # 2. Wrong key in header -> 403 Forbidden
    r_wrong_header = client.get("/endpoints/custom-secret-endpoint", headers={"X-API-Key": "incorrect-secret"})
    assert r_wrong_header.status_code == 403
    assert "invalid" in r_wrong_header.json()["detail"].lower()

    # 3. Wrong key in param -> 403 Forbidden
    r_wrong_param = client.get("/endpoints/custom-secret-endpoint?api_key=bad-key")
    assert r_wrong_param.status_code == 403
    assert "invalid" in r_wrong_param.json()["detail"].lower()

    # 4. Correct key in header -> 200 OK
    r_ok_header = client.get("/endpoints/custom-secret-endpoint", headers={"X-API-Key": "crt-sec-9a8b7c6d5e"})
    assert r_ok_header.status_code == 200
    assert r_ok_header.json()[0]["status"] == "confidential"

    # 5. Correct key in param -> 200 OK
    r_ok_param = client.get("/endpoints/custom-secret-endpoint?api_key=crt-sec-9a8b7c6d5e")
    assert r_ok_param.status_code == 200
    assert r_ok_param.json()[0]["status"] == "confidential"
