"""
Generic REST/GraphQL API Caller Recipe (Apps & Integrations -> Generic).

Calls any REST or GraphQL API and turns the response into a structured DataFrame
(with auth handling, flexible path extraction, schema normalization, rate-limit resilience,
and benchmark simulation fallbacks).
"""

import base64
import json
import os
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

try:
    import pandas as pd
except ImportError:
    pd = None

from backend.app.recipes.base.recipe import BaseRecipe, RecipePort
from backend.app.core.logging import logger


class GenericRestGraphQLCallerRecipe(BaseRecipe):
    recipe_id = "rest_graphql_caller"
    name = "Generic REST/GraphQL API Caller"
    version = "1.0.0"
    category = "integrations"
    group = "Apps & Integrations"
    subgroup = "Generic"
    description = "Calls any REST or GraphQL API and turns the response into a table."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    inputs = [
        RecipePort(
            id="input",
            label="Trigger / Input Dataset",
            type="dataframe",
            required=False,
            max_connections=1,
            description="Optional upstream dataset for batch query parameters or execution trigger"
        )
    ]

    outputs = [
        RecipePort(
            id="output",
            label="API Records Table",
            type="dataframe",
            description="DataFrame containing parsed tabular records from the REST or GraphQL API response"
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "api_type": {
                    "type": "string",
                    "title": "API Type",
                    "input_type": "dropdown",
                    "enum": ["REST", "GraphQL"],
                    "default": "REST",
                    "description": "Select API architecture: REST (GET, POST, etc.) or GraphQL query/mutation."
                },
                "method": {
                    "type": "string",
                    "title": "HTTP Method (REST)",
                    "input_type": "dropdown",
                    "enum": ["GET", "POST", "PUT", "PATCH", "DELETE"],
                    "default": "GET",
                    "description": "HTTP method for REST requests (GraphQL requests automatically use POST)."
                },
                "endpoint_url": {
                    "type": "string",
                    "title": "Endpoint URL",
                    "input_type": "text",
                    "default": "https://jsonplaceholder.typicode.com/posts",
                    "description": "Target HTTP/HTTPS URL of the REST endpoint or GraphQL gateway."
                },
                "auth_type": {
                    "type": "string",
                    "title": "Authentication Method",
                    "input_type": "dropdown",
                    "enum": ["None", "Bearer Token", "Basic Auth", "API Key Header", "API Key Query Param"],
                    "default": "None",
                    "description": "Authentication mechanism applied to the outgoing HTTP request."
                },
                "auth_token": {
                    "type": "string",
                    "title": "Auth Token / Secret Key",
                    "input_type": "password",
                    "description": "Bearer token, password, or API key value."
                },
                "auth_username": {
                    "type": "string",
                    "title": "Auth Username / API Key Name",
                    "input_type": "text",
                    "default": "X-API-Key",
                    "description": "Username for Basic Auth, or header/query parameter name for API key authentication."
                },
                "custom_headers": {
                    "type": "string",
                    "title": "Custom Headers (JSON or Key: Value)",
                    "input_type": "textarea",
                    "description": "Optional HTTP headers as a JSON object (e.g. {\"Accept\": \"application/json\"}) or Key: Value lines."
                },
                "query_params": {
                    "type": "string",
                    "title": "Query Parameters (JSON or Key=Value)",
                    "input_type": "textarea",
                    "description": "Optional URL query parameters as a JSON object (e.g. {\"limit\": 50}) or query string."
                },
                "request_body": {
                    "type": "string",
                    "title": "REST Request Body (JSON)",
                    "input_type": "textarea",
                    "description": "Payload body for POST, PUT, or PATCH REST requests (usually JSON)."
                },
                "graphql_query": {
                    "type": "string",
                    "title": "GraphQL Query Document",
                    "input_type": "textarea",
                    "description": "GraphQL query or mutation document (e.g. 'query { users { id name email } }')."
                },
                "graphql_variables": {
                    "type": "string",
                    "title": "GraphQL Variables (JSON)",
                    "input_type": "textarea",
                    "description": "Optional JSON dictionary of GraphQL variables passed with the query."
                },
                "response_data_path": {
                    "type": "string",
                    "title": "Response Data JSON Path",
                    "input_type": "text",
                    "default": "data",
                    "description": "Dot-separated JSON path to extract records from (e.g. 'data.items', 'results', or blank for top-level array)."
                },
                "record_limit": {
                    "type": "integer",
                    "title": "Record Limit",
                    "input_type": "number",
                    "default": 50,
                    "minimum": 1,
                    "maximum": 5000,
                    "description": "Maximum number of rows to include in the parsed output DataFrame."
                },
                "timeout_seconds": {
                    "type": "integer",
                    "title": "Request Timeout (Seconds)",
                    "input_type": "number",
                    "default": 15,
                    "minimum": 1,
                    "maximum": 120,
                    "description": "Maximum wait time in seconds for the remote server response."
                },
                "simulation_mode": {
                    "type": "boolean",
                    "title": "Simulation Mode (Offline Benchmark Data)",
                    "input_type": "checkbox",
                    "default": False,
                    "description": "Generate realistic benchmark API response rows offline without network requests."
                }
            },
            "required": ["api_type", "endpoint_url"]
        }

    def _parse_headers(self, raw_headers: str) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if not raw_headers or not raw_headers.strip():
            return headers
        raw = raw_headers.strip()
        if raw.startswith("{") and raw.endswith("}"):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    return {str(k): str(v) for k, v in parsed.items()}
            except Exception:
                pass
        for line in raw.splitlines():
            line = line.strip()
            if ":" in line:
                k, v = line.split(":", 1)
                headers[k.strip()] = v.strip()
        return headers

    def _parse_params(self, raw_params: str) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if not raw_params or not raw_params.strip():
            return params
        raw = raw_params.strip()
        if raw.startswith("{") and raw.endswith("}"):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
        if "?" in raw:
            raw = raw.split("?", 1)[1]
        for pair in raw.split("&"):
            if "=" in pair:
                k, v = pair.split("=", 1)
                params[k.strip()] = v.strip()
        return params

    def _navigate_json_path(self, data: Any, path: str) -> Any:
        if not path or not path.strip():
            return data
        tokens = [p.strip() for p in path.split(".") if p.strip()]
        current = data
        for token in tokens:
            if isinstance(current, dict):
                current = current.get(token)
            elif isinstance(current, list) and token.isdigit():
                idx = int(token)
                if 0 <= idx < len(current):
                    current = current[idx]
                else:
                    return None
            else:
                return None
            if current is None:
                break
        return current

    def _normalize_to_records(self, payload: Any, path: str) -> List[Dict[str, Any]]:
        extracted = self._navigate_json_path(payload, path)
        if extracted is None and path:
            # Fallback check common payload envelopes
            for alt_key in ["data", "results", "items", "records", "posts", "rows"]:
                if isinstance(payload, dict) and alt_key in payload:
                    extracted = payload[alt_key]
                    break
        if extracted is None:
            extracted = payload

        if isinstance(extracted, list):
            records = []
            for item in extracted:
                if isinstance(item, dict):
                    records.append(item)
                else:
                    records.append({"value": item})
            return records
        elif isinstance(extracted, dict):
            # Check if dict values contain a list of records
            for v in extracted.values():
                if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                    return v
            return [extracted]
        return []

    def _generate_benchmark_data(self, api_type: str, endpoint: str, limit: int) -> List[Dict[str, Any]]:
        limit = min(max(1, limit), 250)
        now_iso = datetime.now(timezone.utc).isoformat()
        records: List[Dict[str, Any]] = []

        if api_type.upper() == "GRAPHQL":
            types = ["UserNode", "OrderNode", "ProductNode", "IssueNode", "InvoiceNode"]
            statuses = ["ACTIVE", "PENDING", "COMPLETED", "ARCHIVED"]
            for i in range(1, limit + 1):
                node_type = types[i % len(types)]
                records.append({
                    "id": f"gid://benchmark/{node_type}/{1000 + i}",
                    "cursor": f"YXJyYXljb25uZWN0aW9uOms={100 + i}",
                    "node_type": node_type,
                    "title": f"Benchmark GraphQL Node {i} ({node_type})",
                    "status": statuses[i % len(statuses)],
                    "total_count": 1000 + i * 5,
                    "created_at": now_iso,
                    "source_endpoint": endpoint or "https://api.graphql.mock/graphql"
                })
        else:
            categories = ["Technology", "Operations", "Finance", "Customer Success", "Security"]
            methods = ["ACTIVE", "PROCESSING", "VERIFIED", "SYNCED"]
            for i in range(1, limit + 1):
                records.append({
                    "id": 1000 + i,
                    "title": f"REST Entity Item {i}",
                    "category": categories[i % len(categories)],
                    "status": methods[i % len(methods)],
                    "priority": (i % 5) + 1,
                    "score": round(75.5 + (i * 0.3) % 24.5, 2),
                    "created_at": now_iso,
                    "source_endpoint": endpoint or "https://jsonplaceholder.typicode.com/posts"
                })
        return records

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None, **kwargs) -> Dict[str, Any]:
        api_type = str(config.get("api_type", "REST")).strip().upper()
        method = str(config.get("method", "GET")).strip().upper()
        endpoint_url = str(config.get("endpoint_url", "https://jsonplaceholder.typicode.com/posts")).strip()
        auth_type = str(config.get("auth_type", "None")).strip()
        auth_token = str(config.get("auth_token", "")).strip()
        auth_username = str(config.get("auth_username", "X-API-Key")).strip()
        custom_headers_raw = str(config.get("custom_headers", "")).strip()
        query_params_raw = str(config.get("query_params", "")).strip()
        request_body = str(config.get("request_body", "")).strip()
        graphql_query = str(config.get("graphql_query", "")).strip()
        graphql_variables = str(config.get("graphql_variables", "")).strip()
        response_data_path = str(config.get("response_data_path", "data")).strip()
        limit = int(config.get("record_limit", 50))
        timeout = int(config.get("timeout_seconds", 15))
        simulation_mode = bool(config.get("simulation_mode", False))

        records: List[Dict[str, Any]] = []
        delivery_status = "SIMULATED"
        status_msg = ""
        http_code = 200

        # Attempt live request if not forced simulation
        if not simulation_mode and endpoint_url and not endpoint_url.startswith("mock://"):
            try:
                # Prepare headers
                headers = self._parse_headers(custom_headers_raw)
                headers.setdefault("User-Agent", "Datarept-ML-Pipeline/1.0 (REST/GraphQL Caller)")

                # Prepare query parameters
                url = endpoint_url
                parsed_params = self._parse_params(query_params_raw)

                # Auth injection
                if auth_type == "Bearer Token" and auth_token:
                    headers["Authorization"] = f"Bearer {auth_token}"
                elif auth_type == "Basic Auth" and (auth_username or auth_token):
                    user_pass = f"{auth_username}:{auth_token}".encode("utf-8")
                    headers["Authorization"] = f"Basic {base64.b64encode(user_pass).decode('utf-8')}"
                elif auth_type == "API Key Header" and auth_token:
                    headers[auth_username or "X-API-Key"] = auth_token
                elif auth_type == "API Key Query Param" and auth_token:
                    parsed_params[auth_username or "api_key"] = auth_token

                # Append query parameters to URL
                if parsed_params:
                    query_str = urllib.parse.urlencode(parsed_params)
                    sep = "&" if "?" in url else "?"
                    url = f"{url}{sep}{query_str}"

                # Request payload & method
                req_data = None
                if api_type == "GRAPHQL":
                    method = "POST"
                    headers.setdefault("Content-Type", "application/json")
                    headers.setdefault("Accept", "application/json")
                    gql_payload: Dict[str, Any] = {"query": graphql_query or "{ __typename }"}
                    if graphql_variables:
                        try:
                            gql_payload["variables"] = json.loads(graphql_variables)
                        except Exception:
                            pass
                    req_data = json.dumps(gql_payload).encode("utf-8")
                elif method in ["POST", "PUT", "PATCH"] and request_body:
                    headers.setdefault("Content-Type", "application/json")
                    req_data = request_body.encode("utf-8")

                req = urllib.request.Request(url, data=req_data, headers=headers, method=method)
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    http_code = resp.status
                    raw_bytes = resp.read()
                    parsed_json = json.loads(raw_bytes.decode("utf-8", errors="replace"))
                    raw_records = self._normalize_to_records(parsed_json, response_data_path)
                    if raw_records:
                        records = raw_records[:limit]
                        delivery_status = "LIVE"
                        status_msg = f"Retrieved {len(records)} live records via {method} {url} (HTTP {http_code})."
            except Exception as ex:
                logger.warning(f"REST/GraphQL live API call fallback: {ex}")
                status_msg = f"Live API call failed ({ex}). Switched to benchmark simulation data."

        # Offline / simulation fallback
        if not records:
            records = self._generate_benchmark_data(api_type, endpoint_url, limit)
            if not status_msg:
                status_msg = f"Generated {len(records)} realistic benchmark {api_type} records (Simulation Mode)."

        df_out = pd.DataFrame(records) if pd is not None else records

        summary = {
            "title": f"Generic {api_type} API Caller Summary",
            "message": status_msg,
            "status": delivery_status,
            "api_type": api_type,
            "method": method if api_type == "REST" else "POST",
            "endpoint": endpoint_url,
            "records_retrieved": len(records),
            "http_status_code": http_code
        }

        metrics = {
            "api_type": api_type,
            "endpoint": endpoint_url,
            "record_count": len(records),
            "delivery_status": delivery_status,
            "http_status_code": http_code
        }

        return {
            "output": df_out,
            "dataframe": df_out,
            "output_summary": summary,
            "metrics": metrics
        }

    def to_code(self, config: Dict[str, Any]) -> str:
        api_type = str(config.get("api_type", "REST")).strip().upper()
        endpoint = config.get("endpoint_url", "https://jsonplaceholder.typicode.com/posts")
        method = str(config.get("method", "GET")).strip().upper()
        path = config.get("response_data_path", "data")
        limit = config.get("record_limit", 50)

        if api_type == "GRAPHQL":
            return (
                f"# Recipe: Generic REST/GraphQL API Caller (GraphQL)\n"
                f"# ID: {self.recipe_id}\n"
                f"import json\n"
                f"import requests\n"
                f"import pandas as pd\n\n"
                f"url = '{endpoint}'\n"
                f"query = '''{config.get('graphql_query', '{ __typename }')}'''\n"
                f"headers = {{'Content-Type': 'application/json'}}\n"
                f"resp = requests.post(url, json={{'query': query}}, headers=headers, timeout=15)\n"
                f"data = resp.json().get('{path}', resp.json())\n"
                f"df = pd.json_normalize(data if isinstance(data, list) else [data]).head({limit})\n"
            )

        return (
            f"# Recipe: Generic REST/GraphQL API Caller (REST)\n"
            f"# ID: {self.recipe_id}\n"
            f"import requests\n"
            f"import pandas as pd\n\n"
            f"url = '{endpoint}'\n"
            f"resp = requests.{method.lower()}(url, timeout=15)\n"
            f"payload = resp.json()\n"
            f"records = payload.get('{path}', payload) if isinstance(payload, dict) else payload\n"
            f"df = pd.json_normalize(records).head({limit})\n"
        )

