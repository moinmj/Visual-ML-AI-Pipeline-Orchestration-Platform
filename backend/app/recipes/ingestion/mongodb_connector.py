"""
MongoDB Connector Recipe (Ingestion & Orchestration -> Database & Warehouse Connectors).

Reads documents from a MongoDB collection into the pipeline as a DataFrame.
Supports standard URI (mongodb://, mongodb+srv://), host/port, or saved credentials,
MQL filter queries, projections, sorting, BSON ObjectId serialization, nested document
flattening (json_normalize), benchmark simulation fallback, and code generation.
"""

import json
import os
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

try:
    import pandas as pd
except ImportError:
    pd = None

from backend.app.recipes.base.recipe import BaseRecipe, RecipePort
from backend.app.core.logging import logger
from backend.app.core.config import settings


class MongoDBConnectorRecipe(BaseRecipe):
    recipe_id = "mongodb_connector"
    name = "MongoDB Connector"
    version = "1.0.0"
    category = "ingestion"
    group = "Ingestion & Orchestration"
    subgroup = "Database & Warehouse Connectors"
    description = "Reads documents from a MongoDB collection and flattens them into a table."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    inputs = [
        RecipePort(
            id="input",
            label="Trigger / Input Dataset",
            type="dataframe",
            required=False,
            max_connections=1,
            description="Optional upstream dataset for batch IDs or execution trigger"
        )
    ]

    outputs = [
        RecipePort(
            id="output",
            label="MongoDB Dataset",
            type="dataframe",
            description="DataFrame containing retrieved and normalized MongoDB documents"
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "connection_mode": {
                    "type": "string",
                    "title": "Connection Mode",
                    "input_type": "dropdown",
                    "enum": ["connection_string", "host_port", "credential"],
                    "default": "connection_string",
                    "description": "Choose connection method: MongoDB URI (standard or Atlas SRV), host/port, or stored credential."
                },
                "connection_string": {
                    "type": "string",
                    "title": "Connection String (URI)",
                    "input_type": "text",
                    "default": "mongodb://localhost:27017",
                    "description": "MongoDB URI e.g. mongodb://localhost:27017 or mongodb+srv://user:pass@cluster0.mongodb.net/dbname"
                },
                "host": {
                    "type": "string",
                    "title": "Host",
                    "input_type": "text",
                    "default": "localhost",
                    "description": "MongoDB server hostname."
                },
                "port": {
                    "type": "integer",
                    "title": "Port",
                    "input_type": "number",
                    "default": 27017,
                    "description": "MongoDB server port (default 27017)."
                },
                "database": {
                    "type": "string",
                    "title": "Database Name",
                    "input_type": "text",
                    "default": "analytics",
                    "description": "Target MongoDB database name."
                },
                "collection": {
                    "type": "string",
                    "title": "Collection Name",
                    "input_type": "text",
                    "default": "events",
                    "description": "Target MongoDB collection name (e.g. 'events', 'users', 'logs')."
                },
                "username": {
                    "type": "string",
                    "title": "Username",
                    "input_type": "text",
                    "default": "",
                    "description": "MongoDB user role for authentication."
                },
                "password": {
                    "type": "string",
                    "title": "Password",
                    "input_type": "password",
                    "default": "",
                    "description": "MongoDB user password."
                },
                "auth_source": {
                    "type": "string",
                    "title": "Auth Source Database",
                    "input_type": "text",
                    "default": "admin",
                    "description": "Authentication source database (default 'admin')."
                },
                "credential_id": {
                    "type": "string",
                    "title": "Connected MongoDB Credential",
                    "input_type": "credential_dropdown",
                    "provider": "mongodb",
                    "description": "Select a stored MongoDB credential from platform settings."
                },
                "filter_query": {
                    "type": "string",
                    "title": "Filter Query (MQL JSON)",
                    "input_type": "textarea",
                    "default": "{}",
                    "description": "JSON filter query (e.g. '{\"status\": \"active\", \"score\": {\"$gte\": 50}}'). Leave empty or '{}' for all documents."
                },
                "projection": {
                    "type": "string",
                    "title": "Projection (JSON)",
                    "input_type": "textarea",
                    "default": "{}",
                    "description": "JSON field projection (e.g. '{\"_id\": 1, \"event_type\": 1, \"timestamp\": 1}')."
                },
                "sort": {
                    "type": "string",
                    "title": "Sort (JSON)",
                    "input_type": "text",
                    "default": "{}",
                    "description": "Sort expression (e.g. '{\"timestamp\": -1}')."
                },
                "limit": {
                    "type": "integer",
                    "title": "Document Limit",
                    "input_type": "number",
                    "default": 1000,
                    "description": "Maximum number of documents to retrieve (1 to 100000)."
                },
                "flatten_nested": {
                    "type": "boolean",
                    "title": "Flatten Nested Documents",
                    "input_type": "checkbox",
                    "default": True,
                    "description": "Automatically flatten nested dictionaries and objects (e.g. 'user.address.city') into tabular columns."
                },
                "convert_objectid": {
                    "type": "boolean",
                    "title": "Convert ObjectId to String",
                    "input_type": "checkbox",
                    "default": True,
                    "description": "Convert BSON ObjectId values to hexadecimal strings to ensure clean tabular display and downstream recipe compatibility."
                },
                "timeout": {
                    "type": "integer",
                    "title": "Timeout (seconds)",
                    "input_type": "number",
                    "default": 15,
                    "description": "Connection and query timeout in seconds."
                },
                "simulation_mode": {
                    "type": "boolean",
                    "title": "Simulation / Demo Mode",
                    "input_type": "checkbox",
                    "default": False,
                    "description": "Generate realistic benchmark document dataset without connecting to an external MongoDB cluster."
                }
            },
            "required": ["database", "collection"],
            "templates": [
                {
                    "label": "Read All Events",
                    "config": {
                        "database": "analytics",
                        "collection": "events",
                        "filter_query": "{}",
                        "limit": 500,
                        "flatten_nested": True
                    }
                },
                {
                    "label": "Active Users Filter",
                    "config": {
                        "database": "production",
                        "collection": "users",
                        "filter_query": "{\"status\": \"ACTIVE\", \"tier\": \"enterprise\"}",
                        "projection": "{\"_id\": 1, \"username\": 1, \"email\": 1, \"metadata\": 1}",
                        "limit": 1000
                    }
                }
            ]
        }

    def _parse_json_field(self, value: Any, field_name: str, errors: List[str]) -> Dict[str, Any]:
        """Safely parses a string or dict JSON field and appends validation errors."""
        if value is None or value == "":
            return {}
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            v_str = value.strip()
            if not v_str:
                return {}
            try:
                parsed = json.loads(v_str)
                if not isinstance(parsed, dict) and not isinstance(parsed, list):
                    errors.append(f"Field '{field_name}' must be a JSON object or array. Received: {type(parsed).__name__}")
                    return {}
                return parsed if isinstance(parsed, dict) else {k: 1 for k in parsed}
            except json.JSONDecodeError as jde:
                errors.append(f"Invalid JSON in '{field_name}': {jde.msg} at line {jde.lineno} col {jde.colno}.")
                return {}
        errors.append(f"Field '{field_name}' must be a JSON string or dict.")
        return {}

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        errors = []

        database = str(config.get("database", "")).strip()
        if not database:
            errors.append("MongoDB database name is required.")

        collection = str(config.get("collection", "")).strip()
        if not collection:
            errors.append("MongoDB collection name is required.")

        # Validate Port
        port = config.get("port")
        if port is not None:
            try:
                p = int(port)
                if p < 1 or p > 65535:
                    errors.append(f"Invalid MongoDB port '{port}'. Must be between 1 and 65535.")
            except (ValueError, TypeError):
                errors.append(f"Port '{port}' must be a valid integer.")

        # Validate JSON query fields
        self._parse_json_field(config.get("filter_query"), "filter_query", errors)
        self._parse_json_field(config.get("projection"), "projection", errors)
        self._parse_json_field(config.get("sort"), "sort", errors)

        # Validate Connection mode requirements if not simulated
        conn_mode = str(config.get("connection_mode", "connection_string")).strip().lower()
        if not config.get("simulation_mode"):
            if conn_mode == "connection_string" and "connection_string" in config:
                cs = str(config.get("connection_string", "")).strip()
                if not cs and not config.get("host"):
                    errors.append("Connection string (URI) is required in 'connection_string' mode.")
            elif conn_mode == "credential":
                cred_id = str(config.get("credential_id", "")).strip()
                if not cred_id:
                    errors.append("Credential selection is required in 'credential' mode.")

        return errors

    def _resolve_connection_params(self, config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        mode = str(config.get("connection_mode", "connection_string")).strip().lower()
        credential_id = str(config.get("credential_id", "")).strip()

        conn_str = str(config.get("connection_string", "mongodb://localhost:27017")).strip()
        host = str(config.get("host", "localhost")).strip()
        port = int(config.get("port", 27017))
        database = str(config.get("database", "analytics")).strip()
        collection = str(config.get("collection", "events")).strip()
        username = str(config.get("username", "")).strip()
        password = str(config.get("password", "")).strip()
        auth_source = str(config.get("auth_source", "admin")).strip()

        # Context credential lookup
        if credential_id and context and isinstance(context, dict):
            creds = context.get("credentials", {})
            if credential_id in creds:
                c_data = creds[credential_id]
                c_data = c_data.get("data", c_data)
                conn_str = c_data.get("connection_string") or c_data.get("url") or conn_str
                host = c_data.get("host") or host
                port = int(c_data.get("port") or port)
                database = c_data.get("database") or database
                collection = c_data.get("collection") or collection
                username = c_data.get("username") or c_data.get("user") or username
                password = c_data.get("password") or password
                auth_source = c_data.get("auth_source") or auth_source

        # Environment variable overrides
        if not conn_str:
            conn_str = os.getenv("MONGODB_URI", os.getenv("MONGO_URL", ""))

        if not conn_str and host:
            auth_part = f"{username}:{password}@" if username and password else ""
            auth_param = f"?authSource={auth_source}" if username and auth_source else ""
            conn_str = f"mongodb://{auth_part}{host}:{port}/{database}{auth_param}"

        return {
            "mode": mode,
            "connection_string": conn_str,
            "host": host,
            "port": port,
            "database": database,
            "collection": collection,
            "username": username,
            "password": password,
            "auth_source": auth_source
        }

    def _serialize_bson_document(self, doc: Dict[str, Any], convert_objectid: bool = True) -> Dict[str, Any]:
        """Recursively converts BSON ObjectId, datetime, and binary types to clean JSON/Python primitives."""
        clean_doc: Dict[str, Any] = {}
        for k, v in doc.items():
            if convert_objectid and hasattr(v, "__class__") and v.__class__.__name__ == "ObjectId":
                clean_doc[k] = str(v)
            elif isinstance(v, datetime):
                clean_doc[k] = v.isoformat()
            elif isinstance(v, dict):
                clean_doc[k] = self._serialize_bson_document(v, convert_objectid=convert_objectid)
            elif isinstance(v, list):
                clean_list = []
                for item in v:
                    if isinstance(item, dict):
                        clean_list.append(self._serialize_bson_document(item, convert_objectid=convert_objectid))
                    elif convert_objectid and hasattr(item, "__class__") and item.__class__.__name__ == "ObjectId":
                        clean_list.append(str(item))
                    elif isinstance(item, datetime):
                        clean_list.append(item.isoformat())
                    else:
                        clean_list.append(item)
                clean_doc[k] = clean_list
            else:
                clean_doc[k] = v
        return clean_doc

    def _generate_benchmark_data(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        collection = str(config.get("collection", "events")).lower()
        limit = max(1, min(int(config.get("limit", 100)), 500))

        records: List[Dict[str, Any]] = []
        event_types = ["PAGE_VIEW", "PURCHASE_COMPLETED", "SIGN_UP", "SESSION_START", "CLICK"]
        device_types = ["desktop", "mobile", "tablet"]
        os_list = ["macOS", "Windows", "iOS", "Android", "Linux"]

        for i in range(1, limit + 1):
            if "user" in collection or "account" in collection or "profile" in collection:
                records.append({
                    "_id": f"65ef{1000 + i:04d}0000000000000001",
                    "user_id": f"usr_{3000 + i}",
                    "profile": {
                        "full_name": f"Enterprise User {i}",
                        "email": f"user{i}@enterprise.org",
                        "role": "admin" if i % 5 == 0 else "member"
                    },
                    "preferences": {
                        "notifications": True,
                        "theme": "dark" if i % 2 == 0 else "light"
                    },
                    "status": "ACTIVE" if i % 8 != 0 else "INACTIVE",
                    "login_count": i * 4,
                    "created_at": f"2026-02-{((i % 28) + 1):02d}T09:12:00Z"
                })
            else:
                records.append({
                    "_id": f"65ee{2000 + i:04d}0000000000000002",
                    "event_type": event_types[i % len(event_types)],
                    "session_id": f"sess_{100 + (i % 25)}",
                    "user": {
                        "id": f"usr_{200 + (i % 30)}",
                        "tier": "enterprise" if i % 3 == 0 else "pro"
                    },
                    "client": {
                        "device": device_types[i % len(device_types)],
                        "os": os_list[i % len(os_list)],
                        "ip": f"192.168.1.{10 + (i % 80)}"
                    },
                    "metrics": {
                        "latency_ms": 12.5 + (i * 1.8) % 45.0,
                        "revenue": round(19.99 + (i * 3.5) % 250.0, 2) if i % 2 == 0 else 0.0
                    },
                    "timestamp": f"2026-03-{((i % 28) + 1):02d}T15:30:{((i % 60)):02d}Z"
                })
        return records

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        start_time = time.time()
        limit = max(1, min(int(config.get("limit", 1000)), 100000))
        flatten_nested = bool(config.get("flatten_nested", True))
        convert_objectid = bool(config.get("convert_objectid", True))
        simulation_mode = bool(config.get("simulation_mode", False))

        params = self._resolve_connection_params(config, context)
        database_name = params["database"]
        collection_name = params["collection"]

        # Parse query, projection, sort
        errs: List[str] = []
        filter_dict = self._parse_json_field(config.get("filter_query"), "filter_query", errs)
        projection_dict = self._parse_json_field(config.get("projection"), "projection", errs)
        sort_dict = self._parse_json_field(config.get("sort"), "sort", errs)

        records: List[Dict[str, Any]] = []
        delivery_status = "SIMULATED"
        status_msg = ""

        # Live Execution Attempt
        if not simulation_mode and params["connection_string"]:
            try:
                import pymongo
                timeout_ms = int(config.get("timeout", 15)) * 1000
                client = pymongo.MongoClient(
                    params["connection_string"],
                    serverSelectionTimeoutMS=timeout_ms,
                    connectTimeoutMS=timeout_ms
                )
                db = client[database_name]
                coll = db[collection_name]

                # Build Cursor
                cursor = coll.find(filter_dict or {}, projection=projection_dict or None)
                if sort_dict:
                    sort_tuples = list(sort_dict.items())
                    cursor = cursor.sort(sort_tuples)
                cursor = cursor.limit(limit)

                raw_docs = list(cursor)
                records = [self._serialize_bson_document(doc, convert_objectid=convert_objectid) for doc in raw_docs]
                client.close()

                delivery_status = "LIVE"
                status_msg = f"Retrieved {len(records)} live documents from MongoDB collection '{database_name}.{collection_name}'."
            except ImportError:
                status_msg = "PyMongo driver not installed. Using benchmark simulation fallback."
                logger.info(status_msg)
            except Exception as e:
                logger.warning(f"MongoDB connection failed: {e}")
                status_msg = f"MongoDB connection failed ({e}). Using benchmark simulation fallback."

        # Fallback to realistic benchmark data
        if not records:
            raw_benchmark = self._generate_benchmark_data(config)
            records = [self._serialize_bson_document(doc, convert_objectid=convert_objectid) for doc in raw_benchmark]
            if not status_msg:
                status_msg = f"Loaded {len(records)} benchmark documents for MongoDB collection '{database_name}.{collection_name}'."

        # Flatten nested documents via json_normalize if requested
        if pd is not None and records:
            if flatten_nested:
                df_out = pd.json_normalize(records)
            else:
                df_out = pd.DataFrame(records)
        elif pd is not None:
            df_out = pd.DataFrame()
        else:
            df_out = records

        duration_ms = int((time.time() - start_time) * 1000)
        col_count = len(df_out.columns) if hasattr(df_out, "columns") else (len(records[0].keys()) if records else 0)

        summary = {
            "title": "MongoDB Connector Summary",
            "message": status_msg,
            "status": delivery_status,
            "database": database_name,
            "collection": collection_name,
            "records_retrieved": len(records),
            "columns_count": col_count,
            "flattened": flatten_nested,
            "duration_ms": duration_ms
        }

        metrics = {
            "records_retrieved": len(records),
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
        conn_str = config.get("connection_string", "mongodb://localhost:27017")
        database = config.get("database", "analytics")
        collection = config.get("collection", "events")
        limit = config.get("limit", 1000)
        filter_q = config.get("filter_query", "{}")
        flatten = config.get("flatten_nested", True)

        return (
            f"# Recipe: MongoDB Connector\n"
            f"# ID: {self.recipe_id}\n"
            f"import os\n"
            f"import json\n"
            f"import pandas as pd\n"
            f"from pymongo import MongoClient\n\n"
            f"uri = os.getenv('MONGODB_URI', '{conn_str}')\n"
            f"client = MongoClient(uri)\n"
            f"db = client['{database}']\n"
            f"coll = db['{collection}']\n\n"
            f"filter_query = {filter_q if isinstance(filter_q, dict) else json.dumps(json.loads(filter_q) if filter_q else {})}\n"
            f"docs = list(coll.find(filter_query).limit({limit}))\n"
            f"client.close()\n\n"
            f"df = pd.json_normalize(docs) if {flatten} else pd.DataFrame(docs)\n"
            f"print(f'Retrieved {{len(df)}} documents from MongoDB.')\n"
        )
