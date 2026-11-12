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
from backend.app.core.exceptions import ValidationException


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
                    "default": "",
                    "placeholder": "e.g. mongodb://user:pass@host:27017/dbname or mongodb+srv://cluster.mongodb.net/dbname",
                    "description": "Full MongoDB connection URI (standard or Atlas SRV)."
                },
                "host": {
                    "type": "string",
                    "title": "Host",
                    "input_type": "text",
                    "default": "",
                    "placeholder": "e.g. 88.222.215.30 or localhost or cluster0.mongodb.net",
                    "description": "MongoDB server hostname or IP address (used in 'host_port' mode)."
                },
                "port": {
                    "type": "integer",
                    "title": "Port",
                    "input_type": "number",
                    "default": 27017,
                    "placeholder": "27017",
                    "description": "MongoDB server port (default 27017)."
                },
                "database": {
                    "type": "string",
                    "title": "Database Name",
                    "input_type": "text",
                    "default": "",
                    "placeholder": "e.g. ai_whatsapp_db or analytics",
                    "description": "Target MongoDB database name."
                },
                "collection": {
                    "type": "string",
                    "title": "Collection Name",
                    "input_type": "text",
                    "default": "",
                    "placeholder": "e.g. conversations or events",
                    "description": "Target MongoDB collection name (e.g. 'conversations', 'events', 'users')."
                },
                "username": {
                    "type": "string",
                    "title": "Username",
                    "input_type": "text",
                    "default": "",
                    "placeholder": "e.g. smartbot",
                    "description": "MongoDB user role for authentication."
                },
                "password": {
                    "type": "string",
                    "title": "Password",
                    "input_type": "password",
                    "default": "",
                    "placeholder": "Enter MongoDB password",
                    "description": "MongoDB user password."
                },
                "auth_source": {
                    "type": "string",
                    "title": "Auth Source Database",
                    "input_type": "text",
                    "default": "admin",
                    "placeholder": "admin",
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

        conn_str = str(config.get("connection_string", "")).strip()
        host = str(config.get("host", "")).strip()
        port = int(config.get("port") or 27017)
        database = str(config.get("database", "")).strip()
        collection = str(config.get("collection", "")).strip()
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

        # Host/Port mode: construct URI directly from components
        if mode == "host_port" or (mode != "connection_string" and host):
            from urllib.parse import quote_plus
            auth_part = f"{quote_plus(username)}:{quote_plus(password)}@" if username and password else ""
            auth_param = f"?authSource={quote_plus(auth_source)}" if auth_source else ""
            db_part = f"/{database}" if database else ""
            conn_str = f"mongodb://{auth_part}{host}:{port}{db_part}{auth_param}"
        elif mode == "connection_string":
            if not conn_str:
                conn_str = os.getenv("MONGODB_URI", os.getenv("MONGO_URL", ""))

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
        last_error = None

        # Live Execution Attempt
        if not simulation_mode:
            if not params["connection_string"]:
                raise ValidationException(
                    f"MongoDB connection failed: Connection URI or host/port configuration must be provided."
                )
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

                # Ping server to verify connectivity before querying
                db.command("ping")

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
            except Exception as e:
                logger.warning(f"MongoDB connection failed: {e}")
                last_error = e

            if last_error is not None:
                raise ValidationException(f"MongoDB connection to '{database_name}.{collection_name}' failed: {str(last_error)}")

        # Simulation Mode (ONLY when simulation_mode is True)
        if simulation_mode:
            raw_benchmark = self._generate_benchmark_data(config)
            records = [self._serialize_bson_document(doc, convert_objectid=convert_objectid) for doc in raw_benchmark]
            delivery_status = "SIMULATED"
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

    def inspect(
        self,
        action: str,
        config: Dict[str, Any],
        context: Optional[Any] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Introspects connected MongoDB server in real-time."""
        action = str(action or "test_connection").strip().lower()
        params = self._resolve_connection_params(config, context)
        simulation_mode = bool(config.get("simulation_mode", False))

        is_live = False
        server_info = ""
        inspect_err = ""
        client = None

        if not simulation_mode and (params.get("connection_string") or params.get("host")):
            try:
                import pymongo
                auth_conn_str = params.get("connection_string") or f"mongodb://{params.get('host')}:{params.get('port')}"
                client = pymongo.MongoClient(auth_conn_str, serverSelectionTimeoutMS=int(config.get("timeout", 10)) * 1000)
                info = client.server_info()
                if info:
                    server_info = f"MongoDB v{info.get('version', 'unknown')}"
                    is_live = True
            except Exception as ex:
                inspect_err = str(ex)
                logger.debug(f"MongoDB inspect connection failed: {ex}")

        if action == "test_connection":
            if is_live:
                if client: client.close()
                return {"success": True, "connected": True, "status": "LIVE", "server_version": server_info, "database": params.get("database"), "message": "Successfully connected to MongoDB."}
            elif simulation_mode:
                return {"success": True, "connected": True, "status": "SIMULATED", "server_version": "MongoDB v7.0.5 [SIMULATION]", "database": params.get("database"), "message": "MongoDB simulation mode active."}
            else:
                return {"success": False, "connected": False, "status": "UNREACHABLE", "error": inspect_err, "message": f"Could not connect to MongoDB: {inspect_err or 'Unreachable'}"}

        if action in ("databases", "get_databases"):
            if not is_live and not simulation_mode:
                return {"success": False, "databases": [], "error": inspect_err or "Connection could not be established."}
            dbs = []
            if is_live and client:
                try:
                    dbs = [d for d in client.list_database_names() if d not in ('admin', 'local', 'config')]
                    client.close()
                except Exception as e:
                    logger.warning(f"MongoDB list_database_names failed: {e}")
            if not dbs and simulation_mode:
                dbs = ["analytics_db", "production_db", "staging_db"]
            return {"success": True, "databases": dbs, "count": len(dbs), "source": "LIVE" if is_live else "SIMULATED"}

        if action in ("tables", "get_tables", "collections", "get_collections"):
            if not is_live and not simulation_mode:
                return {"success": False, "tables": [], "collections": [], "error": inspect_err or "Connection could not be established."}
            target_db = schema_name or config.get("database", "production_db")
            colls = []
            if is_live and client:
                try:
                    colls = client[target_db].list_collection_names()
                    client.close()
                except Exception as e:
                    logger.warning(f"MongoDB list_collection_names failed: {e}")
            if not colls and simulation_mode:
                colls = ["users", "events", "transactions", "devices"]
            return {"success": True, "tables": colls, "collections": colls, "count": len(colls), "database": target_db, "source": "LIVE" if is_live else "SIMULATED"}

        if action in ("columns", "get_columns", "fields", "get_fields"):
            if not is_live and not simulation_mode:
                return {"success": False, "columns": [], "error": inspect_err or "Connection could not be established."}
            target_db = schema_name or config.get("database", "production_db")
            target_coll = table_name or config.get("collection", "users")
            fields = []
            if is_live and client:
                try:
                    sample = client[target_db][target_coll].find_one()
                    if sample:
                        for k, v in sample.items():
                            fields.append({"name": k, "type": type(v).__name__, "nullable": True, "primary_key": (k == "_id")})
                    client.close()
                except Exception as e:
                    logger.warning(f"MongoDB sample schema failed: {e}")
            if not fields and simulation_mode:
                fields = [
                    {"name": "_id", "type": "ObjectId", "nullable": False, "primary_key": True},
                    {"name": "user_id", "type": "str", "nullable": False, "primary_key": False},
                    {"name": "status", "type": "str", "nullable": True, "primary_key": False},
                    {"name": "created_at", "type": "datetime", "nullable": True, "primary_key": False},
                ]
            return {"success": True, "columns": fields, "count": len(fields), "collection": target_coll, "source": "LIVE" if is_live else "SIMULATED"}

        return {"success": False, "error": f"Unknown action '{action}'."}
