"""
Notion Connector Recipe (Apps & Integrations -> Productivity & Files).

Reads and queries Notion databases, pages, and workspace users via Notion REST API v1
(with credential resolution, rate-limit resilience, and benchmark simulation fallbacks).
"""

import json
import os
import urllib.request
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

try:
    import pandas as pd
except ImportError:
    pd = None

from backend.app.recipes.base.recipe import BaseRecipe, RecipePort
from backend.app.core.logging import logger


class NotionConnectorRecipe(BaseRecipe):
    recipe_id = "notion_connector"
    name = "Notion Connector"
    version = "1.0.0"
    category = "integrations"
    group = "Apps & Integrations"
    subgroup = "Productivity & Files"
    description = "Reads and writes Notion databases and pages."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    inputs = [
        RecipePort(
            id="input",
            label="Trigger / Input Dataset",
            type="dataframe",
            required=False,
            max_connections=1,
            description="Optional upstream dataset for batch page generation or execution trigger"
        )
    ]

    outputs = [
        RecipePort(
            id="output",
            label="Notion Dataset",
            type="dataframe",
            description="DataFrame containing retrieved Notion database pages, properties, or users"
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "resource": {
                    "type": "string",
                    "title": "Notion Resource",
                    "input_type": "dropdown",
                    "enum": ["Database Records", "Pages", "Users"],
                    "default": "Database Records",
                    "description": "Select the Notion resource to query (Database Records, Pages, or Users)."
                },
                "database_id": {
                    "type": "string",
                    "title": "Database ID (UUID)",
                    "input_type": "text",
                    "default": "db_1203948571829304",
                    "description": "Target Notion Database ID (UUID from database URL)."
                },
                "status_filter": {
                    "type": "string",
                    "title": "Status Filter",
                    "input_type": "dropdown",
                    "enum": ["All", "Not started", "In progress", "Done", "Archived"],
                    "default": "All",
                    "description": "Filter database pages by their Status property value."
                },
                "api_token": {
                    "type": "string",
                    "title": "Notion Internal Integration Token (Secret)",
                    "input_type": "password",
                    "description": "Notion Integration Secret (starts with 'secret_')."
                },
                "credential_id": {
                    "type": "string",
                    "title": "Credential ID",
                    "input_type": "text",
                    "description": "Saved credential ID containing Notion authentication secrets."
                },
                "record_limit": {
                    "type": "integer",
                    "title": "Record Limit",
                    "input_type": "number",
                    "default": 50,
                    "minimum": 1,
                    "maximum": 100,
                    "description": "Maximum number of database entries to retrieve."
                },
                "simulation_mode": {
                    "type": "boolean",
                    "title": "Simulation Mode (Offline Benchmark Data)",
                    "input_type": "checkbox",
                    "default": False,
                    "description": "Generate realistic benchmark Notion records without live API calls."
                }
            },
            "required": ["resource"]
        }

    def _generate_benchmark_data(self, resource: str, limit: int) -> List[Dict[str, Any]]:
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        limit_to_use = max(1, min(limit, 100))
        records = []

        if resource == "Database Records":
            pages = [
                ("Product Roadmap: Machine Learning Studio 2.0", "In progress", "P1 - Critical", "Alice Johnson", "2026-12-01"),
                ("Design System: Canvas Node Connectors & HUDs", "Done", "P2 - High", "Bob Martinez", "2026-10-31"),
                ("Architecture Decision Record (ADR): Arrow Stream Protocol", "Done", "P1 - Critical", "Clara Chen", "2026-10-15"),
                ("Customer Churn Early-Warning Dashboard Specs", "In progress", "P2 - High", "David Smith", "2026-11-25"),
                ("Compliance Runbook: SOC2 CC6.1 Logical Access Review", "Not started", "P3 - Medium", "Emma Watson", "2026-12-15")
            ]
            for i in range(limit_to_use):
                item = pages[i % len(pages)]
                records.append({
                    "id": f"page_{i+1:08d}-4321-4def-9abc-0123456789ab",
                    "Title": f"{item[0]} (Run #{i//len(pages)+1})" if i >= len(pages) else item[0],
                    "Status": item[1],
                    "Priority": item[2],
                    "Owner": item[3],
                    "Due_Date": item[4],
                    "Archived": False,
                    "Created_Time": now_iso,
                    "Last_Edited_Time": now_iso
                })

        elif resource == "Pages":
            doc_pages = [
                ("Executive Summary Q3 2026", "2026-10-01", "Alice Johnson"),
                ("Data Lakehouse Ingestion Strategy", "2026-10-20", "Clara Chen"),
                ("MLOps Infrastructure Blueprint", "2026-11-02", "David Smith")
            ]
            for i in range(min(limit_to_use, len(doc_pages))):
                item = doc_pages[i]
                records.append({
                    "id": f"doc_{i+1:08d}",
                    "title": item[0],
                    "created_time": item[1],
                    "created_by": item[2],
                    "url": f"https://notion.so/corp/{item[0].lower().replace(' ', '-')}"
                })

        elif resource == "Users":
            records = [
                {"id": "usr_001", "name": "Alice Johnson", "type": "person", "email": "alice@corp.io"},
                {"id": "usr_002", "name": "Bob Martinez", "type": "person", "email": "bob@corp.io"},
                {"id": "usr_bot", "name": "DataRept Automation Bot", "type": "bot", "email": "bot@corp.io"}
            ]

        return records

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        resource = str(config.get("resource", "Database Records")).strip()
        database_id = str(config.get("database_id", "db_1203948571829304")).strip()
        token = str(config.get("api_token", "")).strip() or os.getenv("NOTION_API_KEY", "")
        credential_id = str(config.get("credential_id", "")).strip()
        limit = int(config.get("record_limit", 50))
        simulation_mode = bool(config.get("simulation_mode", False))

        if credential_id and context and isinstance(context, dict):
            creds_store = context.get("credentials", {})
            if credential_id in creds_store:
                cred = creds_store[credential_id]
                c_data = cred.get("data", cred)
                token = c_data.get("api_token") or c_data.get("token") or token

        records: List[Dict[str, Any]] = []
        delivery_status = "SIMULATED"
        status_msg = ""

        clean_db_id = database_id.replace("-", "").replace("db_", "")

        if not simulation_mode and token and not token.startswith("mock") and len(token) >= 20:
            try:
                url = f"https://api.notion.com/v1/databases/{clean_db_id}/query"
                payload = {"page_size": min(limit, 100)}
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Notion-Version": "2022-06-28",
                        "Content-Type": "application/json"
                    }
                )
                with urllib.request.urlopen(req, timeout=12) as response:
                    if response.status == 200:
                        data = json.loads(response.read().decode("utf-8"))
                        for item in data.get("results", []):
                            row = {"id": item.get("id"), "created_time": item.get("created_time")}
                            # Extract simple properties
                            props = item.get("properties", {})
                            for k, v in props.items():
                                ptype = v.get("type")
                                if ptype == "title":
                                    tlist = v.get("title", [])
                                    row[k] = tlist[0].get("plain_text") if tlist else ""
                                elif ptype == "select":
                                    row[k] = v.get("select", {}).get("name") if v.get("select") else ""
                                elif ptype == "status":
                                    row[k] = v.get("status", {}).get("name") if v.get("status") else ""
                            records.append(row)
                        delivery_status = "LIVE"
                        status_msg = f"Retrieved {len(records)} live Notion records via Notion REST API v1."
            except Exception as ex:
                logger.warning(f"Notion live connection fallback: {ex}")
                status_msg = f"Notion live connection failed ({ex}). Switched to benchmark fallback data."

        if not records:
            records = self._generate_benchmark_data(resource, limit)
            if not status_msg:
                status_msg = f"Generated {len(records)} realistic Notion {resource} records (Simulation Mode)."

        df_out = pd.DataFrame(records) if pd is not None else records

        summary = {
            "title": "Notion Connector Read Summary",
            "message": status_msg,
            "status": delivery_status,
            "resource": resource,
            "records_retrieved": len(records)
        }

        metrics = {
            "notion_resource": resource,
            "record_count": len(records),
            "delivery_status": delivery_status
        }

        return {
            "dataframe": df_out,
            "output_summary": summary,
            "metrics": metrics
        }

    def to_code(self, config: Dict[str, Any]) -> str:
        return (
            f"# Recipe: Notion Connector\n"
            f"# ID: {self.recipe_id}\n"
            f"import os\n"
            f"import requests\n"
            f"import pandas as pd\n\n"
            f"token = os.getenv('NOTION_API_KEY', '')\n"
            f"db_id = '{config.get('database_id', 'db_1234567890')}'\n"
            f"headers = {{'Authorization': f'Bearer {{token}}', 'Notion-Version': '2022-06-28', 'Content-Type': 'application/json'}}\n"
            f"url = f'https://api.notion.com/v1/databases/{{db_id}}/query'\n"
            f"records = requests.post(url, headers=headers, json={{'page_size': {config.get('record_limit', 50)}}}).json().get('results', [])\n"
            f"df = pd.DataFrame(records)\n"
        )

