"""
Freshdesk Connector Recipe (Apps & Integrations -> CRM / Support / Business).

Reads Freshdesk support tickets and customer contacts via Freshdesk REST API v2
(with credential resolution, rate-limit resilience, and benchmark simulation fallbacks).
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
from backend.app.core.config import settings


class FreshdeskConnectorRecipe(BaseRecipe):
    recipe_id = "freshdesk_connector"
    name = "Freshdesk Connector"
    version = "1.0.0"
    category = "integrations"
    group = "Apps & Integrations"
    subgroup = "CRM / Support / Business"
    description = "Reads Freshdesk tickets and contacts."
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
            label="Freshdesk Dataset",
            type="dataframe",
            description="DataFrame containing retrieved Freshdesk support tickets or customer contacts"
        )
    ]

    RESOURCE_ENDPOINTS = {
        "Tickets": "/api/v2/tickets",
        "Contacts": "/api/v2/contacts"
    }

    # Human-readable label mappings for Freshdesk numeric codes
    STATUS_LABELS = {
        2: "Open",
        3: "Pending",
        4: "Resolved",
        5: "Closed"
    }

    PRIORITY_LABELS = {
        1: "Low",
        2: "Medium",
        3: "High",
        4: "Urgent"
    }

    SOURCE_LABELS = {
        1: "Email",
        2: "Portal",
        3: "Phone",
        7: "Chat",
        8: "Mobihelp",
        9: "Feedback Widget",
        10: "Outbound Email"
    }

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "resource": {
                    "type": "string",
                    "title": "Freshdesk Resource",
                    "input_type": "dropdown",
                    "enum": ["Tickets", "Contacts"],
                    "default": "Tickets",
                    "description": "Select the Freshdesk resource to retrieve (Tickets or Contacts)."
                },
                "domain": {
                    "type": "string",
                    "title": "Freshdesk Domain / URL",
                    "input_type": "text",
                    "default": "mycompany",
                    "description": "Your Freshdesk account domain (e.g. 'acme' for https://acme.freshdesk.com) or full URL."
                },
                "api_key": {
                    "type": "string",
                    "title": "Freshdesk API Key",
                    "input_type": "password",
                    "description": "Your Freshdesk API Key (found under Profile Settings, or uses FRESHDESK_API_KEY from .env)."
                },
                "credential_id": {
                    "type": "string",
                    "title": "Connected Freshdesk Credential",
                    "input_type": "credential_dropdown",
                    "provider": "freshdesk",
                    "description": "Select a stored Freshdesk account credential or enter manual API key above."
                },
                "status_filter": {
                    "type": "string",
                    "title": "Ticket Status Filter",
                    "input_type": "dropdown",
                    "enum": ["All", "Open", "Pending", "Resolved", "Closed"],
                    "default": "All",
                    "description": "Filter tickets by status (applies when resource is 'Tickets')."
                },
                "priority_filter": {
                    "type": "string",
                    "title": "Ticket Priority Filter",
                    "input_type": "dropdown",
                    "enum": ["All", "Low", "Medium", "High", "Urgent"],
                    "default": "All",
                    "description": "Filter tickets by priority (applies when resource is 'Tickets')."
                },
                "record_limit": {
                    "type": "integer",
                    "title": "Record Limit",
                    "input_type": "number",
                    "default": 50,
                    "minimum": 1,
                    "maximum": 100,
                    "description": "Maximum number of records to retrieve (1-100)."
                },
                "output_format": {
                    "type": "string",
                    "title": "Output Format",
                    "input_type": "dropdown",
                    "enum": ["DataFrame", "JSON Records", "JSON Dict"],
                    "default": "DataFrame",
                    "description": "Output format: standard DataFrame or JSON format for downstream API responses."
                },
                "simulation_mode": {
                    "type": "boolean",
                    "title": "Simulation Mode (Offline Benchmark Data)",
                    "input_type": "checkbox",
                    "default": False,
                    "description": "Generate realistic benchmark customer support records without making live Freshdesk API calls."
                }
            },
            "required": ["resource"]
        }

    def _generate_benchmark_data(self, resource: str, limit: int, status_filter: str = "All", priority_filter: str = "All") -> List[Dict[str, Any]]:
        """Generates realistic Freshdesk tickets and contacts for testing."""
        limit_to_use = max(1, min(limit, 100))
        records: List[Dict[str, Any]] = []

        now_iso = datetime.now(timezone.utc).isoformat()

        if resource == "Tickets":
            ticket_templates = [
                ("Payment gateway 400 Bad Request during checkout", "Customer unable to complete enterprise renewal with Amex card", 4, 2, 1, "billing,payment,urgent"),
                ("Feature request: Bulk user role import via CSV", "Admin wants to upload 500 team members with pre-assigned groups", 2, 4, 2, "feature,users"),
                ("Dashboard metrics loading very slow on Safari", "Charts take over 8 seconds to render for team leads", 3, 2, 7, "performance,ui"),
                ("Cannot reset password via mobile verification", "SMS OTP token expires before arriving on customer phone", 3, 3, 3, "auth,sms,bug"),
                ("Webhook delivery failing with HTTP 504 gateway timeout", "Customer downstream endpoint receiving timeout errors", 4, 2, 1, "webhook,api"),
                ("Inquiry about GDPR data deletion request SLA", "Legal team inquiring about automated customer record purge", 1, 4, 2, "compliance,gdpr"),
                ("Database sync failure: PostgreSQL connector timeout", "Nightly batch synchronization dropped connection after 30s", 4, 3, 1, "database,connector"),
                ("Custom domain SSL certificate renewal notification", "Customer wants to ensure automated Let's Encrypt renews without downtime", 2, 5, 2, "domain,ssl")
            ]

            status_map_rev = {"Open": 2, "Pending": 3, "Resolved": 4, "Closed": 5}
            priority_map_rev = {"Low": 1, "Medium": 2, "High": 3, "Urgent": 4}

            for i in range(limit_to_use):
                tmpl = ticket_templates[i % len(ticket_templates)]
                pri_code = priority_map_rev.get(priority_filter, tmpl[2])
                stat_code = status_map_rev.get(status_filter, tmpl[3])

                records.append({
                    "id": 200001 + i,
                    "subject": f"{tmpl[0]} (#{i+1})",
                    "description_text": tmpl[1],
                    "status_code": stat_code,
                    "status": self.STATUS_LABELS.get(stat_code, "Open"),
                    "priority_code": pri_code,
                    "priority": self.PRIORITY_LABELS.get(pri_code, "Medium"),
                    "source_code": tmpl[4],
                    "source": self.SOURCE_LABELS.get(tmpl[4], "Email"),
                    "requester_id": 10001 + (i % 8),
                    "responder_id": 30001 + (i % 3),
                    "company_id": 50001 + (i % 4),
                    "type": "Incident" if pri_code >= 3 else "Feature Request",
                    "tags": tmpl[5].split(","),
                    "created_at": now_iso,
                    "updated_at": now_iso
                })

        elif resource == "Contacts":
            contact_templates = [
                ("Alexander Vance", "alex.vance@blackmesa.org", "+1-555-0201", "Director of Research", "Black Mesa Research"),
                ("Beatrix Kiddo", "bkiddo@viperassassins.com", "+1-555-0202", "Senior Operations Lead", "Viper Global Logistics"),
                ("Carlos Santana", "carlos.s@soundwave.io", "+1-555-0203", "VP of Engineering", "Soundwave Digital"),
                ("Dana Scully", "dana.scully@fbi-sci.gov", "+1-555-0204", "Chief Forensic Analyst", "Forensics & Insights Corp"),
                ("Edward Nygma", "edward.n@riddlercorp.com", "+1-555-0205", "Principal Data Architect", "Enigma Analytics"),
                ("Fiona Gallagher", "fiona@gallagherlogistics.com", "+1-555-0206", "Head of Customer Success", "Gallagher Group"),
                ("George Costanza", "george@vandelayindustries.com", "+1-555-0207", "Lead Architect", "Vandelay Industries"),
                ("Helena Wayne", "helena@gothamsolutions.net", "+1-555-0208", "Compliance Officer", "Wayne Technologies")
            ]
            for i in range(limit_to_use):
                ct = contact_templates[i % len(contact_templates)]
                records.append({
                    "id": 10001 + i,
                    "name": f"{ct[0]} {i+1}" if i >= len(contact_templates) else ct[0],
                    "email": f"contact{i+1}_{ct[1]}",
                    "phone": ct[2],
                    "mobile": ct[2],
                    "job_title": ct[3],
                    "company_id": 50001 + (i % 4),
                    "active": True,
                    "time_zone": "UTC",
                    "created_at": now_iso,
                    "updated_at": now_iso
                })

        return records

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        resource = str(config.get("resource", "Tickets")).strip()
        domain = str(config.get("domain", "mycompany")).strip()
        api_key = str(config.get("api_key", "")).strip() or getattr(settings, "FRESHDESK_API_KEY", None) or os.getenv("FRESHDESK_API_KEY", "")
        credential_id = str(config.get("credential_id", "")).strip()
        status_filter = str(config.get("status_filter", "All")).strip()
        priority_filter = str(config.get("priority_filter", "All")).strip()
        limit = int(config.get("record_limit", config.get("limit", 50)))
        simulation_mode = bool(config.get("simulation_mode", config.get("simulate_if_no_auth", False)))
        output_format = str(config.get("output_format", "dataframe")).strip().lower()

        # Resolve credentials if stored in context
        if credential_id and context and isinstance(context, dict):
            creds_store = context.get("credentials", {})
            if credential_id in creds_store:
                cred_info = creds_store[credential_id]
                c_data = cred_info.get("data", cred_info)
                api_key = c_data.get("api_key") or c_data.get("token") or api_key
                domain = c_data.get("domain") or domain

        # Clean domain / base URL
        if domain.startswith("http://") or domain.startswith("https://"):
            base_url = domain.rstrip("/")
        else:
            clean_dom = domain.replace(".freshdesk.com", "").strip()
            base_url = f"https://{clean_dom}.freshdesk.com"

        records: List[Dict[str, Any]] = []
        delivery_status = "SIMULATED"
        status_msg = ""
        endpoint_path = self.RESOURCE_ENDPOINTS.get(resource, "/api/v2/tickets")

        # Live Freshdesk API Call Attempt
        if not simulation_mode and api_key and len(api_key) >= 10 and not api_key.startswith("mock"):
            try:
                per_page = max(1, min(limit, 100))
                query_params: Dict[str, Any] = {"per_page": per_page}

                status_map = {"Open": 2, "Pending": 3, "Resolved": 4, "Closed": 5}
                priority_map = {"Low": 1, "Medium": 2, "High": 3, "Urgent": 4}

                if resource == "Tickets":
                    if status_filter in status_map:
                        query_params["filter"] = f"status:{status_map[status_filter]}"
                    if priority_filter in priority_map:
                        query_params["priority"] = priority_map[priority_filter]

                encoded_params = urllib.parse.urlencode(query_params)
                full_url = f"{base_url}{endpoint_path}?{encoded_params}"

                # Freshdesk auth uses Basic Auth: api_key:X
                auth_str = f"{api_key}:X"
                encoded_auth = base64.b64encode(auth_str.encode("utf-8")).decode("ascii")

                headers = {
                    "Authorization": f"Basic {encoded_auth}",
                    "Accept": "application/json",
                    "User-Agent": "Datarept-ML-Platform/1.0"
                }

                req = urllib.request.Request(full_url, headers=headers)
                with urllib.request.urlopen(req, timeout=12) as response:
                    if response.status == 200:
                        raw_records = json.loads(response.read().decode("utf-8"))
                        if isinstance(raw_records, list):
                            for r in raw_records:
                                # Enrich with human-readable labels if tickets
                                if resource == "Tickets":
                                    st_code = r.get("status")
                                    pr_code = r.get("priority")
                                    src_code = r.get("source")
                                    if st_code in self.STATUS_LABELS:
                                        r["status_label"] = self.STATUS_LABELS[st_code]
                                    if pr_code in self.PRIORITY_LABELS:
                                        r["priority_label"] = self.PRIORITY_LABELS[pr_code]
                                    if src_code in self.SOURCE_LABELS:
                                        r["source_label"] = self.SOURCE_LABELS[src_code]
                                records.append(r)
                        elif isinstance(raw_records, dict):
                            records = [raw_records]

                        delivery_status = "LIVE"
                        status_msg = f"Retrieved {len(records)} live Freshdesk {resource} records from {base_url}."
            except urllib.error.HTTPError as he:
                err_body = ""
                try:
                    err_body = he.read().decode("utf-8")
                except Exception:
                    pass
                logger.warning(f"Freshdesk API returned HTTP {he.code}: {err_body[:200]}")
                status_msg = f"Freshdesk live API query failed (HTTP {he.code}). Using realistic benchmark fallback."
            except Exception as ex:
                logger.warning(f"Freshdesk connection failed: {ex}")
                status_msg = f"Freshdesk live connection failed ({ex}). Using realistic benchmark fallback."

        # Offline / Benchmark Fallback
        if not records:
            records = self._generate_benchmark_data(resource, limit, status_filter, priority_filter)
            if not status_msg:
                status_msg = f"Generated {len(records)} realistic Freshdesk {resource} benchmark records (Simulation Mode)."

        # Convert to Pandas DataFrame
        if pd is not None:
            df_out = pd.DataFrame(records)
        else:
            df_out = records

        summary = {
            "title": "Freshdesk Connector Read Summary",
            "message": status_msg,
            "status": delivery_status,
            "resource": resource,
            "domain": domain,
            "records_retrieved": len(records),
            "endpoint": endpoint_path
        }

        metrics = {
            "freshdesk_resource": resource,
            "record_count": len(records),
            "delivery_status": delivery_status,
            "domain": domain
        }

        if "json_dict" in output_format or output_format == "json dict":
            output_val = df_out.to_dict() if hasattr(df_out, "to_dict") else records
        elif "json" in output_format:
            output_val = records
        else:
            output_val = df_out

        return {
            "dataframe": df_out,
            "output": output_val,
            "output_summary": summary,
            "metrics": metrics
        }

    def to_code(self, config: Dict[str, Any]) -> str:
        resource = config.get("resource", "Tickets")
        dom = config.get("domain", "mycompany")
        lim = config.get("record_limit", 50)
        return (
            f"# Recipe: Freshdesk Connector\n"
            f"# ID: {self.recipe_id}\n"
            f"import requests\n"
            f"import pandas as pd\n\n"
            f"url = 'https://{dom}.freshdesk.com{self.RESOURCE_ENDPOINTS.get(resource, '/api/v2/tickets')}'\n"
            f"auth = (os.getenv('FRESHDESK_API_KEY'), 'X')\n"
            f"response = requests.get(url, auth=auth, params={{'per_page': {lim}}})\n"
            f"df = pd.DataFrame(response.json())\n"
        )
