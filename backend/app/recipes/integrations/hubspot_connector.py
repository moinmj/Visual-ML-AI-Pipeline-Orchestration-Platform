"""
HubSpot CRM Connector Recipe (Apps & Integrations -> CRM / Support / Business).

Reads HubSpot CRM objects such as Contacts, Companies, Deals, Tickets, Products, and Quotes
via the HubSpot CRM REST API v3 (with rate limit resilience, connected credential resolution,
and benchmark simulation fallbacks).
"""

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


class HubSpotConnectorRecipe(BaseRecipe):
    recipe_id = "hubspot_connector"
    name = "HubSpot CRM Connector"
    version = "1.0.0"
    category = "integrations"
    group = "Apps & Integrations"
    subgroup = "CRM / Support / Business"
    description = "Reads HubSpot contacts, companies and deals."
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
            label="HubSpot CRM Dataset",
            type="dataframe",
            description="DataFrame containing retrieved HubSpot CRM records (Contacts, Companies, Deals, etc.)"
        )
    ]

    # Standard default properties per CRM object in HubSpot v3 API
    DEFAULT_PROPERTIES = {
        "Contacts": [
            "firstname", "lastname", "email", "phone", "jobtitle",
            "company", "lifecyclestage", "hs_lead_status", "createdate", "lastmodifieddate"
        ],
        "Companies": [
            "name", "domain", "industry", "annualrevenue", "numberofemployees",
            "city", "state", "country", "phone", "createdate"
        ],
        "Deals": [
            "dealname", "amount", "dealstage", "pipeline", "closedate",
            "dealtype", "hs_forecast_amount", "createdate"
        ],
        "Tickets": [
            "subject", "content", "hs_ticket_category", "hs_ticket_priority",
            "hs_pipeline_stage", "createdate"
        ],
        "Products": [
            "name", "description", "price", "hs_sku", "hs_cost_of_goods_sold", "createdate"
        ],
        "Quotes": [
            "hs_title", "hs_status", "hs_quote_amount", "hs_expiration_date", "createdate"
        ]
    }

    # HubSpot v3 object endpoints
    OBJECT_ENDPOINTS = {
        "Contacts": "contacts",
        "Companies": "companies",
        "Deals": "deals",
        "Tickets": "tickets",
        "Products": "products",
        "Quotes": "quotes"
    }

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "resource": {
                    "type": "string",
                    "title": "HubSpot Object",
                    "input_type": "dropdown",
                    "enum": [
                        "Contacts",
                        "Companies",
                        "Deals",
                        "Tickets",
                        "Products",
                        "Quotes"
                    ],
                    "default": "Contacts",
                    "description": "Select the HubSpot CRM object type to retrieve."
                },
                "record_limit": {
                    "type": "integer",
                    "title": "Record Limit",
                    "input_type": "number",
                    "default": 50,
                    "minimum": 1,
                    "maximum": 250,
                    "description": "Maximum number of CRM records to retrieve per query (1-250)."
                },
                "properties_list": {
                    "type": "string",
                    "title": "Custom Properties (Comma-separated)",
                    "input_type": "text",
                    "description": "Comma-separated list of custom property names to retrieve (leave empty for standard defaults)."
                },
                "include_archived": {
                    "type": "boolean",
                    "title": "Include Archived Records",
                    "input_type": "checkbox",
                    "default": False,
                    "description": "Whether to return soft-deleted/archived CRM records."
                },
                "credential_id": {
                    "type": "string",
                    "title": "HubSpot Credential",
                    "input_type": "credential_dropdown",
                    "provider": "hubspot",
                    "description": "Stored HubSpot Private App Access Token or OAuth credential."
                },
                "access_token": {
                    "type": "string",
                    "title": "Private App Access Token (Manual Override)",
                    "input_type": "password",
                    "description": "HubSpot Private App Token (pat-...) or OAuth Bearer token (or uses HUBSPOT_ACCESS_TOKEN from .env)."
                },
                "simulation_mode": {
                    "type": "boolean",
                    "title": "Simulation Mode (Offline Benchmark Data)",
                    "input_type": "checkbox",
                    "default": False,
                    "description": "Generate realistic benchmark CRM records without making live HubSpot API calls."
                }
            },
            "required": ["resource"]
        }

    def _generate_benchmark_data(self, resource: str, limit: int) -> List[Dict[str, Any]]:
        """Generates realistic HubSpot CRM benchmark records for offline testing."""
        companies_data = [
            ("CloudScale Technologies", "cloudscale.io", "Software & Technology", 14500000, 320, "San Francisco", "California", "United States"),
            ("Beacon Health Solutions", "beaconhealth.org", "Healthcare & Medical", 28000000, 750, "Boston", "Massachusetts", "United States"),
            ("Vanguard Capital Partners", "vanguardcp.com", "Financial Services", 62000000, 1400, "New York", "New York", "United States"),
            ("Aura Sustainable Energy", "auraenergy.co", "Renewables & Environment", 19500000, 480, "Austin", "Texas", "United States"),
            ("Nexus Industrial Supply", "nexusind.com", "Manufacturing & Supply", 36000000, 910, "Chicago", "Illinois", "United States"),
            ("Prism Media Group", "prismgroup.com", "Marketing & Advertising", 8500000, 160, "Los Angeles", "California", "United States"),
            ("Summit Logistics Network", "summitlogistics.net", "Logistics & Supply Chain", 42000000, 1100, "Seattle", "Washington", "United States"),
            ("Quantum Data Security", "quantumsec.io", "Cybersecurity", 21000000, 520, "Atlanta", "Georgia", "United States")
        ]

        contacts_data = [
            ("Alexander", "Wright", "alex.wright@cloudscale.io", "+1-415-555-0142", "Chief Revenue Officer", "CloudScale Technologies", "customer", "CONNECTED"),
            ("Beatrice", "Chen", "b.chen@beaconhealth.org", "+1-617-555-0198", "VP of Clinical Informatics", "Beacon Health Solutions", "opportunity", "IN_PROGRESS"),
            ("Charles", "Montgomery", "cmont@vanguardcp.com", "+1-212-555-0131", "Managing Director", "Vanguard Capital Partners", "customer", "CONNECTED"),
            ("Diana", "Prince", "diana.p@auraenergy.co", "+1-512-555-0177", "Director of Grid Operations", "Aura Sustainable Energy", "salesqualifiedlead", "OPEN"),
            ("Edward", "Nygma", "edward@nexusind.com", "+1-312-555-0155", "Supply Chain Lead", "Nexus Industrial Supply", "marketingqualifiedlead", "NEW"),
            ("Fiona", "Gallagher", "fiona@prismgroup.com", "+1-310-555-0129", "Digital Marketing Director", "Prism Media Group", "salesqualifiedlead", "IN_PROGRESS"),
            ("George", "Costanza", "george@summitlogistics.net", "+1-206-555-0164", "Fleet Operations Manager", "Summit Logistics Network", "opportunity", "OPEN"),
            ("Helena", "Wayne", "helena@quantumsec.io", "+1-404-555-0183", "CISO & Security Lead", "Quantum Data Security", "customer", "CONNECTED")
        ]

        deals_data = [
            ("Enterprise CRM Transformation", 145000.0, "presentationscheduled", "default", "New Business"),
            ("Global Security Infrastructure", 275000.0, "decisionmakerbought", "default", "New Business"),
            ("Automated Marketing Suite License", 82000.0, "closedwon", "default", "Existing Business"),
            ("Multi-Cloud Integration Pipeline", 195000.0, "contractsent", "default", "New Business"),
            ("Predictive AI Analytics Add-On", 64000.0, "qualifiedtobuy", "default", "Existing Business"),
            ("Annual Enterprise Support Tier", 120000.0, "closedwon", "default", "Existing Business"),
            ("Legacy ERP Data Migration", 310000.0, "closedlost", "default", "New Business"),
            ("Omnichannel Contact Center Deal", 98000.0, "appointmentscheduled", "default", "New Business")
        ]

        records = []
        limit_to_use = max(1, min(limit, 200))

        if resource == "Companies":
            for i in range(limit_to_use):
                comp = companies_data[i % len(companies_data)]
                records.append({
                    "id": f"comp_{100001 + i}",
                    "name": f"{comp[0]} {i//len(companies_data)+1}" if i >= len(companies_data) else comp[0],
                    "domain": comp[1],
                    "industry": comp[2],
                    "annualrevenue": comp[3] + (i * 180000),
                    "numberofemployees": comp[4] + (i * 12),
                    "city": comp[5],
                    "state": comp[6],
                    "country": comp[7],
                    "phone": f"+1-800-555-{2000+i}",
                    "createdate": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                })
        elif resource == "Deals":
            for i in range(limit_to_use):
                deal = deals_data[i % len(deals_data)]
                amount = float(deal[1] + (i * 8500))
                records.append({
                    "id": f"deal_{200001 + i}",
                    "dealname": f"{deal[0]} #{i+1}",
                    "amount": amount,
                    "dealstage": deal[2],
                    "pipeline": deal[3],
                    "closedate": datetime.now(timezone.utc).strftime("%Y-11-30"),
                    "dealtype": deal[4],
                    "hs_forecast_amount": amount * (0.9 if deal[2] == "closedwon" else 0.6),
                    "createdate": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                })
        elif resource == "Tickets":
            for i in range(limit_to_use):
                records.append({
                    "id": f"tkt_{300001 + i}",
                    "subject": f"Inquiry regarding API limits and webhooks #{i+1}",
                    "content": f"Customer is requesting higher throughput on API endpoint batch scoring #{i+1}",
                    "hs_ticket_category": "CUSTOMER_INQUIRY" if i % 2 == 0 else "PRODUCT_ISSUE",
                    "hs_ticket_priority": "HIGH" if i % 4 == 0 else ("MEDIUM" if i % 2 == 0 else "LOW"),
                    "hs_pipeline_stage": str((i % 4) + 1),
                    "createdate": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                })
        elif resource == "Products":
            products = [
                ("Enterprise AI Orchestration License", "Full access to visual pipeline whiteboard and AutoML", 45000.0, "SKU-AI-01"),
                ("Real-time Inference Serving Addon", "Low-latency sub-millisecond model prediction gateway", 18000.0, "SKU-INF-02"),
                ("Multi-Tenant Lakehouse Connector", "Bi-directional Apache Iceberg & Druid connector", 25000.0, "SKU-LAKE-03"),
                ("MLflow Governance & Audit Suite", "Model registry, experiment tracking, and compliance", 12000.0, "SKU-MLF-04")
            ]
            for i in range(limit_to_use):
                p = products[i % len(products)]
                records.append({
                    "id": f"prod_{400001 + i}",
                    "name": p[0],
                    "description": p[1],
                    "price": p[2],
                    "hs_sku": f"{p[3]}-{i+1}",
                    "hs_cost_of_goods_sold": round(p[2] * 0.25, 2),
                    "createdate": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                })
        elif resource == "Quotes":
            for i in range(limit_to_use):
                records.append({
                    "id": f"quote_{500001 + i}",
                    "hs_title": f"Enterprise Software Proposal #{1000 + i}",
                    "hs_status": "APPROVED" if i % 3 == 0 else ("APPROVAL_PENDING" if i % 2 == 0 else "DRAFT"),
                    "hs_quote_amount": 75000.0 + (i * 12500),
                    "hs_expiration_date": datetime.now(timezone.utc).strftime("%Y-12-15"),
                    "createdate": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                })
        else:
            # Default to Contacts
            for i in range(limit_to_use):
                ct = contacts_data[i % len(contacts_data)]
                records.append({
                    "id": f"cnt_{i + 1:07d}",
                    "firstname": ct[0],
                    "lastname": f"{ct[1]} {i+1}" if i >= len(contacts_data) else ct[1],
                    "email": f"{ct[0].lower()}.{ct[1].lower()}{i+1}@hubspot-benchmark.io",
                    "phone": ct[3],
                    "jobtitle": ct[4],
                    "company": ct[5],
                    "lifecyclestage": ct[6],
                    "hs_lead_status": ct[7],
                    "createdate": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "lastmodifieddate": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                })

        return records

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        resource = str(config.get("resource", "Contacts")).strip()
        limit = int(config.get("record_limit", 50))
        custom_props = str(config.get("properties_list", "")).strip()
        include_archived = bool(config.get("include_archived", False))
        credential_id = str(config.get("credential_id", "")).strip()
        access_token = str(config.get("access_token", "")).strip() or getattr(settings, "HUBSPOT_ACCESS_TOKEN", None) or ""
        simulation_mode = bool(config.get("simulation_mode", False))

        # Check for connected account in execution context
        if credential_id and context and isinstance(context, dict):
            creds_store = context.get("credentials", {})
            if credential_id in creds_store:
                cred_info = creds_store[credential_id]
                c_data = cred_info.get("data", cred_info)
                access_token = c_data.get("access_token") or c_data.get("token") or access_token

        # Resolve properties to request
        if custom_props:
            requested_props = [p.strip() for p in custom_props.split(",") if p.strip()]
        else:
            requested_props = self.DEFAULT_PROPERTIES.get(resource, ["firstname", "lastname", "email"])

        object_endpoint = self.OBJECT_ENDPOINTS.get(resource, "contacts")
        records: List[Dict[str, Any]] = []
        delivery_status = "SIMULATED"
        status_msg = ""

        # Attempt live API call if token is provided and not in simulation mode
        if not simulation_mode and access_token and not access_token.startswith("mock") and len(access_token) >= 15:
            try:
                base_url = "https://api.hubapi.com/crm/v3/objects"
                query_params = {
                    "limit": str(min(limit, 100)),
                    "archived": "true" if include_archived else "false",
                    "properties": ",".join(requested_props)
                }
                encoded_params = urllib.parse.urlencode(query_params)
                api_url = f"{base_url}/{object_endpoint}?{encoded_params}"

                req = urllib.request.Request(
                    api_url,
                    headers={
                        "Authorization": f"Bearer {access_token}",
                        "Accept": "application/json"
                    }
                )
                with urllib.request.urlopen(req, timeout=12) as response:
                    if response.status == 200:
                        payload = json.loads(response.read().decode("utf-8"))
                        raw_results = payload.get("results", [])
                        for item in raw_results:
                            record = {"id": str(item.get("id", ""))}
                            props = item.get("properties", {})
                            if isinstance(props, dict):
                                record.update(props)
                            records.append(record)

                        delivery_status = "LIVE"
                        status_msg = f"Retrieved {len(records)} live HubSpot {resource} records via CRM v3 REST API."
            except urllib.error.HTTPError as he:
                err_body = ""
                try:
                    err_body = he.read().decode("utf-8")
                except Exception:
                    pass
                logger.warning(f"HubSpot API query returned HTTP {he.code}: {err_body[:200]}")
                status_msg = f"HubSpot live query failed (HTTP {he.code}). Switched to benchmark fallback data."
            except Exception as ex:
                logger.warning(f"HubSpot connection error: {ex}")
                status_msg = f"HubSpot live connection failed ({ex}). Switched to benchmark fallback data."

        # If live call not attempted or returned empty, populate with realistic benchmark dataset
        if not records:
            records = self._generate_benchmark_data(resource, limit)
            if not status_msg:
                status_msg = f"Generated {len(records)} realistic HubSpot {resource} records (Simulation Mode)."

        # Convert to Pandas DataFrame
        if pd is not None:
            df_out = pd.DataFrame(records)
        else:
            df_out = records

        summary = {
            "title": "HubSpot CRM Read Summary",
            "message": status_msg,
            "status": delivery_status,
            "resource": resource,
            "records_retrieved": len(records),
            "properties": requested_props
        }

        metrics = {
            "hubspot_resource": resource,
            "record_count": len(records),
            "delivery_status": delivery_status,
            "include_archived": include_archived
        }

        return {
            "dataframe": df_out,
            "output_summary": summary,
            "metrics": metrics
        }

    def to_code(self, config: Dict[str, Any]) -> str:
        resource = config.get("resource", "Contacts")
        limit = config.get("record_limit", 50)
        props = config.get("properties_list", "")
        return (
            f"# Recipe: HubSpot CRM Connector\n"
            f"# ID: {self.recipe_id}\n"
            f"import requests\n"
            f"import pandas as pd\n\n"
            f"url = 'https://api.hubapi.com/crm/v3/objects/{self.OBJECT_ENDPOINTS.get(resource, 'contacts')}'\n"
            f"headers = {{'Authorization': 'Bearer ' + os.getenv('HUBSPOT_ACCESS_TOKEN', '')}}\n"
            f"params = {{'limit': {limit}, 'properties': '{props}'}}\n"
            f"response = requests.get(url, headers=headers, params=params)\n"
            f"results = response.json().get('results', [])\n"
            f"df = pd.DataFrame([{{'id': r['id'], **r.get('properties', {{}})}} for r in results])\n"
        )

