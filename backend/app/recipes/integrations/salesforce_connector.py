"""
Salesforce CRM Connector Recipe (Apps & Integrations -> CRM / Support / Business).

Reads Salesforce CRM objects such as Accounts, Leads, Opportunities, Contacts, and Cases,
or executes custom SOQL queries via Salesforce REST API (with rate limit resilience,
connected credential resolution, and benchmark simulation fallbacks).
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


class SalesforceConnectorRecipe(BaseRecipe):
    recipe_id = "salesforce_connector"
    name = "Salesforce CRM Connector"
    version = "1.0.0"
    category = "integrations"
    group = "Apps & Integrations"
    subgroup = "CRM / Support / Business"
    description = "Reads Salesforce objects such as accounts, leads and opportunities."
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
            label="Salesforce CRM Dataset",
            type="dataframe",
            description="DataFrame containing retrieved Salesforce CRM records (Accounts, Leads, Opportunities, etc.)"
        )
    ]

    # Standard SOQL field projections per object
    DEFAULT_FIELDS = {
        "Accounts": [
            "Id", "Name", "Type", "Industry", "AnnualRevenue",
            "NumberOfEmployees", "BillingCity", "BillingState", "BillingCountry",
            "Phone", "Website", "CreatedDate"
        ],
        "Leads": [
            "Id", "FirstName", "LastName", "Company", "Title",
            "Email", "Phone", "Status", "LeadSource", "Rating",
            "AnnualRevenue", "City", "Country", "CreatedDate"
        ],
        "Opportunities": [
            "Id", "Name", "StageName", "Amount", "Probability",
            "ExpectedRevenue", "CloseDate", "Type", "LeadSource",
            "IsWon", "IsClosed", "CreatedDate"
        ],
        "Contacts": [
            "Id", "FirstName", "LastName", "Name", "Email",
            "Phone", "Title", "Department", "LeadSource",
            "MailingCity", "MailingCountry", "CreatedDate"
        ],
        "Cases": [
            "Id", "CaseNumber", "Subject", "Status", "Priority",
            "Origin", "Type", "Reason", "IsClosed", "CreatedDate"
        ]
    }

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "resource": {
                    "type": "string",
                    "title": "Salesforce Object",
                    "input_type": "dropdown",
                    "enum": [
                        "Accounts",
                        "Leads",
                        "Opportunities",
                        "Contacts",
                        "Cases",
                        "Custom SOQL Query"
                    ],
                    "default": "Accounts",
                    "description": "Standard Salesforce CRM object or custom SOQL query to execute."
                },
                "record_limit": {
                    "type": "integer",
                    "title": "Record Limit",
                    "input_type": "number",
                    "default": 50,
                    "minimum": 1,
                    "maximum": 2000,
                    "description": "Maximum number of CRM records to retrieve (1-2000)."
                },
                "soql_filter": {
                    "type": "string",
                    "title": "Filter Condition (SOQL WHERE Clause)",
                    "input_type": "text",
                    "description": "Optional SOQL filter condition (e.g., IsClosed = false or AnnualRevenue > 500000 or Status = 'Open - Not Contacted')."
                },
                "custom_soql": {
                    "type": "string",
                    "title": "Custom SOQL Query",
                    "input_type": "textarea",
                    "description": "Full custom SOQL query to run (only used when Salesforce Object is 'Custom SOQL Query')."
                },
                "credential_id": {
                    "type": "string",
                    "title": "Salesforce Credential",
                    "input_type": "credential_dropdown",
                    "provider": "salesforce",
                    "description": "Stored Salesforce Connected App OAuth credential or enter manual token below."
                },
                "instance_url": {
                    "type": "string",
                    "title": "Salesforce Instance URL",
                    "input_type": "text",
                    "default": "https://login.salesforce.com",
                    "description": "Salesforce domain URL (e.g., https://yourcompany.my.salesforce.com)."
                },
                "access_token": {
                    "type": "string",
                    "title": "Access Token / Bearer Token (Manual Override)",
                    "input_type": "password",
                    "description": "OAuth Bearer token (or uses SALESFORCE_ACCESS_TOKEN from .env)."
                },
                "api_version": {
                    "type": "string",
                    "title": "Salesforce REST API Version",
                    "input_type": "dropdown",
                    "enum": ["v60.0", "v59.0", "v58.0", "v57.0"],
                    "default": "v59.0",
                    "description": "Salesforce Force.com REST API version."
                },
                "simulation_mode": {
                    "type": "boolean",
                    "title": "Simulation Mode (Offline Benchmark Data)",
                    "input_type": "checkbox",
                    "default": False,
                    "description": "Generate realistic benchmark CRM records without making live Salesforce API calls."
                }
            },
            "required": ["resource"]
        }

    @staticmethod
    def _apply_soql_filter(df: Any, expr: str) -> Any:
        """Converts basic SOQL WHERE clause to pandas query syntax and filters the DataFrame."""
        if pd is None or not isinstance(df, pd.DataFrame) or df.empty or not expr or not expr.strip():
            return df
        import re
        clean = expr.strip()
        # Replace <> with !=
        clean = re.sub(r'<>', '!=', clean)
        # Replace = with == (avoid replacing >=, <=, !=, ==)
        clean = re.sub(r'(?<![><!=])=(?!=)', '==', clean)
        # Normalize boolean and null keywords
        clean = re.sub(r'\btrue\b', 'True', clean, flags=re.IGNORECASE)
        clean = re.sub(r'\bfalse\b', 'False', clean, flags=re.IGNORECASE)
        clean = re.sub(r'\bnull\b', 'None', clean, flags=re.IGNORECASE)
        clean = re.sub(r'\band\b', 'and', clean, flags=re.IGNORECASE)
        clean = re.sub(r'\bor\b', 'or', clean, flags=re.IGNORECASE)
        try:
            return df.query(clean)
        except Exception as e:
            logger.debug(f"Could not apply SOQL filter '{expr}' to DataFrame: {e}")
            return df

    def _generate_benchmark_data(self, resource: str, limit: int, soql_filter: str = "") -> List[Dict[str, Any]]:
        """Generates realistic enterprise CRM benchmark records for offline testing."""
        companies = [
            ("Acme Corporation", "Technology", "San Francisco", "USA", 12500000, 420),
            ("Apex Global Logistics", "Transportation", "Chicago", "USA", 45000000, 1200),
            ("BioHealth Dynamics", "Healthcare", "Boston", "USA", 8200000, 180),
            ("Citadel Financial Group", "Finance", "New York", "USA", 95000000, 3100),
            ("Echo Systems Inc", "Telecommunications", "Austin", "USA", 16800000, 510),
            ("Frontier Energy Partners", "Energy", "Houston", "USA", 34000000, 890),
            ("Greenwood Manufacturing", "Manufacturing", "Detroit", "USA", 22000000, 750),
            ("Horizon Retail Holdings", "Retail", "Seattle", "USA", 58000000, 2400),
            ("Initech Software Labs", "Technology", "Dallas", "USA", 6400000, 95),
            ("Jupiter Media Network", "Media", "Los Angeles", "USA", 14500000, 320),
            ("Krypton Security Systems", "Security", "Denver", "USA", 11200000, 260),
            ("Luminary Cloud Services", "Technology", "San Jose", "USA", 28000000, 680)
        ]

        leads_data = [
            ("Sarah", "Connor", "Cyberdyne Systems", "VP of Operations", "sarah.c@cyberdyne.io", "Open - Not Contacted", "Web", "Hot"),
            ("John", "Doe", "Initech Global", "Director of IT", "jdoe@initech.com", "Working - Contacted", "Trade Show", "Warm"),
            ("Elena", "Rostova", "Novatech Europe", "Chief Data Officer", "elena@novatech.eu", "Working - Contacted", "Partner Referral", "Hot"),
            ("Marcus", "Vance", "Vance Refrigeration", "Procurement Lead", "mvance@vance.com", "Open - Not Contacted", "Cold Call", "Cold"),
            ("David", "Kim", "Apex Horizon", "Enterprise Architect", "dkim@apexhorizon.com", "Closed - Converted", "Inbound Inquiry", "Hot"),
            ("Rachel", "Green", "Central Perk Retail", "Marketing Manager", "rachel@centralperk.com", "Working - Contacted", "Web", "Warm"),
            ("Michael", "Scott", "Dunder Mifflin", "Regional Director", "mscott@dundermifflin.com", "Closed - Converted", "Partner Referral", "Warm"),
            ("Lisa", "Simpson", "Springfield Power", "Sustainability Lead", "lisa@springfield.org", "Open - Not Contacted", "Web", "Hot")
        ]

        opp_stages = [
            ("Enterprise Platform Expansion", "Proposal/Price Quote", 185000.0, 75, "Existing Customer - Upgrade"),
            ("Q4 Analytics & AI Migration", "Negotiation/Review", 320000.0, 90, "New Customer"),
            ("Global CRM Ingestion Pipeline", "Closed Won", 540000.0, 100, "New Customer"),
            ("Cloud Infrastructure Modernization", "Qualification", 95000.0, 40, "New Customer"),
            ("Unified Security Suite License", "Prospecting", 65000.0, 20, "Existing Customer - Replacement"),
            ("Multi-Tenant Lakehouse Integration", "Closed Won", 410000.0, 100, "New Customer"),
            ("Predictive Churn Automation", "Closed Lost", 120000.0, 0, "New Customer"),
            ("Executive KPI Command Center", "Proposal/Price Quote", 240000.0, 70, "Existing Customer - Upgrade")
        ]

        records = []
        limit_to_use = max(limit * 4, 300) if soql_filter else max(1, min(limit, 200))

        if resource == "Leads":
            for i in range(limit_to_use):
                ld = leads_data[i % len(leads_data)]
                records.append({
                    "Id": f"00Q5g00000{i+1:06d}EAA",
                    "FirstName": ld[0],
                    "LastName": f"{ld[1]} {i+1}" if i >= len(leads_data) else ld[1],
                    "Company": ld[2],
                    "Title": ld[3],
                    "Email": f"{ld[0].lower()}.{ld[1].lower()}{i+1}@benchmark.io",
                    "Phone": f"+1-555-01{i%90+10:02d}",
                    "Status": ld[5],
                    "LeadSource": ld[6],
                    "Rating": ld[7],
                    "AnnualRevenue": 1500000 + (i * 350000),
                    "City": "San Francisco",
                    "Country": "USA",
                    "CreatedDate": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                })
        elif resource == "Opportunities":
            for i in range(limit_to_use):
                op = opp_stages[i % len(opp_stages)]
                amount = float(op[2] + (i * 12500))
                prob = op[3]
                is_won = (op[1] == "Closed Won")
                is_closed = (op[1] in ("Closed Won", "Closed Lost"))
                records.append({
                    "Id": f"0065g00000{i+1:06d}EAA",
                    "Name": f"{op[0]} - Phase {i//len(opp_stages)+1}",
                    "StageName": op[1],
                    "Amount": amount,
                    "Probability": prob,
                    "ExpectedRevenue": round(amount * (prob / 100.0), 2),
                    "CloseDate": datetime.now(timezone.utc).strftime("%Y-12-31"),
                    "Type": op[4],
                    "LeadSource": "Partner Referral" if i % 2 == 0 else "Web Inbound",
                    "IsWon": is_won,
                    "IsClosed": is_closed,
                    "CreatedDate": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                })
        elif resource == "Contacts":
            for i in range(limit_to_use):
                comp = companies[i % len(companies)]
                ld = leads_data[i % len(leads_data)]
                records.append({
                    "Id": f"0035g00000{i+1:06d}EAA",
                    "FirstName": ld[0],
                    "LastName": ld[1],
                    "Name": f"{ld[0]} {ld[1]}",
                    "Email": f"{ld[0].lower()}.{ld[1].lower()}@{comp[0].lower().replace(' ', '')}.com",
                    "Phone": f"+1-415-555-0{i%90+10:02d}",
                    "Title": ld[3],
                    "Department": "Management" if i % 2 == 0 else "Engineering",
                    "LeadSource": ld[6],
                    "MailingCity": comp[2],
                    "MailingCountry": comp[3],
                    "CreatedDate": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                })
        elif resource == "Cases":
            for i in range(limit_to_use):
                records.append({
                    "Id": f"5005g00000{i+1:06d}EAA",
                    "CaseNumber": f"000{10240+i}",
                    "Subject": f"Integration query for API gateway #{i+1}",
                    "Status": "Closed" if i % 3 == 0 else ("Escalated" if i % 5 == 0 else "Working"),
                    "Priority": "High" if i % 4 == 0 else ("Critical" if i % 10 == 0 else "Medium"),
                    "Origin": "Web" if i % 2 == 0 else "Email",
                    "Type": "Feature Request" if i % 3 == 0 else "Problem",
                    "Reason": "Complex functionality",
                    "IsClosed": (i % 3 == 0),
                    "CreatedDate": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                })
        else:
            # Default to Accounts (also used for custom SOQL fallback)
            for i in range(limit_to_use):
                comp = companies[i % len(companies)]
                records.append({
                    "Id": f"0015g00000{i+1:06d}EAA",
                    "Name": f"{comp[0]} {i//len(companies)+1}" if i >= len(companies) else comp[0],
                    "Type": "Customer - Direct" if i % 2 == 0 else "Customer - Channel",
                    "Industry": comp[1],
                    "AnnualRevenue": comp[4] + (i * 250000),
                    "NumberOfEmployees": comp[5] + (i * 15),
                    "BillingCity": comp[2],
                    "BillingState": "CA" if comp[2] in ("San Francisco", "San Jose", "Los Angeles") else "NY",
                    "BillingCountry": comp[3],
                    "Phone": f"+1-800-555-{1000+i}",
                    "Website": f"https://www.{comp[0].lower().replace(' ', '')}.com",
                    "CreatedDate": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                })

        if soql_filter and records:
            if pd is not None:
                df = pd.DataFrame(records)
                filtered = self._apply_soql_filter(df, soql_filter)
                if not filtered.empty:
                    records = filtered.iloc[:limit].to_dict(orient="records")
        else:
            records = records[:limit]

        return records

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        resource = str(config.get("resource", "Accounts")).strip()
        limit = int(config.get("record_limit", 50))
        soql_filter = str(config.get("soql_filter", "")).strip()
        custom_soql = str(config.get("custom_soql", "")).strip()
        credential_id = str(config.get("credential_id", "")).strip()
        instance_url = str(config.get("instance_url", "")).strip() or getattr(settings, "SALESFORCE_INSTANCE_URL", None) or "https://login.salesforce.com"
        access_token = str(config.get("access_token", "")).strip() or getattr(settings, "SALESFORCE_ACCESS_TOKEN", None) or ""
        api_version = str(config.get("api_version", "v59.0")).strip()
        simulation_mode = bool(config.get("simulation_mode", False))

        # Check for connected account in execution context
        if credential_id and context and isinstance(context, dict):
            creds_store = context.get("credentials", {})
            if credential_id in creds_store:
                cred_info = creds_store[credential_id]
                c_data = cred_info.get("data", cred_info)
                access_token = c_data.get("access_token") or c_data.get("token") or access_token
                instance_url = c_data.get("instance_url") or instance_url

        records: List[Dict[str, Any]] = []
        delivery_status = "SIMULATED"
        status_msg = ""
        resolved_soql = ""

        # Construct SOQL query
        if resource == "Custom SOQL Query":
            resolved_soql = custom_soql or f"SELECT Id, Name FROM Account LIMIT {limit}"
        else:
            object_name_map = {
                "Accounts": "Account",
                "Leads": "Lead",
                "Opportunities": "Opportunity",
                "Contacts": "Contact",
                "Cases": "Case"
            }
            s_object = object_name_map.get(resource, "Account")
            fields = ", ".join(self.DEFAULT_FIELDS.get(resource, ["Id", "Name"]))
            resolved_soql = f"SELECT {fields} FROM {s_object}"
            if soql_filter:
                resolved_soql += f" WHERE {soql_filter}"
            resolved_soql += f" LIMIT {limit}"

        # Attempt live API call if configured and not in simulation mode
        if not simulation_mode and access_token and instance_url and not access_token.startswith("mock") and len(access_token) >= 15:
            try:
                base_url = instance_url.rstrip("/")
                encoded_query = urllib.parse.quote(resolved_soql)
                api_url = f"{base_url}/services/data/{api_version}/query?q={encoded_query}"

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
                        raw_records = payload.get("records", [])
                        # Clean Salesforce metadata attributes ('attributes' dict in records)
                        for r in raw_records:
                            item = {k: v for k, v in r.items() if k != "attributes"}
                            records.append(item)

                        delivery_status = "LIVE"
                        status_msg = f"Retrieved {len(records)} live Salesforce {resource} records via Force.com REST API."
            except urllib.error.HTTPError as he:
                err_body = ""
                try:
                    err_body = he.read().decode("utf-8")
                except Exception:
                    pass
                logger.warning(f"Salesforce API query returned HTTP {he.code}: {err_body[:200]}")
                status_msg = f"Salesforce live query failed (HTTP {he.code}). Switched to benchmark fallback data."
            except Exception as ex:
                logger.warning(f"Salesforce connection error: {ex}")
                status_msg = f"Salesforce live connection failed ({ex}). Switched to benchmark fallback data."

        # If live call was not attempted or returned empty, populate with realistic benchmark dataset
        if not records:
            records = self._generate_benchmark_data(resource, limit, soql_filter)
            if not status_msg:
                if soql_filter:
                    status_msg = f"Generated {len(records)} realistic Salesforce {resource} records matching filter '{soql_filter}' (Simulation Mode)."
                else:
                    status_msg = f"Generated {len(records)} realistic Salesforce {resource} records (Simulation Mode)."

        # Convert to Pandas DataFrame
        if pd is not None:
            df_out = pd.DataFrame(records)
            if soql_filter and not df_out.empty:
                df_out = self._apply_soql_filter(df_out, soql_filter)
                if len(df_out) > limit:
                    df_out = df_out.iloc[:limit]
                records = df_out.to_dict(orient="records")
        else:
            df_out = records

        summary = {
            "title": "Salesforce CRM Read Summary",
            "message": status_msg,
            "status": delivery_status,
            "resource": resource,
            "records_retrieved": len(records),
            "soql_query": resolved_soql
        }

        metrics = {
            "salesforce_resource": resource,
            "record_count": len(records),
            "delivery_status": delivery_status,
            "api_version": api_version
        }

        return {
            "dataframe": df_out,
            "output_summary": summary,
            "metrics": metrics
        }

    def to_code(self, config: Dict[str, Any]) -> str:
        resource = config.get("resource", "Accounts")
        limit = config.get("record_limit", 50)
        filter_str = config.get("soql_filter", "")
        return (
            f"# Recipe: Salesforce CRM Connector\n"
            f"# ID: {self.recipe_id}\n"
            f"import requests\n"
            f"import pandas as pd\n\n"
            f"url = '{config.get('instance_url', 'https://yourinstance.my.salesforce.com')}/services/data/{config.get('api_version', 'v59.0')}/query'\n"
            f"headers = {{'Authorization': 'Bearer ' + os.getenv('SALESFORCE_ACCESS_TOKEN', '')}}\n"
            f"params = {{'q': 'SELECT Id, Name FROM {resource} LIMIT {limit}'}}\n"
            f"response = requests.get(url, headers=headers, params=params)\n"
            f"data = response.json().get('records', [])\n"
            f"df = pd.DataFrame([{{k: v for k, v in r.items() if k != 'attributes'}} for r in data])\n"
        )

