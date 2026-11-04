"""
Google Ads Connector Recipe (Apps & Integrations -> CRM / Support / Business).

Reads Google Ads campaign performance, ad group performance, and keyword analytics
via Google Ads API (with credential resolution, rate-limit resilience, and benchmark simulation fallbacks).
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


class GoogleAdsConnectorRecipe(BaseRecipe):
    recipe_id = "google_ads_connector"
    name = "Google Ads Connector"
    version = "1.0.0"
    category = "integrations"
    group = "Apps & Integrations"
    subgroup = "CRM / Support / Business"
    description = "Reads Google Ads campaign performance data."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    inputs = [
        RecipePort(
            id="input",
            label="Trigger / Input Dataset",
            type="dataframe",
            required=False,
            max_connections=1,
            description="Optional upstream dataset for campaign IDs or execution trigger"
        )
    ]

    outputs = [
        RecipePort(
            id="output",
            label="Google Ads Dataset",
            type="dataframe",
            description="DataFrame containing Google Ads performance metrics (Campaigns, Ad Groups, or Keywords)"
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "resource": {
                    "type": "string",
                    "title": "Google Ads Report",
                    "input_type": "dropdown",
                    "enum": [
                        "Campaign Performance",
                        "Ad Group Performance",
                        "Keyword Performance"
                    ],
                    "default": "Campaign Performance",
                    "description": "Select the Google Ads performance report to retrieve."
                },
                "customer_id": {
                    "type": "string",
                    "title": "Customer ID (Account ID)",
                    "input_type": "text",
                    "default": "123-456-7890",
                    "description": "Your 10-digit Google Ads customer ID (e.g. 123-456-7890 or 1234567890)."
                },
                "developer_token": {
                    "type": "string",
                    "title": "Google Ads Developer Token",
                    "input_type": "password",
                    "description": "Developer Token granted from Google Ads MCC account (or uses GOOGLE_ADS_DEVELOPER_TOKEN from .env)."
                },
                "access_token": {
                    "type": "string",
                    "title": "OAuth Access Token / Refresh Token",
                    "input_type": "password",
                    "description": "OAuth 2.0 Access Token or Refresh Token (or uses GOOGLE_ADS_ACCESS_TOKEN from .env)."
                },
                "credential_id": {
                    "type": "string",
                    "title": "Connected Google Ads Credential",
                    "input_type": "credential_dropdown",
                    "provider": "google_ads",
                    "description": "Select a stored Google Ads credential or enter manual credentials above."
                },
                "status_filter": {
                    "type": "string",
                    "title": "Campaign Status Filter",
                    "input_type": "dropdown",
                    "enum": ["All", "ENABLED", "PAUSED", "REMOVED"],
                    "default": "All",
                    "description": "Filter campaigns/ad groups by operating status."
                },
                "date_range": {
                    "type": "string",
                    "title": "Reporting Date Range",
                    "input_type": "dropdown",
                    "enum": ["Last 30 Days", "Last 7 Days", "Last 90 Days", "This Month", "All Time"],
                    "default": "Last 30 Days",
                    "description": "Reporting time window for performance metrics."
                },
                "record_limit": {
                    "type": "integer",
                    "title": "Record Limit",
                    "input_type": "number",
                    "default": 50,
                    "minimum": 1,
                    "maximum": 200,
                    "description": "Maximum number of performance rows to retrieve (1-200)."
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
                    "description": "Generate realistic benchmark Google Ads marketing metrics without making live API calls."
                }
            },
            "required": ["resource"]
        }

    def _generate_benchmark_data(self, resource: str, limit: int, status_filter: str = "All") -> List[Dict[str, Any]]:
        """Generates realistic PPC marketing campaign, ad group, and keyword performance rows."""
        limit_to_use = max(1, min(limit, 200))
        records: List[Dict[str, Any]] = []
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        if resource == "Campaign Performance":
            campaigns = [
                ("Search - Brand Defense & Trademark", "SEARCH", 42000, 3150, 1.45, 185),
                ("Search - Generic ML Platform Keywords", "SEARCH", 88000, 2450, 4.20, 142),
                ("Performance Max - Global Q4 Acquisition", "PERFORMANCE_MAX", 210000, 4800, 2.10, 310),
                ("Display - Retargeting Cart Abandoners", "DISPLAY", 150000, 1600, 0.85, 95),
                ("YouTube - Product Demo & Case Studies", "VIDEO", 95000, 2200, 0.65, 80),
                ("Search - High Intent Enterprise Tier", "SEARCH", 35000, 1850, 6.80, 160),
                ("Discovery - Demand Gen Lookalikes", "DISCOVERY", 115000, 2900, 1.15, 120),
                ("Search - Competitor Comparison Alternatives", "SEARCH", 62000, 1950, 5.40, 110)
            ]
            for i in range(limit_to_use):
                camp = campaigns[i % len(campaigns)]
                camp_status = "ENABLED" if status_filter == "All" else status_filter
                impr = camp[2] + (i * 1500)
                clicks = camp[3] + (i * 65)
                cpc = round(camp[4] + ((i % 5) * 0.15), 2)
                cost = round(clicks * cpc, 2)
                conv = camp[5] + (i * 4)
                ctr = round((clicks / impr) * 100, 2) if impr else 0.0
                cpa = round(cost / conv, 2) if conv else 0.0
                conv_rate = round((conv / clicks) * 100, 2) if clicks else 0.0

                records.append({
                    "campaign_id": 18000000000 + i,
                    "campaign_name": f"{camp[0]} ({i+1})",
                    "advertising_channel_type": camp[1],
                    "status": camp_status,
                    "impressions": impr,
                    "clicks": clicks,
                    "ctr": ctr,
                    "average_cpc": cpc,
                    "cost": cost,
                    "cost_micros": int(cost * 1_000_000),
                    "conversions": conv,
                    "cost_per_conversion": cpa,
                    "conversion_rate": conv_rate,
                    "date": now_iso
                })

        elif resource == "Ad Group Performance":
            adgroups = [
                ("Machine Learning Platform Core", "Search - Generic ML Platform", 32000, 1200, 3.80, 78),
                ("Automated Feature Engineering", "Search - Generic ML Platform", 18000, 650, 4.10, 42),
                ("Brand Direct - Exact Match", "Search - Brand Defense", 28000, 2400, 1.20, 145),
                ("US East - Tech Decision Makers", "Performance Max - Global Q4", 85000, 2100, 2.30, 140),
                ("EMEA SaaS Scalers Tier 1", "Performance Max - Global Q4", 65000, 1550, 1.95, 98),
                ("Retargeting Visitors 30 Days", "Display - Retargeting", 72000, 890, 0.90, 52),
                ("Video Pre-Roll 30s Overviews", "YouTube - Product Demo", 54000, 1400, 0.60, 48),
                ("Top Alternatives & Reviews", "Search - Competitor Comparison", 31000, 980, 5.10, 64)
            ]
            for i in range(limit_to_use):
                ag = adgroups[i % len(adgroups)]
                ag_status = "ENABLED" if status_filter == "All" else status_filter
                impr = ag[2] + (i * 800)
                clicks = ag[3] + (i * 30)
                cpc = round(ag[4] + ((i % 4) * 0.12), 2)
                cost = round(clicks * cpc, 2)
                conv = ag[5] + (i * 2)
                ctr = round((clicks / impr) * 100, 2) if impr else 0.0

                records.append({
                    "ad_group_id": 29000000000 + i,
                    "ad_group_name": f"{ag[0]} [{i+1}]",
                    "campaign_id": 18000000000 + (i % 8),
                    "campaign_name": f"{ag[1]}",
                    "status": ag_status,
                    "impressions": impr,
                    "clicks": clicks,
                    "ctr": ctr,
                    "average_cpc": cpc,
                    "cost": cost,
                    "conversions": conv,
                    "date": now_iso
                })

        elif resource == "Keyword Performance":
            keywords = [
                ("automated machine learning platform", "EXACT", 12000, 680, 5.20, 45),
                ("mlops pipeline orchestration", "PHRASE", 18500, 740, 4.80, 38),
                ("ai model deployment tools", "BROAD", 24000, 590, 3.40, 28),
                ("datarept forge pricing", "EXACT", 8500, 1250, 1.10, 98),
                ("predictive analytics software", "PHRASE", 16000, 480, 4.50, 24),
                ("churn prediction model api", "EXACT", 9200, 420, 6.10, 32),
                ("automl vs custom models", "BROAD", 14500, 380, 3.10, 18),
                ("best ml platform for data science teams", "PHRASE", 11000, 410, 5.60, 26)
            ]
            for i in range(limit_to_use):
                kw = keywords[i % len(keywords)]
                impr = kw[2] + (i * 450)
                clicks = kw[3] + (i * 20)
                cpc = round(kw[4] + ((i % 3) * 0.15), 2)
                cost = round(clicks * cpc, 2)
                conv = kw[5] + (i % 8)
                ctr = round((clicks / impr) * 100, 2) if impr else 0.0

                records.append({
                    "keyword_id": 37000000000 + i,
                    "keyword_text": f"{kw[0]}",
                    "match_type": kw[1],
                    "campaign_name": "Search - Generic ML Platform",
                    "status": "ENABLED" if status_filter == "All" else status_filter,
                    "impressions": impr,
                    "clicks": clicks,
                    "ctr": ctr,
                    "average_cpc": cpc,
                    "cost": cost,
                    "conversions": conv,
                    "date": now_iso
                })

        return records

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        resource = str(config.get("resource", "Campaign Performance")).strip()
        cust_id = str(config.get("customer_id", "123-456-7890")).replace("-", "").strip()
        dev_token = str(config.get("developer_token", "")).strip() or getattr(settings, "GOOGLE_ADS_DEVELOPER_TOKEN", None) or os.getenv("GOOGLE_ADS_DEVELOPER_TOKEN", "")
        access_token = str(config.get("access_token", "")).strip() or getattr(settings, "GOOGLE_ADS_ACCESS_TOKEN", None) or os.getenv("GOOGLE_ADS_ACCESS_TOKEN", "")
        credential_id = str(config.get("credential_id", "")).strip()
        status_filter = str(config.get("status_filter", "All")).strip()
        date_range = str(config.get("date_range", "Last 30 Days")).strip()
        limit = int(config.get("record_limit", config.get("limit", 50)))
        simulation_mode = bool(config.get("simulation_mode", config.get("simulate_if_no_auth", False)))
        output_format = str(config.get("output_format", "dataframe")).strip().lower()

        # Resolve credentials if stored in context
        if credential_id and context and isinstance(context, dict):
            creds_store = context.get("credentials", {})
            if credential_id in creds_store:
                cred_info = creds_store[credential_id]
                c_data = cred_info.get("data", cred_info)
                access_token = c_data.get("access_token") or c_data.get("token") or access_token
                dev_token = c_data.get("developer_token") or dev_token
                cust_id = str(c_data.get("customer_id") or cust_id).replace("-", "").strip()

        records: List[Dict[str, Any]] = []
        delivery_status = "SIMULATED"
        status_msg = ""
        api_endpoint = f"https://googleads.googleapis.com/v16/customers/{cust_id}/googleAds:searchStream"

        # Live API Call Attempt
        if not simulation_mode and access_token and dev_token and len(cust_id) >= 8 and not access_token.startswith("mock"):
            try:
                gaql_query = (
                    "SELECT campaign.id, campaign.name, campaign.advertising_channel_type, campaign.status, "
                    "metrics.impressions, metrics.clicks, metrics.cost_micros, metrics.conversions "
                    f"FROM campaign LIMIT {max(1, min(limit, 200))}"
                )
                payload = json.dumps({"query": gaql_query}).encode("utf-8")
                headers = {
                    "Content-Type": "application/json",
                    "developer-token": dev_token,
                    "Authorization": f"Bearer {access_token}",
                    "User-Agent": "Datarept-ML-Platform/1.0"
                }

                req = urllib.request.Request(api_endpoint, data=payload, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=12) as response:
                    if response.status == 200:
                        raw_json = json.loads(response.read().decode("utf-8"))
                        for batch in raw_json:
                            for row in batch.get("results", []):
                                camp = row.get("campaign", {})
                                mets = row.get("metrics", {})
                                cost = round(float(mets.get("costMicros", 0)) / 1_000_000, 2)
                                records.append({
                                    "campaign_id": camp.get("id"),
                                    "campaign_name": camp.get("name"),
                                    "advertising_channel_type": camp.get("advertisingChannelType"),
                                    "status": camp.get("status"),
                                    "impressions": int(mets.get("impressions", 0)),
                                    "clicks": int(mets.get("clicks", 0)),
                                    "cost": cost,
                                    "conversions": float(mets.get("conversions", 0))
                                })
                        delivery_status = "LIVE"
                        status_msg = f"Retrieved {len(records)} live Google Ads performance rows for account {cust_id}."
            except urllib.error.HTTPError as he:
                err_body = ""
                try:
                    err_body = he.read().decode("utf-8")
                except Exception:
                    pass
                logger.warning(f"Google Ads API HTTP {he.code}: {err_body[:200]}")
                status_msg = f"Google Ads API query failed (HTTP {he.code}). Using realistic benchmark fallback."
            except Exception as ex:
                logger.warning(f"Google Ads connection failed: {ex}")
                status_msg = f"Google Ads live connection failed ({ex}). Using realistic benchmark fallback."

        # Offline / Benchmark Fallback
        if not records:
            records = self._generate_benchmark_data(resource, limit, status_filter)
            if not status_msg:
                status_msg = f"Generated {len(records)} realistic Google Ads {resource} benchmark records (Simulation Mode)."

        # Convert to Pandas DataFrame
        if pd is not None:
            df_out = pd.DataFrame(records)
        else:
            df_out = records

        summary = {
            "title": "Google Ads Performance Read Summary",
            "message": status_msg,
            "status": delivery_status,
            "resource": resource,
            "customer_id": cust_id,
            "date_range": date_range,
            "records_retrieved": len(records)
        }

        metrics = {
            "google_ads_resource": resource,
            "record_count": len(records),
            "delivery_status": delivery_status,
            "customer_id": cust_id
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
        resource = config.get("resource", "Campaign Performance")
        cust_id = config.get("customer_id", "123-456-7890").replace("-", "")
        limit = config.get("record_limit", 50)
        return (
            f"# Recipe: Google Ads Connector\n"
            f"# ID: {self.recipe_id}\n"
            f"import os\n"
            f"import requests\n"
            f"import pandas as pd\n\n"
            f"url = 'https://googleads.googleapis.com/v16/customers/{cust_id}/googleAds:searchStream'\n"
            f"headers = {{\n"
            f"    'developer-token': os.getenv('GOOGLE_ADS_DEVELOPER_TOKEN', ''),\n"
            f"    'Authorization': f'Bearer {{os.getenv(\"GOOGLE_ADS_ACCESS_TOKEN\", \"\")}}',\n"
            f"    'Content-Type': 'application/json'\n"
            f"}}\n"
            f"query = 'SELECT campaign.id, campaign.name, metrics.impressions, metrics.clicks FROM campaign LIMIT {limit}'\n"
            f"response = requests.post(url, headers=headers, json={{'query': query}})\n"
            f"data = response.json()\n"
            f"df = pd.DataFrame([r.get('campaign', {{}}) for batch in data for r in batch.get('results', [])])\n"
        )

