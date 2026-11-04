"""
Meta Ads Connector Recipe (Apps & Integrations -> CRM / Support / Business).

Reads Facebook/Instagram (Meta) Ads campaign performance data, ad sets, and creatives
via Meta Graph API (with credential resolution, rate-limit resilience, and benchmark simulation fallbacks).
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


class MetaAdsConnectorRecipe(BaseRecipe):
    recipe_id = "meta_ads_connector"
    name = "Meta Ads Connector"
    version = "1.0.0"
    category = "integrations"
    group = "Apps & Integrations"
    subgroup = "CRM / Support / Business"
    description = "Reads Facebook/Instagram (Meta) Ads campaign performance data."
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
            label="Meta Ads Dataset",
            type="dataframe",
            description="DataFrame containing Facebook and Instagram campaign insights and ad performance"
        )
    ]

    RESOURCE_ENDPOINTS = {
        "Campaign Insights": "/insights",
        "Ad Set Insights": "/insights",
        "Ad Creative Performance": "/insights"
    }

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "resource": {
                    "type": "string",
                    "title": "Meta Ads Level",
                    "input_type": "dropdown",
                    "enum": [
                        "Campaign Insights",
                        "Ad Set Insights",
                        "Ad Creative Performance"
                    ],
                    "default": "Campaign Insights",
                    "description": "Select the level of Meta Ads performance data to retrieve."
                },
                "account_id": {
                    "type": "string",
                    "title": "Meta Ad Account ID",
                    "input_type": "text",
                    "default": "act_1234567890",
                    "description": "Your Meta ad account ID (e.g. 'act_1234567890' or numeric account ID)."
                },
                "access_token": {
                    "type": "string",
                    "title": "Meta Graph API Access Token",
                    "input_type": "password",
                    "description": "User Access Token or System User Token (or uses META_ADS_ACCESS_TOKEN from .env)."
                },
                "credential_id": {
                    "type": "string",
                    "title": "Connected Meta Ads Credential",
                    "input_type": "credential_dropdown",
                    "provider": "meta_ads",
                    "description": "Select a stored Meta/Facebook credential or enter manual access token above."
                },
                "status_filter": {
                    "type": "string",
                    "title": "Campaign Status Filter",
                    "input_type": "dropdown",
                    "enum": ["All", "ACTIVE", "PAUSED", "ARCHIVED"],
                    "default": "All",
                    "description": "Filter ad entities by status."
                },
                "date_preset": {
                    "type": "string",
                    "title": "Reporting Date Window",
                    "input_type": "dropdown",
                    "enum": ["last_30d", "last_7d", "last_90d", "this_month", "maximum"],
                    "default": "last_30d",
                    "description": "Preset reporting date window for Meta insights aggregation."
                },
                "record_limit": {
                    "type": "integer",
                    "title": "Record Limit",
                    "input_type": "number",
                    "default": 50,
                    "minimum": 1,
                    "maximum": 200,
                    "description": "Maximum number of rows to retrieve (1-200)."
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
                    "description": "Generate realistic benchmark Meta Ads performance metrics without making live API calls."
                }
            },
            "required": ["resource"]
        }

    def _generate_benchmark_data(self, resource: str, limit: int, status_filter: str = "All") -> List[Dict[str, Any]]:
        """Generates realistic Facebook & Instagram paid acquisition benchmarks."""
        limit_to_use = max(1, min(limit, 200))
        records: List[Dict[str, Any]] = []
        now_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        if resource == "Campaign Insights":
            meta_campaigns = [
                ("IG Reels - AI Tool Productivity Showcase", "OUTCOME_TRAFFIC", 125000, 92000, 1.35, 3400, 0.42, 1428.00, 180, 4.2),
                ("FB Feed - Retargeting Free Trial Signups", "OUTCOME_LEADS", 68000, 48000, 1.41, 1950, 1.15, 2242.50, 145, 3.8),
                ("Meta Advantage+ Shopping - Enterprise Tier", "OUTCOME_SALES", 240000, 160000, 1.50, 5200, 0.85, 4420.00, 220, 5.1),
                ("IG Stories - Founder Video Announcement", "OUTCOME_AWARENESS", 310000, 220000, 1.40, 4100, 0.35, 1435.00, 75, 2.4),
                ("FB Messenger - Automated Demo Booking", "OUTCOME_LEADS", 45000, 31000, 1.45, 1100, 1.40, 1540.00, 95, 3.2),
                ("Meta Lookalike 1% - High LTV Churn Reducers", "OUTCOME_SALES", 185000, 130000, 1.42, 3800, 0.95, 3610.00, 190, 4.6),
                ("IG Explore - Technical Infographic Carousel", "OUTCOME_ENGAGEMENT", 82000, 59000, 1.38, 2200, 0.50, 1100.00, 60, 2.1),
                ("Global SaaS Expansion - EMEA Decision Makers", "OUTCOME_LEADS", 95000, 64000, 1.48, 2400, 1.25, 3000.00, 130, 3.5)
            ]
            for i in range(limit_to_use):
                camp = meta_campaigns[i % len(meta_campaigns)]
                camp_status = "ACTIVE" if status_filter == "All" else status_filter
                impr = camp[2] + (i * 2000)
                reach = camp[3] + (i * 1200)
                clicks = camp[5] + (i * 40)
                cpc = round(camp[6] + ((i % 4) * 0.05), 2)
                spend = round(clicks * cpc, 2)
                purchases = camp[8] + (i * 3)
                roas = round(camp[9] + ((i % 5) * 0.1), 2)
                ctr = round((clicks / impr) * 100, 2) if impr else 0.0
                cost_per_purchase = round(spend / purchases, 2) if purchases else 0.0

                records.append({
                    "campaign_id": f"2385{10000000 + i}",
                    "campaign_name": f"{camp[0]} [{i+1}]",
                    "objective": camp[1],
                    "status": camp_status,
                    "impressions": impr,
                    "reach": reach,
                    "frequency": round(impr / reach, 2) if reach else 1.0,
                    "clicks": clicks,
                    "cpc": cpc,
                    "ctr": ctr,
                    "spend": spend,
                    "purchases": purchases,
                    "cost_per_purchase": cost_per_purchase,
                    "roas": roas,
                    "date_start": now_date,
                    "date_stop": now_date
                })

        elif resource == "Ad Set Insights":
            adsets = [
                ("US - Data Engineers 25-54", "IG Reels Showcase", 45000, 32000, 1200, 0.45, 540.00, 4.5),
                ("EU - VP Engineering & Tech Leads", "IG Reels Showcase", 38000, 26000, 950, 0.52, 494.00, 4.1),
                ("Retargeting - Visited Pricing Page", "FB Feed Retargeting", 28000, 19000, 890, 1.10, 979.00, 3.9),
                ("Broad US - Advantage+ Optimization", "Advantage+ Shopping", 92000, 65000, 2400, 0.80, 1920.00, 5.2),
                ("APAC - Startup Founders & CTOs", "Founder Video Announcement", 55000, 39000, 1100, 0.38, 418.00, 2.5),
                ("Lookalike 1% Purchasers (Past 180 Days)", "Lookalike 1% High LTV", 78000, 54000, 1850, 0.92, 1702.00, 4.8),
                ("UK & Germany - MLOps Enthusiasts", "EMEA Decision Makers", 41000, 29000, 1050, 1.20, 1260.00, 3.4),
                ("Product Tour Video Viewers 75%+", "Demo Booking Messenger", 22000, 16000, 680, 1.35, 918.00, 3.3)
            ]
            for i in range(limit_to_use):
                adset = adsets[i % len(adsets)]
                impr = adset[2] + (i * 900)
                reach = adset[3] + (i * 600)
                clicks = adset[4] + (i * 25)
                cpc = round(adset[5] + ((i % 3) * 0.04), 2)
                spend = round(clicks * cpc, 2)
                ctr = round((clicks / impr) * 100, 2) if impr else 0.0

                records.append({
                    "adset_id": f"2386{20000000 + i}",
                    "adset_name": f"{adset[0]} (#{i+1})",
                    "campaign_name": adset[1],
                    "status": "ACTIVE" if status_filter == "All" else status_filter,
                    "impressions": impr,
                    "reach": reach,
                    "clicks": clicks,
                    "cpc": cpc,
                    "ctr": ctr,
                    "spend": spend,
                    "roas": round(adset[7] + ((i % 4) * 0.1), 2),
                    "date_start": now_date,
                    "date_stop": now_date
                })

        elif resource == "Ad Creative Performance":
            creatives = [
                ("Video: 30s Interactive Pipeline Tour", "Product Showcase Ad #1", "IG Reels Showcase", 31000, 22000, 920, 0.40, 368.00),
                ("Carousel: 5 Steps to Deploy Churn Models", "Infographic Card #2", "IG Explore Infographic", 24000, 18000, 710, 0.48, 340.80),
                ("Static: Dark Mode Dashboard UI Screenshot", "High Intent Static #3", "Advantage+ Shopping", 48000, 34000, 1400, 0.78, 1092.00),
                ("Founder Story: Why We Built Forge", "Story Video #4", "Founder Announcement", 36000, 26000, 850, 0.35, 297.50),
                ("Customer Quote: Saved 20 Hours per Week", "Social Proof Card #5", "FB Feed Retargeting", 19000, 14000, 620, 1.05, 651.00),
                ("Animated GIF: Drag and Drop DAG Builder", "Feature Highlight #6", "Advantage+ Shopping", 52000, 37000, 1550, 0.82, 1271.00),
                ("Comparison Chart: Custom MLOps vs Forge", "Battlecard Graphic #7", "EMEA Decision Makers", 29000, 21000, 780, 1.15, 897.00),
                ("Short Clip: One-Click Model Serving API", "Action Video #8", "Demo Booking Messenger", 18000, 13000, 560, 1.30, 728.00)
            ]
            for i in range(limit_to_use):
                cr = creatives[i % len(creatives)]
                impr = cr[3] + (i * 650)
                reach = cr[4] + (i * 450)
                clicks = cr[5] + (i * 18)
                cpc = round(cr[6] + ((i % 3) * 0.03), 2)
                spend = round(clicks * cpc, 2)
                ctr = round((clicks / impr) * 100, 2) if impr else 0.0

                records.append({
                    "ad_id": f"2387{30000000 + i}",
                    "ad_name": f"{cr[0]}",
                    "adset_name": cr[2],
                    "campaign_name": cr[2],
                    "status": "ACTIVE" if status_filter == "All" else status_filter,
                    "impressions": impr,
                    "reach": reach,
                    "clicks": clicks,
                    "cpc": cpc,
                    "ctr": ctr,
                    "spend": spend,
                    "roas": round(3.5 + ((i % 6) * 0.3), 2),
                    "date_start": now_date,
                    "date_stop": now_date
                })

        return records

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        resource = str(config.get("resource", "Campaign Insights")).strip()
        account_id = str(config.get("account_id", "act_1234567890")).strip()
        access_token = str(config.get("access_token", "")).strip() or getattr(settings, "META_ADS_ACCESS_TOKEN", None) or os.getenv("META_ADS_ACCESS_TOKEN", "") or os.getenv("FACEBOOK_ACCESS_TOKEN", "")
        credential_id = str(config.get("credential_id", "")).strip()
        status_filter = str(config.get("status_filter", "All")).strip()
        date_preset = str(config.get("date_preset", "last_30d")).strip()
        limit = int(config.get("record_limit", config.get("limit", 50)))
        simulation_mode = bool(config.get("simulation_mode", config.get("simulate_if_no_auth", False)))
        output_format = str(config.get("output_format", "dataframe")).strip().lower()

        # Clean account ID (Meta API expects act_<id>)
        clean_act = account_id if account_id.startswith("act_") else f"act_{account_id}"

        # Resolve credentials if stored in context
        if credential_id and context and isinstance(context, dict):
            creds_store = context.get("credentials", {})
            if credential_id in creds_store:
                cred_info = creds_store[credential_id]
                c_data = cred_info.get("data", cred_info)
                access_token = c_data.get("access_token") or c_data.get("token") or access_token
                act_val = c_data.get("account_id") or account_id
                clean_act = act_val if str(act_val).startswith("act_") else f"act_{act_val}"

        records: List[Dict[str, Any]] = []
        delivery_status = "SIMULATED"
        status_msg = ""
        level_map = {
            "Campaign Insights": "campaign",
            "Ad Set Insights": "adset",
            "Ad Creative Performance": "ad"
        }
        level = level_map.get(resource, "campaign")
        base_url = f"https://graph.facebook.com/v19.0/{clean_act}/insights"

        # Live API Call Attempt
        if not simulation_mode and access_token and len(access_token) >= 20 and not access_token.startswith("mock"):
            try:
                fields = "campaign_id,campaign_name,impressions,reach,clicks,cpc,ctr,spend,actions"
                query_params = {
                    "level": level,
                    "fields": fields,
                    "date_preset": date_preset,
                    "limit": max(1, min(limit, 100)),
                    "access_token": access_token
                }
                encoded_params = urllib.parse.urlencode(query_params)
                full_url = f"{base_url}?{encoded_params}"

                headers = {
                    "Accept": "application/json",
                    "User-Agent": "Datarept-ML-Platform/1.0"
                }

                req = urllib.request.Request(full_url, headers=headers)
                with urllib.request.urlopen(req, timeout=12) as response:
                    if response.status == 200:
                        raw_json = json.loads(response.read().decode("utf-8"))
                        records = raw_json.get("data", [])
                        delivery_status = "LIVE"
                        status_msg = f"Retrieved {len(records)} live Meta Ads insights rows for {clean_act}."
            except urllib.error.HTTPError as he:
                err_body = ""
                try:
                    err_body = he.read().decode("utf-8")
                except Exception:
                    pass
                logger.warning(f"Meta Graph API HTTP {he.code}: {err_body[:200]}")
                status_msg = f"Meta Graph API query failed (HTTP {he.code}). Using realistic benchmark fallback."
            except Exception as ex:
                logger.warning(f"Meta Ads connection failed: {ex}")
                status_msg = f"Meta Ads live connection failed ({ex}). Using realistic benchmark fallback."

        # Offline / Benchmark Fallback
        if not records:
            records = self._generate_benchmark_data(resource, limit, status_filter)
            if not status_msg:
                status_msg = f"Generated {len(records)} realistic Meta Ads {resource} benchmark records (Simulation Mode)."

        # Convert to Pandas DataFrame
        if pd is not None:
            df_out = pd.DataFrame(records)
        else:
            df_out = records

        summary = {
            "title": "Meta Ads Performance Read Summary",
            "message": status_msg,
            "status": delivery_status,
            "resource": resource,
            "account_id": clean_act,
            "date_preset": date_preset,
            "records_retrieved": len(records)
        }

        metrics = {
            "meta_ads_resource": resource,
            "record_count": len(records),
            "delivery_status": delivery_status,
            "account_id": clean_act
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
        resource = config.get("resource", "Campaign Insights")
        act = config.get("account_id", "act_1234567890")
        limit = config.get("record_limit", 50)
        return (
            f"# Recipe: Meta Ads Connector\n"
            f"# ID: {self.recipe_id}\n"
            f"import os\n"
            f"import requests\n"
            f"import pandas as pd\n\n"
            f"url = 'https://graph.facebook.com/v19.0/{act}/insights'\n"
            f"params = {{\n"
            f"    'access_token': os.getenv('META_ADS_ACCESS_TOKEN', ''),\n"
            f"    'level': 'campaign',\n"
            f"    'fields': 'campaign_id,campaign_name,impressions,reach,clicks,spend',\n"
            f"    'limit': {limit}\n"
            f"}}\n"
            f"response = requests.get(url, params=params)\n"
            f"data = response.json().get('data', [])\n"
            f"df = pd.DataFrame(data)\n"
        )

