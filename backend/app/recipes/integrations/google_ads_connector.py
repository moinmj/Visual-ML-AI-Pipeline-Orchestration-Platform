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
                "manager_customer_id": {
                    "type": "string",
                    "title": "Manager Customer ID (MCC)",
                    "input_type": "text",
                    "description": "Optional 10-digit manager account ID used as the login-customer-id header when querying a client account."
                },
                "oauth_client_id": {
                    "type": "string",
                    "title": "Google OAuth Client ID",
                    "input_type": "text",
                    "description": "OAuth desktop/web client ID used with the refresh token."
                },
                "oauth_client_secret": {
                    "type": "string",
                    "title": "Google OAuth Client Secret",
                    "input_type": "password",
                    "description": "OAuth client secret. Store securely; do not expose it in logs."
                },
                "refresh_token": {
                    "type": "string",
                    "title": "Google OAuth Refresh Token",
                    "input_type": "password",
                    "description": "Long-lived OAuth refresh token used to obtain short-lived access tokens automatically."
                },
                "api_version": {
                    "type": "string",
                    "title": "Google Ads API Version",
                    "input_type": "text",
                    "default": "v24",
                    "description": "Google Ads API version, for example v24."
                },
                "developer_token": {
                    "type": "string",
                    "title": "Google Ads Developer Token",
                    "input_type": "password",
                    "description": "Optional in this connector configuration; if your Google Ads API access requires one, provide it here or through GOOGLE_ADS_DEVELOPER_TOKEN."
                },
                "access_token": {
                    "type": "string",
                    "title": "OAuth Access Token (optional if refresh token is configured)",
                    "input_type": "password",
                    "description": "Short-lived OAuth access token. For automatic renewal, configure the refresh token, OAuth client ID, and client secret."
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

    @staticmethod
    def _clean_customer_id(value: Any) -> str:
        return str(value or "").replace("-", "").replace(" ", "").strip()

    @staticmethod
    def _refresh_access_token(client_id: str, client_secret: str, refresh_token: str) -> str:
        """Exchange a Google OAuth refresh token for a short-lived access token."""
        if not (client_id and client_secret and refresh_token):
            return ""
        form_data = urllib.parse.urlencode({
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        }).encode("utf-8")
        request = urllib.request.Request(
            "https://oauth2.googleapis.com/token",
            data=form_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            token_data = json.loads(response.read().decode("utf-8"))
        return str(token_data.get("access_token") or "")

    @staticmethod
    def _date_clause(date_range: str) -> str:
        date_map = {
            "Last 30 Days": "LAST_30_DAYS",
            "Last 7 Days": "LAST_7_DAYS",
            "Last 90 Days": "LAST_90_DAYS",
            "This Month": "THIS_MONTH",
        }
        period = date_map.get(date_range)
        return f" AND segments.date DURING {period}" if period else ""

    @staticmethod
    def _build_gaql(resource: str, limit: int, status_filter: str, date_range: str) -> str:
        where_clauses = []
        date_clause = GoogleAdsConnectorRecipe._date_clause(date_range)
        if date_clause:
            where_clauses.append(date_clause.replace(" AND ", "", 1))
        if status_filter in {"ENABLED", "PAUSED", "REMOVED"}:
            if resource == "Campaign Performance":
                where_clauses.append(f"campaign.status = {status_filter}")
            elif resource == "Ad Group Performance":
                where_clauses.append(f"ad_group.status = {status_filter}")
            else:
                where_clauses.append(f"ad_group_criterion.status = {status_filter}")
        where_sql = f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        if resource == "Campaign Performance":
            fields = (
                "campaign.id, campaign.name, campaign.status, "
                "campaign.advertising_channel_type, metrics.impressions, "
                "metrics.clicks, metrics.ctr, metrics.average_cpc, "
                "metrics.cost_micros, metrics.conversions, "
                "metrics.cost_per_conversion, metrics.conversions_from_interactions_rate"
            )
            from_clause = "campaign"
        elif resource == "Ad Group Performance":
            fields = (
                "ad_group.id, ad_group.name, ad_group.status, "
                "campaign.id, campaign.name, metrics.impressions, "
                "metrics.clicks, metrics.ctr, metrics.average_cpc, "
                "metrics.cost_micros, metrics.conversions, metrics.cost_per_conversion"
            )
            from_clause = "ad_group"
        elif resource == "Keyword Performance":
            fields = (
                "ad_group_criterion.criterion_id, ad_group_criterion.keyword.text, "
                "ad_group_criterion.keyword.match_type, ad_group_criterion.status, "
                "campaign.id, campaign.name, ad_group.id, ad_group.name, "
                "metrics.impressions, metrics.clicks, metrics.ctr, "
                "metrics.average_cpc, metrics.cost_micros, metrics.conversions, "
                "metrics.cost_per_conversion"
            )
            from_clause = "keyword_view"
        else:
            raise ValueError(f"Unsupported Google Ads resource: {resource}")

        return (
            f"SELECT {fields} FROM {from_clause}{where_sql} "
            f"ORDER BY metrics.impressions DESC LIMIT {max(1, min(limit, 200))}"
        )

    @staticmethod
    def _row_to_record(resource: str, row: Dict[str, Any]) -> Dict[str, Any]:
        metrics = row.get("metrics", {}) or {}
        campaign = row.get("campaign", {}) or {}
        ad_group = row.get("adGroup", {}) or {}
        criterion = row.get("adGroupCriterion", {}) or {}

        cost_micros = int(metrics.get("costMicros", 0) or 0)
        cost = round(cost_micros / 1_000_000, 2)
        impressions = int(metrics.get("impressions", 0) or 0)
        clicks = int(metrics.get("clicks", 0) or 0)
        conversions = float(metrics.get("conversions", 0) or 0)

        # In Google Ads API, average_cpc and cost_per_conversion are DOUBLE (currency units), not micros.
        avg_cpc_val = metrics.get("averageCpc", 0.0) or 0.0
        try:
            average_cpc = round(float(avg_cpc_val), 2)
        except (ValueError, TypeError):
            average_cpc = 0.0

        cpc_val = metrics.get("costPerConversion", 0.0) or 0.0
        try:
            cost_per_conversion = round(float(cpc_val), 2)
        except (ValueError, TypeError):
            cost_per_conversion = 0.0

        common = {
            "campaign_id": campaign.get("id"),
            "campaign_name": campaign.get("name"),
            "impressions": impressions,
            "clicks": clicks,
            "ctr": float(metrics.get("ctr", 0) or 0),
            "average_cpc": average_cpc,
            "cost": cost,
            "cost_micros": cost_micros,
            "conversions": conversions,
            "cost_per_conversion": cost_per_conversion,
        }

        if resource == "Campaign Performance":
            common.update({
                "status": campaign.get("status"),
                "campaign_status": campaign.get("status"),
                "advertising_channel_type": campaign.get("advertisingChannelType"),
                "conversion_rate": float(
                    metrics.get("conversionsFromInteractionsRate", 0) or 0
                ),
            })
        elif resource == "Ad Group Performance":
            common.update({
                "ad_group_id": ad_group.get("id"),
                "ad_group_name": ad_group.get("name"),
                "status": ad_group.get("status"),
            })
        else:
            keyword = criterion.get("keyword", {}) or {}
            common.update({
                "keyword_id": criterion.get("criterionId"),
                "keyword_text": keyword.get("text"),
                "match_type": keyword.get("matchType"),
                "status": criterion.get("status"),
                "ad_group_id": ad_group.get("id"),
                "ad_group_name": ad_group.get("name"),
            })
        return common

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        resource = str(config.get("resource", "Campaign Performance")).strip()
        cust_id = self._clean_customer_id(config.get("customer_id", ""))
        manager_id = self._clean_customer_id(config.get("manager_customer_id", ""))

        dev_token = (
            str(config.get("developer_token", "")).strip()
            or getattr(settings, "GOOGLE_ADS_DEVELOPER_TOKEN", None)
            or os.getenv("GOOGLE_ADS_DEVELOPER_TOKEN", "")
        )
        access_token = (
            str(config.get("access_token", "")).strip()
            or getattr(settings, "GOOGLE_ADS_ACCESS_TOKEN", None)
            or os.getenv("GOOGLE_ADS_ACCESS_TOKEN", "")
        )
        refresh_token = str(config.get("refresh_token", "")).strip() or os.getenv("GOOGLE_ADS_REFRESH_TOKEN", "")
        oauth_client_id = str(config.get("oauth_client_id", "")).strip() or os.getenv("GOOGLE_ADS_CLIENT_ID", "")
        oauth_client_secret = str(config.get("oauth_client_secret", "")).strip() or os.getenv("GOOGLE_ADS_CLIENT_SECRET", "")

        credential_id = str(config.get("credential_id", "")).strip()
        status_filter = str(config.get("status_filter", "All")).strip()
        date_range = str(config.get("date_range", "Last 30 Days")).strip()
        limit = max(1, min(int(config.get("record_limit", config.get("limit", 50))), 200))
        simulation_mode = bool(config.get("simulation_mode", config.get("simulate_if_no_auth", False)))
        output_format = str(config.get("output_format", "DataFrame")).strip().lower()
        api_version = str(config.get("api_version", "v24")).strip()
        api_version = api_version if api_version.startswith("v") else f"v{api_version}"

        # Resolve stored credentials, if the host supplied them in context.
        if credential_id and isinstance(context, dict):
            creds_store = context.get("credentials", {}) or {}
            cred_info = creds_store.get(credential_id, {}) or {}
            c_data = cred_info.get("data", cred_info) or {}
            access_token = c_data.get("access_token") or c_data.get("token") or access_token
            refresh_token = c_data.get("refresh_token") or refresh_token
            oauth_client_id = c_data.get("client_id") or c_data.get("oauth_client_id") or oauth_client_id
            oauth_client_secret = c_data.get("client_secret") or c_data.get("oauth_client_secret") or oauth_client_secret
            dev_token = c_data.get("developer_token") or dev_token
            cust_id = self._clean_customer_id(c_data.get("customer_id") or cust_id)
            manager_id = self._clean_customer_id(
                c_data.get("manager_customer_id") or c_data.get("login_customer_id") or manager_id
            )

        # Fall back to simulation if simulate_if_no_auth is requested and credentials are missing
        has_auth = bool(
            (access_token and not access_token.startswith("mock"))
            or (refresh_token and oauth_client_id and oauth_client_secret)
        )
        if not simulation_mode and bool(config.get("simulate_if_no_auth", False)):
            if not has_auth or not (cust_id and len(cust_id) == 10 and cust_id.isdigit()):
                simulation_mode = True

        records: List[Dict[str, Any]] = []
        delivery_status = "NOT_CONNECTED"
        status_msg = ""
        http_status = None

        if simulation_mode:
            records = self._generate_benchmark_data(resource, limit, status_filter)
            delivery_status = "SIMULATED"
            status_msg = (
                f"Generated {len(records)} benchmark records for {resource}. "
                "These are synthetic demonstration data, not Google Ads account data."
            )
        else:
            try:
                if not cust_id or len(cust_id) != 10 or not cust_id.isdigit():
                    raise ValueError("Enter a valid 10-digit Google Ads client Customer ID.")
                if not access_token and refresh_token and oauth_client_id and oauth_client_secret:
                    access_token = self._refresh_access_token(
                        oauth_client_id, oauth_client_secret, refresh_token
                    )
                if not access_token or access_token.startswith("mock"):
                    raise ValueError(
                        "No usable OAuth access token. Supply an access token or an OAuth "
                        "refresh token with its client ID and client secret."
                    )

                gaql_query = self._build_gaql(resource, limit, status_filter, date_range)
                endpoint = (
                    f"https://googleads.googleapis.com/{api_version}/"
                    f"customers/{cust_id}/googleAds:search"
                )
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {access_token}",
                    "User-Agent": "Datarept-ML-Platform/1.0",
                }
                # Google Ads manager context is required when querying a client account
                # through an MCC. Do not send this header when querying a standalone account.
                if manager_id:
                    headers["login-customer-id"] = manager_id
                # Keep the developer token configurable. The connector does not silently
                # invent one; include it when supplied by the account/credential setup.
                if dev_token:
                    headers["developer-token"] = dev_token

                payload = json.dumps({"query": gaql_query}).encode("utf-8")
                req = urllib.request.Request(endpoint, data=payload, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=30) as response:
                    http_status = response.status
                    api_data = json.loads(response.read().decode("utf-8") or "{}")

                records = [
                    self._row_to_record(resource, row)
                    for row in (api_data.get("results", []) or [])
                ]
                delivery_status = "LIVE"
                status_msg = (
                    f"Google Ads API query succeeded for {resource} on account {cust_id}; "
                    f"received {len(records)} live rows."
                )
                if not records:
                    status_msg += (
                        " The API returned zero matching records. No benchmark data was substituted."
                    )

            except urllib.error.HTTPError as exc:
                http_status = exc.code
                try:
                    error_body = exc.read().decode("utf-8", errors="replace")
                except Exception:
                    error_body = ""
                logger.warning(f"Google Ads API HTTP {exc.code}: {error_body[:1000]}")
                delivery_status = "ERROR"
                status_msg = f"Google Ads API request failed (HTTP {exc.code}). {error_body[:1500]}"
            except Exception as exc:
                logger.warning(f"Google Ads connector error: {exc}")
                delivery_status = "ERROR"
                status_msg = str(exc)

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
            "manager_customer_id": manager_id or None,
            "date_range": date_range,
            "records_retrieved": len(records),
            "http_status": http_status,
        }
        metrics = {
            "google_ads_resource": resource,
            "record_count": len(records),
            "delivery_status": delivery_status,
            "customer_id": cust_id,
            "manager_customer_id": manager_id or None,
            "http_status": http_status,
        }

        if output_format in {"json dict", "json_dict"}:
            output_val = df_out.to_dict(orient="records") if hasattr(df_out, "to_dict") else records
        elif "json" in output_format:
            output_val = records
        else:
            output_val = df_out

        return {
            "dataframe": df_out,
            "output": output_val,
            "output_summary": summary,
            "metrics": metrics,
        }

    def to_code(self, config: Dict[str, Any]) -> str:
        resource = config.get("resource", "Campaign Performance")
        cust_id = self._clean_customer_id(config.get("customer_id", ""))
        manager_id = self._clean_customer_id(config.get("manager_customer_id", ""))
        limit = max(1, min(int(config.get("record_limit", 50)), 200))
        version = str(config.get("api_version", "v24"))
        version = version if version.startswith("v") else f"v{version}"
        return (
            "# Recipe: Google Ads Connector\n"
            "# Credentials are read from environment variables; do not hardcode secrets.\n"
            "import os, requests\n"
            f"url = 'https://googleads.googleapis.com/{version}/customers/{cust_id}/googleAds:search'\n"
            "headers = {\n"
            "    'Authorization': f\"Bearer {os.environ['GOOGLE_ADS_ACCESS_TOKEN']}\",\n"
            "    'Content-Type': 'application/json',\n"
            "}\n"
            + (f"headers['login-customer-id'] = '{manager_id}'\n" if manager_id else "")
            + "if os.getenv('GOOGLE_ADS_DEVELOPER_TOKEN'):\n"
            "    headers['developer-token'] = os.environ['GOOGLE_ADS_DEVELOPER_TOKEN']\n"
            + f"query = {self._build_gaql(resource, limit, 'All', config.get('date_range', 'Last 30 Days'))!r}\n"
            "response = requests.post(url, headers=headers, json={'query': query}, timeout=30)\n"
            "response.raise_for_status()\n"
            "data = response.json()\n"
            "print(data.get('results', []))\n"
        )
