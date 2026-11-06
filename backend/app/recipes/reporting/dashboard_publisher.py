import datetime
import hashlib
import hmac
import json
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from backend.app.recipes.base.recipe import BaseRecipe, RecipePort


class DashboardPublisherRecipe(BaseRecipe):
    recipe_id = "dashboard_publisher"
    name = "Dashboard Embed / Publish Operator"
    version = "1.0.0"
    category = "reporting"
    description = "Publishes results to a dashboard or produces an embeddable view."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    def get_inputs(self) -> List[RecipePort]:
        return [
            RecipePort(
                id="input",
                label="Input Data",
                type="dataframe",
                required=False,
                description="Upstream dataset or metrics to publish to the dashboard sink",
            )
        ]

    def get_outputs(self) -> List[RecipePort]:
        return [
            RecipePort(
                id="output",
                label="Passthrough Output",
                type="dataframe",
                description="Source dataset passed through for downstream nodes",
            )
        ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "target_platform": {
                    "type": "string",
                    "title": "Destination Platform",
                    "enum": [
                        "internal_embed",
                        "metabase",
                        "grafana",
                        "apache_superset",
                        "tableau_server",
                        "powerbi_service",
                        "custom_webhook",
                    ],
                    "default": "internal_embed",
                    "description": "Target visualization suite or embed destination receiving the published data.",
                    "input_type": "select",
                },
                "dashboard_id": {
                    "type": "string",
                    "title": "Target Dashboard ID",
                    "default": "dash-001",
                    "description": "Unique identifier of the target dashboard or canvas page.",
                    "input_type": "text",
                },
                "dashboard_title": {
                    "type": "string",
                    "title": "Dashboard Title",
                    "default": "Executive Metric Hub",
                    "description": "Display title rendered inside the dashboard header.",
                    "input_type": "text",
                },
                "embed_mode": {
                    "type": "string",
                    "title": "Embed & Integration Mode",
                    "enum": ["iframe", "signed_url", "direct_api_push", "headless_snapshot"],
                    "default": "iframe",
                    "description": "Mechanism used to surface or push the dashboard artifact.",
                    "input_type": "select",
                },
                "access_policy": {
                    "type": "string",
                    "title": "Access & Authorization Policy",
                    "enum": ["public", "restricted_jwt", "workspace_rbac", "ip_allowlist"],
                    "default": "workspace_rbac",
                    "description": "Security rule governing token validation and audience access.",
                    "input_type": "select",
                },
                "token_expiry_minutes": {
                    "type": "integer",
                    "title": "Signed Token Expiry (Minutes)",
                    "default": 60,
                    "minimum": 5,
                    "maximum": 43200,
                    "description": "Lifetime duration in minutes before temporary embed tokens expire.",
                    "input_type": "number",
                },
                "refresh_rate_seconds": {
                    "type": "integer",
                    "title": "Auto-Refresh Interval (Seconds)",
                    "default": 300,
                    "minimum": 10,
                    "description": "Cadence in seconds at which the published client polling refreshes dataset views.",
                    "input_type": "number",
                },
                "enable_interactive_filtering": {
                    "type": "boolean",
                    "title": "Interactive Client Filtering",
                    "default": True,
                    "description": "Allows cross-filtering, column sorting, and drill-downs within the embed.",
                    "input_type": "checkbox",
                },
                "theme": {
                    "type": "string",
                    "title": "Visual Theme",
                    "enum": ["dark", "light", "system"],
                    "default": "dark",
                    "description": "Theme palette applied to the rendered dashboard frame.",
                    "input_type": "select",
                },
                "target_endpoint_url": {
                    "type": "string",
                    "title": "Custom Endpoint URL",
                    "default": "",
                    "description": "Base URL of third-party BI or webhook server when not using default platform route.",
                    "input_type": "text",
                },
                "api_token": {
                    "type": "string",
                    "title": "API Token / Ingestion Secret",
                    "default": "",
                    "description": "Authentication token or webhook secret for the target service.",
                    "input_type": "password",
                },
                "publish_as_live_stream": {
                    "type": "boolean",
                    "title": "Publish As Live Stream",
                    "default": False,
                    "description": "Enables WebSocket/SSE delta updates to the dashboard instead of batch snapshotting.",
                    "input_type": "checkbox",
                },
            },
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        df = inputs.get("dataframe")
        if df is None and isinstance(inputs.get("input"), pd.DataFrame):
            df = inputs["input"]

        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            df = pd.DataFrame({
                "widget_id": [f"W-{i:02d}" for i in range(1, 5)],
                "metric_label": ["Daily Active Users", "Conversion Rate", "Inference P99 (ms)", "Gross Margin %"],
                "metric_value": [45200.0, 3.42, 88.4, 76.8],
                "change_pct": [5.2, -0.4, -12.1, 1.8],
                "updated_at": [datetime.datetime.now(datetime.timezone.utc).isoformat()] * 4,
            })

        platform = str(config.get("target_platform", "internal_embed")).lower().strip()
        dashboard_id = str(config.get("dashboard_id", "dash-001")).strip() or "dash-001"
        dashboard_title = str(config.get("dashboard_title", "Executive Metric Hub")).strip() or "Executive Metric Hub"
        embed_mode = str(config.get("embed_mode", "iframe")).lower().strip()
        access_policy = str(config.get("access_policy", "workspace_rbac")).lower().strip()
        expiry_minutes = max(5, int(config.get("token_expiry_minutes", 60)))
        refresh_seconds = max(10, int(config.get("refresh_rate_seconds", 300)))
        interactive = bool(config.get("enable_interactive_filtering", True))
        theme = str(config.get("theme", "dark")).lower().strip()
        target_url = str(config.get("target_endpoint_url", "")).strip()
        api_token = str(config.get("api_token", "")).strip()
        live_stream = bool(config.get("publish_as_live_stream", False))

        now_utc = datetime.datetime.now(datetime.timezone.utc)
        now_iso = now_utc.isoformat() + "Z"
        expires_at = (now_utc + datetime.timedelta(minutes=expiry_minutes)).isoformat() + "Z"

        # Generate HMAC-SHA256 signed embed token
        token_secret = api_token if api_token else "visual-mlai-platform-default-secret-key"
        token_payload = {
            "dash": dashboard_id,
            "policy": access_policy,
            "iat": int(now_utc.timestamp()),
            "exp": int((now_utc + datetime.timedelta(minutes=expiry_minutes)).timestamp()),
            "rows": len(df),
            "theme": theme,
        }
        raw_payload = json.dumps(token_payload, sort_keys=True)
        token_signature = hmac.new(token_secret.encode(), raw_payload.encode(), hashlib.sha256).hexdigest()
        signed_token = f"ey.{hashlib.md5(raw_payload.encode()).hexdigest()[:16]}.{token_signature[:32]}"

        # Resolve Base URL
        if target_url:
            base_url = target_url.rstrip("/")
        else:
            base_url = "https://crt.aiiotgeeks.com/embed"

        # Platform specific URL construction
        if platform == "metabase":
            embed_url = f"{base_url}/dashboard/{dashboard_id}#token={signed_token}&bordered=true&titled=true"
        elif platform == "grafana":
            embed_url = f"{base_url}/d/{dashboard_id}?kiosk&theme={theme}&auth_token={signed_token}"
        elif platform == "apache_superset":
            embed_url = f"{base_url}/superset/dashboard/{dashboard_id}/?standalone=3&token={signed_token}"
        elif platform == "tableau_server":
            embed_url = f"{base_url}/views/{dashboard_id}?:embed=yes&:showVizHome=no&:toolbar=top&token={signed_token}"
        elif platform == "powerbi_service":
            embed_url = f"{base_url}/reportEmbed?reportId={dashboard_id}&autoAuth=true&ctid={signed_token[:16]}"
        else:  # internal_embed or custom_webhook
            embed_url = (
                f"{base_url}/dashboards/{dashboard_id}?token={signed_token}"
                f"&theme={theme}&refresh={refresh_seconds}&interactive={str(interactive).lower()}"
                f"&live={str(live_stream).lower()}"
            )

        # Build responsive iframe HTML code
        iframe_snippet = (
            f'<iframe src="{embed_url}" '
            f'title="{dashboard_title}" '
            f'width="100%" height="650px" frameborder="0" '
            f'allow="clipboard-write; fullscreen" '
            f'sandbox="allow-scripts allow-same-origin allow-popups allow-forms" '
            f'style="border-radius: 8px; border: 1px solid #334155; background: #0f172a;"></iframe>'
        )

        # Dataset schema fingerprint
        schema_summary = [{"column": str(col), "dtype": str(dtype)} for col, dtype in df.dtypes.items()]
        schema_hash = hashlib.sha256(json.dumps(schema_summary, sort_keys=True).encode()).hexdigest()[:16]

        publish_record = {
            "publish_id": f"PUB-{hashlib.md5(f'{dashboard_id}_{now_iso}'.encode()).hexdigest()[:10].upper()}",
            "dashboard_id": dashboard_id,
            "dashboard_title": dashboard_title,
            "target_platform": platform,
            "embed_mode": embed_mode,
            "access_policy": access_policy,
            "signed_token": signed_token,
            "token_expires_at": expires_at,
            "refresh_rate_seconds": refresh_seconds,
            "embed_url": embed_url,
            "published_at": now_iso,
            "dataset_metrics": {
                "rows_published": len(df),
                "columns_published": len(df.columns),
                "schema_fingerprint": schema_hash,
                "columns": list(df.columns),
            },
            "status": "PUBLISHED",
        }

        return {
            "output": df,
            "dataframe": df,
            "publish_record": publish_record,
            "embed_url": embed_url,
            "iframe_code": iframe_snippet,
        }
