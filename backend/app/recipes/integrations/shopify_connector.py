"""
Shopify E-commerce Connector Recipe (Apps & Integrations -> CRM / Support / Business).

Reads Shopify orders, products, and customer profiles via Shopify Admin REST API
(with credential resolution, rate-limit resilience, and benchmark simulation fallbacks).
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


class ShopifyConnectorRecipe(BaseRecipe):
    recipe_id = "shopify_connector"
    name = "Shopify E-commerce Connector"
    version = "1.0.0"
    category = "integrations"
    group = "Apps & Integrations"
    subgroup = "CRM / Support / Business"
    description = "Reads Shopify orders, products and customers."
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
            label="Shopify Dataset",
            type="dataframe",
            description="DataFrame containing retrieved Shopify orders, products, or customers"
        )
    ]

    API_VERSION = "2026-10"
    GRAPHQL_ENDPOINT = "/admin/api/2026-10/graphql.json"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "resource": {
                    "type": "string",
                    "title": "Shopify Resource",
                    "input_type": "dropdown",
                    "enum": [
                        "Orders",
                        "Products",
                        "Customers"
                    ],
                    "default": "Orders",
                    "description": "Select the Shopify resource to retrieve (Orders, Products, or Customers)."
                },
                "shop_domain": {
                    "type": "string",
                    "title": "Shopify Store Domain",
                    "input_type": "text",
                    "default": "my-store",
                    "description": "Your Shopify store name or domain (e.g. 'acme-store' for https://acme-store.myshopify.com)."
                },
                "access_token": {
                    "type": "string",
                    "title": "Shopify Admin API Access Token",
                    "input_type": "password",
                    "description": "Shopify Admin Access Token (shpat_... or uses SHOPIFY_ACCESS_TOKEN from .env)."
                },
                "credential_id": {
                    "type": "string",
                    "title": "Connected Shopify Credential",
                    "input_type": "credential_dropdown",
                    "provider": "shopify",
                    "description": "Select a stored Shopify account credential or enter manual access token above."
                },
                "status_filter": {
                    "type": "string",
                    "title": "Status Filter",
                    "input_type": "dropdown",
                    "enum": ["All", "open", "closed", "paid", "pending", "fulfilled", "active", "draft"],
                    "default": "All",
                    "description": "Filter retrieved records by status (e.g. open/closed/paid for orders, active/draft for products)."
                },
                "record_limit": {
                    "type": "integer",
                    "title": "Record Limit",
                    "input_type": "number",
                    "default": 50,
                    "minimum": 1,
                    "maximum": 250,
                    "description": "Maximum number of records to retrieve (1-250 per Shopify API page)."
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
                    "description": "Generate realistic benchmark Shopify e-commerce records without making live API calls."
                }
            },
            "required": ["resource"]
        }

    def _generate_benchmark_data(self, resource: str, limit: int, status_filter: str = "All") -> List[Dict[str, Any]]:
        """Generates realistic Shopify orders, products, and customer records for testing."""
        limit_to_use = max(1, min(limit, 250))
        records: List[Dict[str, Any]] = []
        now_iso = datetime.now(timezone.utc).isoformat()

        if resource == "Orders":
            order_totals = [45.00, 129.95, 89.50, 249.00, 19.99, 399.00, 74.50, 520.00]
            financial_statuses = ["paid", "paid", "paid", "pending", "refunded", "paid"]
            fulfillment_statuses = ["fulfilled", "fulfilled", "unfulfilled", "fulfilled", "partial"]
            for i in range(limit_to_use):
                total = order_totals[i % len(order_totals)]
                subtotal = round(total * 0.9, 2)
                f_status = financial_statuses[i % len(financial_statuses)] if status_filter == "All" else status_filter
                records.append({
                    "id": 5500000000 + i,
                    "order_number": 1001 + i,
                    "name": f"#100{1 + i}",
                    "email": f"shopper_{i+1}@gmail.com",
                    "total_price": str(total),
                    "subtotal_price": str(subtotal),
                    "currency": "USD",
                    "financial_status": f_status,
                    "fulfillment_status": fulfillment_statuses[i % len(fulfillment_statuses)],
                    "line_items_count": 1 + (i % 4),
                    "tags": "VIP, Repeat" if i % 3 == 0 else "First-time",
                    "created_at": now_iso,
                    "updated_at": now_iso
                })

        elif resource == "Products":
            product_catalog = [
                ("Ergonomic Bamboo Standing Desk", "Furniture", "Active Ergonomics", 499.00, 35, "DESK-BAM-01"),
                ("Wireless Noise-Cancelling Headphones", "Electronics", "AudioWave Tech", 199.95, 120, "HEAD-NC-02"),
                ("Organic Fair-Trade Coffee Beans 1kg", "Food & Beverage", "Terra Roast Co", 24.50, 450, "COFF-ORG-01"),
                ("Stainless Steel Insulated Water Bottle", "Accessories", "HydroPeak", 34.00, 280, "BOTT-SS-03"),
                ("Mechanical RGB Gaming Keyboard", "Electronics", "ClickCraft Pro", 149.00, 65, "KEY-MEC-04"),
                ("Breathable Merino Wool Running Socks", "Apparel", "StrideWear", 18.00, 600, "SOCK-MER-05"),
                ("USB-C Ultra-Fast GaN 100W Charger", "Electronics", "VoltCraft", 59.99, 180, "CHG-100W-06"),
                ("Minimalist Leather Cardholder Wallet", "Accessories", "Artisan Hide", 45.00, 95, "WAL-LTH-07")
            ]
            for i in range(limit_to_use):
                item = product_catalog[i % len(product_catalog)]
                prod_status = "active" if status_filter == "All" else status_filter
                records.append({
                    "id": 8800000000 + i,
                    "title": f"{item[0]} ({i+1})",
                    "vendor": item[2],
                    "product_type": item[1],
                    "status": prod_status,
                    "price": str(item[3]),
                    "sku": f"{item[5]}-{i+1}",
                    "inventory_quantity": item[4] + (i * 2),
                    "tags": f"{item[1].lower()}, best-seller",
                    "created_at": now_iso,
                    "updated_at": now_iso
                })

        elif resource == "Customers":
            customer_names = [
                ("Jessica", "Alba", "jessica.alba@starmail.com", "+1-555-8001", 12, 1420.50),
                ("Marcus", "Aurelius", "marcus@meditations.org", "+1-555-8002", 4, 380.00),
                ("Sophia", "Loren", "sophia@cinema.it", "+1-555-8003", 26, 3850.00),
                ("Leonardo", "DaVinci", "leo@renaissance.art", "+1-555-8004", 1, 45.00),
                ("Amelia", "Earhart", "amelia@flyer.org", "+1-555-8005", 8, 890.25),
                ("Nikola", "Tesla", "nikola@wardenclyffe.tech", "+1-555-8006", 18, 2200.00),
                ("Marie", "Curie", "marie@radium.fr", "+1-555-8007", 7, 760.00),
                ("Alexander", "Hamilton", "alex@treasury.gov", "+1-555-8008", 15, 1680.00)
            ]
            for i in range(limit_to_use):
                cust = customer_names[i % len(customer_names)]
                records.append({
                    "id": 6600000000 + i,
                    "first_name": cust[0],
                    "last_name": f"{cust[1]} {i+1}",
                    "email": f"cust_{i+1}_{cust[2]}",
                    "phone": cust[3],
                    "orders_count": cust[4] + (i % 5),
                    "total_spent": str(round(cust[5] + (i * 45.20), 2)),
                    "currency": "USD",
                    "state": "enabled" if i % 10 != 0 else "disabled",
                    "tags": "Loyal, High-LTV" if cust[4] > 10 else "General",
                    "created_at": now_iso
                })

        return records

    def _graphql_request(self, base_url: str, access_token: str,
                          query: str, variables: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one Shopify Admin GraphQL request."""
        url = f"{base_url}{self.GRAPHQL_ENDPOINT}"
        payload = json.dumps({"query": query, "variables": variables}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "X-Shopify-Access-Token": access_token,
                "User-Agent": "Datarept-ML-Platform/1.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Shopify GraphQL HTTP {exc.code}: {detail[:1000]}") from exc
        except Exception as exc:
            raise RuntimeError(f"Shopify GraphQL request failed: {exc}") from exc

        if body.get("errors"):
            messages = "; ".join(
                item.get("message", "Unknown GraphQL error")
                for item in body["errors"]
            )
            raise RuntimeError(f"Shopify GraphQL error: {messages}")
        if body.get("data") is None:
            raise RuntimeError("Shopify GraphQL response did not contain data.")
        return body["data"]

    @staticmethod
    def _money(money_set: Any, key: str = "amount") -> str:
        if not isinstance(money_set, dict):
            return ""
        shop_money = money_set.get("shopMoney") or {}
        return str(shop_money.get(key, ""))

    def _fetch_graphql_records(self, resource: str, base_url: str,
                               access_token: str, limit: int,
                               status_filter: str) -> List[Dict[str, Any]]:
        """Fetch up to limit records with cursor pagination and normalize fields."""
        if resource == "Orders":
            query_filter = ""
            if status_filter in ("open", "closed"):
                query_filter = f"status:{status_filter}"
            elif status_filter in ("paid", "pending"):
                query_filter = f"financial_status:{status_filter}"
            elif status_filter == "fulfilled":
                query_filter = "fulfillment_status:fulfilled"

            query = """
            query Orders($first: Int!, $after: String, $search: String) {
              orders(first: $first, after: $after, sortKey: CREATED_AT,
                     reverse: true, query: $search) {
                edges { cursor node {
                  id name email createdAt updatedAt tags
                  displayFinancialStatus displayFulfillmentStatus
                  totalPriceSet { shopMoney { amount currencyCode } }
                  subtotalPriceSet { shopMoney { amount } }
                  lineItems(first: 250) {
                    edges { node { quantity } }
                  }
                } }
                pageInfo { hasNextPage endCursor }
              }
            }
            """
        elif resource == "Products":
            query_filter = ""
            if status_filter in ("active", "draft"):
                query_filter = f"status:{status_filter}"
            query = """
            query Products($first: Int!, $after: String, $search: String) {
              products(first: $first, after: $after, sortKey: CREATED_AT,
                       reverse: true, query: $search) {
                edges { cursor node {
                  id title vendor productType status tags createdAt updatedAt
                  variants(first: 1) {
                    edges { node { sku price inventoryQuantity } }
                  }
                } }
                pageInfo { hasNextPage endCursor }
              }
            }
            """
        elif resource == "Customers":
            query_filter = ""
            query = """
            query Customers($first: Int!, $after: String, $search: String) {
              customers(first: $first, after: $after, sortKey: CREATED_AT,
                        reverse: true, query: $search) {
                edges { cursor node {
                  id firstName lastName email phone numberOfOrders
                  amountSpent { amount currencyCode }
                  state tags createdAt
                } }
                pageInfo { hasNextPage endCursor }
              }
            }
            """
        else:
            raise ValueError(f"Unsupported Shopify resource: {resource}")

        records: List[Dict[str, Any]] = []
        cursor = None
        connection_name = resource.lower()
        while len(records) < limit:
            page_size = min(250, limit - len(records))
            data = self._graphql_request(
                base_url, access_token, query,
                {"first": page_size, "after": cursor, "search": query_filter or None},
            )
            connection = data.get(connection_name) or {}
            edges = connection.get("edges") or []
            for edge in edges:
                node = edge.get("node") or {}
                if resource == "Orders":
                    total_set = node.get("totalPriceSet") or {}
                    subtotal_set = node.get("subtotalPriceSet") or {}
                    total_money = total_set.get("shopMoney") or {}
                    records.append({
                        "id": node.get("id", "").rsplit("/", 1)[-1],
                        "order_number": node.get("name", "").lstrip("#"),
                        "name": node.get("name", ""),
                        "email": node.get("email") or "",
                        "total_price": self._money(total_set),
                        "subtotal_price": self._money(subtotal_set),
                        "currency": total_money.get("currencyCode", ""),
                        "financial_status": (node.get("displayFinancialStatus") or "").lower(),
                        "fulfillment_status": (node.get("displayFulfillmentStatus") or "").lower(),
                        "line_items_count": sum(
                            int((item.get("node") or {}).get("quantity", 0))
                            for item in ((node.get("lineItems") or {}).get("edges") or [])
                        ),
                        "tags": ", ".join(node.get("tags") or []),
                        "created_at": node.get("createdAt"),
                        "updated_at": node.get("updatedAt"),
                    })
                elif resource == "Products":
                    variants = (node.get("variants") or {}).get("edges") or []
                    variant = (variants[0].get("node") or {}) if variants else {}
                    records.append({
                        "id": node.get("id", "").rsplit("/", 1)[-1],
                        "title": node.get("title", ""),
                        "vendor": node.get("vendor", ""),
                        "product_type": node.get("productType", ""),
                        "status": (node.get("status") or "").lower(),
                        "price": str(variant.get("price", "")),
                        "sku": variant.get("sku") or "",
                        "inventory_quantity": variant.get("inventoryQuantity", 0),
                        "tags": ", ".join(node.get("tags") or []),
                        "created_at": node.get("createdAt"),
                        "updated_at": node.get("updatedAt"),
                    })
                else:
                    spent = node.get("amountSpent") or {}
                    records.append({
                        "id": node.get("id", "").rsplit("/", 1)[-1],
                        "first_name": node.get("firstName") or "",
                        "last_name": node.get("lastName") or "",
                        "email": node.get("email") or "",
                        "phone": node.get("phone") or "",
                        "orders_count": node.get("numberOfOrders", 0),
                        "total_spent": str(spent.get("amount", "")),
                        "currency": spent.get("currencyCode", ""),
                        "state": (node.get("state") or "").lower(),
                        "tags": ", ".join(node.get("tags") or []),
                        "created_at": node.get("createdAt"),
                    })
                if len(records) >= limit:
                    break

            page_info = connection.get("pageInfo") or {}
            if not page_info.get("hasNextPage") or not page_info.get("endCursor"):
                break
            cursor = page_info["endCursor"]
        return records

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any],
                context: Optional[Any] = None) -> Dict[str, Any]:
        resource = str(config.get("resource", "Orders")).strip()
        shop_domain = str(config.get("shop_domain", "my-store")).strip()
        access_token = (
            str(config.get("access_token", "")).strip()
            or getattr(settings, "SHOPIFY_ACCESS_TOKEN", None)
            or os.getenv("SHOPIFY_ACCESS_TOKEN", "")
        )
        credential_id = str(config.get("credential_id", "")).strip()
        status_filter = str(config.get("status_filter", "All")).strip()
        limit = max(1, min(int(config.get("record_limit", config.get("limit", 50))), 250))
        simulation_mode = bool(config.get("simulation_mode", config.get("simulate_if_no_auth", False)))
        output_format = str(config.get("output_format", "dataframe")).strip().lower()

        if credential_id and context and isinstance(context, dict):
            creds_store = context.get("credentials", {})
            if credential_id in creds_store:
                cred_info = creds_store[credential_id]
                c_data = cred_info.get("data", cred_info)
                access_token = c_data.get("access_token") or c_data.get("token") or access_token
                shop_domain = c_data.get("shop_domain") or c_data.get("domain") or shop_domain

        if shop_domain.startswith(("http://", "https://")):
            base_url = shop_domain.rstrip("/")
        else:
            clean_shop = shop_domain.replace(".myshopify.com", "").strip()
            base_url = f"https://{clean_shop}.myshopify.com"

        if simulation_mode:
            records = self._generate_benchmark_data(resource, limit, status_filter)
            delivery_status = "SIMULATED"
            status_msg = f"Generated {len(records)} benchmark {resource} records because Simulation Mode is enabled."
        else:
            if not access_token:
                raise RuntimeError(
                    "Shopify access token is missing. Configure access_token or SHOPIFY_ACCESS_TOKEN, "
                    "or explicitly enable Simulation Mode."
                )
            records = self._fetch_graphql_records(
                resource, base_url, access_token, limit, status_filter
            )
            delivery_status = "LIVE"
            status_msg = f"Retrieved {len(records)} live Shopify {resource} records from {base_url}."

        # Keep a consistent schema even when Shopify legitimately returns zero records.
        columns = {
            "Orders": ["id", "order_number", "name", "email", "total_price", "subtotal_price",
                       "currency", "financial_status", "fulfillment_status", "line_items_count",
                       "tags", "created_at", "updated_at"],
            "Products": ["id", "title", "vendor", "product_type", "status", "price", "sku",
                         "inventory_quantity", "tags", "created_at", "updated_at"],
            "Customers": ["id", "first_name", "last_name", "email", "phone", "orders_count",
                          "total_spent", "currency", "state", "tags", "created_at"],
        }
        if resource not in columns:
            raise ValueError(f"Unsupported Shopify resource: {resource}")
        df_out = pd.DataFrame(records, columns=columns[resource]) if pd is not None else records

        summary = {
            "title": "Shopify E-commerce Read Summary",
            "message": status_msg,
            "status": delivery_status,
            "resource": resource,
            "shop_domain": shop_domain,
            "records_retrieved": len(records),
            "endpoint": self.GRAPHQL_ENDPOINT,
        }
        metrics = {
            "shopify_resource": resource,
            "record_count": len(records),
            "delivery_status": delivery_status,
            "shop_domain": shop_domain,
        }

        if "json_dict" in output_format or output_format == "json dict":
            output_val = df_out.to_dict() if hasattr(df_out, "to_dict") else records
        elif "json" in output_format:
            output_val = records
        else:
            output_val = df_out

        return {"dataframe": df_out, "output": output_val,
                "output_summary": summary, "metrics": metrics}

    def to_code(self, config: Dict[str, Any]) -> str:
        resource = config.get("resource", "Orders")
        shop = config.get("shop_domain", "my-store")
        limit = max(1, min(int(config.get("record_limit", 50)), 250))
        return (
            "# Recipe: Shopify E-commerce Connector\n"
            f"# Resource: {resource}\n"
            f"# Store: {shop}\n"
            "# Uses Shopify Admin GraphQL API 2026-10.\n"
            "import os, requests, pandas as pd\n"
            f"url = 'https://{shop}.myshopify.com{self.GRAPHQL_ENDPOINT}'\n"
            "headers = {'X-Shopify-Access-Token': os.getenv('SHOPIFY_ACCESS_TOKEN', ''), "
            "'Content-Type': 'application/json'}\n"
            f"# Configure the resource-specific GraphQL query and paginate up to {limit} records.\n"
            "raise NotImplementedError('Use the recipe runtime for the full GraphQL query and pagination.')\n"
        )
