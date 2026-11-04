import pytest
import pandas as pd
from backend.app.recipes import recipe_registry
from backend.app.recipes.integrations.intercom_connector import IntercomConnectorRecipe
from backend.app.recipes.integrations.stripe_connector import StripeConnectorRecipe
from backend.app.recipes.integrations.shopify_connector import ShopifyConnectorRecipe
from backend.app.recipes.integrations.google_ads_connector import GoogleAdsConnectorRecipe
from backend.app.recipes.integrations.meta_ads_connector import MetaAdsConnectorRecipe


def test_registry_registration_and_aliases_all_five():
    """Verify all 5 new business connectors and their aliases are properly registered."""
    connectors = [
        ("intercom_connector", "intercom", IntercomConnectorRecipe),
        ("stripe_connector", "stripe", StripeConnectorRecipe),
        ("shopify_connector", "shopify", ShopifyConnectorRecipe),
        ("google_ads_connector", "google_ads", GoogleAdsConnectorRecipe),
        ("meta_ads_connector", "meta_ads", MetaAdsConnectorRecipe),
    ]

    for rid, alias, cls in connectors:
        assert recipe_registry.has(rid), f"Registry missing {rid}"
        assert recipe_registry.has(alias), f"Registry missing alias {alias}"
        rec = recipe_registry.get(rid)
        assert isinstance(rec, cls)
        meta = recipe_registry.get_metadata(rid)
        assert meta.group == "Apps & Integrations"
        assert meta.subgroup == "CRM / Support / Business"

    # Extra aliases
    assert recipe_registry.has("facebook_ads")
    assert recipe_registry.has("meta")


def test_intercom_connector_schema_and_execution():
    """Verify Intercom schema structure and simulation execution."""
    recipe = IntercomConnectorRecipe()
    schema = recipe.get_schema()
    for prop_name, prop_data in schema["properties"].items():
        assert "input_type" in prop_data, f"Intercom missing input_type for {prop_name}"

    # Conversations
    res_conv = recipe.execute(
        inputs={},
        config={"resource": "Conversations", "limit": 6, "simulate_if_no_auth": True}
    )
    df_conv = res_conv["output"]
    assert isinstance(df_conv, pd.DataFrame)
    assert len(df_conv) == 6
    assert "source_body" in df_conv.columns
    assert "state" in df_conv.columns

    # Users
    res_usr = recipe.execute(
        inputs={},
        config={"resource": "Users", "limit": 4, "simulate_if_no_auth": True}
    )
    df_usr = res_usr["output"]
    assert len(df_usr) == 4
    assert "email" in df_usr.columns

    # Companies
    res_comp = recipe.execute(
        inputs={},
        config={"resource": "Companies", "limit": 3, "simulate_if_no_auth": True}
    )
    df_comp = res_comp["output"]
    assert len(df_comp) == 3
    assert "monthly_spend" in df_comp.columns

    # JSON output
    res_json = recipe.execute(
        inputs={},
        config={"resource": "Conversations", "limit": 2, "output_format": "json_records", "simulate_if_no_auth": True}
    )
    assert isinstance(res_json["output"], list)
    assert len(res_json["output"]) == 2


def test_stripe_connector_schema_and_execution():
    """Verify Stripe Payments schema structure and simulation execution."""
    recipe = StripeConnectorRecipe()
    schema = recipe.get_schema()
    for prop_name, prop_data in schema["properties"].items():
        assert "input_type" in prop_data, f"Stripe missing input_type for {prop_name}"

    # Payment Intents
    res_pay = recipe.execute(
        inputs={},
        config={"resource": "Payment Intents", "limit": 8, "simulate_if_no_auth": True}
    )
    df_pay = res_pay["output"]
    assert isinstance(df_pay, pd.DataFrame)
    assert len(df_pay) == 8
    assert "amount" in df_pay.columns
    assert "currency" in df_pay.columns

    # Customers
    res_cust = recipe.execute(
        inputs={},
        config={"resource": "Customers", "limit": 5, "simulate_if_no_auth": True}
    )
    df_cust = res_cust["output"]
    assert len(df_cust) == 5
    assert "email" in df_cust.columns

    # Subscriptions
    res_sub = recipe.execute(
        inputs={},
        config={"resource": "Subscriptions", "limit": 4, "simulate_if_no_auth": True}
    )
    df_sub = res_sub["output"]
    assert len(df_sub) == 4
    assert "plan_name" in df_sub.columns

    # JSON output
    res_json = recipe.execute(
        inputs={},
        config={"resource": "Subscriptions", "limit": 3, "output_format": "json_records", "simulate_if_no_auth": True}
    )
    assert isinstance(res_json["output"], list)
    assert len(res_json["output"]) == 3


def test_shopify_connector_schema_and_execution():
    """Verify Shopify schema structure and simulation execution."""
    recipe = ShopifyConnectorRecipe()
    schema = recipe.get_schema()
    for prop_name, prop_data in schema["properties"].items():
        assert "input_type" in prop_data, f"Shopify missing input_type for {prop_name}"

    # Orders
    res_ord = recipe.execute(
        inputs={},
        config={"resource": "Orders", "limit": 7, "simulate_if_no_auth": True}
    )
    df_ord = res_ord["output"]
    assert isinstance(df_ord, pd.DataFrame)
    assert len(df_ord) == 7
    assert "order_number" in df_ord.columns
    assert "total_price" in df_ord.columns

    # Products
    res_prod = recipe.execute(
        inputs={},
        config={"resource": "Products", "limit": 6, "simulate_if_no_auth": True}
    )
    df_prod = res_prod["output"]
    assert len(df_prod) == 6
    assert "title" in df_prod.columns
    assert "price" in df_prod.columns

    # Customers
    res_cust = recipe.execute(
        inputs={},
        config={"resource": "Customers", "limit": 5, "simulate_if_no_auth": True}
    )
    df_cust = res_cust["output"]
    assert len(df_cust) == 5
    assert "total_spent" in df_cust.columns

    # JSON output
    res_json = recipe.execute(
        inputs={},
        config={"resource": "Orders", "limit": 2, "output_format": "json_records", "simulate_if_no_auth": True}
    )
    assert isinstance(res_json["output"], list)
    assert len(res_json["output"]) == 2


def test_google_ads_connector_schema_and_execution():
    """Verify Google Ads schema structure and simulation execution."""
    recipe = GoogleAdsConnectorRecipe()
    schema = recipe.get_schema()
    for prop_name, prop_data in schema["properties"].items():
        assert "input_type" in prop_data, f"Google Ads missing input_type for {prop_name}"

    # Campaign Performance
    res_camp = recipe.execute(
        inputs={},
        config={"resource": "Campaign Performance", "limit": 8, "simulate_if_no_auth": True}
    )
    df_camp = res_camp["output"]
    assert isinstance(df_camp, pd.DataFrame)
    assert len(df_camp) == 8
    assert "campaign_name" in df_camp.columns
    assert "impressions" in df_camp.columns
    assert "clicks" in df_camp.columns
    assert "cost" in df_camp.columns

    # Ad Group Performance
    res_ag = recipe.execute(
        inputs={},
        config={"resource": "Ad Group Performance", "limit": 5, "simulate_if_no_auth": True}
    )
    df_ag = res_ag["output"]
    assert len(df_ag) == 5
    assert "ad_group_name" in df_ag.columns

    # Keyword Performance
    res_kw = recipe.execute(
        inputs={},
        config={"resource": "Keyword Performance", "limit": 4, "simulate_if_no_auth": True}
    )
    df_kw = res_kw["output"]
    assert len(df_kw) == 4
    assert "keyword_text" in df_kw.columns

    # JSON output
    res_json = recipe.execute(
        inputs={},
        config={"resource": "Campaign Performance", "limit": 3, "output_format": "json_records", "simulate_if_no_auth": True}
    )
    assert isinstance(res_json["output"], list)
    assert len(res_json["output"]) == 3


def test_meta_ads_connector_schema_and_execution():
    """Verify Meta Ads schema structure and simulation execution."""
    recipe = MetaAdsConnectorRecipe()
    schema = recipe.get_schema()
    for prop_name, prop_data in schema["properties"].items():
        assert "input_type" in prop_data, f"Meta Ads missing input_type for {prop_name}"

    # Campaign Insights
    res_camp = recipe.execute(
        inputs={},
        config={"resource": "Campaign Insights", "limit": 7, "simulate_if_no_auth": True}
    )
    df_camp = res_camp["output"]
    assert isinstance(df_camp, pd.DataFrame)
    assert len(df_camp) == 7
    assert "campaign_name" in df_camp.columns
    assert "impressions" in df_camp.columns
    assert "reach" in df_camp.columns
    assert "spend" in df_camp.columns
    assert "roas" in df_camp.columns

    # Ad Set Insights
    res_adset = recipe.execute(
        inputs={},
        config={"resource": "Ad Set Insights", "limit": 4, "simulate_if_no_auth": True}
    )
    df_adset = res_adset["output"]
    assert len(df_adset) == 4
    assert "adset_name" in df_adset.columns

    # Ad Creative Performance
    res_cr = recipe.execute(
        inputs={},
        config={"resource": "Ad Creative Performance", "limit": 5, "simulate_if_no_auth": True}
    )
    df_cr = res_cr["output"]
    assert len(df_cr) == 5
    assert "ad_name" in df_cr.columns

    # JSON output
    res_json = recipe.execute(
        inputs={},
        config={"resource": "Campaign Insights", "limit": 2, "output_format": "json_records", "simulate_if_no_auth": True}
    )
    assert isinstance(res_json["output"], list)
    assert len(res_json["output"]) == 2

