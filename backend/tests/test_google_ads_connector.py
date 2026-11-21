import json
import urllib.error
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from backend.app.recipes.integrations.google_ads_connector import GoogleAdsConnectorRecipe


def test_google_ads_schema():
    recipe = GoogleAdsConnectorRecipe()
    schema = recipe.get_schema()
    assert schema["type"] == "object"
    props = schema["properties"]

    assert "customer_id" in props
    assert props["customer_id"]["input_type"] == "text"
    assert "manager_customer_id" in props
    assert props["manager_customer_id"]["input_type"] == "text"
    assert "oauth_client_id" in props
    assert "oauth_client_secret" in props
    assert props["oauth_client_secret"]["input_type"] == "password"
    assert "refresh_token" in props
    assert props["refresh_token"]["input_type"] == "password"
    assert "api_version" in props
    assert props["api_version"]["default"] == "v24"
    assert "developer_token" in props
    assert props["developer_token"]["input_type"] == "password"
    assert "access_token" in props
    assert props["access_token"]["input_type"] == "password"
    assert "credential_id" in props
    assert props["credential_id"]["input_type"] == "credential_dropdown"
    assert "simulation_mode" in props
    assert props["simulation_mode"]["input_type"] == "checkbox"


def test_clean_customer_id():
    assert GoogleAdsConnectorRecipe._clean_customer_id("123-456-7890") == "1234567890"
    assert GoogleAdsConnectorRecipe._clean_customer_id(" 987 654 3210 ") == "9876543210"
    assert GoogleAdsConnectorRecipe._clean_customer_id(None) == ""
    assert GoogleAdsConnectorRecipe._clean_customer_id(1234567890) == "1234567890"


def test_build_gaql_queries():
    # Campaign Performance with date and status filters
    gaql_camp = GoogleAdsConnectorRecipe._build_gaql(
        resource="Campaign Performance",
        limit=50,
        status_filter="ENABLED",
        date_range="Last 30 Days",
    )
    assert "SELECT campaign.id, campaign.name" in gaql_camp
    assert "FROM campaign" in gaql_camp
    assert "segments.date DURING LAST_30_DAYS" in gaql_camp
    assert "campaign.status = ENABLED" in gaql_camp
    assert "ORDER BY metrics.impressions DESC LIMIT 50" in gaql_camp

    # Ad Group Performance
    gaql_ag = GoogleAdsConnectorRecipe._build_gaql(
        resource="Ad Group Performance",
        limit=25,
        status_filter="PAUSED",
        date_range="Last 7 Days",
    )
    assert "FROM ad_group" in gaql_ag
    assert "ad_group.status = PAUSED" in gaql_ag
    assert "segments.date DURING LAST_7_DAYS" in gaql_ag
    assert "LIMIT 25" in gaql_ag

    # Keyword Performance
    gaql_kw = GoogleAdsConnectorRecipe._build_gaql(
        resource="Keyword Performance",
        limit=10,
        status_filter="All",
        date_range="All Time",
    )
    assert "FROM keyword_view" in gaql_kw
    assert "ad_group_criterion.keyword.text" in gaql_kw
    assert "LIMIT 10" in gaql_kw


def test_row_to_record_parsing():
    sample_api_row = {
        "campaign": {
            "id": "18999999999",
            "name": "Summer Promotion",
            "status": "ENABLED",
            "advertisingChannelType": "SEARCH",
        },
        "metrics": {
            "impressions": "15000",
            "clicks": "450",
            "ctr": 0.03,
            "costMicros": "450000000",  # $450.00
            "averageCpc": 1.0,           # $1.00 directly in currency units (not micros)
            "conversions": 30.0,
            "costPerConversion": 15.0,   # $15.00 directly in currency units (not micros)
            "conversionsFromInteractionsRate": 0.0667,
        },
    }

    record = GoogleAdsConnectorRecipe._row_to_record("Campaign Performance", sample_api_row)
    assert record["campaign_id"] == "18999999999"
    assert record["campaign_name"] == "Summer Promotion"
    assert record["status"] == "ENABLED"
    assert record["impressions"] == 15000
    assert record["clicks"] == 450
    assert record["cost"] == 450.0
    assert record["cost_micros"] == 450000000
    assert record["average_cpc"] == 1.0
    assert record["cost_per_conversion"] == 15.0
    assert record["conversions"] == 30.0


def test_live_execution_mocked_success():
    recipe = GoogleAdsConnectorRecipe()

    fake_response_data = {
        "results": [
            {
                "campaign": {
                    "id": "18001002003",
                    "name": "Q4 Enterprise Search",
                    "status": "ENABLED",
                    "advertisingChannelType": "SEARCH",
                },
                "metrics": {
                    "impressions": "5000",
                    "clicks": "200",
                    "ctr": 0.04,
                    "costMicros": "240000000",
                    "averageCpc": 1.20,
                    "conversions": 16.0,
                    "costPerConversion": 15.0,
                    "conversionsFromInteractionsRate": 0.08,
                },
            }
        ]
    }

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = json.dumps(fake_response_data).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        result = recipe.execute(
            inputs={},
            config={
                "resource": "Campaign Performance",
                "customer_id": "123-456-7890",
                "manager_customer_id": "999-888-7777",
                "access_token": "ya29.valid_live_token",
                "developer_token": "dev-token-abc",
                "record_limit": 10,
            },
        )

        assert result["output_summary"]["status"] == "LIVE"
        assert result["output_summary"]["http_status"] == 200
        assert len(result["dataframe"]) == 1
        row = result["dataframe"].iloc[0]
        assert row["campaign_name"] == "Q4 Enterprise Search"
        assert row["clicks"] == 200
        assert row["cost"] == 240.0
        assert row["average_cpc"] == 1.20

        # Verify outgoing request headers
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert req.get_header("Authorization") == "Bearer ya29.valid_live_token"
        assert req.get_header("Login-customer-id") == "9998887777"
        assert req.get_header("Developer-token") == "dev-token-abc"
        assert req.full_url == "https://googleads.googleapis.com/v24/customers/1234567890/googleAds:search"


def test_live_execution_token_refresh():
    recipe = GoogleAdsConnectorRecipe()

    mock_refresh_resp = MagicMock()
    mock_refresh_resp.read.return_value = json.dumps({"access_token": "ya29.fresh_access_token"}).encode("utf-8")
    mock_refresh_resp.__enter__.return_value = mock_refresh_resp

    mock_search_resp = MagicMock()
    mock_search_resp.status = 200
    mock_search_resp.read.return_value = json.dumps({"results": []}).encode("utf-8")
    mock_search_resp.__enter__.return_value = mock_search_resp

    with patch("urllib.request.urlopen", side_effect=[mock_refresh_resp, mock_search_resp]) as mock_urlopen:
        result = recipe.execute(
            inputs={},
            config={
                "resource": "Campaign Performance",
                "customer_id": "1234567890",
                "oauth_client_id": "client-id-xyz.apps.googleusercontent.com",
                "oauth_client_secret": "secret-xyz",
                "refresh_token": "1//refresh_token_xyz",
            },
        )

        assert result["output_summary"]["status"] == "LIVE"
        assert mock_urlopen.call_count == 2
        search_req = mock_urlopen.call_args_list[1][0][0]
        assert search_req.get_header("Authorization") == "Bearer ya29.fresh_access_token"


def test_simulation_fallback_when_simulate_if_no_auth():
    recipe = GoogleAdsConnectorRecipe()

    # When no real tokens and simulate_if_no_auth is given
    res = recipe.execute(
        inputs={},
        config={"resource": "Campaign Performance", "limit": 5, "simulate_if_no_auth": True},
    )
    assert res["output_summary"]["status"] == "SIMULATED"
    assert len(res["dataframe"]) == 5
    assert "campaign_name" in res["dataframe"].columns


def test_to_code_output():
    recipe = GoogleAdsConnectorRecipe()
    code = recipe.to_code({
        "resource": "Campaign Performance",
        "customer_id": "111-222-3333",
        "manager_customer_id": "444-555-6666",
        "api_version": "v24",
    })
    assert "https://googleads.googleapis.com/v24/customers/1112223333/googleAds:search" in code
    assert "headers['login-customer-id'] = '4445556666'" in code
    assert "os.environ['GOOGLE_ADS_ACCESS_TOKEN']" in code
