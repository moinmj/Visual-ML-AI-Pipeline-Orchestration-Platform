import pandas as pd
import pytest

from backend.app.recipes.reporting.dashboard_publisher import DashboardPublisherRecipe


def test_dashboard_publisher_metadata_and_schema():
    recipe = DashboardPublisherRecipe()
    assert recipe.recipe_id == "dashboard_publisher"
    assert recipe.category == "reporting"
    assert "dataframe" in recipe.output_types

    schema = recipe.get_schema()
    assert schema["type"] == "object"
    props = schema["properties"]
    assert "target_platform" in props
    assert "dashboard_id" in props
    assert "embed_mode" in props
    assert "access_policy" in props

    for prop_name, prop_data in props.items():
        assert "description" in prop_data, f"Property {prop_name} missing description"
        assert "input_type" in prop_data, f"Property {prop_name} missing input_type"


def test_dashboard_publisher_fallback_internal_embed():
    recipe = DashboardPublisherRecipe()
    result = recipe.execute({}, {})

    assert "output" in result
    assert "dataframe" in result
    assert "publish_record" in result
    assert "embed_url" in result
    assert "iframe_code" in result

    pub = result["publish_record"]
    assert pub["status"] == "PUBLISHED"
    assert pub["target_platform"] == "internal_embed"
    assert "signed_token" in pub
    assert pub["signed_token"].startswith("ey.")
    assert "crt.aiiotgeeks.com/embed/dashboards/dash-001" in result["embed_url"]
    assert "<iframe" in result["iframe_code"]
    assert 'sandbox="allow-scripts allow-same-origin allow-popups allow-forms"' in result["iframe_code"]


def test_dashboard_publisher_third_party_platforms():
    recipe = DashboardPublisherRecipe()
    df = pd.DataFrame({
        "channel": ["Organic Search", "Paid Social", "Direct"],
        "conversions": [1400, 850, 620],
        "cac": [12.4, 45.1, 0.0],
    })

    # Test Metabase
    res_meta = recipe.execute(
        {"input": df},
        {
            "target_platform": "metabase",
            "dashboard_id": "88",
            "target_endpoint_url": "https://analytics.internal.corp",
            "api_token": "secret_meta_key_123",
        },
    )
    assert "analytics.internal.corp/dashboard/88#token=" in res_meta["embed_url"]
    assert res_meta["dataframe"].equals(df)

    # Test Grafana
    res_grafana = recipe.execute(
        {"dataframe": df},
        {
            "target_platform": "grafana",
            "dashboard_id": "cpu-fleet-dash",
            "theme": "light",
        },
    )
    assert "/d/cpu-fleet-dash?kiosk&theme=light" in res_grafana["embed_url"]

    # Test PowerBI
    res_powerbi = recipe.execute(
        {"input": df},
        {
            "target_platform": "powerbi_service",
            "dashboard_id": "rep-abc-999",
        },
    )
    assert "reportEmbed?reportId=rep-abc-999" in res_powerbi["embed_url"]

