import pytest
import pandas as pd
from backend.app.recipes import recipe_registry
from backend.app.recipes.integrations.zendesk_connector import ZendeskConnectorRecipe
from backend.app.recipes.integrations.freshdesk_connector import FreshdeskConnectorRecipe


def test_registry_registration_and_aliases():
    """Verify Zendesk and Freshdesk connectors and aliases are registered."""
    # Zendesk
    assert recipe_registry.has("zendesk_connector")
    assert recipe_registry.has("zendesk")
    zendesk_rec = recipe_registry.get("zendesk_connector")
    assert isinstance(zendesk_rec, ZendeskConnectorRecipe)

    # Freshdesk
    assert recipe_registry.has("freshdesk_connector")
    assert recipe_registry.has("freshdesk")
    freshdesk_rec = recipe_registry.get("freshdesk_connector")
    assert isinstance(freshdesk_rec, FreshdeskConnectorRecipe)

    # Groups & Subgroups
    zd_meta = recipe_registry.get_metadata("zendesk_connector")
    assert zd_meta.group == "Apps & Integrations"
    assert zd_meta.subgroup == "CRM / Support / Business"

    fd_meta = recipe_registry.get_metadata("freshdesk_connector")
    assert fd_meta.group == "Apps & Integrations"
    assert fd_meta.subgroup == "CRM / Support / Business"


def test_zendesk_connector_schema():
    """Verify Zendesk schema structure and all input_type properties."""
    recipe = ZendeskConnectorRecipe()
    schema = recipe.get_schema()

    assert schema["type"] == "object"
    props = schema["properties"]
    expected_props = [
        "resource",
        "subdomain",
        "auth_type",
        "email",
        "api_token",
        "credential_id",
        "status_filter",
        "record_limit",
        "output_format",
        "simulation_mode",
    ]
    for prop in expected_props:
        assert prop in props, f"Missing property: {prop}"
        assert "input_type" in props[prop], f"Missing input_type for {prop}"


def test_zendesk_connector_simulation_tickets():
    """Verify Zendesk simulation for Tickets."""
    recipe = ZendeskConnectorRecipe()
    result = recipe.execute(
        inputs={},
        config={"resource": "Tickets", "limit": 10, "simulate_if_no_auth": True}
    )
    assert "output" in result
    assert "dataframe" in result
    df = result["output"]
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10
    for col in ["id", "subject", "status", "priority", "created_at"]:
        assert col in df.columns


def test_zendesk_connector_simulation_users():
    """Verify Zendesk simulation for Users."""
    recipe = ZendeskConnectorRecipe()
    result = recipe.execute(
        inputs={},
        config={"resource": "Users", "limit": 5, "simulate_if_no_auth": True}
    )
    df = result["output"]
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    for col in ["id", "name", "email", "role"]:
        assert col in df.columns


def test_zendesk_connector_simulation_satisfaction_ratings():
    """Verify Zendesk simulation for Satisfaction Ratings."""
    recipe = ZendeskConnectorRecipe()
    result = recipe.execute(
        inputs={},
        config={"resource": "Satisfaction Ratings", "limit": 7, "simulate_if_no_auth": True}
    )
    df = result["output"]
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 7
    for col in ["id", "ticket_id", "score"]:
        assert col in df.columns


def test_zendesk_connector_json_output():
    """Verify Zendesk JSON output format."""
    recipe = ZendeskConnectorRecipe()
    result = recipe.execute(
        inputs={},
        config={"resource": "Tickets", "limit": 4, "output_format": "json_records", "simulate_if_no_auth": True}
    )
    records = result["output"]
    assert isinstance(records, list)
    assert len(records) == 4
    assert "subject" in records[0]


def test_freshdesk_connector_schema():
    """Verify Freshdesk schema structure and all input_type properties."""
    recipe = FreshdeskConnectorRecipe()
    schema = recipe.get_schema()

    assert schema["type"] == "object"
    props = schema["properties"]
    expected_props = [
        "resource",
        "domain",
        "api_key",
        "credential_id",
        "status_filter",
        "priority_filter",
        "record_limit",
        "output_format",
        "simulation_mode",
    ]
    for prop in expected_props:
        assert prop in props, f"Missing property: {prop}"
        assert "input_type" in props[prop], f"Missing input_type for {prop}"


def test_freshdesk_connector_simulation_tickets():
    """Verify Freshdesk simulation for Tickets."""
    recipe = FreshdeskConnectorRecipe()
    result = recipe.execute(
        inputs={},
        config={"resource": "Tickets", "limit": 8, "simulate_if_no_auth": True}
    )
    assert "output" in result
    assert "dataframe" in result
    df = result["output"]
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 8
    for col in ["id", "subject", "status", "priority", "source", "created_at"]:
        assert col in df.columns


def test_freshdesk_connector_simulation_contacts():
    """Verify Freshdesk simulation for Contacts."""
    recipe = FreshdeskConnectorRecipe()
    result = recipe.execute(
        inputs={},
        config={"resource": "Contacts", "limit": 6, "simulate_if_no_auth": True}
    )
    df = result["output"]
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 6
    for col in ["id", "name", "email", "phone", "active"]:
        assert col in df.columns


def test_freshdesk_connector_json_output():
    """Verify Freshdesk JSON output format."""
    recipe = FreshdeskConnectorRecipe()
    result = recipe.execute(
        inputs={},
        config={"resource": "Contacts", "limit": 3, "output_format": "json_records", "simulate_if_no_auth": True}
    )
    records = result["output"]
    assert isinstance(records, list)
    assert len(records) == 3
    assert "email" in records[0]
