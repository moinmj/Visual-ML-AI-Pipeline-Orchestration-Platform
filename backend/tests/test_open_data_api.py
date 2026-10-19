"""
Unit tests for Public / Open-Data (Government) API Connector recipe (open_data_api).
"""
import pytest
import pandas as pd
import numpy as np

from backend.app.recipes.base.registry import recipe_registry
from backend.app.recipes.integrations.open_data_api import (
    OpenDataApiRecipe,
    _fetch_world_bank_indicator,
    _fetch_ckan_catalog,
    _fetch_soda_dataset
)
from backend.app.workflows.schema_inference import WorkflowSchemaInferencer


def test_open_data_api_registered_in_catalog():
    """Verify recipe registration, metadata, group, and aliases."""
    recipe = recipe_registry.get("open_data_api")
    assert recipe is not None
    assert recipe.recipe_id == "open_data_api"
    assert recipe.name == "Public / Open-Data (Government) API Connector"
    assert recipe.category == "integrations"

    # Metadata and category grouping
    meta = recipe_registry.get_metadata("open_data_api")
    assert meta.enabled is True
    assert meta.group == "Apps & Integrations"
    assert meta.subgroup == "Market & Reference Data"

    # Aliases
    for alias in ["open_data", "government_data", "gov_data", "public_data", "data_gov"]:
        aliased = recipe_registry.get(alias)
        assert aliased.recipe_id == "open_data_api"


def test_open_data_api_schema_and_preflight_guardrails():
    """Ensure all configuration parameters have valid input_type and no empty defaults."""
    recipe = OpenDataApiRecipe()
    schema = recipe.get_schema()
    assert "properties" in schema
    props = schema["properties"]

    valid_types = {
        "dropdown", "text", "password", "number", "column_select", "multi_column_select", "checkbox", "slider"
    }

    for pname, pdef in props.items():
        assert "input_type" in pdef, f"Property '{pname}' missing input_type"
        assert pdef["input_type"] in valid_types, f"Property '{pname}' has invalid input_type '{pdef['input_type']}'"

        # Critical guardrail
        low = pname.lower()
        if any(k in low for k in ("column", "metric", "target")):
            assert pdef.get("default") != "", f"Property '{pname}' has forbidden empty string default"

    # Specific property assertions
    assert props["portal_type"]["input_type"] == "dropdown"
    assert props["preset"]["input_type"] == "dropdown"
    assert props["search_query"]["input_type"] == "text"
    assert props["dataset_id_or_url"]["input_type"] == "text"
    assert props["country_code"]["input_type"] == "text"
    assert props["date_range"]["input_type"] == "text"
    assert props["limit"]["input_type"] == "number"
    assert props["api_key"]["input_type"] == "password"
    assert props["query_field"]["input_type"] == "column_select"


def test_open_data_api_curated_preset_execution():
    """Test standalone execution using curated preset (World Bank GDP)."""
    recipe = OpenDataApiRecipe()
    res = recipe.execute(
        inputs={},
        config={
            "portal_type": "Curated Presets (World Bank & US Government)",
            "preset": "World Bank: Real GDP (Current US$)",
            "country_code": "US, CN, DE",
            "date_range": "2020:2023",
            "limit": 50
        }
    )

    assert "dataframe" in res
    df = res["dataframe"]
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "country_name" in df.columns
    assert "value" in df.columns
    assert "year" in df.columns
    assert "indicator_id" in df.columns


def test_open_data_api_ckan_catalog_search():
    """Test CKAN / Data.gov catalog discovery mode."""
    recipe = OpenDataApiRecipe()
    res = recipe.execute(
        inputs={},
        config={
            "portal_type": "Data.gov / CKAN Catalog Search",
            "search_query": "climate",
            "limit": 10
        }
    )

    assert "dataframe" in res
    df = res["dataframe"]
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "title" in df.columns
    assert "dataset_id" in df.columns
    assert "organization" in df.columns


def test_open_data_api_soda_ev_dataset():
    """Test Socrata SODA endpoint parsing."""
    recipe = OpenDataApiRecipe()
    res = recipe.execute(
        inputs={},
        config={
            "portal_type": "Socrata Open Data (SODA API)",
            "dataset_id_or_url": "f6w7-q2d2",
            "limit": 20
        }
    )

    assert "dataframe" in res
    df = res["dataframe"]
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert len(df) > 0


def test_open_data_api_batch_upstream_mode():
    """Test batch-row enrichment when upstream queries are passed."""
    recipe = OpenDataApiRecipe()
    upstream_df = pd.DataFrame({
        "country_iso": ["US", "DEU"]
    })

    res = recipe.execute(
        inputs={"dataframe": upstream_df},
        config={
            "portal_type": "World Bank Open Data API",
            "query_field": "country_iso",
            "date_range": "2022:2023",
            "limit": 10
        }
    )

    assert "dataframe" in res
    df = res["dataframe"]
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "source_query" in df.columns


def test_open_data_api_to_code():
    """Test Python code representation generation."""
    recipe = OpenDataApiRecipe()
    code = recipe.to_code({"portal_type": "World Bank Open Data API", "country_code": "US"})
    assert "open_data_api" in code
    assert "World Bank Open Data API" in code


@pytest.mark.asyncio
async def test_open_data_api_schema_inferencer():
    """Test that WorkflowSchemaInferencer correctly infers columns for open_data_api."""
    nodes = [
        {"id": "open_data_1", "recipe_id": "open_data_api", "config": {}},
        {"id": "filter_1", "recipe_id": "row_filter", "config": {}}
    ]
    edges = [
        {"source": "open_data_1", "target": "filter_1", "sourceHandle": "output", "targetHandle": "input"}
    ]
    res = await WorkflowSchemaInferencer.infer_schema(nodes, edges)
    assert res.success is True
    cols = res.node_schemas["open_data_1"].columns
    assert "country_name" in cols
    assert "value" in cols
    assert "year" in cols
