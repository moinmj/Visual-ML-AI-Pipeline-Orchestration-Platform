"""
Unit tests for Web Scraper / HTML Table Extractor Recipe.
"""

import pytest
import pandas as pd
from unittest.mock import MagicMock, patch

from backend.app.core.exceptions import ExecutionException, ValidationException
from backend.app.recipes.ingestion.web_scraper import WebScraperRecipe
from backend.app.recipes.base.registry import recipe_registry


def test_web_scraper_registration():
    """Verify recipe registration and catalog metadata."""
    recipe = recipe_registry.get("web_scraper")
    assert recipe is not None
    assert recipe.recipe_id == "web_scraper"
    assert recipe.group == "Ingestion & Orchestration"
    assert recipe.subgroup == "File & Web Ingestion"


def test_web_scraper_schema_contract():
    """Verify all schema properties have input_type and description."""
    recipe = WebScraperRecipe()
    schema = recipe.get_schema()
    assert schema["type"] == "object"
    props = schema["properties"]
    for field_name, spec in props.items():
        assert "input_type" in spec, f"Field '{field_name}' missing input_type"
        assert "description" in spec, f"Field '{field_name}' missing description"


def test_web_scraper_validation_errors():
    """Test required parameter and enum validations."""
    recipe = WebScraperRecipe()

    # Missing url
    with pytest.raises(ValidationException) as exc_1:
        recipe.execute(inputs={}, config={"url": ""})
    assert "Parameter 'url' cannot be empty" in str(exc_1.value)

    # Invalid extraction mode
    with pytest.raises(ValidationException) as exc_2:
        recipe.execute(inputs={}, config={"url": "https://test.com", "extraction_mode": "video_stream"})
    assert "Unsupported extraction_mode" in str(exc_2.value)


def test_web_scraper_simulation_fallback():
    """Verify synthetic table fallback when network request fails."""
    recipe = WebScraperRecipe()
    config = {
        "url": "https://unreachable.domain.internal/data",
        "simulation_fallback": True,
        "include_metadata_columns": True,
    }
    with patch.object(recipe, "_fetch_html", side_effect=ConnectionError("Host unreachable")):
        result = recipe.execute(inputs={}, config=config)

    df = result["dataframe"]
    metrics = result["metrics"]

    assert not df.empty
    assert metrics["is_simulation"] is True
    assert "_source_url" in df.columns
    assert "_extracted_at" in df.columns
    assert "country" in df.columns or "rank" in df.columns


def test_web_scraper_mock_html_table():
    """Verify parsing HTML tables from mock HTML string."""
    recipe = WebScraperRecipe()
    mock_html = """
    <html>
      <body>
        <table id="leaderboard">
          <thead>
            <tr><th>Player</th><th>Score</th><th>Tier</th></tr>
          </thead>
          <tbody>
            <tr><td>Alpha</td><td>950</td><td>Master</td></tr>
            <tr><td>Beta</td><td>870</td><td>Diamond</td></tr>
          </tbody>
        </table>
      </body>
    </html>
    """
    config = {
        "url": "https://mock.site/leaderboard",
        "mock_html": mock_html,
        "extraction_mode": "html_tables",
        "include_metadata_columns": True,
    }
    result = recipe.execute(inputs={}, config=config)
    df = result["dataframe"]
    assert len(df) == 2
    assert "Player" in df.columns
    assert list(df["Player"]) == ["Alpha", "Beta"]
    assert df.iloc[0]["_source_url"] == "https://mock.site/leaderboard"


def test_web_scraper_multiple_tables_indexing():
    """Verify selecting table by table_index."""
    recipe = WebScraperRecipe()
    mock_html = """
    <html>
      <body>
        <table id="table0"><tr><th>Col0</th></tr><tr><td>A0</td></tr></table>
        <table id="table1"><tr><th>Col1</th></tr><tr><td>B1</td></tr></table>
      </body>
    </html>
    """
    # Select table index 1
    config = {
        "url": "https://mock.site",
        "mock_html": mock_html,
        "table_index": 1,
    }
    result = recipe.execute(inputs={}, config=config)
    df = result["dataframe"]
    assert "Col1" in df.columns
    assert df.iloc[0]["Col1"] == "B1"


def test_web_scraper_page_links_mode():
    """Verify page_links extraction mode extracts hyperlinks and attributes."""
    recipe = WebScraperRecipe()
    mock_html = """
    <html>
      <body>
        <a href="/about">About Us</a>
        <a href="https://external.com/partner">Partner Site</a>
      </body>
    </html>
    """
    config = {
        "url": "https://mysite.com",
        "mock_html": mock_html,
        "extraction_mode": "page_links",
    }
    result = recipe.execute(inputs={}, config=config)
    df = result["dataframe"]
    assert len(df) == 2
    assert "link_text" in df.columns
    assert "url" in df.columns
    assert "is_external" in df.columns
    assert list(df["link_text"]) == ["About Us", "Partner Site"]


def test_web_scraper_raw_text_blocks_mode():
    """Verify raw_text_blocks extraction mode parses structured paragraphs and headings."""
    recipe = WebScraperRecipe()
    mock_html = """
    <html>
      <body>
        <h1>Title of Research Paper</h1>
        <p>This is the introduction paragraph describing the experimental setup.</p>
      </body>
    </html>
    """
    config = {
        "url": "https://arxiv.org/abs/123",
        "mock_html": mock_html,
        "extraction_mode": "raw_text_blocks",
    }
    result = recipe.execute(inputs={}, config=config)
    df = result["dataframe"]
    assert len(df) == 2
    assert "tag" in df.columns
    assert "content" in df.columns
    assert df.iloc[0]["tag"] == "h1"
    assert df.iloc[1]["tag"] == "p"


def test_web_scraper_simulation_disabled_raises():
    """Verify error raised when network fails and simulation fallback is false."""
    recipe = WebScraperRecipe()
    config = {
        "url": "https://fail.domain.com",
        "simulation_fallback": False,
    }
    with patch.object(recipe, "_fetch_html", side_effect=ConnectionError("Failed")):
        with pytest.raises(ExecutionException) as exc:
            recipe.execute(inputs={}, config=config)
        assert "Failed to fetch content from URL" in str(exc.value)
