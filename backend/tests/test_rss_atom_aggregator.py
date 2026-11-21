"""
Unit tests for RSS/Atom Aggregator Recipe.
"""

import pytest
import pandas as pd
from unittest.mock import MagicMock, patch

from backend.app.core.exceptions import ExecutionException, ValidationException
from backend.app.recipes.ingestion.rss_atom_aggregator import RSSAtomAggregatorRecipe
from backend.app.recipes.base.registry import recipe_registry


def test_rss_atom_aggregator_registration():
    """Verify recipe registration and catalog metadata."""
    recipe = recipe_registry.get("rss_atom_aggregator")
    assert recipe is not None
    assert recipe.recipe_id == "rss_atom_aggregator"
    assert recipe.group == "Ingestion & Orchestration"
    assert recipe.subgroup == "File & Web Ingestion"


def test_rss_atom_aggregator_schema_contract():
    """Verify all schema properties have input_type and description."""
    recipe = RSSAtomAggregatorRecipe()
    schema = recipe.get_schema()
    assert schema["type"] == "object"
    props = schema["properties"]
    for field_name, spec in props.items():
        assert "input_type" in spec, f"Field '{field_name}' missing input_type"
        assert "description" in spec, f"Field '{field_name}' missing description"


def test_rss_atom_aggregator_validation_errors():
    """Test required parameter validations."""
    recipe = RSSAtomAggregatorRecipe()

    # Missing feed URLs and no mock
    with pytest.raises(ValidationException) as exc:
        recipe.execute(inputs={}, config={"feed_urls": ""})
    assert "At least one feed URL must be provided" in str(exc.value)


def test_rss_atom_aggregator_simulation_fallback():
    """Verify simulation fallback generates deduplicated tech news feeds."""
    recipe = RSSAtomAggregatorRecipe()
    config = {
        "feed_urls": "https://offline.feed.news/rss.xml",
        "simulation_fallback": True,
        "deduplication_key": "link_or_guid",
    }
    with patch.object(recipe, "_fetch_url_bytes", side_effect=ConnectionError("Offline")):
        result = recipe.execute(inputs={}, config=config)

    df = result["dataframe"]
    metrics = result["metrics"]

    assert not df.empty
    assert metrics["is_simulation"] is True
    # The synthetic feed includes 3 items with 1 duplicate, so unique items should be 2
    assert len(df) == 2
    assert metrics["duplicates_removed"] == 1
    assert "feed_title" in df.columns
    assert "title" in df.columns
    assert "link" in df.columns


def test_rss_atom_aggregator_mock_rss2_xml():
    """Verify parsing standard RSS 2.0 channel and items."""
    recipe = RSSAtomAggregatorRecipe()
    rss_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
      <channel>
        <title>Engineering Digest</title>
        <item>
          <title>Release 2.5 Deployed</title>
          <link>https://eng.example.com/posts/release-2-5</link>
          <guid>post-250</guid>
          <pubDate>Fri, 06 Nov 2026 09:30:00 GMT</pubDate>
          <description>&lt;p&gt;Zero downtime &lt;b&gt;deployment&lt;/b&gt; succeeded.&lt;/p&gt;</description>
          <author>devops@example.com</author>
        </item>
      </channel>
    </rss>
    """
    config = {
        "mock_feeds_xml": rss_xml,
        "strip_html": True,
    }
    result = recipe.execute(inputs={}, config=config)
    df = result["dataframe"]

    assert len(df) == 1
    row = df.iloc[0]
    assert row["feed_title"] == "Engineering Digest"
    assert row["title"] == "Release 2.5 Deployed"
    assert row["link"] == "https://eng.example.com/posts/release-2-5"
    assert row["summary"] == "Zero downtime deployment succeeded."
    assert row["author"] == "devops@example.com"


def test_rss_atom_aggregator_mock_atom_xml():
    """Verify parsing Atom 1.0 feed and entries."""
    recipe = RSSAtomAggregatorRecipe()
    atom_xml = """<?xml version="1.0" encoding="utf-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <title>Science Daily</title>
      <entry>
        <title>Quantum Computing Milestone</title>
        <link href="https://sciencedaily.internal/quantum-milestone"/>
        <id>urn:uuid:1225c695-cfb8-4ebb-aaaa-80da344efa6a</id>
        <updated>2026-11-06T10:00:00Z</updated>
        <summary>Coherence time increased tenfold.</summary>
        <author><name>Dr. Thorne</name></author>
      </entry>
    </feed>
    """
    config = {
        "mock_feeds_xml": atom_xml,
    }
    result = recipe.execute(inputs={}, config=config)
    df = result["dataframe"]

    assert len(df) == 1
    row = df.iloc[0]
    assert row["feed_title"] == "Science Daily"
    assert row["title"] == "Quantum Computing Milestone"
    assert row["link"] == "https://sciencedaily.internal/quantum-milestone"
    assert row["author"] == "Dr. Thorne"


def test_rss_atom_aggregator_deduplication():
    """Verify duplicate detection and elimination across entries."""
    recipe = RSSAtomAggregatorRecipe()
    rss_with_dups = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
      <channel>
        <title>Tech Feed</title>
        <item>
          <title>Article A</title>
          <link>https://news.com/a</link>
          <pubDate>Fri, 06 Nov 2026 08:00:00 GMT</pubDate>
        </item>
        <item>
          <title>Article A Duplicate</title>
          <link>https://news.com/a</link>
          <pubDate>Fri, 06 Nov 2026 09:00:00 GMT</pubDate>
        </item>
        <item>
          <title>Article B</title>
          <link>https://news.com/b</link>
          <pubDate>Fri, 06 Nov 2026 07:00:00 GMT</pubDate>
        </item>
      </channel>
    </rss>
    """
    config = {
        "mock_feeds_xml": rss_with_dups,
        "deduplication_key": "link",
        "dedup_strategy": "keep_newest",
    }
    result = recipe.execute(inputs={}, config=config)
    df = result["dataframe"]
    metrics = result["metrics"]

    assert len(df) == 2
    assert metrics["duplicates_removed"] == 1
    # keep_newest should have retained the 09:00 version with title "Article A Duplicate"
    titles = set(df["title"])
    assert "Article B" in titles
    assert "Article A Duplicate" in titles


def test_rss_atom_aggregator_simulation_disabled_raises():
    """Verify error raised when feeds fail and simulation fallback is false."""
    recipe = RSSAtomAggregatorRecipe()
    config = {
        "feed_urls": "https://unreachable.host/feed.xml",
        "simulation_fallback": False,
    }
    with patch.object(recipe, "_fetch_url_bytes", side_effect=ConnectionError("Refused")):
        with pytest.raises(ExecutionException) as exc:
            recipe.execute(inputs={}, config=config)
        assert "Failed to retrieve entries from all provided feed URLs" in str(exc.value)


def test_rss_atom_aggregator_rdf_rss10_and_category_parsing():
    """Verify namespace-agnostic parsing of RDF/RSS 1.0 feeds and category extraction."""
    recipe = RSSAtomAggregatorRecipe()
    rdf_xml = """<?xml version="1.0"?>
    <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns="http://purl.org/rss/1.0/" xmlns:dc="http://purl.org/dc/elements/1.1/">
      <channel rdf:about="https://slashdot.org/">
        <title>Slashdot Headlines</title>
        <link>https://slashdot.org/</link>
      </channel>
      <item rdf:about="https://slashdot.org/story/1">
        <title>New AI Chip Unveiled</title>
        <link>https://slashdot.org/story/1</link>
        <description>Semiconductor startup introduces ultra-low power ASIC.</description>
        <dc:creator>Roblimo</dc:creator>
        <dc:date>2026-11-21T08:00:00Z</dc:date>
        <category>Hardware</category>
        <category>AI</category>
      </item>
    </rdf:RDF>
    """
    config = {
        "mock_feeds_xml": rdf_xml,
    }
    result = recipe.execute(inputs={}, config=config)
    df = result["dataframe"]
    metrics = result["metrics"]

    assert len(df) == 1
    assert metrics["title"] == "RSS / Atom Feed Ingestion Metrics"
    assert metrics["tag"] == "RSS"
    assert isinstance(metrics["unique_items"], int)
    assert df.iloc[0]["feed_title"] == "Slashdot Headlines"
    assert df.iloc[0]["title"] == "New AI Chip Unveiled"
    assert df.iloc[0]["author"] == "Roblimo"
    assert "Hardware" in df.iloc[0]["category"]


def test_rss_atom_aggregator_auth_and_bearer_token():
    """Verify bearer token and basic authentication parameter forwarding."""
    recipe = RSSAtomAggregatorRecipe()

    with patch("requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"""<rss version="2.0"><channel><title>Protected Feed</title><item><title>Alert 1</title><link>http://sec.internal/1</link></item></channel></rss>"""
        mock_get.return_value = mock_resp

        # 1. Bearer Token
        config_bearer = {
            "feed_urls": "https://sec.internal/rss",
            "auth_mode": "bearer_token",
            "bearer_token": "feed-secret-jwt-1234",
            "simulation_fallback": False,
        }
        res_bearer = recipe.execute(inputs={}, config=config_bearer)
        assert len(res_bearer["dataframe"]) == 1
        args, kwargs = mock_get.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer feed-secret-jwt-1234"

        # 2. Basic Auth
        mock_get.reset_mock()
        mock_get.return_value = mock_resp
        config_basic = {
            "feed_urls": "https://sec.internal/rss",
            "auth_mode": "basic_auth",
            "auth_username": "rss_admin",
            "auth_password": "rss_password",
            "simulation_fallback": False,
        }
        res_basic = recipe.execute(inputs={}, config=config_basic)
        assert len(res_basic["dataframe"]) == 1
        args, kwargs = mock_get.call_args
        assert kwargs["auth"] == ("rss_admin", "rss_password")


def test_rss_atom_aggregator_dataframe_input_url_chaining():
    """Verify reading feed URLs dynamically from an upstream DataFrame."""
    recipe = RSSAtomAggregatorRecipe()
    upstream_df = pd.DataFrame({
        "channel_name": ["BBC Tech"],
        "rss_url": ["https://feeds.bbci.co.uk/news/technology/rss.xml"]
    })

    with patch.object(recipe, "_fetch_url_bytes") as mock_fetch:
        mock_fetch.return_value = b"""<rss version="2.0"><channel><title>BBC Tech</title><item><title>Tech News</title><link>https://bbc.com/tech</link></item></channel></rss>"""
        result = recipe.execute(inputs={"dataframe": upstream_df}, config={"simulation_fallback": False})

    df = result["dataframe"]
    assert len(df) == 1
    assert df.iloc[0]["title"] == "Tech News"
    mock_fetch.assert_called_once_with(
        url="https://feeds.bbci.co.uk/news/technology/rss.xml",
        timeout=10,
        user_agent="Mozilla/5.0",
        auth_mode="none",
        auth_username="",
        auth_password="",
        bearer_token="",
        custom_headers="",
    )

