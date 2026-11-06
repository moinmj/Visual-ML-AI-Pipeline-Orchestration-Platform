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
