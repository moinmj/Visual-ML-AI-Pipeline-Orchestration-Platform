"""
Unit tests for RSS/News Feed Reader recipe (rss_news_reader).
"""
import pytest
import pandas as pd
from backend.app.recipes.base.registry import recipe_registry
from backend.app.recipes.integrations.rss_news_reader import (
    RssNewsReaderRecipe,
    parse_feed_xml,
    fetch_single_feed,
    FEED_PRESETS
)
from backend.app.workflows.schema_inference import WorkflowSchemaInferencer


SAMPLE_RSS_2_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/" xmlns:dc="http://purl.org/dc/elements/1.1/">
  <channel>
    <title>Tech News Feed</title>
    <link>https://technews.example.com</link>
    <description>Daily tech and engineering stories</description>
    <item>
      <title>Breakthrough in Edge AI Accelerators</title>
      <link>https://technews.example.com/edge-ai-breakthrough</link>
      <pubDate>Fri, 17 Oct 2026 08:30:00 GMT</pubDate>
      <description>&lt;p&gt;Silicon researchers unveiled sub-watt neural hardware.&lt;/p&gt;</description>
      <content:encoded>&lt;p&gt;Full technical analysis of the novel architecture.&lt;/p&gt;</content:encoded>
      <guid isPermaLink="false">guid-edge-ai-101</guid>
      <dc:creator>Dr. Elena Smith</dc:creator>
      <category>Hardware</category>
    </item>
    <item>
      <title>Next-Gen High Speed Distributed Cache Announced</title>
      <link>https://technews.example.com/distributed-cache</link>
      <pubDate>Thu, 16 Oct 2026 14:00:00 GMT</pubDate>
      <description>Distributed in-memory storage achieves 10x throughput.</description>
      <guid>guid-cache-102</guid>
      <category>Databases</category>
    </item>
  </channel>
</rss>
"""

SAMPLE_ATOM_XML = b"""<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Engineering Blog Feed</title>
  <link href="https://blog.example.com"/>
  <updated>2026-10-17T09:00:00Z</updated>
  <entry>
    <title>Scaling Graph Databases to Billions of Nodes</title>
    <link rel="alternate" href="https://blog.example.com/scaling-graph-db"/>
    <id>urn:uuid:60a76c80-d399-11d9-b93C-0004ac961c37</id>
    <published>2026-10-17T07:15:00Z</published>
    <summary>Architectural deep-dive on partitioning graph state.</summary>
    <author>
      <name>Alex Chen</name>
    </author>
    <category term="Architecture"/>
  </entry>
</feed>
"""


def test_rss_news_reader_registered_in_catalog():
    """Recipe must be loaded and registered in the global catalog."""
    recipe = recipe_registry.get("rss_news_reader")
    assert recipe is not None
    assert recipe.recipe_id == "rss_news_reader"
    assert recipe.name == "RSS/News Feed Reader"
    assert recipe.category == "integrations"
    assert recipe.group == "Apps & Integrations"
    assert recipe.subgroup == "Market & Reference Data"


def test_rss_news_reader_schema_and_preflight_guardrails():
    """Verify schema properties and frontend pre-flight guardrail safety."""
    recipe = RssNewsReaderRecipe()
    schema = recipe.get_schema()
    assert schema["type"] == "object"
    props = schema["properties"]

    assert "feed_preset" in props
    assert "feed_url" in props
    assert "max_items" in props
    assert "sort_order" in props
    assert "feed_url_field" in props

    # Preflight guardrail test: no key with substring "column", "metric", or "target" should have empty default
    for k, v in props.items():
        if any(sub in k for sub in ["column", "metric", "target"]):
            assert False, f"Key '{k}' contains guardrail substring! Rename to safely avoid client error."


def test_parse_rss_2_xml():
    """Test standard RSS 2.0 XML parsing with namespaces and clean text."""
    feed_title, items = parse_feed_xml(SAMPLE_RSS_2_XML, "https://technews.example.com/rss", max_items=10, extract_full=True)
    assert feed_title == "Tech News Feed"
    assert len(items) == 2

    item0 = items[0]
    assert item0["title"] == "Breakthrough in Edge AI Accelerators"
    assert item0["link"] == "https://technews.example.com/edge-ai-breakthrough"
    assert item0["guid"] == "guid-edge-ai-101"
    assert item0["author"] == "Dr. Elena Smith"
    assert item0["category"] == "Hardware"
    assert "Silicon researchers" in item0["description"]
    assert "Full technical analysis" in item0["content"]
    assert "2026-10-17" in item0["published_at"]


def test_parse_atom_xml():
    """Test Atom 1.0 XML parsing."""
    feed_title, items = parse_feed_xml(SAMPLE_ATOM_XML, "https://blog.example.com/feed.atom", max_items=10)
    assert feed_title == "Engineering Blog Feed"
    assert len(items) == 1

    entry = items[0]
    assert entry["title"] == "Scaling Graph Databases to Billions of Nodes"
    assert entry["link"] == "https://blog.example.com/scaling-graph-db"
    assert entry["author"] == "Alex Chen"
    assert entry["category"] == "Architecture"
    assert "Architectural deep-dive" in entry["description"]
    assert "2026-10-17" in entry["published_at"]


def test_rss_news_reader_standalone_execution(monkeypatch):
    """Test execute() standalone mode returning valid DataFrame and output summary."""
    import io
    import urllib.request

    class MockResp:
        status = 200
        def read(self):
            return SAMPLE_RSS_2_XML
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: MockResp())

    recipe = RssNewsReaderRecipe()
    config = {
        "feed_preset": "Hacker News (Tech)",
        "max_items": 10,
        "sort_order": "Newest First",
        "extract_full_content": True
    }

    result = recipe.execute(inputs={}, config=config)
    assert "dataframe" in result
    df = result["dataframe"]
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "title" in df.columns
    assert "link" in df.columns
    assert "published_at" in df.columns
    assert "description" in df.columns
    assert "guid" in df.columns
    assert "content" in df.columns

    out_sum = result["output_summary"]
    assert out_sum["article_count"] == 2
    assert out_sum["feed_title"] == "Tech News Feed"
    assert len(out_sum["preview_data"]) == 2


def test_rss_news_reader_batch_row_mode(monkeypatch):
    """Test batch mode when an incoming DataFrame with feed URLs is passed."""
    import urllib.request

    class MockResp:
        status = 200
        def read(self):
            return SAMPLE_ATOM_XML
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: MockResp())

    recipe = RssNewsReaderRecipe()
    incoming = pd.DataFrame({
        "topic": ["Tech", "World"],
        "rss_link": [
            "https://news.ycombinator.com/rss",
            "http://feeds.bbci.co.uk/news/world/rss.xml"
        ]
    })

    config = {
        "feed_preset": "Custom URL",
        "feed_url_field": "rss_link",
        "max_items": 10
    }

    result = recipe.execute(inputs={"dataframe": incoming}, config=config)
    df = result["dataframe"]
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "feed_url" in df.columns


def test_rss_news_reader_offline_fallback(monkeypatch):
    """Verify resilient offline benchmark fallback when feed URL fails."""
    import urllib.request

    def fail_urlopen(req, timeout=None):
        raise urllib.error.URLError("Simulated connection timeout")

    monkeypatch.setattr(urllib.request, "urlopen", fail_urlopen)

    recipe = RssNewsReaderRecipe()
    config = {
        "feed_preset": "Custom URL",
        "feed_url": "https://unreachable-domain-xyz.com/feed.xml",
        "max_items": 5
    }

    result = recipe.execute(inputs={}, config=config)
    df = result["dataframe"]
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    assert result["output_summary"]["status"] == "OFFLINE_BENCHMARK"


def test_rss_news_reader_to_code():
    """Test code generation string for standalone execution."""
    recipe = RssNewsReaderRecipe()
    code = recipe.to_code({"feed_preset": "BBC News - Technology", "feed_url": "http://feeds.bbci.co.uk/news/technology/rss.xml", "max_items": 15})
    assert "urllib.request" in code
    assert "pd.DataFrame" in code
    assert "http://feeds.bbci.co.uk/news/technology/rss.xml" in code


@pytest.mark.asyncio
async def test_rss_news_reader_schema_inferencer():
    """Test that WorkflowSchemaInferencer recognizes rss_news_reader columns."""
    nodes = [
        {"id": "rss_1", "recipe_id": "rss_news_reader", "config": {}},
        {"id": "filter_1", "recipe_id": "row_filter", "config": {}}
    ]
    edges = [
        {"source": "rss_1", "target": "filter_1", "sourceHandle": "output", "targetHandle": "input"}
    ]

    res = await WorkflowSchemaInferencer.infer_schema(nodes=nodes, edges=edges)
    assert res.success is True
    rss_schema = res.node_schemas["rss_1"]
    assert "title" in rss_schema.columns
    assert "link" in rss_schema.columns
    assert "published_at" in rss_schema.columns
    assert "description" in rss_schema.columns

    filter_schema = res.node_schemas["filter_1"]
    assert "title" in filter_schema.available_left_columns
    assert "description" in filter_schema.available_left_columns
