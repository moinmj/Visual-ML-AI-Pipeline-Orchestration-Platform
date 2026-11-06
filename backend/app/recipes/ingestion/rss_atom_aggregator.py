"""
RSS/Atom Aggregator with Deduplication Recipe (Ingestion & Orchestration -> File & Web Ingestion).

Aggregates news, blog posts, and alert feeds across multiple RSS and Atom channels,
extracting standard story metadata (title, link, published_date, summary, author, guid, category),
cleaning HTML markup, and deduplicating cross-syndicated articles using canonical links or GUIDs.
Features multi-feed aggregation, simulation fallback, and offline XML injection for unit testing.
"""

import email.utils
import html
import io
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from backend.app.core.exceptions import ExecutionException, ValidationException
from backend.app.core.logging import logger
from backend.app.recipes.base.recipe import BaseRecipe, RecipePort


class RSSAtomAggregatorRecipe(BaseRecipe):
    recipe_id = "rss_atom_aggregator"
    name = "RSS/Atom Aggregator (with dedup)"
    version = "1.0.0"
    category = "ingestion"
    group = "Ingestion & Orchestration"
    subgroup = "File & Web Ingestion"
    description = "Aggregates items from multiple RSS/Atom feeds and removes duplicates into a DataFrame."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    inputs = [
        RecipePort(
            id="input",
            label="Trigger / Feed URLs",
            type="dataframe",
            required=False,
            max_connections=1,
            description="Optional upstream dataset containing a 'feed_url' column",
        )
    ]

    outputs = [
        RecipePort(
            id="output",
            label="Output Table",
            type="dataframe",
            description="Aggregated and deduplicated feed items dataset",
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "feed_urls": {
                    "type": "string",
                    "title": "Feed URLs",
                    "default": "https://feeds.bbci.co.uk/news/technology/rss.xml\nhttps://techcrunch.com/feed/",
                    "input_type": "textarea",
                    "description": "Newline- or comma-separated list of RSS (2.0/1.0) or Atom feed URLs.",
                },
                "deduplication_key": {
                    "type": "string",
                    "title": "Deduplication Key",
                    "enum": ["link_or_guid", "link", "guid", "title"],
                    "default": "link_or_guid",
                    "input_type": "select",
                    "description": "Identifier used to identify duplicate articles across feeds.",
                },
                "dedup_strategy": {
                    "type": "string",
                    "title": "Duplicate Resolution",
                    "enum": ["keep_newest", "keep_first"],
                    "default": "keep_newest",
                    "input_type": "select",
                    "description": "Which record to retain when multiple feeds publish the same article.",
                },
                "max_items_per_feed": {
                    "type": "integer",
                    "title": "Max Items Per Feed",
                    "default": 25,
                    "minimum": 1,
                    "maximum": 500,
                    "input_type": "number",
                    "description": "Limit of articles extracted from each individual feed stream.",
                },
                "max_total_items": {
                    "type": "integer",
                    "title": "Max Total Items",
                    "default": 100,
                    "minimum": 1,
                    "maximum": 2000,
                    "input_type": "number",
                    "description": "Maximum number of total deduplicated entries in the output dataset.",
                },
                "strip_html": {
                    "type": "boolean",
                    "title": "Clean HTML Markup",
                    "default": True,
                    "input_type": "checkbox",
                    "description": "Remove HTML tags and unescape entities from summary and content.",
                },
                "sort_by_date": {
                    "type": "boolean",
                    "title": "Sort by Published Date",
                    "default": True,
                    "input_type": "checkbox",
                    "description": "Sort final dataset in descending order by published timestamp.",
                },
                "simulation_fallback": {
                    "type": "boolean",
                    "title": "Simulation Fallback",
                    "default": True,
                    "input_type": "checkbox",
                    "description": "Return realistic synthetic tech news feed items if remote feeds are unreachable.",
                },
                "mock_feeds_xml": {
                    "type": "string",
                    "title": "Mock Feeds XML",
                    "default": "",
                    "input_type": "textarea",
                    "description": "Optional raw RSS or Atom XML content for offline testing without network calls.",
                },
            },
            "required": ["feed_urls"],
        }

    def _clean_text(self, text: Optional[str], strip_html: bool) -> str:
        if not text:
            return ""
        val = str(text)
        if strip_html:
            val = re.sub(r"<[^>]+>", " ", val)
            val = html.unescape(val)
        return " ".join(val.split())

    def _parse_xml_date(self, date_str: Optional[str]) -> Tuple[str, float]:
        if not date_str:
            now = datetime.now(timezone.utc)
            return now.isoformat(), now.timestamp()
        clean = date_str.strip()
        # 1. Try RFC 2822 (common in RSS 2.0: 'Wed, 02 Oct 2026 13:00:00 GMT')
        try:
            parsed_tuple = email.utils.parsedate_to_datetime(clean)
            if parsed_tuple:
                utc_dt = parsed_tuple.astimezone(timezone.utc)
                return utc_dt.isoformat(), utc_dt.timestamp()
        except Exception:
            pass

        # 2. Try ISO 8601 (common in Atom: '2026-10-02T13:00:00Z')
        try:
            clean_iso = clean.replace("Z", "+00:00")
            dt = datetime.fromisoformat(clean_iso)
            if not dt.tzinfo:
                dt = dt.replace(tzinfo=timezone.utc)
            utc_dt = dt.astimezone(timezone.utc)
            return utc_dt.isoformat(), utc_dt.timestamp()
        except Exception:
            pass

        now = datetime.now(timezone.utc)
        return clean, now.timestamp()

    def _parse_single_feed_xml(
        self,
        xml_bytes: bytes,
        feed_source: str,
        max_items: int,
        strip_html: bool,
    ) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        try:
            root = ET.fromstring(xml_bytes)
        except Exception as e:
            logger.warning(f"Failed parsing XML for {feed_source}: {e}")
            return []

        # Remove namespaces for straightforward tag queries
        # Tag root could be 'rss', 'feed', '{http://www.w3.org/2005/Atom}feed'
        clean_tag = root.tag.split("}")[-1].lower()

        # Branch 1: Atom Feed
        if clean_tag == "feed":
            channel_title = ""
            for child in root:
                tag = child.tag.split("}")[-1].lower()
                if tag == "title":
                    channel_title = self._clean_text(child.text, strip_html=True)
                    break

            for child in root:
                if len(entries) >= max_items:
                    break
                tag = child.tag.split("}")[-1].lower()
                if tag == "entry":
                    title = ""
                    link = ""
                    summary = ""
                    guid = ""
                    author = ""
                    pub_str = ""

                    for sub in child:
                        sub_tag = sub.tag.split("}")[-1].lower()
                        if sub_tag == "title":
                            title = self._clean_text(sub.text, strip_html)
                        elif sub_tag == "link":
                            link = sub.attrib.get("href", sub.text or "")
                        elif sub_tag in ("summary", "content"):
                            if not summary:
                                summary = self._clean_text(sub.text, strip_html)
                        elif sub_tag == "id":
                            guid = (sub.text or "").strip()
                        elif sub_tag == "author":
                            for a_sub in sub:
                                if a_sub.tag.split("}")[-1].lower() == "name":
                                    author = self._clean_text(a_sub.text, strip_html=True)
                        elif sub_tag in ("updated", "published"):
                            if not pub_str:
                                pub_str = (sub.text or "").strip()

                    iso_date, ts = self._parse_xml_date(pub_str)
                    entries.append({
                        "feed_title": channel_title or feed_source,
                        "title": title,
                        "link": link or guid,
                        "guid": guid or link,
                        "author": author,
                        "published_date": iso_date,
                        "timestamp": ts,
                        "summary": summary,
                        "feed_source": feed_source,
                    })

        # Branch 2: RSS 2.0 / 0.91 / 1.0 Feed
        else:
            channel = root.find("channel")
            channel_title = ""
            if channel is not None:
                title_el = channel.find("title")
                if title_el is not None and title_el.text:
                    channel_title = self._clean_text(title_el.text, strip_html=True)

                items = channel.findall("item")
                for item in items:
                    if len(entries) >= max_items:
                        break
                    title = ""
                    link = ""
                    guid = ""
                    summary = ""
                    author = ""
                    pub_str = ""

                    for child in item:
                        tag = child.tag.split("}")[-1].lower()
                        if tag == "title":
                            title = self._clean_text(child.text, strip_html)
                        elif tag == "link":
                            link = (child.text or "").strip()
                        elif tag == "guid":
                            guid = (child.text or "").strip()
                        elif tag in ("description", "encoded"):
                            if not summary:
                                summary = self._clean_text(child.text, strip_html)
                        elif tag in ("creator", "author"):
                            author = self._clean_text(child.text, strip_html=True)
                        elif tag in ("pubdate", "date"):
                            pub_str = (child.text or "").strip()

                    iso_date, ts = self._parse_xml_date(pub_str)
                    entries.append({
                        "feed_title": channel_title or feed_source,
                        "title": title,
                        "link": link or guid,
                        "guid": guid or link,
                        "author": author,
                        "published_date": iso_date,
                        "timestamp": ts,
                        "summary": summary,
                        "feed_source": feed_source,
                    })

        return entries

    def _fetch_url_bytes(self, url: str) -> bytes:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) FeedAggregator/1.0"},
        )
        with urllib.request.urlopen(req, timeout=8.0) as resp:
            return resp.read()

    def _generate_synthetic_feed(self) -> List[Dict[str, Any]]:
        now = datetime.now(timezone.utc)
        return [
            {
                "feed_title": "Global AI & Data Science News",
                "title": "Breakthrough in Efficient Transformer Distillation",
                "link": "https://techpulse.internal/news/transformer-distillation-2026",
                "guid": "guid-techpulse-101",
                "author": "Dr. Elena Rostova",
                "published_date": now.isoformat(),
                "timestamp": now.timestamp(),
                "summary": "Researchers demonstrate an 80% reduction in inference latency with zero accuracy degradation.",
                "feed_source": "https://techpulse.internal/rss.xml",
            },
            {
                "feed_title": "Enterprise ML Monitor",
                "title": "Automated Pipeline Governance Adopts Zero-Trust Standards",
                "link": "https://mlgovernance.org/articles/zero-trust-mlops",
                "guid": "guid-mlgov-204",
                "author": "Marcus Vance",
                "published_date": now.isoformat(),
                "timestamp": now.timestamp() - 3600,
                "summary": "New regulatory guidelines mandate cryptographic data lineage audits across training clusters.",
                "feed_source": "https://mlgovernance.org/feed.atom",
            },
            {
                "feed_title": "Global AI & Data Science News",
                "title": "Breakthrough in Efficient Transformer Distillation",  # Duplicate syndicated item
                "link": "https://techpulse.internal/news/transformer-distillation-2026",
                "guid": "guid-techpulse-101",
                "author": "Dr. Elena Rostova",
                "published_date": now.isoformat(),
                "timestamp": now.timestamp() - 60,
                "summary": "Syndicated mirror of transformer distillation article.",
                "feed_source": "https://syndicate.internal/rss.xml",
            },
        ]

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.time()
        raw_feed_urls = str(config.get("feed_urls", "")).strip()
        dedup_key = str(config.get("deduplication_key", "link_or_guid")).strip().lower()
        dedup_strat = str(config.get("dedup_strategy", "keep_newest")).strip().lower()
        max_per_feed = max(1, min(int(config.get("max_items_per_feed", 25)), 500))
        max_total = max(1, min(int(config.get("max_total_items", 100)), 2000))
        strip_html = bool(config.get("strip_html", True))
        sort_by_date = bool(config.get("sort_by_date", True))
        simulation_fallback = bool(config.get("simulation_fallback", True))
        mock_xml = str(config.get("mock_feeds_xml", "")).strip()

        # Parse URLs from input dataframe or config
        urls: List[str] = []
        in_df = inputs.get("dataframe") if inputs else None
        if isinstance(in_df, pd.DataFrame) and not in_df.empty:
            for col in in_df.columns:
                if "url" in col.lower() or "feed" in col.lower():
                    urls.extend(in_df[col].dropna().astype(str).tolist())
                    break

        if not urls and raw_feed_urls:
            # Split by newline or comma
            split_urls = [u.strip() for u in re.split(r"[\n,;]", raw_feed_urls) if u.strip()]
            urls.extend(split_urls)

        if not urls and not mock_xml:
            raise ValidationException("At least one feed URL must be provided in 'feed_urls' or input DataFrame.")

        all_entries: List[Dict[str, Any]] = []
        is_simulation = False

        # 1. Mock XML provided
        if mock_xml:
            entries = self._parse_single_feed_xml(
                xml_bytes=mock_xml.encode("utf-8"),
                feed_source="mock://inline.xml",
                max_items=max_per_feed,
                strip_html=strip_html,
            )
            all_entries.extend(entries)

        # 2. Fetch URLs
        else:
            for url in urls:
                try:
                    xml_data = self._fetch_url_bytes(url)
                    feed_items = self._parse_single_feed_xml(
                        xml_bytes=xml_data,
                        feed_source=url,
                        max_items=max_per_feed,
                        strip_html=strip_html,
                    )
                    all_entries.extend(feed_items)
                except Exception as e:
                    logger.warning(f"Failed to fetch feed from {url}: {e}")

            if not all_entries:
                if not simulation_fallback:
                    raise ExecutionException(f"Failed to retrieve entries from all provided feed URLs: {urls}")
                all_entries = self._generate_synthetic_feed()
                is_simulation = True

        raw_count = len(all_entries)

        # 3. Deduplication
        deduped: List[Dict[str, Any]] = []
        seen_keys: Dict[str, Dict[str, Any]] = {}

        for item in all_entries:
            key = ""
            if dedup_key == "link":
                key = item.get("link", "").strip().lower()
            elif dedup_key == "guid":
                key = item.get("guid", "").strip()
            elif dedup_key == "title":
                key = item.get("title", "").strip().lower()
            else:  # link_or_guid
                key = (item.get("link") or item.get("guid") or item.get("title") or "").strip().lower()

            if not key:
                deduped.append(item)
                continue

            if key not in seen_keys:
                seen_keys[key] = item
            else:
                existing = seen_keys[key]
                if dedup_strat == "keep_newest":
                    if item.get("timestamp", 0) > existing.get("timestamp", 0):
                        seen_keys[key] = item
                # if keep_first, keep existing

        deduped.extend(seen_keys.values())

        # 4. Sort and limit
        if sort_by_date:
            deduped.sort(key=lambda x: x.get("timestamp", 0), reverse=True)

        final_items = deduped[:max_total]
        out_df = pd.DataFrame(final_items)

        if not out_df.empty and "timestamp" in out_df.columns:
            out_df = out_df.drop(columns=["timestamp"])

        if out_df.empty:
            out_df = pd.DataFrame(columns=["feed_title", "title", "link", "guid", "author", "published_date", "summary", "feed_source"])

        elapsed = round((time.time() - t0) * 1000, 2)
        duplicates_removed = raw_count - len(final_items)

        metrics = {
            "execution_time_ms": elapsed,
            "feeds_queried": len(urls) if urls else 1,
            "raw_items_fetched": raw_count,
            "unique_items": len(out_df),
            "duplicates_removed": max(0, duplicates_removed),
            "is_simulation": is_simulation,
            "deduplication_key": dedup_key,
        }

        return {"dataframe": out_df, "metrics": metrics}
