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
    credential_types = ["api_key", "basic_auth", "bearer_token"]
    input_types: List[str] = []
    output_types = ["dataframe"]

    inputs: List[RecipePort] = []

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
            "sections": [
                {
                    "id": "sources",
                    "title": "Feed Sources & Endpoints",
                    "description": "Configure RSS and Atom feed endpoint URLs and network timeouts",
                },
                {
                    "id": "auth",
                    "title": "Authentication & Headers (Optional)",
                    "description": "Configure credentials or custom headers for protected/private feeds",
                },
                {
                    "id": "deduplication",
                    "title": "Deduplication & Formatting",
                    "description": "Duplicate detection keys, resolution strategies, and HTML cleanup",
                },
                {
                    "id": "runtime_limits",
                    "title": "Limits & Simulation",
                    "description": "Article batch limits, offline simulation fallback, and mock XML injection",
                },
            ],
            "properties": {
                "feed_urls": {
                    "type": "string",
                    "title": "Feed URLs",
                    "default": "https://feeds.bbci.co.uk/news/technology/rss.xml\nhttps://techcrunch.com/feed/",
                    "input_type": "textarea",
                    "section": "sources",
                    "description": "Newline- or comma-separated list of RSS (2.0/1.0) or Atom feed URLs.",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "title": "Timeout (Seconds)",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 60,
                    "input_type": "number",
                    "section": "sources",
                    "description": "Maximum seconds to wait for each feed HTTP response.",
                },
                "user_agent": {
                    "type": "string",
                    "title": "HTTP User-Agent",
                    "default": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                    "input_type": "text",
                    "section": "sources",
                    "description": "Custom User-Agent header string sent with HTTP feed requests.",
                },
                "auth_mode": {
                    "type": "string",
                    "title": "Authentication Mode",
                    "enum": ["none", "basic_auth", "bearer_token", "custom_headers"],
                    "default": "none",
                    "input_type": "dropdown",
                    "section": "auth",
                    "description": "Optional authentication for internal or subscriber-only feed endpoints.",
                    "enum_titles": {
                        "none": "None (Public Feeds)",
                        "basic_auth": "HTTP Basic Authentication",
                        "bearer_token": "Bearer / Access Token",
                        "custom_headers": "Custom HTTP Headers",
                    },
                },
                "auth_username": {
                    "type": "string",
                    "title": "Username",
                    "default": "",
                    "input_type": "text",
                    "section": "auth",
                    "description": "Username for HTTP Basic Authentication.",
                    "visible_when": {
                        "field": "auth_mode",
                        "equals": "basic_auth",
                    },
                },
                "auth_password": {
                    "type": "string",
                    "title": "Password",
                    "default": "",
                    "input_type": "password",
                    "section": "auth",
                    "description": "Password for HTTP Basic Authentication.",
                    "visible_when": {
                        "field": "auth_mode",
                        "equals": "basic_auth",
                    },
                },
                "bearer_token": {
                    "type": "string",
                    "title": "Bearer Token",
                    "default": "",
                    "input_type": "password",
                    "section": "auth",
                    "placeholder": "ey...",
                    "description": "Secret access token passed in 'Authorization: Bearer <token>' header.",
                    "visible_when": {
                        "field": "auth_mode",
                        "equals": "bearer_token",
                    },
                },
                "custom_headers": {
                    "type": "string",
                    "title": "Custom Headers (JSON or Key: Value)",
                    "default": "",
                    "input_type": "textarea",
                    "section": "auth",
                    "placeholder": "{\n  \"X-Api-Key\": \"secret\"\n}",
                    "description": "Custom headers passed with requests to all configured feed URLs.",
                    "visible_when": {
                        "field": "auth_mode",
                        "in": ["bearer_token", "custom_headers"],
                    },
                },
                "deduplication_key": {
                    "type": "string",
                    "title": "Deduplication Key",
                    "enum": ["link_or_guid", "link", "guid", "title"],
                    "default": "link_or_guid",
                    "input_type": "dropdown",
                    "section": "deduplication",
                    "description": "Identifier used to identify duplicate articles across feeds.",
                },
                "dedup_strategy": {
                    "type": "string",
                    "title": "Duplicate Resolution",
                    "enum": ["keep_newest", "keep_first"],
                    "default": "keep_newest",
                    "input_type": "dropdown",
                    "section": "deduplication",
                    "description": "Which record to retain when multiple feeds publish the same article.",
                },
                "max_items_per_feed": {
                    "type": "integer",
                    "title": "Max Items Per Feed",
                    "default": 25,
                    "minimum": 1,
                    "maximum": 500,
                    "input_type": "number",
                    "section": "runtime_limits",
                    "description": "Limit of articles extracted from each individual feed stream.",
                },
                "max_total_items": {
                    "type": "integer",
                    "title": "Max Total Items",
                    "default": 100,
                    "minimum": 1,
                    "maximum": 2000,
                    "input_type": "number",
                    "section": "runtime_limits",
                    "description": "Maximum number of total deduplicated entries in the output dataset.",
                },
                "strip_html": {
                    "type": "boolean",
                    "title": "Clean HTML Markup",
                    "default": True,
                    "input_type": "checkbox",
                    "section": "deduplication",
                    "description": "Remove HTML tags and unescape entities from summary and content.",
                },
                "sort_by_date": {
                    "type": "boolean",
                    "title": "Sort by Published Date",
                    "default": True,
                    "input_type": "checkbox",
                    "section": "deduplication",
                    "description": "Sort final dataset in descending order by published timestamp.",
                },
                "simulation_fallback": {
                    "type": "boolean",
                    "title": "Simulation Fallback",
                    "default": True,
                    "input_type": "checkbox",
                    "section": "runtime_limits",
                    "description": "Return realistic synthetic tech news feed items if remote feeds are unreachable.",
                },
                "mock_feeds_xml": {
                    "type": "string",
                    "title": "Mock Feeds XML",
                    "default": "",
                    "input_type": "textarea",
                    "section": "runtime_limits",
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
            raw_bytes = xml_bytes.strip().encode("utf-8") if isinstance(xml_bytes, str) else xml_bytes.strip()
            root = ET.fromstring(raw_bytes)
        except Exception as e:
            logger.warning(f"Failed parsing XML with ElementTree for {feed_source}: {e}, trying BeautifulSoup fallback")
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(xml_bytes, "xml")
                return self._parse_with_bs4(soup, feed_source, max_items, strip_html)
            except Exception as bs_e:
                logger.warning(f"BeautifulSoup parser fallback also failed for {feed_source}: {bs_e}")
                return []

        # Find channel/feed title and locate item/entry elements namespace-agnostically
        channel_title = ""
        items_or_entries: List[ET.Element] = []

        for el in root.iter():
            tag = el.tag.split("}")[-1].lower()
            if tag in ("channel", "feed") and not channel_title:
                for ch in el:
                    if ch.tag.split("}")[-1].lower() == "title":
                        channel_title = self._clean_text(ch.text, strip_html=True)
                        break
            elif tag in ("item", "entry"):
                items_or_entries.append(el)

        for item in items_or_entries[:max_items]:
            title = ""
            link = ""
            guid = ""
            summary = ""
            author = ""
            pub_str = ""
            categories: List[str] = []

            for child in item:
                tag = child.tag.split("}")[-1].lower()
                if tag == "title":
                    title = self._clean_text(child.text, strip_html)
                elif tag == "link":
                    link = (child.attrib.get("href") or child.text or "").strip()
                elif tag in ("guid", "id"):
                    guid = (child.text or "").strip()
                elif tag in ("description", "summary", "content", "encoded"):
                    if not summary:
                        summary = self._clean_text(child.text, strip_html)
                elif tag in ("creator", "author"):
                    # Check nested <name> inside author (common in Atom)
                    has_subname = False
                    for sub in child:
                        if sub.tag.split("}")[-1].lower() == "name":
                            author = self._clean_text(sub.text, strip_html=True)
                            has_subname = True
                            break
                    if not has_subname:
                        author = self._clean_text(child.text, strip_html=True)
                elif tag in ("pubdate", "date", "updated", "published"):
                    if not pub_str:
                        pub_str = (child.text or "").strip()
                elif tag == "category":
                    cat_val = (child.attrib.get("term") or child.text or "").strip()
                    if cat_val:
                        categories.append(self._clean_text(cat_val, strip_html=True))

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
                "category": ", ".join(categories) if categories else "",
                "feed_source": feed_source,
            })

        return entries

    def _parse_with_bs4(self, soup: Any, feed_source: str, max_items: int, strip_html: bool) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        channel_title = ""
        title_tag = soup.find(["channel", "feed"])
        if title_tag:
            t = title_tag.find("title")
            if t and t.text:
                channel_title = self._clean_text(t.text, strip_html=True)

        items = soup.find_all(["item", "entry"])
        for item in items[:max_items]:
            t_el = item.find("title")
            title = self._clean_text(t_el.text if t_el else "", strip_html)

            l_el = item.find("link")
            link = ""
            if l_el:
                link = (l_el.get("href") or l_el.text or "").strip()

            g_el = item.find(["guid", "id"])
            guid = (g_el.text if g_el else "").strip()

            s_el = item.find(["description", "summary", "content", "encoded"])
            summary = self._clean_text(s_el.text if s_el else "", strip_html)

            a_el = item.find(["creator", "author"])
            author = self._clean_text(a_el.text if a_el else "", strip_html=True)

            d_el = item.find(["pubdate", "pubDate", "date", "updated", "published"])
            pub_str = (d_el.text if d_el else "").strip()
            iso_date, ts = self._parse_xml_date(pub_str)

            cats = [self._clean_text(c.get("term") or c.text or "", strip_html=True) for c in item.find_all("category")]
            clean_cats = [c for c in cats if c]

            entries.append({
                "feed_title": channel_title or feed_source,
                "title": title,
                "link": link or guid,
                "guid": guid or link,
                "author": author,
                "published_date": iso_date,
                "timestamp": ts,
                "summary": summary,
                "category": ", ".join(clean_cats) if clean_cats else "",
                "feed_source": feed_source,
            })

        return entries

    def _fetch_url_bytes(
        self,
        url: str,
        timeout: int = 10,
        user_agent: str = "",
        auth_mode: str = "none",
        auth_username: str = "",
        auth_password: str = "",
        bearer_token: str = "",
        custom_headers: str = "",
    ) -> bytes:
        default_ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ua = user_agent.strip() if user_agent and user_agent.strip() else default_ua
        headers = {
            "User-Agent": ua,
            "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml, */*",
            "Accept-Language": "en-US,en;q=0.9",
        }

        # Custom headers
        if custom_headers and custom_headers.strip():
            raw_h = custom_headers.strip()
            if raw_h.startswith("{") and raw_h.endswith("}"):
                try:
                    parsed_h = json.loads(raw_h)
                    if isinstance(parsed_h, dict):
                        headers.update({str(k): str(v) for k, v in parsed_h.items()})
                except Exception as je:
                    logger.warning(f"Failed to parse custom_headers as JSON: {je}")
            else:
                for line in raw_h.splitlines():
                    if ":" in line:
                        hk, hv = line.split(":", 1)
                        headers[hk.strip()] = hv.strip()

        # Bearer token
        if auth_mode == "bearer_token" and bearer_token:
            tok = bearer_token.strip()
            headers["Authorization"] = tok if tok.lower().startswith("bearer ") else f"Bearer {tok}"

        # 1. Try requests first
        try:
            import requests
            auth = (auth_username, auth_password) if auth_mode == "basic_auth" and (auth_username or auth_password) else None
            resp = requests.get(url, headers=headers, auth=auth, timeout=timeout, allow_redirects=True)
            resp.raise_for_status()
            return resp.content
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"requests.get failed for '{url}': {e}, attempting urllib fallback")

        # 2. Urllib fallback
        req = urllib.request.Request(url, headers=headers)
        if auth_mode == "basic_auth" and (auth_username or auth_password):
            import base64
            token = base64.b64encode(f"{auth_username}:{auth_password}".encode()).decode("ascii")
            req.add_header("Authorization", f"Basic {token}")

        with urllib.request.urlopen(req, timeout=timeout) as resp:
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
                "category": "Artificial Intelligence",
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
                "category": "MLOps & Security",
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
                "category": "Artificial Intelligence",
                "feed_source": "https://syndicate.internal/rss.xml",
            },
        ]

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        t0 = time.time()
        raw_feed_urls = str(config.get("feed_urls", "")).strip()
        timeout = max(1, min(int(config.get("timeout_seconds", 10)), 60))
        user_agent = str(config.get("user_agent", "Mozilla/5.0"))
        auth_mode = str(config.get("auth_mode", "none")).strip().lower()
        auth_username = str(config.get("auth_username", "")).strip()
        auth_password = str(config.get("auth_password", "")).strip()
        bearer_token = str(config.get("bearer_token", "")).strip()
        custom_headers = str(config.get("custom_headers", "")).strip()

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
                if any(k in col.lower() for k in ("url", "feed", "rss", "atom", "link")):
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
                    xml_data = self._fetch_url_bytes(
                        url=url,
                        timeout=timeout,
                        user_agent=user_agent,
                        auth_mode=auth_mode,
                        auth_username=auth_username,
                        auth_password=auth_password,
                        bearer_token=bearer_token,
                        custom_headers=custom_headers,
                    )
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
            out_df = pd.DataFrame(columns=["feed_title", "title", "link", "guid", "author", "published_date", "summary", "category", "feed_source"])

        # Guarantee column names are strings
        out_df.columns = [str(c) for c in out_df.columns]

        elapsed = round((time.time() - t0) * 1000, 2)
        duplicates_removed = raw_count - len(final_items)

        summary = {
            "total_articles": int(len(out_df)),
            "unique_feeds": int(out_df["feed_title"].nunique() if not out_df.empty else 0),
            "duplicates_filtered": int(max(0, duplicates_removed)),
            "is_simulation": bool(is_simulation),
        }

        metrics = {
            "title": "RSS / Atom Feed Ingestion Metrics",
            "tag": "RSS",
            "feeds_queried": int(len(urls) if urls else 1),
            "raw_items_fetched": int(raw_count),
            "unique_items": int(len(out_df)),
            "duplicates_removed": int(max(0, duplicates_removed)),
            "is_simulation": bool(is_simulation),
            "deduplication_key": str(dedup_key),
            "execution_time_ms": float(elapsed),
        }

        return {
            "dataframe": out_df,
            "output": out_df,
            "output_summary": summary,
            "metrics": metrics,
        }
