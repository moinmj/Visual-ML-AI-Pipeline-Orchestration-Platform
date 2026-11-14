"""
Web Scraper & HTML Table Extractor Recipe (Ingestion & Orchestration -> File & Web Ingestion).

Extracts structured data from web pages, including HTML tables, CSS selector target lists,
hyperlinks, and text blocks.
Features realistic HTTP request simulation, custom headers/User-Agent, resilient fallback parsing,
mock HTML string injection for unit testing, and simulation mode when external networks are offline.
"""

import io
import json
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from backend.app.core.exceptions import ExecutionException, ValidationException
from backend.app.core.logging import logger
from backend.app.recipes.base.recipe import BaseRecipe, RecipePort


class WebScraperRecipe(BaseRecipe):
    recipe_id = "web_scraper"
    name = "Web Scraper / HTML Table Extractor"
    version = "1.0.0"
    category = "ingestion"
    group = "Ingestion & Orchestration"
    subgroup = "File & Web Ingestion"
    description = "Extracts HTML tables or page content from a web page into a DataFrame."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    inputs: List[RecipePort] = []

    outputs = [
        RecipePort(
            id="output",
            label="Output Table",
            type="dataframe",
            description="Extracted table or page elements dataset",
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "sections": [
                {
                    "id": "target",
                    "title": "Target Webpage & HTTP",
                    "description": "Configure target webpage URL, custom User-Agent, and network timeouts",
                },
                {
                    "id": "extraction",
                    "title": "Extraction Mode & Selectors",
                    "description": "HTML table extraction, CSS selectors, links, or text block options",
                },
                {
                    "id": "runtime_limits",
                    "title": "Metadata & Simulation",
                    "description": "Provenance metadata, simulation fallbacks, and mock HTML testing",
                },
            ],
            "properties": {
                "url": {
                    "type": "string",
                    "title": "Target Webpage URL",
                    "default": "https://en.wikipedia.org/wiki/List_of_countries_by_GDP_(nominal)",
                    "input_type": "text",
                    "section": "target",
                    "description": "Full HTTP or HTTPS web URL to scrape data from.",
                },
                "extraction_mode": {
                    "type": "string",
                    "title": "Extraction Mode",
                    "enum": ["html_tables", "css_selector_list", "page_links", "raw_text_blocks"],
                    "default": "html_tables",
                    "input_type": "dropdown",
                    "section": "extraction",
                    "description": "Method of data extraction: HTML tables, CSS selector matches, page links, or text blocks.",
                },
                "table_index": {
                    "type": "integer",
                    "title": "Table Index (0-based)",
                    "default": 0,
                    "minimum": 0,
                    "maximum": 50,
                    "input_type": "number",
                    "section": "extraction",
                    "description": "Zero-based index of the table to extract when extraction_mode is 'html_tables'.",
                    "visible_when": {
                        "field": "extraction_mode",
                        "equals": "html_tables",
                    },
                },
                "css_selector": {
                    "type": "string",
                    "title": "CSS Selector",
                    "default": "",
                    "input_type": "text",
                    "section": "extraction",
                    "description": "Target CSS selector when extraction_mode is 'css_selector_list' (e.g. '.product-card', 'article').",
                    "visible_when": {
                        "field": "extraction_mode",
                        "equals": "css_selector_list",
                    },
                },
                "user_agent": {
                    "type": "string",
                    "title": "HTTP User-Agent",
                    "default": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "input_type": "text",
                    "section": "target",
                    "description": "Custom User-Agent header string sent with HTTP request.",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "title": "Timeout (Seconds)",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 60,
                    "input_type": "number",
                    "section": "target",
                    "description": "Maximum seconds to wait for network HTTP response.",
                },
                "include_metadata_columns": {
                    "type": "boolean",
                    "title": "Include Metadata Columns",
                    "default": True,
                    "input_type": "checkbox",
                    "section": "runtime_limits",
                    "description": "When true, appends '_source_url' and '_extracted_at' columns to parsed rows.",
                },
                "simulation_fallback": {
                    "type": "boolean",
                    "title": "Simulation Fallback",
                    "default": True,
                    "input_type": "checkbox",
                    "section": "runtime_limits",
                    "description": "Generate synthetic dataset if webpage is offline or inaccessible.",
                },
                "mock_html": {
                    "type": "string",
                    "title": "Mock HTML Content",
                    "default": "",
                    "input_type": "textarea",
                    "section": "runtime_limits",
                    "description": "Optional raw HTML string to scrape directly without making network calls.",
                },
            },
            "required": ["url"],
        }

    def _fetch_html(self, url: str, user_agent: str, timeout: int) -> str:
        parsed = urllib.parse.urlparse(url)
        if not parsed.scheme or parsed.scheme not in ("http", "https"):
            raise ValidationException(f"Invalid URL '{url}'. Must start with 'http://' or 'https://'.")

        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            encoding = response.headers.get_content_charset() or "utf-8"
            return response.read().decode(encoding, errors="replace")

    def _extract_html_tables(self, html: str, table_index: int) -> pd.DataFrame:
        try:
            dfs = pd.read_html(io.StringIO(html))
            if not dfs:
                raise ExecutionException("No <table> elements found in the HTML content.")
            idx = min(table_index, len(dfs) - 1)
            selected_df = dfs[idx]
            # Flatten multi-level columns if present
            if isinstance(selected_df.columns, pd.MultiIndex):
                selected_df.columns = ["_".join(str(c) for c in col).strip("_") for col in selected_df.columns]
            return selected_df
        except Exception as e:
            # Fallback simple table regex extractor if read_html fails
            return self._regex_parse_tables(html, table_index, e)

    def _regex_parse_tables(self, html: str, table_index: int, original_error: Exception) -> pd.DataFrame:
        table_matches = re.findall(r"<table[^>]*>(.*?)</table>", html, re.DOTALL | re.IGNORECASE)
        if not table_matches:
            raise ExecutionException(f"Failed to parse HTML tables: {original_error}")
        idx = min(table_index, len(table_matches) - 1)
        raw_table = table_matches[idx]

        # Extract rows
        rows: List[List[str]] = []
        for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", raw_table, re.DOTALL | re.IGNORECASE):
            cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", tr, re.DOTALL | re.IGNORECASE)
            clean_cells = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
            if clean_cells:
                rows.append(clean_cells)

        if not rows:
            raise ExecutionException("Found table tag but no rows could be parsed.")

        header = rows[0]
        data = rows[1:] if len(rows) > 1 else []
        return pd.DataFrame(data, columns=header)

    def _extract_css_selector_list(self, html: str, selector: str) -> pd.DataFrame:
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            elements = soup.select(selector) if selector else soup.find_all(["div", "article", "section"])
            rows: List[Dict[str, Any]] = []
            for idx, el in enumerate(elements):
                rows.append({
                    "element_index": idx,
                    "tag_name": el.name,
                    "text": el.get_text(separator=" ", strip=True),
                    "class": " ".join(el.get("class", [])),
                    "id": el.get("id", ""),
                })
            return pd.DataFrame(rows)
        except ImportError:
            # Fallback without bs4
            pattern = rf'class=["\'][^"\']*{re.escape(selector.lstrip("."))}[^"\']*["\'][^>]*>(.*?)<' if selector.startswith(".") else r"<p[^>]*>(.*?)</p>"
            matches = re.findall(pattern, html, re.DOTALL | re.IGNORECASE)
            return pd.DataFrame([{"element_index": i, "text": re.sub(r"<[^>]+>", "", m).strip()} for i, m in enumerate(matches)])

    def _extract_page_links(self, html: str, base_url: str) -> pd.DataFrame:
        link_matches = re.findall(r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', html, re.DOTALL | re.IGNORECASE)
        rows: List[Dict[str, Any]] = []
        for href, text in link_matches:
            clean_text = re.sub(r"<[^>]+>", "", text).strip()
            absolute_url = urllib.parse.urljoin(base_url, href)
            rows.append({
                "link_text": clean_text,
                "url": absolute_url,
                "raw_href": href,
                "is_external": not href.startswith("/") and not href.startswith("#") and not urllib.parse.urlparse(base_url).netloc in absolute_url,
            })
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["link_text", "url", "raw_href", "is_external"])

    def _extract_raw_text_blocks(self, html: str) -> pd.DataFrame:
        tag_pattern = r"<(p|h1|h2|h3|h4|li)[^>]*>(.*?)</\1>"
        matches = re.findall(tag_pattern, html, re.DOTALL | re.IGNORECASE)
        rows: List[Dict[str, Any]] = []
        for idx, (tag, content) in enumerate(matches):
            clean_text = re.sub(r"<[^>]+>", "", content).strip()
            if clean_text:
                rows.append({
                    "block_id": idx + 1,
                    "tag": tag.lower(),
                    "word_count": len(clean_text.split()),
                    "content": clean_text,
                })
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["block_id", "tag", "word_count", "content"])

    def _generate_synthetic_table(self, extraction_mode: str) -> pd.DataFrame:
        if extraction_mode == "page_links":
            return pd.DataFrame([
                {"link_text": "Documentation Home", "url": "https://example.com/docs", "raw_href": "/docs", "is_external": False},
                {"link_text": "API Reference", "url": "https://example.com/api", "raw_href": "/api", "is_external": False},
                {"link_text": "GitHub Repository", "url": "https://github.com/example/repo", "raw_href": "https://github.com", "is_external": True},
            ])
        elif extraction_mode == "raw_text_blocks":
            return pd.DataFrame([
                {"block_id": 1, "tag": "h1", "word_count": 5, "content": "Visual ML/AI Pipeline Architecture Overview"},
                {"block_id": 2, "tag": "p", "word_count": 14, "content": "This document outlines the distributed data ingestion and real-time inference workflow."},
                {"block_id": 3, "tag": "li", "word_count": 4, "content": "Zero downtime schema migrations"},
            ])
        elif extraction_mode == "css_selector_list":
            return pd.DataFrame([
                {"element_index": 0, "tag_name": "div", "text": "Product Alpha - $199.00 - In Stock", "class": "product-card", "id": "p-1"},
                {"element_index": 1, "tag_name": "div", "text": "Product Beta - $299.00 - Low Stock", "class": "product-card", "id": "p-2"},
            ])
        else:
            return pd.DataFrame([
                {"rank": 1, "country": "United States", "gdp_nominal_trillion": 26.9, "share_global_pct": 25.4},
                {"rank": 2, "country": "China", "gdp_nominal_trillion": 17.7, "share_global_pct": 16.7},
                {"rank": 3, "country": "Germany", "gdp_nominal_trillion": 4.4, "share_global_pct": 4.2},
                {"rank": 4, "country": "Japan", "gdp_nominal_trillion": 4.2, "share_global_pct": 4.0},
                {"rank": 5, "country": "India", "gdp_nominal_trillion": 3.7, "share_global_pct": 3.5},
            ])

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.time()
        url = str(config.get("url", "")).strip()
        if not url:
            raise ValidationException("Parameter 'url' cannot be empty.")

        extraction_mode = str(config.get("extraction_mode", "html_tables")).strip().lower()
        valid_modes = ("html_tables", "css_selector_list", "page_links", "raw_text_blocks")
        if extraction_mode not in valid_modes:
            raise ValidationException(f"Unsupported extraction_mode '{extraction_mode}'. Must be one of {valid_modes}.")

        table_index = max(0, int(config.get("table_index", 0)))
        css_selector = str(config.get("css_selector", "")).strip()
        user_agent = str(config.get("user_agent", "Mozilla/5.0"))
        timeout = max(1, min(int(config.get("timeout_seconds", 10)), 60))
        include_meta = bool(config.get("include_metadata_columns", True))
        simulation_fallback = bool(config.get("simulation_fallback", True))
        mock_html = str(config.get("mock_html", "")).strip()

        html_content = ""
        is_simulation = False

        # 1. Use mock HTML if provided
        if mock_html:
            html_content = mock_html
        else:
            # 2. Try fetching from network URL
            try:
                html_content = self._fetch_html(url=url, user_agent=user_agent, timeout=timeout)
            except Exception as e:
                logger.warning(f"Failed to scrape webpage '{url}': {e}")
                if not simulation_fallback:
                    raise ExecutionException(f"Failed to fetch content from URL '{url}': {e}")
                is_simulation = True

        # Extract data
        if is_simulation:
            df = self._generate_synthetic_table(extraction_mode)
        else:
            try:
                if extraction_mode == "html_tables":
                    df = self._extract_html_tables(html_content, table_index)
                elif extraction_mode == "css_selector_list":
                    df = self._extract_css_selector_list(html_content, css_selector)
                elif extraction_mode == "page_links":
                    df = self._extract_page_links(html_content, url)
                elif extraction_mode == "raw_text_blocks":
                    df = self._extract_raw_text_blocks(html_content)
                else:
                    df = self._extract_html_tables(html_content, table_index)
            except Exception as e:
                logger.warning(f"Error parsing HTML: {e}")
                if not simulation_fallback:
                    raise ExecutionException(f"Failed to extract '{extraction_mode}' from webpage: {e}")
                df = self._generate_synthetic_table(extraction_mode)
                is_simulation = True

        if include_meta and not df.empty:
            now_iso = datetime.now(timezone.utc).isoformat()
            df["_source_url"] = url
            df["_extracted_at"] = now_iso

        elapsed = round((time.time() - t0) * 1000, 2)
        metrics = {
            "execution_time_ms": elapsed,
            "url": url,
            "extraction_mode": extraction_mode,
            "rows_extracted": len(df),
            "columns_count": len(df.columns),
            "is_simulation": is_simulation,
        }

        return {"dataframe": df, "metrics": metrics}
