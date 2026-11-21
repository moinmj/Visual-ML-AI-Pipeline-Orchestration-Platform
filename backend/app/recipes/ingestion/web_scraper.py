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
    input_types: List[str] = []
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
                    "id": "auth",
                    "title": "Authentication & Session",
                    "description": "Configure session cookies, Bearer tokens, HTTP Basic Auth, or automated form login & logout",
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
                "user_agent": {
                    "type": "string",
                    "title": "HTTP User-Agent",
                    "default": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
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
                "auth_mode": {
                    "type": "string",
                    "title": "Authentication Mode",
                    "enum": ["none", "session_cookie", "bearer_token", "basic_auth", "form_login"],
                    "default": "none",
                    "input_type": "dropdown",
                    "section": "auth",
                    "description": "Authentication method to access protected pages: Public, Cookie Session, Bearer Token, Basic Auth, or Automated Form Login.",
                    "enum_titles": {
                        "none": "None (Public Webpage)",
                        "session_cookie": "Session Cookie / Custom Headers",
                        "bearer_token": "Bearer Token / API Secret",
                        "basic_auth": "HTTP Basic Authentication",
                        "form_login": "Automated Form Login & Session",
                    },
                },
                "session_cookies": {
                    "type": "string",
                    "title": "Session Cookies",
                    "default": "",
                    "input_type": "textarea",
                    "section": "auth",
                    "placeholder": "session_id=s%3Axyz; auth_token=abc12345",
                    "description": "Active session cookie string to attach to HTTP requests.",
                    "visible_when": {
                        "field": "auth_mode",
                        "equals": "session_cookie",
                    },
                },
                "custom_headers": {
                    "type": "string",
                    "title": "Custom HTTP Headers (JSON or Key: Value)",
                    "default": "",
                    "input_type": "textarea",
                    "section": "auth",
                    "placeholder": "{\n  \"Authorization\": \"Bearer <token>\"\n}",
                    "description": "Custom HTTP headers as JSON dictionary or newline-delimited 'Header-Name: value'.",
                    "visible_when": {
                        "field": "auth_mode",
                        "equals": "session_cookie",
                    },
                },
                "bearer_token": {
                    "type": "string",
                    "title": "Bearer / Access Token",
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
                "login_url": {
                    "type": "string",
                    "title": "Login URL",
                    "default": "",
                    "input_type": "text",
                    "section": "auth",
                    "placeholder": "https://example.com/login",
                    "description": "URL of the login page or authentication form endpoint.",
                    "visible_when": {
                        "field": "auth_mode",
                        "equals": "form_login",
                    },
                },
                "auth_username": {
                    "type": "string",
                    "title": "Username / Email",
                    "default": "",
                    "input_type": "text",
                    "section": "auth",
                    "description": "Username or email for Basic Authentication or Form Login.",
                    "visible_when": {
                        "field": "auth_mode",
                        "in": ["basic_auth", "form_login"],
                    },
                },
                "auth_password": {
                    "type": "string",
                    "title": "Password",
                    "default": "",
                    "input_type": "password",
                    "section": "auth",
                    "description": "Password for Basic Authentication or Form Login.",
                    "visible_when": {
                        "field": "auth_mode",
                        "in": ["basic_auth", "form_login"],
                    },
                },
                "login_username_field": {
                    "type": "string",
                    "title": "Username Field Name",
                    "default": "username",
                    "input_type": "text",
                    "section": "auth",
                    "placeholder": "username, email, or user[email]",
                    "description": "HTML form input name attribute for the username field.",
                    "visible_when": {
                        "field": "auth_mode",
                        "equals": "form_login",
                    },
                },
                "login_password_field": {
                    "type": "string",
                    "title": "Password Field Name",
                    "default": "password",
                    "input_type": "text",
                    "section": "auth",
                    "placeholder": "password or user[password]",
                    "description": "HTML form input name attribute for the password field.",
                    "visible_when": {
                        "field": "auth_mode",
                        "equals": "form_login",
                    },
                },
                "csrf_token_field": {
                    "type": "string",
                    "title": "CSRF Token Field Name",
                    "default": "csrfmiddlewaretoken",
                    "input_type": "text",
                    "section": "auth",
                    "placeholder": "csrfmiddlewaretoken, _csrf, or authenticity_token",
                    "description": "Optional CSRF field name to extract from the login form and include in submission.",
                    "visible_when": {
                        "field": "auth_mode",
                        "equals": "form_login",
                    },
                },
                "logout_url": {
                    "type": "string",
                    "title": "Logout URL (Optional)",
                    "default": "",
                    "input_type": "text",
                    "section": "auth",
                    "placeholder": "https://example.com/logout",
                    "description": "Optional URL to call after scraping to cleanly terminate the session.",
                    "visible_when": {
                        "field": "auth_mode",
                        "equals": "form_login",
                    },
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

    def _fetch_html(
        self,
        url: str,
        user_agent: str,
        timeout: int,
        auth_mode: str = "none",
        session_cookies: str = "",
        custom_headers: str = "",
        bearer_token: str = "",
        auth_username: str = "",
        auth_password: str = "",
        login_url: str = "",
        login_username_field: str = "username",
        login_password_field: str = "password",
        csrf_token_field: str = "csrfmiddlewaretoken",
        logout_url: str = "",
    ) -> str:
        parsed = urllib.parse.urlparse(url)
        if not parsed.scheme or parsed.scheme not in ("http", "https"):
            raise ValidationException(f"Invalid URL '{url}'. Must start with 'http://' or 'https://'.")

        default_ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ua = user_agent.strip() if user_agent and user_agent.strip() else default_ua

        # 1. Try requests.Session first for stateful session, cookie, and form login support
        try:
            import requests
            session = requests.Session()
            session.headers.update({
                "User-Agent": ua,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            })

            # Custom headers
            if custom_headers and custom_headers.strip():
                raw_h = custom_headers.strip()
                if raw_h.startswith("{") and raw_h.endswith("}"):
                    try:
                        parsed_h = json.loads(raw_h)
                        if isinstance(parsed_h, dict):
                            session.headers.update({str(k): str(v) for k, v in parsed_h.items()})
                    except Exception as je:
                        logger.warning(f"Failed to parse custom_headers as JSON: {je}")
                else:
                    for line in raw_h.splitlines():
                        if ":" in line:
                            hk, hv = line.split(":", 1)
                            session.headers[hk.strip()] = hv.strip()

            # Authentication handling
            if auth_mode == "session_cookie":
                if session_cookies and session_cookies.strip():
                    sc_str = session_cookies.strip()
                    session.headers["Cookie"] = sc_str
                    for part in sc_str.split(";"):
                        if "=" in part:
                            ck, cv = part.strip().split("=", 1)
                            session.cookies.set(ck.strip(), cv.strip())

            elif auth_mode == "bearer_token":
                if bearer_token and bearer_token.strip():
                    tok = bearer_token.strip()
                    if not tok.lower().startswith("bearer "):
                        session.headers["Authorization"] = f"Bearer {tok}"
                    else:
                        session.headers["Authorization"] = tok

            elif auth_mode == "basic_auth":
                if auth_username or auth_password:
                    session.auth = (auth_username, auth_password)

            elif auth_mode == "form_login":
                clean_login_url = login_url.strip() if login_url else ""
                if not clean_login_url:
                    raise ValidationException("Parameter 'login_url' cannot be empty when Authentication Mode is 'form_login'.")
                parsed_login = urllib.parse.urlparse(clean_login_url)
                if not parsed_login.scheme or parsed_login.scheme not in ("http", "https"):
                    raise ValidationException(f"Invalid login URL '{clean_login_url}'. Must start with 'http://' or 'https://'.")

                # Step A: GET login page to prime cookies and discover CSRF token
                try:
                    init_login_resp = session.get(clean_login_url, timeout=timeout, allow_redirects=True)
                except Exception as e:
                    raise ExecutionException(f"Failed to connect to login page '{clean_login_url}': {e}")

                csrf_val = None
                # Check cookies first
                for cand in (csrf_token_field, "csrftoken", "_csrf", "csrf_token", "authenticity_token"):
                    if cand in session.cookies:
                        csrf_val = session.cookies[cand]
                        break

                # Extract from login page HTML if needed
                if not csrf_val and init_login_resp.text:
                    patterns = [
                        rf'<input[^>]+name=["\']{re.escape(csrf_token_field)}["\'][^>]+value=["\']([^"\']+)["\']',
                        rf'<input[^>]+value=["\']([^"\']+)["\'][^>]+name=["\']{re.escape(csrf_token_field)}["\']',
                        r'<input[^>]+name=["\'](?:csrf[-_]?token|authenticity[-_]?token|_token)["\'][^>]+value=["\']([^"\']+)["\']',
                    ]
                    for pat in patterns:
                        m = re.search(pat, init_login_resp.text, re.IGNORECASE)
                        if m:
                            csrf_val = m.group(1)
                            break

                # Step B: POST credentials
                post_data = {
                    login_username_field or "username": auth_username,
                    login_password_field or "password": auth_password,
                }
                if csrf_val:
                    post_data[csrf_token_field or "csrfmiddlewaretoken"] = csrf_val

                try:
                    login_post_resp = session.post(clean_login_url, data=post_data, timeout=timeout, allow_redirects=True)
                    if login_post_resp.status_code >= 400:
                        raise ExecutionException(
                            f"Login rejected at '{clean_login_url}' with HTTP {login_post_resp.status_code}: {login_post_resp.reason}"
                        )
                except Exception as e:
                    raise ExecutionException(f"Failed to submit login credentials to '{clean_login_url}': {e}")

            # Fetch Target URL
            try:
                target_resp = session.get(url, timeout=timeout, allow_redirects=True)
                if target_resp.status_code in (401, 403):
                    if "wikipedia" in url.lower():
                        raise ExecutionException(
                            f"HTTP 403 Forbidden accessing '{url}'. Wikimedia policy blocks automated scrapers without custom contact headers. "
                            "Specify a descriptive User-Agent or enable Simulation Fallback."
                        )
                    raise ExecutionException(
                        f"HTTP {target_resp.status_code} {target_resp.reason} accessing '{url}'. "
                        "Target page rejected access. Verify authentication credentials, session cookies, or form login."
                    )
                target_resp.raise_for_status()
                html_text = target_resp.text
            finally:
                # Optional Logout
                if logout_url and logout_url.strip():
                    try:
                        session.get(logout_url.strip(), timeout=timeout, allow_redirects=True)
                    except Exception as logout_err:
                        logger.warning(f"Failed to trigger logout at '{logout_url}': {logout_err}")
                session.close()

            return html_text

        except ImportError:
            pass
        except requests.exceptions.RequestException as req_err:
            err_str = str(req_err)
            if "111" in err_str or "connection refused" in err_str.lower():
                raise ExecutionException(
                    f"Connection refused connecting to '{url}'. The server environment may lack outbound internet access "
                    f"or the target host actively rejected the socket. Enable Simulation Fallback to use benchmark data."
                )
            if "403" in err_str and "wikipedia" in url.lower():
                raise ExecutionException(
                    f"HTTP 403 Forbidden accessing '{url}'. Wikimedia policy blocks automated scrapers. "
                    "Specify a custom contact User-Agent or enable Simulation Fallback."
                )
            raise ExecutionException(f"Failed to fetch content from URL '{url}': {req_err}")

        # 2. Urllib fallback
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": ua,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                encoding = response.headers.get_content_charset() or "utf-8"
                return response.read().decode(encoding, errors="replace")
        except urllib.error.HTTPError as he:
            if he.code == 403 and "wikipedia" in url.lower():
                raise ExecutionException(
                    f"HTTP 403 Forbidden accessing '{url}'. Wikimedia policy blocks automated scrapers. "
                    "Specify a custom contact User-Agent or enable Simulation Fallback."
                )
            raise ExecutionException(f"HTTP Error {he.code} {he.reason} fetching '{url}'")
        except urllib.error.URLError as ue:
            err_reason = str(ue.reason)
            if "111" in err_reason or "connection refused" in err_reason.lower():
                raise ExecutionException(
                    f"Connection refused connecting to '{url}'. The server environment may lack outbound internet access "
                    f"or the target host actively rejected the socket. Enable Simulation Fallback to use benchmark data."
                )
            raise ExecutionException(f"Failed to fetch content from URL '{url}': {err_reason}")

    def _extract_html_tables(self, html: str, table_index: int) -> pd.DataFrame:
        # 1. Try standard pandas read_html (uses lxml or html5lib if available)
        try:
            dfs = pd.read_html(io.StringIO(html))
            if dfs:
                idx = min(table_index, len(dfs) - 1)
                selected_df = dfs[idx]
                if isinstance(selected_df.columns, pd.MultiIndex):
                    selected_df.columns = ["_".join(str(c) for c in col).strip("_") for col in selected_df.columns]
                # Guarantee all column names are non-empty strings (prevents integer RangeIndex Pydantic validation errors)
                selected_df.columns = [
                    str(c).strip() if str(c).strip() and str(c).lower() != "nan" else f"col_{i+1}"
                    for i, c in enumerate(selected_df.columns)
                ]
                return selected_df
        except Exception as e:
            logger.info(f"pd.read_html failed ({e}), falling back to HTML parser")

        # 2. Resilient BeautifulSoup / built-in parser fallback
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            tables = soup.find_all("table")
            if tables:
                idx = min(table_index, len(tables) - 1)
                target_table = tables[idx]
                rows: List[List[str]] = []
                for tr in target_table.find_all("tr"):
                    cells = [td.get_text(strip=True) for td in tr.find_all(["th", "td"])]
                    if cells:
                        rows.append(cells)
                if rows:
                    header = rows[0]
                    seen: Dict[str, int] = {}
                    clean_headers = []
                    for h in header:
                        name = str(h) if h else "col"
                        if name in seen:
                            seen[name] += 1
                            name = f"{name}_{seen[name]}"
                        else:
                            seen[name] = 0
                        clean_headers.append(name)
                    data = rows[1:] if len(rows) > 1 else []
                    padded_data = []
                    for r in data:
                        if len(r) < len(clean_headers):
                            r = r + [""] * (len(clean_headers) - len(r))
                        elif len(r) > len(clean_headers):
                            r = r[:len(clean_headers)]
                        padded_data.append(r)
                    return pd.DataFrame(padded_data, columns=clean_headers)
        except Exception as bs_err:
            logger.warning(f"BeautifulSoup table parse failed: {bs_err}")

        # 3. Final regex fallback
        return self._regex_parse_tables(html, table_index)

    def _regex_parse_tables(self, html: str, table_index: int) -> pd.DataFrame:
        table_matches = re.findall(r"<table[^>]*>(.*?)</table>", html, re.DOTALL | re.IGNORECASE)
        if not table_matches:
            raise ExecutionException("No <table> elements found in the HTML content.")
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

        if len(rows) == 1:
            clean_headers = [f"col_{i+1}" for i in range(len(rows[0]))]
            return pd.DataFrame([rows[0]], columns=clean_headers)

        header = rows[0]
        data = rows[1:]
        seen: Dict[str, int] = {}
        clean_headers = []
        for i, h in enumerate(header):
            name = str(h).strip() if str(h).strip() and str(h).lower() != "nan" else f"col_{i+1}"
            if name in seen:
                seen[name] += 1
                name = f"{name}_{seen[name]}"
            else:
                seen[name] = 0
            clean_headers.append(name)

        padded_data = []
        for r in data:
            if len(r) < len(clean_headers):
                r = r + [""] * (len(clean_headers) - len(r))
            elif len(r) > len(clean_headers):
                r = r[:len(clean_headers)]
            padded_data.append(r)
        return pd.DataFrame(padded_data, columns=clean_headers)

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

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
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

        # Authentication and Session parameters
        auth_mode = str(config.get("auth_mode", "none")).strip().lower()
        session_cookies = str(config.get("session_cookies", "")).strip()
        custom_headers = str(config.get("custom_headers", "")).strip()
        bearer_token = str(config.get("bearer_token", "")).strip()
        auth_username = str(config.get("auth_username", "")).strip()
        auth_password = str(config.get("auth_password", "")).strip()
        login_url = str(config.get("login_url", "")).strip()
        login_username_field = str(config.get("login_username_field", "username")).strip()
        login_password_field = str(config.get("login_password_field", "password")).strip()
        csrf_token_field = str(config.get("csrf_token_field", "csrfmiddlewaretoken")).strip()
        logout_url = str(config.get("logout_url", "")).strip()

        html_content = ""
        is_simulation = False

        # 1. Use mock HTML if provided
        if mock_html:
            html_content = mock_html
        else:
            # 2. Try fetching from network URL
            try:
                html_content = self._fetch_html(
                    url=url,
                    user_agent=user_agent,
                    timeout=timeout,
                    auth_mode=auth_mode,
                    session_cookies=session_cookies,
                    custom_headers=custom_headers,
                    bearer_token=bearer_token,
                    auth_username=auth_username,
                    auth_password=auth_password,
                    login_url=login_url,
                    login_username_field=login_username_field,
                    login_password_field=login_password_field,
                    csrf_token_field=csrf_token_field,
                    logout_url=logout_url,
                )
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

        # Ensure all DataFrame columns are strictly strings
        if df is not None and not df.empty:
            df.columns = [
                str(c).strip() if str(c).strip() and str(c).lower() != "nan" else f"col_{i+1}"
                for i, c in enumerate(df.columns)
            ]

        if include_meta and not df.empty:
            now_iso = datetime.now(timezone.utc).isoformat()
            df["_source_url"] = url
            df["_extracted_at"] = now_iso

        elapsed = round((time.time() - t0) * 1000, 2)
        metrics = {
            "execution_time_ms": elapsed,
            "url": url,
            "extraction_mode": extraction_mode,
            "auth_mode": auth_mode,
            "rows_extracted": len(df),
            "columns_count": len(df.columns),
            "is_simulation": is_simulation,
        }

        return {"dataframe": df, "metrics": metrics}
