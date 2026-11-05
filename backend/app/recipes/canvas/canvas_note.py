"""
Note (canvas annotation) Recipe (Canvas Tools -> Documentation & Layout).

Adds a text note to the canvas to document the pipeline, record SLAs, design assumptions,
and team notes with color themes, markdown formatting, and optional passthrough wiring.
"""

import html
import re
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

try:
    import pandas as pd
except ImportError:
    pd = None

from backend.app.recipes.base.recipe import BaseRecipe, RecipePort


class CanvasNoteRecipe(BaseRecipe):
    recipe_id = "canvas_note"
    name = "Note (canvas annotation)"
    version = "1.0.0"
    category = "canvas"
    group = "Canvas Tools"
    subgroup = "Documentation & Layout"
    description = "Adds a text note to the canvas to document the pipeline."
    input_types = ["dataframe"]
    output_types = ["dataframe", "report"]

    inputs: List[RecipePort] = [
        RecipePort(
            id="input",
            label="Upstream Data / Trigger (Optional)",
            type="dataframe",
            required=False,
            max_connections=1,
            description="Optional upstream dataset passed through untouched without mutation."
        )
    ]

    outputs: List[RecipePort] = [
        RecipePort(
            id="output",
            label="Passthrough Table",
            type="dataframe",
            description="Preserved upstream table passed transparently to downstream nodes."
        ),
        RecipePort(
            id="report",
            label="Documentation Report",
            type="report",
            description="Formatted documentation markdown card and metadata."
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "title": "Note Title",
                    "input_type": "text",
                    "default": "Pipeline Documentation Note",
                    "description": "Short headline or section title displayed at the top of the canvas note."
                },
                "content": {
                    "type": "string",
                    "title": "Note Body Content",
                    "input_type": "textarea",
                    "default": "Document your data flow assumptions, source origins, business rules, and SLAs here.",
                    "description": "Full Markdown or plain-text body content of the note."
                },
                "color_theme": {
                    "type": "string",
                    "title": "Card Color Theme",
                    "input_type": "dropdown",
                    "enum": ["yellow", "blue", "green", "purple", "gray", "red"],
                    "default": "yellow",
                    "description": "Visual sticky note background color styling on the canvas."
                },
                "font_size": {
                    "type": "string",
                    "title": "Typography Size",
                    "input_type": "dropdown",
                    "enum": ["small", "medium", "large"],
                    "default": "medium",
                    "description": "Text size rendering for readability across different canvas zoom levels."
                },
                "pinned": {
                    "type": "boolean",
                    "title": "Pin to Viewport",
                    "input_type": "checkbox",
                    "default": False,
                    "description": "Keep this documentation note permanently visible in viewport regardless of pan/zoom."
                },
                "author": {
                    "type": "string",
                    "title": "Author / Contributor",
                    "input_type": "text",
                    "default": "Data Science Team",
                    "description": "Name or handle of team member who authored or approved this note."
                },
                "tags": {
                    "type": "array",
                    "title": "Topic Tags",
                    "input_type": "tags",
                    "items": {"type": "string"},
                    "default": ["documentation", "sla"],
                    "description": "Searchable classification tags (e.g. data-quality, compliance, sla, assumptions)."
                }
            }
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        # Edge Case 1: Handle upstream dataframe passthrough cleanly
        upstream_df = inputs.get("dataframe") if "dataframe" in inputs else inputs.get("input")
        if upstream_df is None and pd is not None:
            # Handle list/dict inputs or create empty DataFrame fallback
            raw_data = inputs.get("data")
            if isinstance(raw_data, list):
                upstream_df = pd.DataFrame(raw_data)

        # Edge Case 2: Config extraction with complete fallback defaults
        raw_title = config.get("title")
        title = str(raw_title).strip() if raw_title is not None and str(raw_title).strip() else "Pipeline Documentation Note"

        raw_content = config.get("content")
        content = str(raw_content).strip() if raw_content is not None and str(raw_content).strip() else "No content provided."

        color_theme = str(config.get("color_theme", "yellow")).lower()
        if color_theme not in ["yellow", "blue", "green", "purple", "gray", "red"]:
            color_theme = "yellow"

        font_size = str(config.get("font_size", "medium")).lower()
        if font_size not in ["small", "medium", "large"]:
            font_size = "medium"

        pinned = bool(config.get("pinned", False))
        author = str(config.get("author", "Data Science Team")).strip() or "Data Science Team"

        # Edge Case 3: Tags handling (accept list, comma-separated string, or empty)
        raw_tags = config.get("tags", ["documentation"])
        if isinstance(raw_tags, str):
            tags = [t.strip() for t in raw_tags.split(",") if t.strip()]
        elif isinstance(raw_tags, (list, tuple)):
            tags = [str(t).strip() for t in raw_tags if str(t).strip()]
        else:
            tags = ["documentation"]

        # Word count & metrics
        word_count = len(re.findall(r"\w+", content))
        char_count = len(content)
        now_iso = datetime.now(timezone.utc).isoformat()

        # Edge Case 4: Safe HTML Escaping for display badge
        safe_title = html.escape(title)
        safe_content = html.escape(content)
        theme_hex = {
            "yellow": "#FEF08A",
            "blue": "#BAE6FD",
            "green": "#BBF7D0",
            "purple": "#E9D5FF",
            "gray": "#E2E8F0",
            "red": "#FECDD3"
        }.get(color_theme, "#FEF08A")

        rendered_markdown = (
            f"### 📝 {title}\n"
            f"**Author:** `{author}` | **Updated:** `{now_iso[:10]}` | **Theme:** `{color_theme}`\n\n"
            f"{content}\n\n"
            f"---\n"
            f"*Tags:* " + " ".join([f"`#{t}`" for t in tags])
        )

        rendered_html = (
            f"<div style='background-color:{theme_hex}; border-left:4px solid #475569; padding:12px; border-radius:6px; color:#1E293B;'>"
            f"<div style='font-weight:700; font-size:1.05rem; margin-bottom:4px;'>📌 {safe_title}</div>"
            f"<div style='font-size:0.88rem; line-height:1.5; white-space:pre-wrap;'>{safe_content}</div>"
            f"<div style='margin-top:8px; font-size:0.75rem; color:#64748B;'>By: {html.escape(author)} • {now_iso[:19]}</div>"
            f"</div>"
        )

        result: Dict[str, Any] = {
            "title": title,
            "content": content,
            "color_theme": color_theme,
            "font_size": font_size,
            "pinned": pinned,
            "author": author,
            "tags": tags,
            "metrics": {
                "word_count": word_count,
                "character_count": char_count,
                "tag_count": len(tags),
                "pinned": int(pinned)
            },
            "report": {
                "title": title,
                "markdown": rendered_markdown,
                "html": rendered_html,
                "timestamp": now_iso
            },
            "output_summary": {
                "status": "DOCUMENTED",
                "title": title,
                "word_count": word_count,
                "theme": color_theme
            }
        }

        # Return upstream dataframe if present
        if upstream_df is not None:
            result["dataframe"] = upstream_df
            result["output"] = upstream_df

        return result
