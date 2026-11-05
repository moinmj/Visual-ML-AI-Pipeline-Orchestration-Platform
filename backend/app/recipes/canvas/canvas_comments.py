"""
Comment / Annotation Threads Recipe (Canvas Tools -> Documentation & Layout).

Adds collaborative review and audit annotation threads to nodes or global pipelines,
tracking reviewer feedback, resolution status (Open, In Review, Resolved), and priorities.
"""

import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

try:
    import pandas as pd
except ImportError:
    pd = None

from backend.app.recipes.base.recipe import BaseRecipe, RecipePort


class CanvasCommentsRecipe(BaseRecipe):
    recipe_id = "canvas_comments"
    name = "Comment / Annotation Threads"
    version = "1.0.0"
    category = "canvas"
    group = "Canvas Tools"
    subgroup = "Documentation & Layout"
    description = "Adds discussion threads to nodes for team collaboration."
    input_types = ["dataframe"]
    output_types = ["dataframe", "report"]

    inputs: List[RecipePort] = [
        RecipePort(
            id="input",
            label="Pipeline Data (Optional)",
            type="dataframe",
            required=False,
            max_connections=1,
            description="Optional input table passing through annotated checkpoint."
        )
    ]

    outputs: List[RecipePort] = [
        RecipePort(
            id="output",
            label="Passthrough Table",
            type="dataframe",
            description="Preserved table passed unchanged to downstream nodes."
        ),
        RecipePort(
            id="report",
            label="Thread Discussion Report",
            type="report",
            description="Audit trail of discussion notes, feedback, and sign-offs."
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "target_node_id": {
                    "type": "string",
                    "title": "Target Node ID",
                    "input_type": "text",
                    "default": "",
                    "description": "Specific Canvas Node ID this comment attaches to (leave empty for canvas-wide thread)."
                },
                "author": {
                    "type": "string",
                    "title": "Reviewer Name / Email",
                    "input_type": "text",
                    "default": "Lead ML Engineer",
                    "description": "Name or handle of team reviewer leaving the comment."
                },
                "comment_text": {
                    "type": "string",
                    "title": "Review Comment / Feedback",
                    "input_type": "textarea",
                    "default": "Verified target leak guards and hyperparameter bounds. Approved for staging deployment.",
                    "description": "Feedback, audit question, or architectural review notes."
                },
                "status": {
                    "type": "string",
                    "title": "Thread Status",
                    "input_type": "dropdown",
                    "enum": ["Open", "In Review", "Resolved", "Archived"],
                    "default": "Open",
                    "description": "Current resolution status of this review thread."
                },
                "priority": {
                    "type": "string",
                    "title": "Issue Priority",
                    "input_type": "dropdown",
                    "enum": ["Low", "Medium", "High", "Blocker"],
                    "default": "Medium",
                    "description": "Severity level of items raised in this thread."
                },
                "thread_id": {
                    "type": "string",
                    "title": "Thread ID (Optional)",
                    "input_type": "text",
                    "default": "",
                    "description": "Unique identifier for this discussion thread (auto-generated if omitted)."
                },
                "tags": {
                    "type": "array",
                    "title": "Review Tags",
                    "input_type": "tags",
                    "items": {"type": "string"},
                    "default": ["qa_review", "signoff"],
                    "description": "Categorization tags (e.g. data_quality, governance, performance, blocker)."
                }
            }
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        upstream_df = inputs.get("dataframe") if "dataframe" in inputs else inputs.get("input")

        # Edge Case 1: Config extraction with clean fallbacks
        target_node = str(config.get("target_node_id", "")).strip()
        author = str(config.get("author", "Reviewer")).strip() or "Reviewer"
        raw_text = config.get("comment_text", "")
        comment_text = str(raw_text).strip() if raw_text is not None and str(raw_text).strip() else "No review comment provided."

        status = str(config.get("status", "Open")).title()
        if status not in ["Open", "In Review", "Resolved", "Archived"]:
            status = "Open"

        priority = str(config.get("priority", "Medium")).title()
        if priority not in ["Low", "Medium", "High", "Blocker"]:
            priority = "Medium"

        raw_tid = config.get("thread_id", "")
        thread_id = str(raw_tid).strip() if raw_tid and str(raw_tid).strip() else f"th_{uuid.uuid4().hex[:8]}"

        raw_tags = config.get("tags", ["qa_review"])
        if isinstance(raw_tags, str):
            tags = [t.strip() for t in raw_tags.split(",") if t.strip()]
        elif isinstance(raw_tags, (list, tuple)):
            tags = [str(t).strip() for t in raw_tags if str(t).strip()]
        else:
            tags = ["qa_review"]

        now_iso = datetime.now(timezone.utc).isoformat()
        priority_icons = {"Low": "🟢", "Medium": "🟡", "High": "🟠", "Blocker": "🔴"}
        status_icons = {"Open": "⭕", "In Review": "🔍", "Resolved": "✅", "Archived": "📁"}

        p_ico = priority_icons.get(priority, "🟡")
        s_ico = status_icons.get(status, "⭕")

        target_display = f"`{target_node}`" if target_node else "*Global Canvas Thread*"

        md_report = (
            f"### 💬 Thread #{thread_id}: {s_ico} **{status.upper()}**\n"
            f"**Target:** {target_display} | **Priority:** {p_ico} `{priority}` | **Author:** `{author}`\n\n"
            f"> {comment_text}\n\n"
            f"---\n"
            f"*Recorded at:* `{now_iso[:19]}` | *Tags:* " + " ".join([f"`#{t}`" for t in tags])
        )

        result: Dict[str, Any] = {
            "thread_id": thread_id,
            "target_node_id": target_node,
            "author": author,
            "comment_text": comment_text,
            "status": status,
            "priority": priority,
            "tags": tags,
            "created_at": now_iso,
            "metrics": {
                "character_count": len(comment_text),
                "is_resolved": int(status == "Resolved"),
                "is_blocker": int(priority == "Blocker"),
                "tag_count": len(tags)
            },
            "report": {
                "title": f"Review Comment ({status})",
                "markdown": md_report,
                "thread_id": thread_id,
                "timestamp": now_iso
            },
            "output_summary": {
                "status": status.upper(),
                "priority": priority,
                "author": author,
                "target_node": target_node or "Global"
            }
        }

        if upstream_df is not None:
            result["dataframe"] = upstream_df
            result["output"] = upstream_df

        return result
