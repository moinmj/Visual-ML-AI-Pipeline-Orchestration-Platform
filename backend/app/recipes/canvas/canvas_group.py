"""
Group (visual grouping) Recipe (Canvas Tools -> Documentation & Layout).

Visually groups related nodes on the canvas into colored logical stages (e.g. Ingestion,
Preprocessing, Feature Engineering, Training, Evaluation) with bounding metadata and status summaries.
"""

from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

try:
    import pandas as pd
except ImportError:
    pd = None

from backend.app.recipes.base.recipe import BaseRecipe, RecipePort


class CanvasGroupRecipe(BaseRecipe):
    recipe_id = "canvas_group"
    name = "Group (visual grouping)"
    version = "1.0.0"
    category = "canvas"
    group = "Canvas Tools"
    subgroup = "Documentation & Layout"
    description = "Visually groups related nodes on the canvas."
    input_types = ["dataframe"]
    output_types = ["dataframe", "report"]

    inputs: List[RecipePort] = [
        RecipePort(
            id="input",
            label="Stage Data (Optional)",
            type="dataframe",
            required=False,
            max_connections=1,
            description="Optional input table passing through this stage group."
        )
    ]

    outputs: List[RecipePort] = [
        RecipePort(
            id="output",
            label="Passthrough Table",
            type="dataframe",
            description="Upstream table forwarded unchanged to downstream nodes."
        ),
        RecipePort(
            id="report",
            label="Group Membership Summary",
            type="report",
            description="Inspection report of enclosed nodes and group status."
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "group_name": {
                    "type": "string",
                    "title": "Group Stage Title",
                    "input_type": "text",
                    "default": "Data Cleaning & Prep Stage",
                    "description": "Readable title displayed in the group header badge on the canvas."
                },
                "description": {
                    "type": "string",
                    "title": "Stage Description",
                    "input_type": "text",
                    "default": "Encompasses missing value imputation, scaling, and categorical encoding.",
                    "description": "Functional description of pipeline transformations enclosed in this group."
                },
                "color": {
                    "type": "string",
                    "title": "Bounding Box Theme",
                    "input_type": "dropdown",
                    "enum": ["slate", "blue", "emerald", "amber", "purple", "rose"],
                    "default": "slate",
                    "description": "Border and background tint for the canvas visual bounding container."
                },
                "collapsed": {
                    "type": "boolean",
                    "title": "Collapse Nodes in View",
                    "input_type": "checkbox",
                    "default": False,
                    "description": "Whether to collapse the group container into a single macro summary node."
                },
                "node_ids": {
                    "type": "string",
                    "title": "Enclosed Node IDs",
                    "input_type": "text",
                    "default": "",
                    "description": "Comma-separated list of Canvas Node IDs contained within this group."
                },
                "border_style": {
                    "type": "string",
                    "title": "Border Line Style",
                    "input_type": "dropdown",
                    "enum": ["solid", "dashed", "dotted"],
                    "default": "dashed",
                    "description": "Stroke style for group outline on canvas whiteboard."
                },
                "execution_order_priority": {
                    "type": "integer",
                    "title": "Execution Stage Priority",
                    "input_type": "number",
                    "default": 1,
                    "minimum": 1,
                    "maximum": 100,
                    "description": "Logical execution sequencing order relative to other pipeline stages."
                }
            }
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        upstream_df = inputs.get("dataframe") if "dataframe" in inputs else inputs.get("input")

        # Edge Case 1: Extract group name and defaults
        group_name = str(config.get("group_name", "Data Cleaning & Prep Stage")).strip() or "Group Stage"
        description = str(config.get("description", "")).strip()
        color = str(config.get("color", "slate")).lower()
        if color not in ["slate", "blue", "emerald", "amber", "purple", "rose"]:
            color = "slate"

        border_style = str(config.get("border_style", "dashed")).lower()
        if border_style not in ["solid", "dashed", "dotted"]:
            border_style = "dashed"

        collapsed = bool(config.get("collapsed", False))

        try:
            priority = int(config.get("execution_order_priority", 1))
        except (ValueError, TypeError):
            priority = 1

        # Edge Case 2: Node IDs extraction from comma-separated string, array, or whitespace
        raw_nodes = config.get("node_ids", "")
        node_list: List[str] = []
        if isinstance(raw_nodes, str):
            node_list = [n.strip() for n in raw_nodes.split(",") if n.strip()]
        elif isinstance(raw_nodes, (list, tuple)):
            node_list = [str(n).strip() for n in raw_nodes if str(n).strip()]

        # Edge Case 3: Deduplicate node IDs while preserving order
        seen = set()
        clean_nodes = []
        for n in node_list:
            if n not in seen:
                seen.add(n)
                clean_nodes.append(n)

        # Edge Case 4: Inspect active DAG context if provided to verify node existence
        missing_in_context = []
        existing_in_context = []
        if context and isinstance(context, dict):
            known_canvas_nodes = context.get("nodes", [])
            known_ids = set()
            for kn in known_canvas_nodes:
                if isinstance(kn, dict) and "id" in kn:
                    known_ids.add(kn["id"])
                elif hasattr(kn, "id"):
                    known_ids.add(kn.id)

            if known_ids:
                for nid in clean_nodes:
                    if nid in known_ids:
                        existing_in_context.append(nid)
                    else:
                        missing_in_context.append(nid)
            else:
                existing_in_context = clean_nodes
        else:
            existing_in_context = clean_nodes

        now_iso = datetime.now(timezone.utc).isoformat()
        md_report = (
            f"### 🗂️ Stage Group: **{group_name}**\n"
            f"**Theme:** `{color}` | **Border:** `{border_style}` | **Priority:** `{priority}` | **Collapsed:** `{collapsed}`\n\n"
            f"**Description:** {description or 'No stage description provided.'}\n\n"
            f"**Enclosed Nodes ({len(clean_nodes)}):**\n"
            + ("\n".join([f"- 🔹 `{nid}`" for nid in clean_nodes]) if clean_nodes else "- *No nodes explicitly assigned to this group yet.*")
        )

        result: Dict[str, Any] = {
            "group_name": group_name,
            "description": description,
            "color": color,
            "border_style": border_style,
            "collapsed": collapsed,
            "priority": priority,
            "enclosed_node_ids": clean_nodes,
            "verified_in_context": existing_in_context,
            "unverified_in_context": missing_in_context,
            "metrics": {
                "enclosed_nodes_count": len(clean_nodes),
                "verified_nodes_count": len(existing_in_context),
                "priority_level": priority,
                "is_collapsed": int(collapsed)
            },
            "report": {
                "title": f"Group: {group_name}",
                "markdown": md_report,
                "timestamp": now_iso
            },
            "output_summary": {
                "status": "GROUPED",
                "group_name": group_name,
                "nodes_count": len(clean_nodes),
                "theme": color
            }
        }

        if upstream_df is not None:
            result["dataframe"] = upstream_df
            result["output"] = upstream_df

        return result
