"""
Pipeline Diff Viewer Recipe (Canvas Tools -> Versioning & Reuse).

Compares two pipeline versions or JSON topologies, identifying added/removed nodes,
rewired edge connections, and modified configuration parameters with executive Markdown reports.
"""

import json
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

try:
    import pandas as pd
except ImportError:
    pd = None

from backend.app.recipes.base.recipe import BaseRecipe, RecipePort


class PipelineDiffViewerRecipe(BaseRecipe):
    recipe_id = "pipeline_diff_viewer"
    name = "Pipeline Diff Viewer"
    version = "1.0.0"
    category = "canvas"
    group = "Canvas Tools"
    subgroup = "Versioning & Reuse"
    description = "Shows the differences between two versions of a pipeline."
    input_types = ["dataframe"]
    output_types = ["dataframe", "report"]

    inputs: List[RecipePort] = [
        RecipePort(
            id="input",
            label="Active Data (Optional)",
            type="dataframe",
            required=False,
            max_connections=1,
            description="Upstream dataset to pass through diff operator."
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
            label="Pipeline Comparison Report",
            type="report",
            description="Structured difference report comparing both pipeline versions."
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "source_version_label": {
                    "type": "string",
                    "title": "Baseline Version Label",
                    "input_type": "text",
                    "default": "v1.0 (Production)",
                    "description": "Descriptive tag for baseline / original pipeline."
                },
                "target_version_label": {
                    "type": "string",
                    "title": "Candidate Version Label",
                    "input_type": "text",
                    "default": "v2.0 (Staging)",
                    "description": "Descriptive tag for candidate / modified pipeline."
                },
                "source_pipeline_json": {
                    "type": "string",
                    "title": "Baseline Pipeline JSON",
                    "input_type": "textarea",
                    "default": "{\"nodes\": [{\"id\": \"imputer_1\", \"recipe_id\": \"missing_value_imputer\", \"config\": {\"strategy\": \"mean\"}}], \"edges\": []}",
                    "description": "JSON representation of the baseline pipeline DAG."
                },
                "target_pipeline_json": {
                    "type": "string",
                    "title": "Candidate Pipeline JSON",
                    "input_type": "textarea",
                    "default": "{\"nodes\": [{\"id\": \"imputer_1\", \"recipe_id\": \"missing_value_imputer\", \"config\": {\"strategy\": \"median\"}}, {\"id\": \"scaler_1\", \"recipe_id\": \"feature_scaler\", \"config\": {\"method\": \"standard\"}}], \"edges\": [{\"source\": \"imputer_1\", \"target\": \"scaler_1\"}]}",
                    "description": "JSON representation of the candidate pipeline DAG."
                },
                "diff_mode": {
                    "type": "string",
                    "title": "Comparison Scope",
                    "input_type": "dropdown",
                    "enum": ["full", "structural", "parameter"],
                    "default": "full",
                    "description": "Scope of analysis: full (nodes, edges & parameters), structural (topology only), or parameter (configs only)."
                },
                "ignore_visual_positions": {
                    "type": "boolean",
                    "title": "Ignore Canvas (X, Y) Coordinates",
                    "input_type": "checkbox",
                    "default": True,
                    "description": "Disregard purely cosmetic coordinate movements and focus on functional logic."
                }
            }
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        upstream_df = inputs.get("dataframe") if "dataframe" in inputs else inputs.get("input")

        # Edge Case 1: Labels & Config extraction
        src_label = str(config.get("source_version_label", "v1.0 (Baseline)")).strip() or "v1.0"
        tgt_label = str(config.get("target_version_label", "v2.0 (Candidate)")).strip() or "v2.0"
        diff_mode = str(config.get("diff_mode", "full")).lower()
        ignore_pos = bool(config.get("ignore_visual_positions", True))

        # Edge Case 2: Parse JSON payloads safely
        def _parse_pipeline(raw_val: Any) -> Dict[str, Any]:
            if isinstance(raw_val, dict):
                return raw_val
            if isinstance(raw_val, str):
                s = raw_val.strip()
                if s:
                    try:
                        parsed = json.loads(s)
                        if isinstance(parsed, dict):
                            return parsed
                    except Exception:
                        pass
            return {"nodes": [], "edges": []}

        src_data = _parse_pipeline(config.get("source_pipeline_json"))
        tgt_data = _parse_pipeline(config.get("target_pipeline_json"))

        # Extract nodes and edges
        src_nodes = {n.get("id"): n for n in src_data.get("nodes", []) if isinstance(n, dict) and "id" in n}
        tgt_nodes = {n.get("id"): n for n in tgt_data.get("nodes", []) if isinstance(n, dict) and "id" in n}

        src_edges = set((e.get("source"), e.get("target")) for e in src_data.get("edges", []) if isinstance(e, dict))
        tgt_edges = set((e.get("source"), e.get("target")) for e in tgt_data.get("edges", []) if isinstance(e, dict))

        # Compute Node Diffs
        added_node_ids = sorted(list(set(tgt_nodes.keys()) - set(src_nodes.keys())))
        removed_node_ids = sorted(list(set(src_nodes.keys()) - set(tgt_nodes.keys())))
        common_node_ids = sorted(list(set(src_nodes.keys()) & set(tgt_nodes.keys())))

        # Compute Parameter Diffs on common nodes
        parameter_changes = []
        for nid in common_node_ids:
            s_cfg = src_nodes[nid].get("config", {}) or {}
            t_cfg = tgt_nodes[nid].get("config", {}) or {}

            all_keys = set(s_cfg.keys()) | set(t_cfg.keys())
            if ignore_pos:
                all_keys -= {"x", "y", "pos_x", "pos_y", "position"}

            for k in sorted(all_keys):
                s_val = s_cfg.get(k)
                t_val = t_cfg.get(k)
                if s_val != t_val:
                    parameter_changes.append({
                        "node_id": nid,
                        "recipe_id": tgt_nodes[nid].get("recipe_id", src_nodes[nid].get("recipe_id")),
                        "parameter": k,
                        "baseline_value": s_val,
                        "candidate_value": t_val
                    })

        # Compute Edge Diffs
        added_edges = sorted(list(tgt_edges - src_edges))
        removed_edges = sorted(list(src_edges - tgt_edges))

        now_iso = datetime.now(timezone.utc).isoformat()

        # Build Executive Markdown Diff
        md_lines = [
            f"### 🔍 Pipeline Version Comparison: `{src_label}` vs `{tgt_label}`",
            f"**Diff Mode:** `{diff_mode.upper()}` | **Evaluated at:** `{now_iso[:19]} UTC`\n",
            "| Metric | Baseline | Candidate | Delta |",
            "| :--- | :---: | :---: | :---: |",
            f"| **Total Nodes** | `{len(src_nodes)}` | `{len(tgt_nodes)}` | `{len(tgt_nodes) - len(src_nodes):+d}` |",
            f"| **Total Connections** | `{len(src_edges)}` | `{len(tgt_edges)}` | `{len(tgt_edges) - len(src_edges):+d}` |",
            f"| **Parameter Changes** | — | — | `{len(parameter_changes)} modified` |\n"
        ]

        if added_node_ids:
            md_lines.append(f"#### ➕ Added Nodes ({len(added_node_ids)}):")
            for nid in added_node_ids:
                rid = tgt_nodes[nid].get("recipe_id", "custom")
                md_lines.append(f"- `+{nid}` *(Recipe: `{rid}`)*")
            md_lines.append("")

        if removed_node_ids:
            md_lines.append(f"#### ➖ Removed Nodes ({len(removed_node_ids)}):")
            for nid in removed_node_ids:
                rid = src_nodes[nid].get("recipe_id", "custom")
                md_lines.append(f"- `-{nid}` *(Recipe: `{rid}`)*")
            md_lines.append("")

        if parameter_changes:
            md_lines.append(f"#### ⚙️ Modified Configuration Parameters ({len(parameter_changes)}):")
            for ch in parameter_changes:
                md_lines.append(f"- **`{ch['node_id']}`** (`{ch['parameter']}`): `{ch['baseline_value']}` ➔ **`{ch['candidate_value']}`**")
            md_lines.append("")

        if added_edges:
            md_lines.append(f"#### 🔗 New Connections:")
            for s, t in added_edges:
                md_lines.append(f"- `+{s}` ➔ `{t}`")
            md_lines.append("")

        if not added_node_ids and not removed_node_ids and not parameter_changes and not added_edges and not removed_edges:
            md_lines.append("✅ **Identical Architectures:** No structural or functional parameter differences detected.")

        md_report = "\n".join(md_lines)

        result: Dict[str, Any] = {
            "source_label": src_label,
            "target_label": tgt_label,
            "diff_summary": {
                "nodes_added": len(added_node_ids),
                "nodes_removed": len(removed_node_ids),
                "parameters_modified": len(parameter_changes),
                "edges_added": len(added_edges),
                "edges_removed": len(removed_edges),
                "is_identical": int(not added_node_ids and not removed_node_ids and not parameter_changes and not added_edges and not removed_edges)
            },
            "added_nodes": added_node_ids,
            "removed_nodes": removed_node_ids,
            "parameter_changes": parameter_changes,
            "added_edges": [f"{s}->{t}" for s, t in added_edges],
            "removed_edges": [f"{s}->{t}" for s, t in removed_edges],
            "metrics": {
                "nodes_added": len(added_node_ids),
                "nodes_removed": len(removed_node_ids),
                "params_changed": len(parameter_changes),
                "is_identical": int(not added_node_ids and not removed_node_ids and not parameter_changes and not added_edges and not removed_edges)
            },
            "report": {
                "title": f"Diff: {src_label} vs {tgt_label}",
                "markdown": md_report,
                "timestamp": now_iso
            },
            "output_summary": {
                "status": "DIFF_COMPLETED",
                "added": len(added_node_ids),
                "removed": len(removed_node_ids),
                "param_diffs": len(parameter_changes)
            }
        }

        if upstream_df is not None:
            result["dataframe"] = upstream_df
            result["output"] = upstream_df

        return result
