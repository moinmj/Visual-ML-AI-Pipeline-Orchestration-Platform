"""
Hierarchical / Tree Flattener Recipe.
Sub-group: Domain-Specific Cleaning
Category: Data Preparation & Transformation (preprocessing)

Flattens recursive parent-child hierarchies (e.g. org charts, product category taxonomies,
account structures) into explicit level columns, breadcrumb paths, depth metrics, and root IDs.
"""
from typing import Dict, Any, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
from backend.app.recipes.base.recipe import BaseRecipe, RecipePort


class HierarchicalTreeFlattenerRecipe(BaseRecipe):
    recipe_id = "hierarchical_tree_flattener"
    name = "Hierarchical / Tree Flattener"
    version = "1.0.0"
    category = "preprocessing"
    group = "Data Preparation & Transformation"
    subgroup = "Domain-Specific Cleaning"
    description = "Flattens parent-child hierarchies into level columns, ancestor breadcrumb paths, and tree depth metrics."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    inputs = [
        RecipePort(
            id="input",
            label="Input Table",
            type="dataframe",
            required=True,
            max_connections=1,
            description="Dataset containing self-referencing parent-child columns"
        )
    ]

    outputs = [
        RecipePort(
            id="dataframe",
            label="Tree Flattened Table",
            type="dataframe",
            description="Dataset enriched with depth levels, hierarchical path, and level columns"
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "id_column": {
                    "type": "string",
                    "title": "Child / Entity ID Column",
                    "default": "id",
                    "description": "Unique identifier column for each node (e.g. 'employee_id', 'category_id')."
                },
                "parent_id_column": {
                    "type": "string",
                    "title": "Parent ID Column",
                    "default": "parent_id",
                    "description": "Column referencing the parent's ID."
                },
                "label_column": {
                    "type": "string",
                    "title": "Label / Name Column (Optional)",
                    "default": "",
                    "description": "Optional text column to use in hierarchical breadcrumbs (e.g. 'name', 'title'). If blank, uses ID."
                },
                "path_separator": {
                    "type": "string",
                    "title": "Path Breadcrumb Separator",
                    "default": " > ",
                    "description": "Separator string used to build the hierarchy path (e.g. ' > ' or ' / ')."
                },
                "max_depth_limit": {
                    "type": "integer",
                    "title": "Max Depth Limit",
                    "default": 10,
                    "description": "Maximum tree depth before stopping (prevents infinite loops in cyclic graphs)."
                },
                "generate_level_columns": {
                    "type": "boolean",
                    "title": "Generate Level Columns (level_0, level_1...)",
                    "default": True,
                    "description": "Creates explicit columns for each tier of the tree."
                }
            },
            "required": ["id_column", "parent_id_column"]
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        df = inputs.get("dataframe") if "dataframe" in inputs else inputs.get("input")
        if df is None:
            df = inputs.get("df")

        if df is None:
            return {
                "dataframe": pd.DataFrame(),
                "metrics": {"rows_processed": 0},
                "output_summary": {"title": "Hierarchical Tree Flattener", "message": "No input table provided.", "metrics": {}}
            }

        df = df.copy()
        if len(df) == 0:
            return {
                "dataframe": df,
                "metrics": {"rows_processed": 0},
                "output_summary": {"title": "Hierarchical Tree Flattener", "message": "Input table is empty.", "metrics": {}}
            }

        id_col = config.get("id_column", "id")
        parent_col = config.get("parent_id_column", "parent_id")
        label_col = config.get("label_column") or id_col
        sep = config.get("path_separator", " > ")
        max_depth = config.get("max_depth_limit", 10)
        gen_levels = config.get("generate_level_columns", True)

        if id_col not in df.columns or parent_col not in df.columns:
            return {
                "dataframe": df,
                "metrics": {"rows_processed": len(df)},
                "output_summary": {"title": "Hierarchical Tree Flattener", "message": f"Required columns '{id_col}' or '{parent_col}' not found.", "metrics": {}}
            }

        # Build lookups
        parent_lookup: Dict[Any, Any] = {}
        label_lookup: Dict[Any, str] = {}
        all_ids: Set[Any] = set()
        children_set: Set[Any] = set()

        for _, row in df.iterrows():
            node_id = row[id_col]
            p_id = row[parent_col]
            lbl = str(row[label_col]) if label_col in df.columns and pd.notna(row[label_col]) else str(node_id)

            all_ids.add(node_id)
            label_lookup[node_id] = lbl

            if pd.notna(p_id) and str(p_id).strip() != "" and p_id != node_id:
                parent_lookup[node_id] = p_id
                children_set.add(p_id)
            else:
                parent_lookup[node_id] = None

        # Trace hierarchy for each node
        depths = []
        paths = []
        root_ids = []
        is_leaf_flags = []
        ancestors_matrix: List[List[str]] = []

        max_observed_depth = 0

        for node_id in df[id_col]:
            curr = node_id
            chain = []
            visited = set()
            depth = 0

            while curr is not None and curr in all_ids and depth <= max_depth:
                if curr in visited:
                    # Circular reference detected
                    break
                visited.add(curr)
                chain.append(curr)
                curr = parent_lookup.get(curr)
                if curr is not None:
                    depth += 1

            # Reverse chain so root is at index 0
            reversed_chain = list(reversed(chain))
            max_observed_depth = max(max_observed_depth, len(reversed_chain) - 1)

            depths.append(len(reversed_chain) - 1)
            root_ids.append(reversed_chain[0] if reversed_chain else node_id)
            path_str = sep.join(label_lookup.get(nid, str(nid)) for nid in reversed_chain)
            paths.append(path_str)
            is_leaf_flags.append(bool(node_id not in children_set))

            ancestors_matrix.append([label_lookup.get(nid, str(nid)) for nid in reversed_chain])

        df["tree_depth"] = depths
        df["tree_path"] = paths
        df["tree_root_id"] = root_ids
        df["is_leaf"] = is_leaf_flags

        # Create level columns (level_0, level_1, etc.)
        if gen_levels:
            for lvl in range(max_observed_depth + 1):
                col_vals = []
                for row_chain in ancestors_matrix:
                    col_vals.append(row_chain[lvl] if lvl < len(row_chain) else None)
                df[f"level_{lvl}"] = col_vals

        output_summary = {
            "title": "Hierarchical Tree Flattener",
            "message": f"Flattened {len(df):,} hierarchy rows across {max_observed_depth + 1} levels.",
            "metrics": {
                "nodes_processed": len(df),
                "max_tree_depth": max_observed_depth,
                "leaf_nodes_count": sum(is_leaf_flags),
                "root_nodes_count": len(set(root_ids))
            }
        }

        return {
            "dataframe": df,
            "metrics": output_summary["metrics"],
            "output_summary": output_summary
        }

    def to_code(self, config: Dict[str, Any]) -> str:
        id_col = repr(config.get("id_column", "id"))
        parent_col = repr(config.get("parent_id_column", "parent_id"))
        return (
            f"# Recipe: {self.name}\n"
            f"# ID: {self.recipe_id}\n"
            f"# ID Column: {id_col}, Parent ID Column: {parent_col}\n"
            f"# Flattens parent-child hierarchy into depth and level columns\n"
        )
