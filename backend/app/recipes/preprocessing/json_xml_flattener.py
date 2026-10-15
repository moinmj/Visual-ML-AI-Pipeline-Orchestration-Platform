"""
JSON / XML Flattener & Nester Recipe.
Sub-group: Domain-Specific Cleaning
Category: Data Preparation & Transformation (preprocessing)

Flattens deeply nested JSON or XML text/objects into structured tabular columns,
or bundles flat tabular columns back into structured nested JSON payloads.
"""
import json
import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
from backend.app.recipes.base.recipe import BaseRecipe, RecipePort


class JsonXmlFlattenerNesterRecipe(BaseRecipe):
    recipe_id = "json_xml_flattener_nester"
    name = "JSON / XML Flattener & Nester"
    version = "1.0.0"
    category = "preprocessing"
    group = "Data Preparation & Transformation"
    subgroup = "Domain-Specific Cleaning"
    description = "Flattens nested JSON/XML into columns, or nests columns back into JSON objects."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    inputs = [
        RecipePort(
            id="input",
            label="Input Table",
            type="dataframe",
            required=True,
            max_connections=1,
            description="Dataset containing JSON/XML columns to flatten or columns to nest"
        )
    ]

    outputs = [
        RecipePort(
            id="dataframe",
            label="Transformed Table",
            type="dataframe",
            description="Dataset with flattened columns or nested JSON attributes"
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "title": "Operation Mode",
                    "enum": ["flatten", "nest"],
                    "default": "flatten",
                    "description": "'flatten' unpacks nested JSON/XML into separate columns; 'nest' groups columns into structured JSON."
                },
                "format": {
                    "type": "string",
                    "title": "Data Format",
                    "enum": ["json", "xml"],
                    "default": "json",
                    "description": "Payload format for parsing ('json' or 'xml')."
                },
                "source_column": {
                    "type": "string",
                    "title": "Source Column (for Flatten mode)",
                    "default": "payload",
                    "description": "Column containing the nested JSON string, dict, or XML string."
                },
                "separator": {
                    "type": "string",
                    "title": "Hierarchy Separator",
                    "default": ".",
                    "description": "Character used to join nested keys (e.g. '.' for 'user.address.city' or '_' for 'user_address_city')."
                },
                "column_prefix": {
                    "type": "string",
                    "title": "Output Column Prefix (Flatten mode)",
                    "default": "",
                    "description": "Optional prefix added to all generated columns."
                },
                "columns_to_nest": {
                    "type": "array",
                    "title": "Columns to Nest (Nest mode)",
                    "items": {"type": "string"},
                    "default": [],
                    "description": "List of columns to combine into a nested JSON object."
                },
                "target_column": {
                    "type": "string",
                    "title": "Target Column Name (Nest mode)",
                    "default": "nested_json",
                    "description": "Name of output column where nested JSON will be saved."
                },
                "drop_source_columns": {
                    "type": "boolean",
                    "title": "Drop Original Source Columns",
                    "default": False,
                    "description": "Whether to drop the unflattened source column or nested input columns."
                }
            },
            "required": ["mode"]
        }

    def _flatten_dict(self, d: Any, parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
        items: List[tuple] = []
        if isinstance(d, dict):
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
                if isinstance(v, dict):
                    items.extend(self._flatten_dict(v, new_key, sep=sep).items())
                elif isinstance(v, list):
                    # For lists, if elements are dicts or primitives, stringify or index
                    if len(v) > 0 and isinstance(v[0], dict):
                        for idx, item in enumerate(v[:5]):  # Cap at first 5 to avoid blowup
                            items.extend(self._flatten_dict(item, f"{new_key}{sep}{idx}", sep=sep).items())
                    else:
                        items.append((new_key, json.dumps(v) if v else None))
                else:
                    items.append((new_key, v))
        elif isinstance(d, list):
            for idx, item in enumerate(d[:5]):
                new_key = f"{parent_key}{sep}{idx}" if parent_key else str(idx)
                if isinstance(item, dict):
                    items.extend(self._flatten_dict(item, new_key, sep=sep).items())
                else:
                    items.append((new_key, item))
        else:
            items.append((parent_key, d))
        return dict(items)

    def _xml_to_dict(self, element: ET.Element) -> Dict[str, Any]:
        node: Dict[str, Any] = {}
        # Include attributes
        for k, v in element.attrib.items():
            node[f"@{k}"] = v
        # Process children
        for child in element:
            child_dict = self._xml_to_dict(child)
            tag = child.tag
            if tag in node:
                if not isinstance(node[tag], list):
                    node[tag] = [node[tag]]
                node[tag].append(child_dict)
            else:
                node[tag] = child_dict
        # Text value
        text = element.text.strip() if element.text else ""
        if text:
            if node:
                node["#text"] = text
            else:
                return text
        return node

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        df = inputs.get("dataframe") if "dataframe" in inputs else inputs.get("input")
        if df is None:
            df = inputs.get("df")

        if df is None:
            return {
                "dataframe": pd.DataFrame(),
                "metrics": {"rows_processed": 0},
                "output_summary": {"title": "JSON / XML Flattener & Nester", "message": "No input table provided.", "metrics": {}}
            }

        df = df.copy()
        if len(df) == 0:
            return {
                "dataframe": df,
                "metrics": {"rows_processed": 0},
                "output_summary": {"title": "JSON / XML Flattener & Nester", "message": "Input table is empty.", "metrics": {}}
            }

        mode = config.get("mode", "flatten")
        data_format = config.get("format", "json")
        source_col = config.get("source_column", "payload")
        sep = config.get("separator", ".")
        prefix = config.get("column_prefix", "")
        cols_to_nest = config.get("columns_to_nest", [])
        target_col = config.get("target_column", "nested_json")
        drop_sources = config.get("drop_source_columns", False)

        # MODE 1: FLATTEN
        if mode == "flatten":
            if source_col not in df.columns:
                return {
                    "dataframe": df,
                    "metrics": {"rows_processed": len(df)},
                    "output_summary": {"title": "JSON / XML Flattener", "message": f"Source column '{source_col}' not found.", "metrics": {}}
                }

            flattened_rows = []
            for val in df[source_col]:
                if pd.isna(val) or val is None or val == "":
                    flattened_rows.append({})
                    continue

                parsed_obj = None
                if isinstance(val, (dict, list)):
                    parsed_obj = val
                elif isinstance(val, str):
                    s = val.strip()
                    if data_format == "xml" or (s.startswith("<") and s.endswith(">")):
                        try:
                            root = ET.fromstring(s)
                            parsed_obj = {root.tag: self._xml_to_dict(root)}
                        except Exception:
                            parsed_obj = {}
                    else:
                        try:
                            parsed_obj = json.loads(s)
                        except Exception:
                            parsed_obj = {}

                flat_dict = self._flatten_dict(parsed_obj, sep=sep) if parsed_obj else {}
                if prefix:
                    flat_dict = {f"{prefix}{k}": v for k, v in flat_dict.items()}
                flattened_rows.append(flat_dict)

            flat_df = pd.DataFrame(flattened_rows, index=df.index)
            # Combine back with original df
            if drop_sources:
                df = df.drop(columns=[source_col])

            new_columns = list(flat_df.columns)
            for c in new_columns:
                df[c] = flat_df[c]

            output_summary = {
                "title": "JSON / XML Flattener",
                "message": f"Flattened '{source_col}' into {len(new_columns)} column(s).",
                "metrics": {
                    "mode": "flatten",
                    "columns_created": len(new_columns),
                    "created_column_names": new_columns[:10],
                    "rows_processed": len(df)
                }
            }

        # MODE 2: NEST
        else:
            active_nest_cols = [c for c in cols_to_nest if c in df.columns]
            if not active_nest_cols:
                # Fallback to columns matching prefix or having separator
                active_nest_cols = [c for c in df.columns if sep in str(c)]

            if not active_nest_cols:
                return {
                    "dataframe": df,
                    "metrics": {"rows_processed": len(df)},
                    "output_summary": {"title": "JSON / XML Nester", "message": "No columns found to nest.", "metrics": {}}
                }

            def nest_row(row):
                root = {}
                for col in active_nest_cols:
                    val = row[col]
                    if pd.isna(val):
                        continue
                    parts = str(col).split(sep)
                    curr = root
                    for part in parts[:-1]:
                        if part not in curr or not isinstance(curr[part], dict):
                            curr[part] = {}
                        curr = curr[part]
                    curr[parts[-1]] = val
                return json.dumps(root)

            df[target_col] = df.apply(nest_row, axis=1)
            if drop_sources:
                df = df.drop(columns=active_nest_cols)

            output_summary = {
                "title": "JSON / XML Nester",
                "message": f"Nested {len(active_nest_cols)} column(s) into structured JSON column '{target_col}'.",
                "metrics": {
                    "mode": "nest",
                    "nested_columns_count": len(active_nest_cols),
                    "target_column": target_col,
                    "rows_processed": len(df)
                }
            }

        return {
            "dataframe": df,
            "metrics": output_summary["metrics"],
            "output_summary": output_summary
        }

    def to_code(self, config: Dict[str, Any]) -> str:
        mode = repr(config.get("mode", "flatten"))
        col = repr(config.get("source_column", "payload"))
        return (
            f"# Recipe: {self.name}\n"
            f"# ID: {self.recipe_id}\n"
            f"# Mode: {mode}, Column: {col}\n"
            f"# Flattens or nests hierarchical data structures\n"
        )
