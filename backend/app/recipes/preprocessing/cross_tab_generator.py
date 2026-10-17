"""
Cross-Tab / Contingency Table Generator Recipe.
Sub-group: Domain-Specific Cleaning
Category: Data Preparation & Transformation (preprocessing)

Builds multi-dimensional cross-tabulation and contingency tables between two or more
categorical features with margin totals, percentage normalization, and value aggregations.
"""
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
from backend.app.recipes.base.recipe import BaseRecipe, RecipePort


class CrossTabGeneratorRecipe(BaseRecipe):
    recipe_id = "cross_tab_generator"
    name = "Cross-Tab / Contingency Table Generator"
    version = "1.0.0"
    category = "preprocessing"
    group = "Data Preparation & Transformation"
    subgroup = "Domain-Specific Cleaning"
    description = "Builds cross-tabulation (contingency) tables between two or more columns with percentages, totals, and aggregations."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    inputs = [
        RecipePort(
            id="input",
            label="Input Table",
            type="dataframe",
            required=True,
            max_connections=1,
            description="Dataset containing categorical and metric fields to cross-tabulate"
        )
    ]

    outputs = [
        RecipePort(
            id="dataframe",
            label="Cross-Tab Table",
            type="dataframe",
            description="Generated contingency matrix with row/column dimensions and totals"
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "index_column": {
                    "type": "string",
                    "title": "Row Dimension Column",
                    "default": "",
                    "description": "Primary column to group by in rows."
                },
                "index_columns": {
                    "type": "array",
                    "title": "Multiple Row Dimension Columns",
                    "items": {"type": "string"},
                    "default": [],
                    "description": "Optional list of multiple columns for nested row hierarchy."
                },
                "column_column": {
                    "type": "string",
                    "title": "Column Dimension Column",
                    "default": "",
                    "description": "Primary column to pivot into columns."
                },
                "column_columns": {
                    "type": "array",
                    "title": "Multiple Column Dimension Columns",
                    "items": {"type": "string"},
                    "default": [],
                    "description": "Optional list of multiple columns for nested column headers."
                },
                "value_column": {
                    "type": "string",
                    "title": "Aggregation Value Column (Optional)",
                    "default": "",
                    "description": "Numeric column to aggregate. If omitted, computes frequency counts."
                },
                "aggfunc": {
                    "type": "string",
                    "title": "Aggregation Function",
                    "enum": ["count", "sum", "mean", "min", "max", "median"],
                    "default": "count",
                    "description": "Function to aggregate values ('count', 'sum', 'mean', etc.)."
                },
                "normalize": {
                    "type": "string",
                    "title": "Percentage Normalization",
                    "enum": ["none", "all", "index", "columns"],
                    "default": "none",
                    "description": "'none' gives raw counts/sums; 'all' percentages of grand total; 'index' row percentages; 'columns' column percentages."
                },
                "margins": {
                    "type": "boolean",
                    "title": "Include Totals / Margins",
                    "default": True,
                    "description": "Adds total row and column margins."
                },
                "margins_name": {
                    "type": "string",
                    "title": "Margins Label",
                    "default": "Total",
                    "description": "Name for the margin row/column."
                }
            },
            "required": []
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        df = inputs.get("dataframe") if "dataframe" in inputs else inputs.get("input")
        if df is None:
            df = inputs.get("df")

        if df is None:
            return {
                "dataframe": pd.DataFrame(),
                "metrics": {"row_count": 0, "col_count": 0},
                "output_summary": {"title": "Cross-Tab Generator", "message": "No input table provided.", "metrics": {}}
            }

        df = df.copy()
        if len(df) == 0:
            return {
                "dataframe": df,
                "metrics": {"row_count": 0, "col_count": 0},
                "output_summary": {"title": "Cross-Tab Generator", "message": "Input table is empty.", "metrics": {}}
            }

        idx_col = config.get("index_column")
        idx_cols = config.get("index_columns", [])
        col_col = config.get("column_column")
        col_cols = config.get("column_columns", [])
        val_col = config.get("value_column")
        aggfunc_name = config.get("aggfunc", "count")
        norm_mode = config.get("normalize", "none")
        margins = config.get("margins", True)
        margins_name = config.get("margins_name", "Total")

        # Resolve index list
        active_index = []
        if idx_col and idx_col in df.columns:
            active_index.append(idx_col)
        for c in idx_cols:
            if c in df.columns and c not in active_index:
                active_index.append(c)

        # Resolve column list
        active_columns = []
        if col_col and col_col in df.columns:
            active_columns.append(col_col)
        for c in col_cols:
            if c in df.columns and c not in active_columns:
                active_columns.append(c)

        # Fallback to first two categorical/object columns if none provided
        if not active_index or not active_columns:
            cats = [c for c in df.columns if df[c].dtype == "object" or df[c].dtype.name in ["str", "string", "category"]]
            if not active_index and cats:
                active_index.append(cats[0])
            if not active_columns and len(cats) > 1:
                active_columns.append(cats[1])

        if not active_index or not active_columns:
            return {
                "dataframe": df,
                "metrics": {"row_count": 0, "col_count": 0},
                "output_summary": {"title": "Cross-Tab Generator", "message": "Please specify at least one row and one column dimension.", "metrics": {}}
            }

        # Setup index series / lists
        index_data = [df[c] for c in active_index]
        cols_data = [df[c] for c in active_columns]

        # Normalize argument
        norm_arg = False if norm_mode == "none" else norm_mode

        # Execute crosstab
        if val_col and val_col in df.columns and aggfunc_name != "count":
            values_series = pd.to_numeric(df[val_col], errors="coerce").fillna(0.0)
            ct = pd.crosstab(
                index=index_data,
                columns=cols_data,
                values=values_series,
                aggfunc=aggfunc_name,
                margins=margins,
                margins_name=margins_name,
                normalize=norm_arg
            )
        else:
            ct = pd.crosstab(
                index=index_data,
                columns=cols_data,
                margins=margins,
                margins_name=margins_name,
                normalize=norm_arg
            )

        if norm_mode != "none":
            ct = ct.round(4)

        # Flatten multi-index columns if present
        if isinstance(ct.columns, pd.MultiIndex):
            ct.columns = ["_".join(str(lvl) for lvl in col_tuple if str(lvl)) for col_tuple in ct.columns]
        else:
            ct.columns = [str(c) for c in ct.columns]

        # Reset index into proper tabular columns
        ct_reset = ct.reset_index()

        output_summary = {
            "title": "Cross-Tab / Contingency Table",
            "message": f"Generated {len(ct_reset)} row(s) x {len(ct_reset.columns)} column(s) cross-tab between {active_index} and {active_columns}.",
            "metrics": {
                "rows_in_crosstab": len(ct_reset),
                "columns_in_crosstab": len(ct_reset.columns),
                "row_dimensions": active_index,
                "col_dimensions": active_columns,
                "normalization": norm_mode,
                "has_margins": margins
            }
        }

        return {
            "dataframe": ct_reset,
            "metrics": output_summary["metrics"],
            "output_summary": output_summary
        }

    def to_code(self, config: Dict[str, Any]) -> str:
        idx = repr(config.get("index_column", ""))
        cols = repr(config.get("column_column", ""))
        return (
            f"# Recipe: {self.name}\n"
            f"# ID: {self.recipe_id}\n"
            f"# Index: {idx}, Columns: {cols}\n"
            f"# Generates cross-tabulation contingency matrix\n"
        )
