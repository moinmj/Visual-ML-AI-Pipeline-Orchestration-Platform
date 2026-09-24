import pandas as pd
from typing import Dict, Any, List, Optional
from backend.app.recipes.base.recipe import BaseRecipe, RecipePort
from backend.app.recipes.flow_control.condition_utils import evaluate_composite_conditions


class RowFilterRecipe(BaseRecipe):
    recipe_id = "row_filter"
    name = "Filter Rows"
    version = "1.0.0"
    category = "flow_control"
    description = "Filters tabular rows based on custom rules (e.g. status == 'ACTIVE', amount > 100). Keeps or drops matching records."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    inputs = [
        RecipePort(
            id="input",
            label="Input Data",
            type="dataframe",
            required=True,
            max_connections=1,
            description="Incoming tabular dataset"
        )
    ]

    outputs = [
        RecipePort(
            id="output",
            label="Filtered Data",
            type="dataframe",
            description="Dataset containing filtered rows"
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "title": "Filter Action",
                    "enum": ["keep", "drop"],
                    "default": "keep",
                    "description": "'keep' retains rows matching conditions; 'drop' removes matching rows."
                },
                "combine_with": {
                    "type": "string",
                    "title": "Combine Conditions",
                    "enum": ["AND", "OR"],
                    "default": "AND"
                },
                "conditions": {
                    "type": "array",
                    "title": "Filter Rules",
                    "items": {
                        "type": "object",
                        "properties": {
                            "column": {"type": "string", "title": "Column Name"},
                            "operator": {
                                "type": "string",
                                "title": "Operator",
                                "enum": [
                                    "==", "!=", ">", ">=", "<", "<=",
                                    "contains", "not_contains", "starts_with", "ends_with",
                                    "in", "not_in", "is_null", "not_null"
                                ],
                                "default": "=="
                            },
                            "value": {"type": "string", "title": "Value"}
                        },
                        "required": ["column", "operator"]
                    },
                    "default": [{"column": "", "operator": "==", "value": ""}]
                }
            },
            "required": ["conditions"]
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        df = inputs.get("dataframe")
        if df is None:
            if context and isinstance(context, dict) and "dataframe" in context:
                df = context["dataframe"]
            else:
                raise ValueError("Row Filter recipe requires an incoming 'dataframe'.")

        action = str(config.get("action", "keep")).lower().strip()
        conditions = config.get("conditions", [])
        combine_with = config.get("combine_with", "AND")

        mask = evaluate_composite_conditions(df, conditions, combine_with=combine_with)

        if action == "drop":
            df_filtered = df[~mask].copy()
        else:
            df_filtered = df[mask].copy()

        initial_rows = len(df)
        retained_rows = len(df_filtered)
        dropped_rows = initial_rows - retained_rows

        summary = {
            "title": "Row Filter Result",
            "message": f"Retained {retained_rows:,} of {initial_rows:,} rows ({dropped_rows:,} dropped).",
            "initial_rows": initial_rows,
            "retained_rows": retained_rows,
            "dropped_rows": dropped_rows
        }

        return {
            "dataframe": df_filtered,
            "output_summary": summary,
            "metrics": {
                "initial_rows": initial_rows,
                "retained_rows": retained_rows,
                "dropped_rows": dropped_rows,
                "retention_rate_pct": round((retained_rows / max(initial_rows, 1)) * 100.0, 2)
            }
        }
