import pandas as pd
from typing import Dict, Any, List, Optional
from backend.app.recipes.base.recipe import BaseRecipe, RecipePort
from backend.app.recipes.flow_control.condition_utils import evaluate_composite_conditions


class IfConditionRecipe(BaseRecipe):
    recipe_id = "if_condition"
    name = "IF / Condition"
    version = "1.0.0"
    category = "flow_control"
    description = "Routes rows into two distinct output branches ('true' and 'false') based on dynamic column rules (matches n8n / Alteryx Filter)."
    input_types = ["dataframe"]
    output_types = ["dataframe", "dataframe"]

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
            id="true",
            label="True (If)",
            type="dataframe",
            description="Rows matching the condition rules"
        ),
        RecipePort(
            id="false",
            label="False (Else)",
            type="dataframe",
            description="Rows not matching the condition rules"
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "combine_with": {
                    "type": "string",
                    "title": "Combine Conditions",
                    "enum": ["AND", "OR"],
                    "default": "AND",
                    "description": "Combine multiple rules with AND (all must match) or OR (any can match)."
                },
                "conditions": {
                    "type": "array",
                    "title": "Conditions",
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
                            "value": {"type": "string", "title": "Comparison Value"}
                        },
                        "required": ["column", "operator"]
                    },
                    "default": [{"column": "", "operator": "==", "value": ""}],
                    "description": "Rules to evaluate against each row."
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
                raise ValueError("IF / Condition recipe requires an incoming 'dataframe'.")

        conditions = config.get("conditions", [])
        combine_with = config.get("combine_with", "AND")

        mask = evaluate_composite_conditions(df, conditions, combine_with=combine_with)
        df_true = df[mask].copy()
        df_false = df[~mask].copy()

        true_count = len(df_true)
        false_count = len(df_false)
        total_count = len(df)

        true_pct = round((true_count / max(total_count, 1)) * 100.0, 1)
        false_pct = round((false_count / max(total_count, 1)) * 100.0, 1)

        summary = {
            "title": "IF / Condition Branching",
            "message": f"Split {total_count:,} total rows into True: {true_count:,} ({true_pct}%) and False: {false_count:,} ({false_pct}%).",
            "true_rows": true_count,
            "false_rows": false_count
        }

        return {
            "dataframe": df_true,  # Primary default output
            "true": df_true,
            "false": df_false,
            "output_summary": summary,
            "metrics": {
                "total_rows": total_count,
                "true_rows": true_count,
                "false_rows": false_count,
                "true_percentage": true_pct,
                "false_percentage": false_pct
            }
        }
