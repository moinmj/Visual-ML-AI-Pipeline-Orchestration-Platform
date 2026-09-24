import pandas as pd
from typing import Dict, Any, List, Optional
from backend.app.recipes.base.recipe import BaseRecipe, RecipePort
from backend.app.recipes.flow_control.condition_utils import evaluate_composite_conditions


class SwitchRecipe(BaseRecipe):
    recipe_id = "switch"
    name = "Switch"
    version = "1.0.0"
    category = "flow_control"
    description = "Evaluates multiple branch cases and routes subsets of rows to corresponding case output ports (Case 1, Case 2, Case 3, Default)."
    input_types = ["dataframe"]
    output_types = ["dataframe", "dataframe", "dataframe", "dataframe"]

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
        RecipePort(id="case_1", label="Case 1 Output", type="dataframe", description="Rows matching Case 1 criteria"),
        RecipePort(id="case_2", label="Case 2 Output", type="dataframe", description="Rows matching Case 2 criteria"),
        RecipePort(id="case_3", label="Case 3 Output", type="dataframe", description="Rows matching Case 3 criteria"),
        RecipePort(id="default", label="Default Output", type="dataframe", description="Rows not matching any case")
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "case_1_conditions": {
                    "type": "array",
                    "title": "Case 1 Conditions",
                    "items": {
                        "type": "object",
                        "properties": {
                            "column": {"type": "string", "title": "Column"},
                            "operator": {"type": "string", "default": "=="},
                            "value": {"type": "string", "title": "Value"}
                        },
                        "required": ["column"]
                    },
                    "default": []
                },
                "case_2_conditions": {
                    "type": "array",
                    "title": "Case 2 Conditions",
                    "items": {
                        "type": "object",
                        "properties": {
                            "column": {"type": "string", "title": "Column"},
                            "operator": {"type": "string", "default": "=="},
                            "value": {"type": "string", "title": "Value"}
                        },
                        "required": ["column"]
                    },
                    "default": []
                },
                "case_3_conditions": {
                    "type": "array",
                    "title": "Case 3 Conditions",
                    "items": {
                        "type": "object",
                        "properties": {
                            "column": {"type": "string", "title": "Column"},
                            "operator": {"type": "string", "default": "=="},
                            "value": {"type": "string", "title": "Value"}
                        },
                        "required": ["column"]
                    },
                    "default": []
                }
            }
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        df = inputs.get("dataframe")
        if df is None:
            if context and isinstance(context, dict) and "dataframe" in context:
                df = context["dataframe"]
            else:
                raise ValueError("Switch recipe requires an incoming 'dataframe'.")

        c1 = config.get("case_1_conditions", [])
        c2 = config.get("case_2_conditions", [])
        c3 = config.get("case_3_conditions", [])

        mask1 = evaluate_composite_conditions(df, c1) if c1 else pd.Series(False, index=df.index)
        mask2 = evaluate_composite_conditions(df, c2) if c2 else pd.Series(False, index=df.index)
        mask3 = evaluate_composite_conditions(df, c3) if c3 else pd.Series(False, index=df.index)

        # Mutually exclusive partition (first matching case wins, like switch-case)
        case_1_df = df[mask1].copy()
        case_2_df = df[~mask1 & mask2].copy()
        case_3_df = df[~mask1 & ~mask2 & mask3].copy()
        default_df = df[~mask1 & ~mask2 & ~mask3].copy()

        summary = {
            "title": "Switch Multi-Branch Result",
            "message": f"Routed {len(df):,} rows -> Case 1: {len(case_1_df):,}, Case 2: {len(case_2_df):,}, Case 3: {len(case_3_df):,}, Default: {len(default_df):,}",
            "case_1_count": len(case_1_df),
            "case_2_count": len(case_2_df),
            "case_3_count": len(case_3_df),
            "default_count": len(default_df)
        }

        return {
            "dataframe": case_1_df if len(case_1_df) > 0 else default_df,
            "case_1": case_1_df,
            "case_2": case_2_df,
            "case_3": case_3_df,
            "default": default_df,
            "output_summary": summary,
            "metrics": {
                "total_rows": len(df),
                "case_1_rows": len(case_1_df),
                "case_2_rows": len(case_2_df),
                "case_3_rows": len(case_3_df),
                "default_rows": len(default_df)
            }
        }
