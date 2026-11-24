"""
Workflow End Anchor Recipe (Ingestion & Orchestration -> Control Flow & Reliability).

Acts as the exit boundary anchor for a sub-process workflow.
Captures the final dataset transformed by internal sub-process nodes and marks it as the
official return value to be handed downstream to Point B in the parent workflow.
"""

from typing import Dict, Any, List, Optional
import pandas as pd

from backend.app.recipes.base.recipe import BaseRecipe, RecipePort
from backend.app.core.logging import logger


class WorkflowEndRecipe(BaseRecipe):
    recipe_id = "workflow_end"
    name = "End Anchor"
    version = "1.0.0"
    category = "flow_control"
    group = "Ingestion & Orchestration"
    subgroup = "Control Flow & Reliability"
    description = (
        "Exit anchor for a sub-process workflow. Captures the final transformed dataset "
        "and emits it downstream to Point B in the parent workflow."
    )
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    inputs = [
        RecipePort(
            id="input",
            label="Input Table",
            type="dataframe",
            required=True,
            max_connections=1,
            description="Final dataset from upstream sub-process steps"
        )
    ]

    outputs = [
        RecipePort(
            id="output",
            label="Final Table",
            type="dataframe",
            description="Final dataset emitted to parent pipeline (Point B)"
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "title": "Anchor Description / Return Notes",
                    "default": "Sub-process output exit point",
                    "description": "Notes describing returned columns and guarantees to caller.",
                    "input_type": "text"
                }
                # 'pass_through' was removed: the End Anchor always hands its input to the parent
                # unchanged, so the checkbox never did anything. Saved configs that still contain
                # it are ignored.
            }
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        in_df = None
        if inputs:
            for k in ("dataframe", "input"):
                if isinstance(inputs.get(k), pd.DataFrame):
                    in_df = inputs[k]
                    break

        if in_df is None and context and isinstance(context, dict):
            for k in ("dataframe", "input"):
                val = context.get(k)
                if isinstance(val, pd.DataFrame):
                    in_df = val
                    break

        if in_df is None:
            in_df = pd.DataFrame()

        out_df = in_df.copy()

        # Hand the final table to the Add Workflow node that started this run. It puts a fresh
        # dict in the context for every call; the executor only shallow-copies the context,
        # so writing into that dict reaches the caller (and only that call).
        if isinstance(context, dict):
            holder = context.get("_sub_workflow_result")
            if isinstance(holder, dict):
                holder["dataframe"] = out_df.copy()
                holder["end_node_completed"] = True

        return {
            "dataframe": out_df,
            "output_summary": {
                "title": "End Anchor (Sub-Process Exit)",
                "status": "COMPLETED",
                "records": len(out_df),
                "columns": list(out_df.columns),
                "message": f"Successfully captured {len(out_df)} rows for parent pipeline (Point B)."
            },
            "metrics": {
                "records_emitted": len(out_df),
                "columns_count": len(out_df.columns),
                "anchor_type": "end"
            }
        }

