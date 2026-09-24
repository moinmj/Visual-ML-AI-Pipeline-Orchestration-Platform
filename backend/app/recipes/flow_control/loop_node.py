import pandas as pd
from typing import Dict, Any, List, Optional
from backend.app.recipes.base.recipe import BaseRecipe, RecipePort


class LoopBatchRecipe(BaseRecipe):
    recipe_id = "loop"
    name = "Loop / Batch Iterator"
    version = "1.0.0"
    category = "flow_control"
    description = "Chunks large datasets into batches or limits iterations for batch processing (matches n8n Loop Over Items)."
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
            id="batch_output",
            label="Batch Output",
            type="dataframe",
            description="Current batch of items to process"
        ),
        RecipePort(
            id="completed",
            label="Complete",
            type="dataframe",
            description="All processed items"
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "batch_size": {
                    "type": "integer",
                    "title": "Batch Size",
                    "default": 100,
                    "minimum": 1,
                    "description": "Number of rows per processing batch."
                },
                "max_rows": {
                    "type": "integer",
                    "title": "Max Rows Limit",
                    "default": 1000,
                    "minimum": 1,
                    "description": "Cap on maximum rows to iterate over (safeguard)."
                }
            }
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        df = inputs.get("dataframe")
        if df is None:
            if context and isinstance(context, dict) and "dataframe" in context:
                df = context["dataframe"]
            else:
                raise ValueError("Loop recipe requires an incoming 'dataframe'.")

        batch_size = int(config.get("batch_size", 100))
        max_rows = int(config.get("max_rows", 1000))

        capped_df = df.head(max_rows).copy()
        batch_df = capped_df.head(batch_size).copy()
        total_batches = (len(capped_df) + batch_size - 1) // max(batch_size, 1)

        summary = {
            "title": "Loop Batch Created",
            "message": f"Processed {len(capped_df):,} rows into {total_batches:,} batches of {batch_size:,} rows. Active batch has {len(batch_df):,} rows.",
            "active_batch_size": len(batch_df),
            "total_batches": total_batches
        }

        return {
            "dataframe": batch_df,
            "batch_output": batch_df,
            "completed": capped_df,
            "output_summary": summary,
            "metrics": {
                "batch_size": batch_size,
                "active_batch_rows": len(batch_df),
                "total_rows": len(capped_df),
                "total_batches": total_batches
            }
        }
