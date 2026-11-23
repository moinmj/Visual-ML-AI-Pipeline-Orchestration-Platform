"""
Workflow Start Anchor Recipe (Ingestion & Orchestration -> Control Flow & Reliability).

Acts as the entry boundary anchor for a sub-process workflow.
When invoked inside a parent workflow via 'Add Workflow', it receives the upstream
DataFrame from Point A and injects it into downstream sub-workflow nodes.
When tested in isolation on the canvas, it generates realistic mock data.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime, timezone

from backend.app.recipes.base.recipe import BaseRecipe, RecipePort
from backend.app.core.logging import logger


class WorkflowStartRecipe(BaseRecipe):
    recipe_id = "workflow_start"
    name = "Start Anchor"
    version = "1.0.0"
    category = "flow_control"
    group = "Ingestion & Orchestration"
    subgroup = "Control Flow & Reliability"
    description = (
        "Entry anchor for a sub-process workflow. Captures data passed from the parent "
        "pipeline (Point A) or generates mock data in standalone test mode."
    )
    input_types = []
    output_types = ["dataframe"]

    inputs = []

    outputs = [
        RecipePort(
            id="output",
            label="Output Table",
            type="dataframe",
            description="Dataset injected into the sub-process"
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "title": "Anchor Description / Contract Notes",
                    "default": "Sub-process input entry point",
                    "description": "Notes describing expected columns and types from the caller.",
                    "input_type": "text"
                },
                "mock_record_count": {
                    "type": "integer",
                    "title": "Standalone Mock Record Count",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 500,
                    "description": "Number of sample rows to generate when testing this sub-process standalone.",
                    "input_type": "number"
                }
            }
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        mock_count = int(config.get("mock_record_count", 5))

        # Called by a parent through Add Workflow: always use the parent's table as-is,
        # even when it has 0 rows (an empty result upstream must stay empty, never become
        # sample data). Sample rows are only for running this sub-process on its own.
        called_by_parent = isinstance(context, dict) and "sub_workflow_input" in context
        in_df = None
        if called_by_parent:
            val = context.get("sub_workflow_input")
            in_df = val if isinstance(val, pd.DataFrame) else pd.DataFrame()
        else:
            if inputs:
                for k in ("dataframe", "input"):
                    if isinstance(inputs.get(k), pd.DataFrame) and not inputs[k].empty:
                        in_df = inputs[k]
                        break
            if in_df is None and isinstance(context, dict):
                for k in ("parent_dataframe", "dataframe"):
                    val = context.get(k)
                    if isinstance(val, pd.DataFrame) and not val.empty:
                        in_df = val
                        break

        mode = "INJECTED"
        if in_df is not None:
            out_df = in_df.copy()
            status_msg = (
                f"Injected {len(out_df)} rows from parent pipeline (Point A)."
                if len(out_df) else "Parent pipeline passed an empty table (0 rows); passing it on unchanged."
            )
        else:
            # Standalone test fallback
            mode = "STANDALONE_MOCK"
            out_df = pd.DataFrame([
                {
                    "id": i,
                    "customer_id": f"CUST-00{100 + i}",
                    "name": ["Alpha Corp", "Beta Solutions", "Gamma Global", "Delta Systems", "Epsilon Tech"][i % 5],
                    "status": "Active" if i % 2 == 0 else "Pending",
                    "amount": 1250.0 + (i * 350.0),
                    "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
                }
                for i in range(1, mock_count + 1)
            ])
            status_msg = f"Running in standalone test mode with {len(out_df)} mock rows."

        return {
            "dataframe": out_df,
            "output_summary": {
                "title": "Start Anchor (Sub-Process Entry)",
                "status": mode,
                "records": len(out_df),
                "columns": list(out_df.columns),
                "message": status_msg
            },
            "metrics": {
                "records_injected": len(out_df),
                "mode": mode,
                "anchor_type": "start"
            }
        }

