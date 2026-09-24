import time
import pandas as pd
from typing import Dict, Any, List, Optional
from backend.app.recipes.base.recipe import BaseRecipe, RecipePort


class DelayRecipe(BaseRecipe):
    recipe_id = "delay"
    name = "Delay / Wait"
    version = "1.0.0"
    category = "flow_control"
    description = "Pauses pipeline execution for a configured duration before passing data downstream (rate limiting / throttler / scheduling)."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    inputs = [
        RecipePort(
            id="input",
            label="Input",
            type="dataframe",
            required=False,
            max_connections=1,
            description="Incoming data to pass through"
        )
    ]

    outputs = [
        RecipePort(
            id="output",
            label="Output",
            type="dataframe",
            description="Data passed through after delay"
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "duration": {
                    "type": "number",
                    "title": "Delay Duration",
                    "default": 1.0,
                    "minimum": 0.1,
                    "maximum": 300.0,
                    "description": "Number of seconds or minutes to pause."
                },
                "unit": {
                    "type": "string",
                    "title": "Time Unit",
                    "enum": ["seconds", "minutes", "milliseconds"],
                    "default": "seconds"
                }
            },
            "required": ["duration"]
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        duration = float(config.get("duration", 1.0))
        unit = str(config.get("unit", "seconds")).lower().strip()

        if unit == "minutes":
            delay_sec = min(duration * 60.0, 300.0)  # Safe guard maximum 5 minutes
        elif unit == "milliseconds":
            delay_sec = duration / 1000.0
        else:
            delay_sec = min(duration, 300.0)

        time.sleep(delay_sec)

        # Passthrough any input artifacts
        df = inputs.get("dataframe")
        if df is None and context and isinstance(context, dict):
            df = context.get("dataframe")

        summary = {
            "title": "Delay Finished",
            "message": f"Paused workflow execution for {duration} {unit} ({delay_sec:.2f}s total)."
        }

        result = {
            "output_summary": summary,
            "metrics": {
                "delayed_seconds": delay_sec,
                "requested_duration": duration,
                "unit": unit
            }
        }
        if df is not None:
            result["dataframe"] = df

        return result
