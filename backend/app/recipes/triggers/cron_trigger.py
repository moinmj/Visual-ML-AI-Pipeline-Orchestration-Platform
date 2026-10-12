from typing import Dict, Any, List, Optional
import pandas as pd
from backend.app.recipes.base.recipe import BaseRecipe

PRESET_MAP = {
    "Every 15 Minutes": "*/15 * * * *",
    "Hourly": "0 * * * *",
    "Daily at Midnight": "0 0 * * *",
    "Weekly on Sunday": "0 0 * * 0",
}


class CronScheduleTriggerRecipe(BaseRecipe):
    """
    Cron / Periodic Schedule Trigger Recipe (n8n / Boomi style).
    Triggers automated pipeline executions on a recurring cron schedule.
    """

    recipe_id = "cron_trigger"
    name = "Cron Schedule Trigger"
    version = "1.1.0"
    category = "triggers"
    description = "Triggers automated pipeline executions on recurring intervals (e.g. daily training or hourly anomaly scans)."
    input_types = []  # Root trigger node
    output_types = ["dataframe"]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "cron_expression": {
                    "type": "string",
                    "title": "Cron Schedule Expression",
                    "default": "0 0 * * *",
                    "description": "Standard 5-part cron syntax: 'minute hour day month day-of-week' (e.g. '*/15 * * * *' = Every 15 min, '0 0 * * *' = Daily Midnight)"
                },
                "interval_preset": {
                    "type": "string",
                    "title": "Quick Interval Preset",
                    "enum": ["Custom Cron", "Every 15 Minutes", "Hourly", "Daily at Midnight", "Weekly on Sunday"],
                    "default": "Every 15 Minutes"
                },
                "timezone": {
                    "type": "string",
                    "title": "Timezone",
                    "enum": ["UTC", "America/New_York", "Europe/London", "Asia/Kolkata", "Asia/Tokyo"],
                    "default": "UTC"
                }
            }
        }

    def _resolve_cron_expression(self, config: Dict[str, Any]) -> str:
        preset = config.get("interval_preset", "")
        expr = config.get("cron_expression", "").strip()

        # If user picked a preset, automatically map it if expr is empty or left at midnight default
        if preset in PRESET_MAP:
            if not expr or expr == "0 0 * * *":
                return PRESET_MAP[preset]
        return expr or "0 0 * * *"

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        resolved_cron = self._resolve_cron_expression(config)
        tz = config.get("timezone", "UTC")

        df = inputs.get("dataframe")
        if df is None:
            if context and isinstance(context, dict) and "dataframe" in context:
                df = context["dataframe"]
            else:
                import datetime
                df = pd.DataFrame([{
                    "cron_trigger_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "schedule": resolved_cron,
                    "timezone": tz,
                    "status": "scheduled_fire"
                }])

        return {
            "dataframe": df,
            "trigger_metadata": {
                "type": "cron_schedule",
                "cron": resolved_cron,
                "timezone": tz
            }
        }

    def to_code(self, config: Dict[str, Any]) -> str:
        cron = self._resolve_cron_expression(config)
        tz = config.get("timezone", "UTC")
        return f"# Recurring Cron Scheduler: '{cron}' ({tz})\n# Triggered automatically by platform worker pool"
