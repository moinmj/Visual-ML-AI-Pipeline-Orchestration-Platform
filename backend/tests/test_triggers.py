import pytest
import pandas as pd
from backend.app.recipes.triggers.webhook_trigger import WebhookTriggerRecipe
from backend.app.recipes.triggers.cron_trigger import CronScheduleTriggerRecipe


def test_webhook_trigger_recipe():
    recipe = WebhookTriggerRecipe()
    assert recipe.recipe_id == "webhook_trigger"
    assert recipe.category == "triggers"
    
    # Test with custom raw payload
    raw_payload = [{"user_id": 1, "amount": 100.5}, {"user_id": 2, "amount": 250.0}]
    output = recipe.execute(
        inputs={},
        config={"webhook_path": "finance_events"},
        context={"raw_payload": raw_payload}
    )
    
    assert "dataframe" in output
    assert len(output["dataframe"]) == 2
    assert "amount" in output["dataframe"].columns
    assert output["trigger_metadata"]["webhook_path"] == "finance_events"


def test_cron_trigger_recipe():
    recipe = CronScheduleTriggerRecipe()
    assert recipe.recipe_id == "cron_trigger"
    assert recipe.category == "triggers"
    
    # 1. Direct cron expression
    output = recipe.execute(
        inputs={},
        config={"cron_expression": "0 12 * * *", "timezone": "UTC"}
    )
    assert "dataframe" in output
    assert len(output["dataframe"]) >= 1
    assert output["trigger_metadata"]["cron"] == "0 12 * * *"

    # 2. Preset auto-mapping: Every 15 Minutes -> */15 * * * *
    out_15 = recipe.execute(
        inputs={},
        config={"interval_preset": "Every 15 Minutes", "cron_expression": "0 0 * * *"}
    )
    assert out_15["trigger_metadata"]["cron"] == "*/15 * * * *"


def test_cron_matcher_and_scheduler_evaluation():
    from backend.app.workflows.scheduler import is_cron_due, match_cron_field, workflow_scheduler
    import datetime

    # 1. 15-minute cron check
    dt_15 = datetime.datetime(2026, 10, 11, 12, 15, 0)
    dt_16 = datetime.datetime(2026, 10, 11, 12, 16, 0)
    assert is_cron_due("*/15 * * * *", dt_15) is True
    assert is_cron_due("*/15 * * * *", dt_16) is False

    # 2. Daily midnight check
    dt_midnight = datetime.datetime(2026, 10, 11, 0, 0, 0)
    assert is_cron_due("0 0 * * *", dt_midnight) is True
    assert is_cron_due("0 0 * * *", dt_15) is False

    # 3. Field matcher steps and ranges
    assert match_cron_field("*/15", 30) is True
    assert match_cron_field("*/15", 7) is False
    assert match_cron_field("1-5", 3) is True
    assert match_cron_field("1-5", 6) is False
    assert match_cron_field("10,20,30", 20) is True

    # 4. Scheduler instance check
    assert workflow_scheduler.check_interval_seconds == 30


@pytest.mark.asyncio
async def test_scheduler_force_tick_and_resilience():
    from backend.app.workflows.models import Workflow
    from backend.app.workflows.scheduler import workflow_scheduler
    
    # Test _extract_cron_config with dict nodes
    wf = Workflow(
        id="test_wf_cron_1",
        name="Cron Test Pipeline",
        is_active=True,
        nodes=[{"id": "c1", "recipe_id": "cron_trigger"}],
        node_configs={"c1": {"recipe_id": "cron_trigger", "config": {"interval_preset": "Every 15 Minutes"}}}
    )
    cfg = workflow_scheduler._extract_cron_config(wf)
    assert cfg is not None
    assert cfg["cron_expression"] == "*/15 * * * *"

    # Test _extract_cron_config with stringified JSON (defensive parsing)
    import json
    wf_str = Workflow(
        id="test_wf_cron_2",
        name="Cron Test Pipeline Str",
        is_active=True,
        nodes=json.dumps([{"id": "c2", "recipe_id": "cron_trigger"}]),
        node_configs=json.dumps({"c2": {"recipe_id": "cron_trigger", "config": {"cron_expression": "*/15 * * * *"}}})
    )
    cfg_str = workflow_scheduler._extract_cron_config(wf_str)
    assert cfg_str is not None
    assert cfg_str["cron_expression"] == "*/15 * * * *"


