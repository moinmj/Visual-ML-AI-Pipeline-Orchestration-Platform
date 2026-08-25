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
    
    output = recipe.execute(
        inputs={},
        config={"cron_expression": "0 12 * * *", "timezone": "UTC"}
    )
    
    assert "dataframe" in output
    assert len(output["dataframe"]) >= 1
    assert output["trigger_metadata"]["cron"] == "0 12 * * *"
