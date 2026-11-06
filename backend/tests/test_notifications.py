import io
import json
import urllib.error
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from backend.app.core.exceptions import ExecutionException
from backend.app.recipes.base.registry import recipe_registry
from backend.app.recipes.ingestion.notifications import NotificationsRecipe


def test_notifications_registration():
    """Verify notifications recipe is registered, enabled, and in Ingestion & Orchestration > Output."""
    recipe = recipe_registry.get("notifications")
    assert isinstance(recipe, NotificationsRecipe)
    assert recipe.recipe_id == "notifications"
    assert recipe_registry.has("notifications")

    meta = recipe_registry._to_metadata_with_group(recipe)
    assert meta.enabled is True
    assert meta.group == "Ingestion & Orchestration"
    assert meta.subgroup == "Output"


def test_notifications_aliases():
    """Verify registered aliases resolve to notifications recipe."""
    assert recipe_registry.get("notification").recipe_id == "notifications"
    assert recipe_registry.get("pipeline_notification").recipe_id == "notifications"
    assert recipe_registry.get("alert_sink").recipe_id == "notifications"
    assert recipe_registry.get("webhook_notification").recipe_id == "notifications"


def test_notifications_schema_contract():
    """Verify all schema properties have input_type and description."""
    recipe = NotificationsRecipe()
    schema = recipe.get_schema()
    props = schema.get("properties", {})
    assert len(props) > 0
    for name, p_data in props.items():
        assert "input_type" in p_data, f"Property '{name}' is missing 'input_type'"
        assert "description" in p_data, f"Property '{name}' is missing 'description'"


def test_notifications_passthrough_dataframe():
    """Verify notifications is an idempotent sink: upstream DataFrame is passed through unchanged."""
    recipe = NotificationsRecipe()
    input_df = pd.DataFrame({"user_id": [101, 102], "score": [0.95, 0.88]})
    result = recipe.execute(
        inputs={"dataframe": input_df},
        config={"channel_type": "webhook", "title": "Model Deployed"},
    )
    assert result["output"].equals(input_df)
    assert result["dataframe"].equals(input_df)
    assert result["status"] == "SIMULATED"


def test_notifications_template_token_interpolation():
    """Test token replacement in title and message: {row_count}, {col_count}, {severity}."""
    recipe = NotificationsRecipe()
    input_df = pd.DataFrame({"col_a": [1, 2, 3], "col_b": [4, 5, 6]})
    config = {
        "channel_type": "webhook",
        "title": "Ingested {row_count} rows ({severity})",
        "message": "Processed table with {col_count} columns across {row_count} records.",
        "severity": "success",
    }
    result = recipe.execute(inputs={"dataframe": input_df}, config=config)
    assert result["title"] == "Ingested 3 rows (SUCCESS)"
    assert result["message"] == "Processed table with 2 columns across 3 records."
    assert result["severity"] == "success"


def test_notifications_slack_payload():
    """Test Slack formatted payload structure with blocks and color mapping."""
    recipe = NotificationsRecipe()
    input_df = pd.DataFrame({"x": [10, 20]})
    config = {
        "channel_type": "slack",
        "title": "Alert: Drift Detected",
        "message": "KS test statistic exceeded threshold.",
        "severity": "warning",
        "include_dataset_summary": True,
        "include_metrics": True,
    }
    result = recipe.execute(
        inputs={"dataframe": input_df, "metrics": {"ks_p_value": 0.0012, "accuracy": 0.94}},
        config=config,
    )
    payload = result["payload"]
    assert "attachments" in payload
    assert payload["attachments"][0]["color"] == "#FF9800"
    blocks = payload["attachments"][0]["blocks"]
    assert any(b.get("type") == "header" for b in blocks)


def test_notifications_teams_payload():
    """Test Microsoft Teams MessageCard format."""
    recipe = NotificationsRecipe()
    config = {
        "channel_type": "teams",
        "title": "Daily Pipeline Complete",
        "message": "All jobs finished without errors.",
        "severity": "success",
    }
    result = recipe.execute(inputs={}, config=config)
    payload = result["payload"]
    assert payload.get("@type") == "MessageCard"
    assert payload.get("themeColor") == "4CAF50"
    assert "sections" in payload


def test_notifications_discord_payload():
    """Test Discord rich embed payload structure."""
    recipe = NotificationsRecipe()
    config = {
        "channel_type": "discord",
        "title": "Incident Detected",
        "message": "Outlier rate is high.",
        "severity": "error",
    }
    result = recipe.execute(inputs={}, config=config)
    payload = result["payload"]
    assert "embeds" in payload
    assert len(payload["embeds"]) == 1
    # Red decimal color code for Discord
    assert payload["embeds"][0]["color"] == int("F44336", 16)


def test_notifications_email_simulated():
    """Test email notifications in simulation mode when no SMTP host is provided."""
    recipe = NotificationsRecipe()
    config = {
        "channel_type": "email",
        "email_recipients": "data-team@example.com, lead@example.com",
        "title": "Weekly Report Ready",
        "message": "See metrics attached.",
        "severity": "info",
    }
    result = recipe.execute(inputs={}, config=config)
    assert result["status"] == "SIMULATED"
    payload = result["payload"]
    assert payload["to"] == ["data-team@example.com", "lead@example.com"]
    assert payload["subject"] == "[INFO] Weekly Report Ready"


@patch("urllib.request.urlopen")
def test_notifications_successful_http_dispatch(mock_urlopen):
    """Test successful webhook dispatch when target URL returns HTTP 200."""
    recipe = NotificationsRecipe()
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.__enter__.return_value = mock_resp
    mock_urlopen.return_value = mock_resp

    config = {
        "channel_type": "webhook",
        "target_url": "https://api.example.com/alerts/webhook",
        "title": "Job Finished",
        "message": "All good.",
    }
    result = recipe.execute(inputs={}, config=config)
    assert result["status"] == "DELIVERED"
    assert result["error"] is None
    assert mock_urlopen.called


@patch("urllib.request.urlopen")
def test_notifications_http_error_fail_on_error_false(mock_urlopen):
    """Test HTTP error when fail_on_error is False (default) logs error and continues."""
    recipe = NotificationsRecipe()
    mock_urlopen.side_effect = urllib.error.HTTPError(
        url="https://api.example.com/fail",
        code=500,
        msg="Internal Server Error",
        hdrs={},
        fp=io.BytesIO(b"error"),
    )

    config = {
        "channel_type": "webhook",
        "target_url": "https://api.example.com/fail",
        "fail_on_error": False,
    }
    result = recipe.execute(inputs={}, config=config)
    assert result["status"] == "FAILED"
    assert "HTTP Error 500" in result["error"]


@patch("urllib.request.urlopen")
def test_notifications_http_error_fail_on_error_true(mock_urlopen):
    """Test HTTP error when fail_on_error is True raises ExecutionException."""
    recipe = NotificationsRecipe()
    mock_urlopen.side_effect = urllib.error.HTTPError(
        url="https://api.example.com/fail",
        code=502,
        msg="Bad Gateway",
        hdrs={},
        fp=io.BytesIO(b"error"),
    )

    config = {
        "channel_type": "webhook",
        "target_url": "https://api.example.com/fail",
        "fail_on_error": True,
    }
    with pytest.raises(ExecutionException) as exc_info:
        recipe.execute(inputs={}, config=config)
    assert "Notification dispatch failed" in str(exc_info.value)
