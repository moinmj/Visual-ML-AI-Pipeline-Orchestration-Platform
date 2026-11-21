"""
Unit tests for WebSocket Listener Recipe.
Covers:
1. Default open URL schema (wss://echo.websocket.org).
2. Flatten JSON toggle:
   - flatten_json=True unpacks live/benchmark JSON fields into separate columns.
   - flatten_json=False retains entire raw message inside single 'payload' text column.
3. Auto-fallback to simulated benchmark records when live connection fails or times out.
4. Proper streaming card title ('WebSocket Stream Ingestion Metrics'), tag ('STREAMING'), and clean integer counts.
5. Code generation respecting flatten toggle.
"""

import json
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from backend.app.recipes.ingestion.websocket_listener import WebSocketListenerRecipe


def test_websocket_listener_schema_defaults():
    recipe = WebSocketListenerRecipe()
    schema = recipe.get_schema()
    props = schema["properties"]

    assert props["url"]["default"] == "wss://echo.websocket.org"
    assert props["flatten_json"]["default"] is True
    assert "payload" in props["flatten_json"]["description"]


def test_websocket_listener_flatten_json_true():
    recipe = WebSocketListenerRecipe()
    config = {
        "url": "wss://echo.websocket.org",
        "max_messages": 10,
        "flatten_json": True,
        "simulation_mode": True,
    }
    result = recipe.execute({}, config)
    df = result["dataframe"]

    assert len(df) == 10
    # Top-level unpacked columns
    assert "symbol" in df.columns
    assert "price" in df.columns
    assert "quantity" in df.columns
    assert "payload" not in df.columns


def test_websocket_listener_flatten_json_false():
    recipe = WebSocketListenerRecipe()
    config = {
        "url": "wss://echo.websocket.org",
        "max_messages": 10,
        "flatten_json": False,
        "simulation_mode": True,
    }
    result = recipe.execute({}, config)
    df = result["dataframe"]

    assert len(df) == 10
    # Single raw text payload column
    assert "payload" in df.columns
    # Ensure individual fields are NOT unpacked into separate columns
    assert "price" not in df.columns
    assert "quantity" not in df.columns

    # Verify payload contains valid parseable JSON string
    first_payload = json.loads(df["payload"].iloc[0])
    assert "price" in first_payload
    assert "symbol" in first_payload


def test_websocket_listener_auto_fallback_on_connection_failure():
    """When a remote WebSocket host is blocked, unreachable, or times out, node auto-switches to demo data."""
    recipe = WebSocketListenerRecipe()
    config = {
        "url": "wss://unreachable-blocked-domain-test.local:9999/ws",
        "max_messages": 5,
        "timeout_seconds": 1,
        "simulation_mode": False,  # live mode attempted
    }
    # Should NOT raise ValidationException; should gracefully auto-fallback to benchmark records
    result = recipe.execute({}, config)

    assert result["output_summary"]["status"] == "SIMULATED"
    assert "Auto-switched" in result["output_summary"]["message"] or "failed" in result["output_summary"]["message"]
    assert len(result["dataframe"]) == 5
    assert result["metrics"]["messages_captured"] == 5


def test_websocket_listener_streaming_metrics_and_clean_integer_counts():
    recipe = WebSocketListenerRecipe()
    config = {
        "url": "wss://echo.websocket.org",
        "max_messages": 12,
        "simulation_mode": True,
    }
    result = recipe.execute({}, config)
    metrics = result["metrics"]
    summary = result["output_summary"]

    # Verify Streaming Header and Tag
    assert metrics["title"] == "WebSocket Stream Ingestion Metrics"
    assert metrics["tag"] == "STREAMING"
    assert metrics["task_type"] == "streaming"
    assert summary["title"] == "WebSocket Stream Ingestion Metrics"
    assert summary["tag"] == "STREAMING"

    # Verify Clean Counts (strictly integer type, avoiding float 12.0)
    assert isinstance(metrics["messages_captured"], int)
    assert isinstance(metrics["columns_count"], int)
    assert isinstance(metrics["duration_ms"], int)
    assert metrics["messages_captured"] == 12
    assert not isinstance(metrics["messages_captured"], float)


def test_websocket_listener_live_mock_with_flatten_toggle():
    recipe = WebSocketListenerRecipe()
    mock_ws = MagicMock()
    mock_messages = [
        json.dumps({"sensor_id": "SN-01", "temp": 24.5, "humidity": 60}),
        json.dumps({"sensor_id": "SN-02", "temp": 25.1, "humidity": 58}),
    ]
    mock_ws.recv.side_effect = mock_messages + [None]

    # Test 1: Live with flatten_json = False
    with patch("websocket.create_connection", return_value=mock_ws):
        res_unflattened = recipe.execute({}, {
            "url": "wss://echo.websocket.org",
            "max_messages": 2,
            "flatten_json": False,
            "simulation_mode": False
        })
        df_unflat = res_unflattened["dataframe"]
        assert len(df_unflat) == 2
        assert "payload" in df_unflat.columns
        assert "temp" not in df_unflat.columns
        assert "SN-01" in df_unflat["payload"].iloc[0]

    # Test 2: Live with flatten_json = True
    mock_ws2 = MagicMock()
    mock_ws2.recv.side_effect = mock_messages + [None]
    with patch("websocket.create_connection", return_value=mock_ws2):
        res_flattened = recipe.execute({}, {
            "url": "wss://echo.websocket.org",
            "max_messages": 2,
            "flatten_json": True,
            "simulation_mode": False
        })
        df_flat = res_flattened["dataframe"]
        assert len(df_flat) == 2
        assert "sensor_id" in df_flat.columns
        assert "temp" in df_flat.columns
        assert df_flat["temp"].iloc[0] == 24.5


def test_websocket_listener_to_code():
    recipe = WebSocketListenerRecipe()
    code_flat = recipe.to_code({"url": "wss://echo.websocket.org", "flatten_json": True})
    assert "wss://echo.websocket.org" in code_flat
    assert "pd.json_normalize" in code_flat

    code_raw = recipe.to_code({"url": "wss://echo.websocket.org", "flatten_json": False})
    assert "pd.DataFrame(records)" in code_raw
    assert "{'payload': msg}" in code_raw
