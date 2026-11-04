"""
WebSocket Listener Recipe (Ingestion & Orchestration -> Streaming & Change Data).

Listens to WebSocket streaming feeds (ws:// or wss://) and collects incoming messages into DataFrame micro-batches.
Supports subscription messages, authorization headers, timeout sampling windows,
JSON payload deserialization with auto-flattening, benchmark simulation fallback, and code generation.
"""

import os
import json
import time
from typing import Dict, Any, List, Optional, Tuple

try:
    import pandas as pd
except ImportError:
    pd = None

from backend.app.recipes.base.recipe import BaseRecipe, RecipePort
from backend.app.core.logging import logger
from backend.app.core.config import settings


class WebSocketListenerRecipe(BaseRecipe):
    recipe_id = "websocket_listener"
    name = "WebSocket Streaming Listener"
    version = "1.0.0"
    category = "ingestion"
    group = "Ingestion & Orchestration"
    subgroup = "Streaming & Change Data"
    description = "Listens to WebSocket streaming feeds and captures incoming messages into tabular micro-batches."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    inputs = [
        RecipePort(
            id="input",
            label="Trigger / Input Dataset",
            type="dataframe",
            required=False,
            max_connections=1,
            description="Optional upstream dataset for batch IDs or execution trigger"
        )
    ]

    outputs = [
        RecipePort(
            id="output",
            label="WebSocket Feed Dataset",
            type="dataframe",
            description="DataFrame containing polled WebSocket streaming messages"
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "title": "WebSocket URL (ws:// or wss://)",
                    "input_type": "text",
                    "default": "wss://stream.binance.com:9443/ws/btcusdt@trade",
                    "description": "Target WebSocket streaming endpoint (must start with 'ws://' or 'wss://')."
                },
                "subscribe_payload": {
                    "type": "string",
                    "title": "Subscription Message (JSON / Text)",
                    "input_type": "textarea",
                    "default": "",
                    "description": "Optional payload sent immediately upon connection (e.g. '{\"method\": \"SUBSCRIBE\", \"params\": [\"btcusdt@trade\"], \"id\": 1}')."
                },
                "headers": {
                    "type": "string",
                    "title": "Custom Headers (JSON)",
                    "input_type": "textarea",
                    "default": "{}",
                    "description": "Optional HTTP headers sent during WebSocket handshake (e.g. '{\"Authorization\": \"Bearer token\"}')."
                },
                "max_messages": {
                    "type": "integer",
                    "title": "Max Messages per Batch",
                    "input_type": "number",
                    "default": 100,
                    "description": "Maximum number of messages to capture before completing the batch (1 to 20,000)."
                },
                "timeout_seconds": {
                    "type": "integer",
                    "title": "Sampling Window (Seconds)",
                    "input_type": "number",
                    "default": 3,
                    "description": "Maximum time to keep the WebSocket open waiting for incoming events."
                },
                "flatten_json": {
                    "type": "boolean",
                    "title": "Flatten JSON Payloads",
                    "input_type": "checkbox",
                    "default": True,
                    "description": "Automatically unpack nested JSON object fields into top-level columns."
                },
                "credential_id": {
                    "type": "string",
                    "title": "Connected WebSocket Credential",
                    "input_type": "credential_dropdown",
                    "provider": "websocket",
                    "description": "Select stored WebSocket API credentials or tokens from platform settings."
                },
                "simulation_mode": {
                    "type": "boolean",
                    "title": "Simulation / Demo Mode",
                    "input_type": "checkbox",
                    "default": False,
                    "description": "When enabled or when WebSocket endpoint is offline, generates realistic market trade stream benchmark records."
                }
            },
            "required": ["url"]
        }

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        errors: List[str] = []
        url = str(config.get("url", "")).strip()

        if not config.get("simulation_mode"):
            if not url:
                errors.append("WebSocket URL is required.")
            elif not url.startswith(("ws://", "wss://")):
                errors.append("WebSocket URL must start with 'ws://' or 'wss://'.")

        raw_headers = config.get("headers")
        if raw_headers:
            try:
                parsed_h = json.loads(raw_headers) if isinstance(raw_headers, str) else raw_headers
                if not isinstance(parsed_h, dict):
                    errors.append("Headers must be a JSON object / dictionary.")
            except Exception as e:
                errors.append(f"Invalid JSON headers: {e}")

        max_msgs = config.get("max_messages")
        if max_msgs is not None:
            try:
                m_val = int(max_msgs)
                if m_val <= 0:
                    errors.append("Max messages must be greater than 0.")
            except (ValueError, TypeError):
                errors.append("Max messages must be an integer.")

        return errors

    def _generate_benchmark_data(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        limit = max(1, min(int(config.get("max_messages", 100)), 5000))
        url = str(config.get("url", "wss://stream.binance.com:9443/ws/btcusdt@trade"))
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT"]

        records: List[Dict[str, Any]] = []
        for i in range(limit):
            sym = symbols[i % len(symbols)]
            records.append({
                "ws_endpoint": url,
                "event_type": "trade",
                "event_time": 1774000000000 + (i * 450),
                "symbol": sym,
                "trade_id": 85000000 + i,
                "price": round(65400.0 + ((i * 12.5) % 1500.0), 2) if "BTC" in sym else round(3450.0 + ((i * 3.5) % 200.0), 2),
                "quantity": round(0.015 + ((i * 0.02) % 1.5), 4),
                "buyer_maker": True if i % 2 == 0 else False,
                "latency_ms": round(2.5 + ((i * 0.2) % 8.0), 2),
                "timestamp": f"2026-03-{((i % 28) + 1):02d}T16:{((i % 60)):02d}:00Z"
            })
        return records

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        start_time = time.time()
        max_messages = max(1, min(int(config.get("max_messages", 100)), 20000))
        timeout_sec = max(1, min(int(config.get("timeout_seconds", 3)), 30))
        flatten_json = bool(config.get("flatten_json", True))
        simulation_mode = bool(config.get("simulation_mode", False))

        url = str(config.get("url", "wss://stream.binance.com:9443/ws/btcusdt@trade")).strip()
        sub_payload = str(config.get("subscribe_payload", "")).strip()

        records: List[Dict[str, Any]] = []
        delivery_status = "SIMULATED"
        status_msg = ""

        # Live Execution Attempt with sync / async websockets
        if not simulation_mode and url.startswith(("ws://", "wss://")):
            try:
                import websocket  # websocket-client
                ws = websocket.create_connection(url, timeout=timeout_sec)
                if sub_payload:
                    ws.send(sub_payload)

                collected = 0
                t_end = time.time() + timeout_sec
                while collected < max_messages and time.time() < t_end:
                    try:
                        raw_msg = ws.recv()
                        if not raw_msg:
                            break
                        try:
                            parsed = json.loads(raw_msg)
                        except Exception:
                            parsed = {"raw_text": raw_msg}
                        records.append({"ws_url": url, **parsed})
                        collected += 1
                    except websocket.WebSocketTimeoutException:
                        break

                ws.close()
                delivery_status = "LIVE"
                status_msg = f"Captured {len(records)} live streaming messages from WebSocket feed '{url}'."
            except ImportError:
                status_msg = "websocket-client driver not installed. Using benchmark simulation fallback."
                logger.info(status_msg)
            except Exception as e:
                logger.warning(f"WebSocket connection failed: {e}")
                status_msg = f"WebSocket connection failed ({e}). Using benchmark simulation fallback."

        # Fallback to realistic benchmark data
        if not records:
            records = self._generate_benchmark_data(config)
            if not status_msg:
                status_msg = f"Loaded {len(records)} benchmark streaming messages for WebSocket feed '{url}'."

        if pd is not None and records:
            if flatten_json:
                df_out = pd.json_normalize(records)
            else:
                df_out = pd.DataFrame(records)
        elif pd is not None:
            df_out = pd.DataFrame()
        else:
            df_out = records

        duration_ms = int((time.time() - start_time) * 1000)
        col_count = len(df_out.columns) if hasattr(df_out, "columns") else (len(records[0].keys()) if records else 0)

        summary = {
            "title": "WebSocket Stream Summary",
            "message": status_msg,
            "status": delivery_status,
            "url": url,
            "messages_captured": len(records),
            "columns_count": col_count,
            "duration_ms": duration_ms
        }

        metrics = {
            "messages_captured": len(records),
            "columns_count": col_count,
            "duration_ms": duration_ms,
            "delivery_status": delivery_status
        }

        return {
            "dataframe": df_out,
            "output": df_out,
            "output_summary": summary,
            "metrics": metrics
        }

    def to_code(self, config: Dict[str, Any]) -> str:
        url = config.get("url", "wss://stream.binance.com:9443/ws/btcusdt@trade")
        max_msgs = config.get("max_messages", 100)

        return (
            f"# Generated WebSocket Real-Time Stream Listener Script\n"
            f"import websocket\n"
            f"import json\n"
            f"import pandas as pd\n\n"
            f"ws = websocket.create_connection('{url}')\n"
            f"records = []\n"
            f"for _ in range({max_msgs}):\n"
            f"    msg = ws.recv()\n"
            f"    records.append(json.loads(msg))\n"
            f"ws.close()\n"
            f"df = pd.json_normalize(records)\n"
            f"print(f'Captured {{len(df)}} messages from WebSocket stream.')\n"
        )
